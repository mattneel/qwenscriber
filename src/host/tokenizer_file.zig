//! Reads a checkpoint's tokenizer files into the token table the runtime
//! consumes.
//!
//! Two checkpoint layouts exist and both must convert:
//!
//!     vocab.json + merges.txt + tokenizer_config.json   (the vLLM export)
//!     tokenizer.json                                    (the native export)
//!
//! `vocab.json` is a flat `token -> id` map and `tokenizer.json` carries the
//! same map under `model.vocab` plus `model.merges`. Both are byte-level BPE
//! vocabularies in the byte-to-unicode alphabet: the strings stored here are
//! exactly the strings the file contains, because `tokenizer.appendToken` is
//! what undoes that alphabet, once, at decode time.
//!
//! The table the runtime reads is `count + 1` little-endian `u32` offsets
//! followed by the concatenated token strings, where `count` is `max_id + 1`
//! over the union of the base vocabulary and the added tokens. Ids are the
//! runtime's array index, so a gap would make some token resolve to a
//! neighbouring token's text: a gap is an error here rather than a silent
//! substitution later.
//!
//! The merge table is copied through untouched. Nothing in the runtime reads it
//! yet -- decoding needs only the vocabulary -- but a checkpoint run is
//! expensive and re-downloading 1.8 GiB to add an encoder would be worse, so the
//! converter writes it while it has it.

const std = @import("std");
const qwenscriber = @import("qwenscriber");

const assert = std.debug.assert;
const tokenizer = qwenscriber.tokenizer;

/// Largest token id this reader accepts, matching `model_config.vocab_max`.
/// The released vocabulary tops out at 151704.
pub const token_id_max: u32 = 1 << 21;

/// Largest tokenizer file this reader accepts. `tokenizer.json` for the native
/// export is 11 MiB, most of it a regex-free merge list.
pub const file_bytes_max: u64 = 64 << 20;

pub const Error = error{
    /// The directory has neither `vocab.json` nor `tokenizer.json`.
    MissingTokenizerFiles,
    /// A vocabulary file exists but has no vocabulary in it.
    MissingVocabulary,
    /// A tokenizer file is not JSON, or a vocabulary entry is not a string to
    /// integer mapping.
    MalformedJson,
    /// Two token strings claim the same id.
    DuplicateTokenId,
    /// The union of vocabulary and added tokens is missing an id below the
    /// maximum. The diagnostic names the first one.
    TokenIdGap,
    /// An id is above `token_id_max`.
    TokenIdTooLarge,
    /// A token string is empty, which cannot happen in a byte-level BPE
    /// vocabulary and would make its offsets indistinguishable from the
    /// previous token's.
    EmptyToken,
    /// Two files describe the same id with different text.
    ConflictingAddedTokens,
} || std.mem.Allocator.Error;

/// What went wrong, in enough detail to fix a checkpoint by hand.
pub const Diagnostics = struct {
    /// First id below the maximum with no token, when the failure was a gap.
    gap_id: u32 = 0,
    /// Number of missing ids.
    gap_count: u32 = 0,
    /// The id claimed twice, when the failure was a duplicate.
    duplicate_id: u32 = 0,
};

/// A token the tokenizer adds on top of the base vocabulary. The name is the
/// literal text, e.g. `<|im_start|>`, which is what a configuration refers to
/// it by.
pub const TokenSpecials = struct {
    name: []const u8,
    id: u32,
};

pub const Data = struct {
    /// `count + 1` entries; `offsets[id]..offsets[id + 1]` delimits token `id`.
    offsets: []const u32,
    /// Concatenated token strings, in id order.
    bytes: []const u8,
    count: u32,
    /// Every added token, including ones the tokenizer does not flag as
    /// `special`: `<asr_text>` is an added token but not a special one, and the
    /// runtime still needs its id.
    specials: []const TokenSpecials,
    /// The merge table as text, verbatim when the checkpoint ships
    /// `merges.txt`.
    merges: ?[]const u8,
    /// File the vocabulary came from, for the manifest.
    source_name: []const u8,

    pub fn table(self: *const Data) tokenizer.TokenTable {
        assert(self.offsets.len == self.count + 1);
        return .{ .offsets = self.offsets, .bytes = self.bytes, .count = self.count };
    }

    /// Bytes the runtime's table occupies: the leading count, the offsets, then
    /// the strings.
    pub fn tableBytes(self: *const Data) u64 {
        return @as(u64, self.count + 1) * 4 + 4 + self.bytes.len;
    }

    /// Looks a token up by its literal text, which is how a configuration
    /// refers to a special token it does not have an id for.
    pub fn idOf(self: *const Data, name: []const u8) ?u32 {
        for (self.specials) |entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry.id;
        }
        return null;
    }
};

/// The characters the checkpoint's files are read into. Every read is bounded
/// so a corrupt or hostile file cannot exhaust memory before it is rejected.
const ReadLimit = struct {
    fn of(bytes: u64) std.Io.Limit {
        return .limited64(@min(bytes, file_bytes_max));
    }
};

/// Reads a checkpoint directory's tokenizer. `dir` must be the checkpoint
/// directory.
pub fn read(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    diagnostics: *Diagnostics,
) !Data {
    const vocab_json = try readOptional(arena, io, dir, "vocab.json");
    const tokenizer_json = try readOptional(arena, io, dir, "tokenizer.json");
    if (vocab_json == null and tokenizer_json == null) return Error.MissingTokenizerFiles;

    var added: std.ArrayList(TokenSpecials) = .empty;
    if (try readOptional(arena, io, dir, "tokenizer_config.json")) |bytes| {
        try collectAddedFromConfig(arena, &added, bytes);
    }
    if (tokenizer_json) |bytes| {
        try collectAddedFromTokenizer(arena, &added, bytes);
    }
    const specials = try added.toOwnedSlice(arena);

    var vocabulary: ?[]const Entry = null;
    if (vocab_json) |bytes| vocabulary = try readVocabFile(arena, bytes);
    if (tokenizer_json) |bytes| {
        if (try readTokenizerVocabulary(arena, bytes)) |entries| vocabulary = entries;
    }
    const entries = vocabulary orelse return Error.MissingVocabulary;

    const merges = try readMerges(arena, io, dir, tokenizer_json);
    return build(arena, entries, specials, merges, vocab_json != null, diagnostics);
}

/// One `token -> id` pair, as parsed from whichever file held the vocabulary.
const Entry = struct { name: []const u8, id: u32 };

fn readOptional(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    name: []const u8,
) !?[]u8 {
    const bytes = dir.readFileAlloc(io, name, arena, ReadLimit.of(file_bytes_max)) catch |err| {
        switch (err) {
            error.FileNotFound => return null,
            else => return err,
        }
    };
    std.debug.assert(bytes.len <= file_bytes_max);
    return bytes;
}

/// `vocab.json` is a flat object of token strings to ids.
fn readVocabFile(arena: std.mem.Allocator, bytes: []const u8) Error![]const Entry {
    const root = parseJson(arena, bytes) catch |err| switch (err) {
        error.OutOfMemory => return Error.OutOfMemory,
        else => return Error.MalformedJson,
    };
    const object = switch (root) {
        .object => |object| object,
        else => return Error.MalformedJson,
    };
    return entriesFromObject(arena, object);
}

/// `tokenizer.json` carries the same vocabulary under `model.vocab`, and is
/// absent for a checkpoint that ships `vocab.json`.
fn readTokenizerVocabulary(arena: std.mem.Allocator, bytes: []const u8) Error!?[]const Entry {
    const root = parseJson(arena, bytes) catch |err| switch (err) {
        error.OutOfMemory => return Error.OutOfMemory,
        else => return Error.MalformedJson,
    };
    const top = switch (root) {
        .object => |object| object,
        else => return Error.MalformedJson,
    };
    const model = switch (top.get("model") orelse return null) {
        .object => |object| object,
        else => return Error.MalformedJson,
    };
    const vocab = switch (model.get("vocab") orelse return null) {
        .object => |object| object,
        else => return Error.MalformedJson,
    };
    return try entriesFromObject(arena, vocab);
}

fn entriesFromObject(
    arena: std.mem.Allocator,
    object: std.json.ObjectMap,
) Error![]const Entry {
    const entries = try arena.alloc(Entry, object.count());
    var iterator = object.iterator();
    var count: usize = 0;
    while (iterator.next()) |pair| {
        const id = switch (pair.value_ptr.*) {
            .integer => |number| number,
            else => return Error.MalformedJson,
        };
        if (id < 0) return Error.MalformedJson;
        if (id > token_id_max) return Error.TokenIdTooLarge;
        const name = pair.key_ptr.*;
        if (name.len == 0) return Error.EmptyToken;
        assert(count < entries.len);
        entries[count] = .{ .name = name, .id = @intCast(id) };
        count += 1;
    }
    assert(count == entries.len);
    assert(entries.len == object.count());
    return entries;
}

/// `tokenizer_config.json` holds added tokens as a string-keyed map of ids to
/// token descriptions.
fn collectAddedFromConfig(
    arena: std.mem.Allocator,
    added: *std.ArrayList(TokenSpecials),
    bytes: []const u8,
) Error!void {
    const root = parseJson(arena, bytes) catch |err| switch (err) {
        error.OutOfMemory => return Error.OutOfMemory,
        else => return Error.MalformedJson,
    };
    const top = switch (root) {
        .object => |object| object,
        else => return Error.MalformedJson,
    };
    const decoder = switch (top.get("added_tokens_decoder") orelse return) {
        .object => |object| object,
        else => return Error.MalformedJson,
    };
    var iterator = decoder.iterator();
    while (iterator.next()) |pair| {
        const id = std.fmt.parseInt(u32, pair.key_ptr.*, 10) catch return Error.MalformedJson;
        const description = switch (pair.value_ptr.*) {
            .object => |object| object,
            else => return Error.MalformedJson,
        };
        const content = switch (description.get("content") orelse return Error.MalformedJson) {
            .string => |text| text,
            else => return Error.MalformedJson,
        };
        try appendAdded(added, arena, .{ .name = content, .id = id });
    }
}

/// `tokenizer.json` holds added tokens as an array of descriptions. Both
/// spellings describe the same tokens; a checkpoint usually has one or the
/// other, and a checkpoint with both must not gain a duplicate entry.
fn collectAddedFromTokenizer(
    arena: std.mem.Allocator,
    added: *std.ArrayList(TokenSpecials),
    bytes: []const u8,
) Error!void {
    const root = parseJson(arena, bytes) catch |err| switch (err) {
        error.OutOfMemory => return Error.OutOfMemory,
        else => return Error.MalformedJson,
    };
    const top = switch (root) {
        .object => |object| object,
        else => return Error.MalformedJson,
    };
    const tokens = switch (top.get("added_tokens") orelse return) {
        .array => |array| array.items,
        else => return Error.MalformedJson,
    };
    for (tokens) |token| {
        const description = switch (token) {
            .object => |object| object,
            else => return Error.MalformedJson,
        };
        const id = switch (description.get("id") orelse return Error.MalformedJson) {
            .integer => |number| number,
            else => return Error.MalformedJson,
        };
        if (id < 0 or id > token_id_max) return Error.TokenIdTooLarge;
        const content = switch (description.get("content") orelse return Error.MalformedJson) {
            .string => |text| text,
            else => return Error.MalformedJson,
        };
        try appendAdded(added, arena, .{ .name = content, .id = @intCast(id) });
    }
}

fn appendAdded(
    added: *std.ArrayList(TokenSpecials),
    arena: std.mem.Allocator,
    entry: TokenSpecials,
) Error!void {
    assert(entry.id <= token_id_max);
    for (added.items) |existing| {
        if (existing.id != entry.id) continue;
        // The same token described twice: keep the first description, but the
        // two files must not disagree about what id 151669 is.
        if (!std.mem.eql(u8, existing.name, entry.name)) {
            return Error.ConflictingAddedTokens;
        }
        return;
    }
    try added.append(arena, entry);
}

/// The merge table as text: the checkpoint's own `merges.txt` when it has one,
/// otherwise the merge list inside `tokenizer.json` rendered one merge per
/// line, which is the same format.
fn readMerges(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    tokenizer_json: ?[]const u8,
) !?[]const u8 {
    if (try readOptional(arena, io, dir, "merges.txt")) |bytes| return bytes;
    const json = tokenizer_json orelse return null;
    const root = parseJson(arena, json) catch |err| switch (err) {
        error.OutOfMemory => return Error.OutOfMemory,
        else => return Error.MalformedJson,
    };
    const top = switch (root) {
        .object => |object| object,
        else => return null,
    };
    const model = switch (top.get("model") orelse return null) {
        .object => |object| object,
        else => return null,
    };
    const merges = switch (model.get("merges") orelse return null) {
        .array => |array| array.items,
        else => return null,
    };
    return try renderMerges(arena, merges);
}

fn renderMerges(arena: std.mem.Allocator, merges: []const std.json.Value) ![]const u8 {
    var text: std.Io.Writer.Allocating = .init(arena);
    defer text.deinit();
    const out = &text.writer;
    // The version line a `merges.txt` carries. Writing it makes a table rendered
    // from `tokenizer.json` byte-identical to the one the other export ships for
    // the same merges, so the two spellings of a checkpoint convert to the same
    // directory.
    try out.writeAll("#version: 0.2\n");
    for (merges) |merge| {
        switch (merge) {
            // The current format is a pair of token strings.
            .array => |pair| {
                if (pair.items.len != 2) return Error.MalformedJson;
                const left = switch (pair.items[0]) {
                    .string => |string| string,
                    else => return Error.MalformedJson,
                };
                const right = switch (pair.items[1]) {
                    .string => |string| string,
                    else => return Error.MalformedJson,
                };
                try out.print("{s} {s}\n", .{ left, right });
            },
            // Older files spell a merge as one space-separated string.
            .string => |string| try out.print("{s}\n", .{string}),
            else => return Error.MalformedJson,
        }
    }
    return text.toOwnedSlice();
}

/// Assembles the offsets and the concatenated strings, rejecting any id that
/// no token covers.
fn build(
    arena: std.mem.Allocator,
    entries: []const Entry,
    specials: []const TokenSpecials,
    merges: ?[]const u8,
    from_vocab_file: bool,
    diagnostics: *Diagnostics,
) Error!Data {
    var id_max: u32 = 0;
    var id_max_set = false;
    for (entries) |entry| {
        if (!id_max_set or entry.id > id_max) id_max = entry.id;
        id_max_set = true;
    }
    for (specials) |entry| {
        if (!id_max_set or entry.id > id_max) id_max = entry.id;
        id_max_set = true;
    }
    if (!id_max_set) return Error.MissingVocabulary;
    const count = id_max + 1;

    const slots = try arena.alloc(?[]const u8, count);
    @memset(slots, null);
    for (entries) |entry| {
        if (slots[entry.id] != null) {
            diagnostics.duplicate_id = entry.id;
            return Error.DuplicateTokenId;
        }
        slots[entry.id] = entry.name;
    }
    for (specials) |entry| {
        if (slots[entry.id] != null) {
            diagnostics.duplicate_id = entry.id;
            return Error.DuplicateTokenId;
        }
        slots[entry.id] = entry.name;
    }

    var bytes_len: u64 = 0;
    var gaps: u32 = 0;
    for (slots, 0..) |slot, id| {
        if (slot) |name| {
            bytes_len += name.len;
        } else {
            if (gaps == 0) diagnostics.gap_id = @intCast(id);
            gaps += 1;
        }
    }
    diagnostics.gap_count = gaps;
    if (gaps != 0) return Error.TokenIdGap;

    const offsets = try arena.alloc(u32, count + 1);
    const text = try arena.alloc(u8, @intCast(bytes_len));
    var written: u32 = 0;
    offsets[0] = 0;
    for (slots, 0..) |slot, id| {
        const name = slot.?;
        assert(written + name.len <= text.len);
        @memcpy(text[written..][0..name.len], name);
        written += @intCast(name.len);
        offsets[id + 1] = written;
    }
    assert(written == text.len);

    return .{
        .offsets = offsets,
        .bytes = text,
        .count = count,
        .specials = specials,
        .merges = merges,
        .source_name = if (from_vocab_file) "vocab.json" else "tokenizer.json",
    };
}

/// Parses with the strict default duplicate-field behavior: a vocabulary that
/// defines one token twice must not silently keep one of the two.
fn parseJson(arena: std.mem.Allocator, bytes: []const u8) !std.json.Value {
    return std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{});
}

/// Writes the runtime's table: a little-endian `count`, then `count + 1`
/// little-endian offsets, then the concatenated token strings. The offsets are
/// relative to the start of the string block, so the reader can index them
/// directly.
///
/// The leading count is not redundant with the manifest: the table is a
/// standalone artifact, and the native reader (`src/cli/transcribe.zig`) has only
/// this file to go on. Writing it only in the manifest is exactly the kind of
/// converter/reader drift that a format version is supposed to prevent.
pub fn writeTable(writer: *std.Io.Writer, data: *const Data) !void {
    assert(data.offsets.len == data.count + 1);
    var count_buffer: [4]u8 = undefined;
    std.mem.writeInt(u32, &count_buffer, data.count, .little);
    try writer.writeAll(&count_buffer);
    for (data.offsets) |offset| {
        var buffer: [4]u8 = undefined;
        std.mem.writeInt(u32, &buffer, offset, .little);
        try writer.writeAll(&buffer);
    }
    try writer.writeAll(data.bytes);
}

fn writeCheckpointFile(dir: std.Io.Dir, io: std.Io, name: []const u8, bytes: []const u8) !void {
    try dir.writeFile(io, .{ .sub_path = name, .data = bytes, .flags = .{ .truncate = true } });
}

test "a vocabulary and its added tokens become one contiguous table" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeCheckpointFile(tmp.dir, io, "vocab.json", "{\"a\":0,\"b\":1,\"c\":2}");
    try writeCheckpointFile(tmp.dir, io, "merges.txt", "#version: 0.2\na b\nb c\n");
    try writeCheckpointFile(tmp.dir, io, "tokenizer_config.json",
        \\{"added_tokens_decoder":{
        \\  "3": {"content":"<|im_end|>","special":true},
        \\  "4": {"content":"<asr_text>","special":false}}}
    );

    var diagnostics: Diagnostics = .{};
    const data = try read(arena, io, tmp.dir, &diagnostics);
    try std.testing.expectEqualStrings("vocab.json", data.source_name);
    try std.testing.expectEqual(@as(u32, 5), data.count);
    try std.testing.expectEqual(@as(usize, 6), data.offsets.len);
    try std.testing.expectEqual(@as(u32, 0), data.offsets[0]);
    try std.testing.expectEqual(@as(u32, 1), data.offsets[1]);
    try std.testing.expectEqual(@as(u32, 3), data.offsets[3]);
    try std.testing.expectEqual(@as(u32, 13), data.offsets[4]);
    try std.testing.expectEqual(@as(u32, 23), data.offsets[5]);
    try std.testing.expectEqualStrings("abc<|im_end|><asr_text>", data.bytes);
    try std.testing.expectEqual(@as(u64, 4 + 6 * 4 + 23), data.tableBytes());
    try std.testing.expectEqual(@as(usize, 0), diagnostics.gap_count);

    const table = data.table();
    try table.validate();
    try std.testing.expectEqualStrings("a", try table.token(0));
    try std.testing.expectEqualStrings("<asr_text>", try table.token(4));
    try std.testing.expectError(error.UnknownToken, table.token(5));

    try std.testing.expectEqual(@as(u32, 3), data.idOf("<|im_end|>").?);
    // `<asr_text>` is added but not flagged special, and the runtime still
    // needs its id, so the list is "added tokens", not "special tokens".
    try std.testing.expectEqual(@as(u32, 4), data.idOf("<asr_text>").?);
    try std.testing.expect(data.idOf("<|im_start|>") == null);
    try std.testing.expectEqualStrings("#version: 0.2\na b\nb c\n", data.merges.?);

    // The table format round trips through the writer.
    var written: std.Io.Writer.Allocating = .init(arena);
    defer written.deinit();
    try writeTable(&written.writer, &data);
    const bytes = written.written();
    try std.testing.expectEqual(@as(usize, 4 + 6 * 4 + 23), bytes.len);
    try std.testing.expectEqual(@as(u32, 5), std.mem.readInt(u32, bytes[0..4], .little));
    try std.testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, bytes[16..20], .little));
    try std.testing.expectEqualStrings("abc<|im_end|><asr_text>", bytes[28..]);
}

test "a vocabulary gap or duplicate id is an error rather than a silent fill" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Id 1 has no token: token 2 would otherwise decode as token 1's text.
    try writeCheckpointFile(tmp.dir, io, "vocab.json", "{\"a\":0,\"c\":2}");
    var diagnostics: Diagnostics = .{};
    try std.testing.expectError(Error.TokenIdGap, read(arena, io, tmp.dir, &diagnostics));
    try std.testing.expectEqual(@as(u32, 1), diagnostics.gap_id);
    try std.testing.expectEqual(@as(u32, 1), diagnostics.gap_count);

    // Two token strings for one id.
    try writeCheckpointFile(tmp.dir, io, "vocab.json", "{\"a\":0,\"b\":0,\"c\":2}");
    diagnostics = .{};
    try std.testing.expectError(Error.DuplicateTokenId, read(arena, io, tmp.dir, &diagnostics));
    try std.testing.expectEqual(@as(u32, 0), diagnostics.duplicate_id);

    // A token string that is not a byte-level BPE token.
    try writeCheckpointFile(tmp.dir, io, "vocab.json", "{\"a\":0,\"\":1}");
    diagnostics = .{};
    try std.testing.expectError(Error.EmptyToken, read(arena, io, tmp.dir, &diagnostics));

    try writeCheckpointFile(tmp.dir, io, "vocab.json", "{\"a\":0,\"b\":-1}");
    diagnostics = .{};
    try std.testing.expectError(Error.MalformedJson, read(arena, io, tmp.dir, &diagnostics));

    // A vocabulary file that is not an object.
    try writeCheckpointFile(tmp.dir, io, "vocab.json", "[1, 2]");
    diagnostics = .{};
    try std.testing.expectError(Error.MalformedJson, read(arena, io, tmp.dir, &diagnostics));
}

test "a directory with no tokenizer files is reported as such" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var diagnostics: Diagnostics = .{};
    try std.testing.expectError(
        Error.MissingTokenizerFiles,
        read(arena, std.testing.io, tmp.dir, &diagnostics),
    );
}

test "a tokenizer.json checkpoint yields the same table and a rendered merge list" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeCheckpointFile(tmp.dir, io, "tokenizer.json",
        \\{"added_tokens":[
        \\  {"id":3,"content":"<|im_end|>","special":true},
        \\  {"id":4,"content":"<asr_text>","special":false}],
        \\ "model":{"vocab":{"a":0,"b":1,"c":2},
        \\  "merges":[["a","b"],["b","c"]]}}
    );

    var diagnostics: Diagnostics = .{};
    const data = try read(arena, io, tmp.dir, &diagnostics);
    try std.testing.expectEqualStrings("tokenizer.json", data.source_name);
    try std.testing.expectEqual(@as(u32, 5), data.count);
    try std.testing.expectEqualStrings("abc<|im_end|><asr_text>", data.bytes);
    try std.testing.expectEqual(@as(u32, 3), data.idOf("<|im_end|>").?);
    // The merge list renders as the same text `merges.txt` would carry,
    // version line included.
    try std.testing.expectEqualStrings("#version: 0.2\na b\nb c\n", data.merges.?);
    try data.table().validate();
    try std.testing.expectEqual(@as(usize, 0), diagnostics.gap_count);
}

test "the tokenizer files agree with each other or the disagreement is reported" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeCheckpointFile(tmp.dir, io, "vocab.json", "{\"a\":0,\"b\":1}");
    try writeCheckpointFile(tmp.dir, io, "tokenizer.json",
        \\{"added_tokens":[{"id":2,"content":"<|im_end|>","special":true}],
        \\ "model":{"vocab":{"a":0,"b":1}}}
    );
    try writeCheckpointFile(tmp.dir, io, "tokenizer_config.json",
        \\{"added_tokens_decoder":{"2":{"content":"<|im_end|>","special":true}}}
    );

    var diagnostics: Diagnostics = .{};
    const data = try read(arena, io, tmp.dir, &diagnostics);
    // The duplicate description of the same token collapses into one entry.
    try std.testing.expectEqual(@as(usize, 1), data.specials.len);
    try std.testing.expectEqual(@as(u32, 3), data.count);
    try std.testing.expectEqualStrings("ab<|im_end|>", data.bytes);

    // Two different tokens claiming one id: the checkpoints disagree, so there
    // is no safe table to build.
    try writeCheckpointFile(tmp.dir, io, "tokenizer_config.json",
        \\{"added_tokens_decoder":{"2":{"content":"<|other|>","special":true}}}
    );
    diagnostics = .{};
    try std.testing.expectError(
        Error.ConflictingAddedTokens,
        read(arena, io, tmp.dir, &diagnostics),
    );
}
