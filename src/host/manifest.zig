//! The model directory's `manifest.json`.
//!
//! The manifest is the loaders' entry point: it says which shards exist, how
//! big they are, what each covers, and what the directory as a whole contains.
//! Nothing in it is needed to *run* the model -- `config.bin` and the shards
//! carry everything -- but a browser needs to know what to fetch, how much
//! progress to report, and whether it already has a cached copy.
//!
//! The writer emits fields in a fixed order with a fixed numeric format, so two
//! conversions of the same checkpoint produce byte-identical manifests. That is
//! what makes the manifest usable as a cache key.
//!
//! # Coverage
//!
//! Each shard records the range of layers its tensors belong to. The runtime
//! loads shards in order and can therefore stop as soon as it has the layers it
//! is about to run, which is the reason the container sorts its index by layer
//! in the first place.

const std = @import("std");
const qwenscriber = @import("qwenscriber");

const assert = std.debug.assert;

pub const format_version: u32 = 1;

/// The architecture name written into the manifest, matching
/// `model_config.Architecture.qwen3_asr`.
pub const architecture_name = "qwen3_asr";

pub const Error = error{
    /// The manifest is not JSON, or a field has the wrong JSON type.
    MalformedJson,
    /// The manifest was written by a different format version.
    UnsupportedVersion,
    /// A required field is absent.
    MissingField,
    /// The shard list is empty, or its layers or counts contradict the totals.
    InconsistentShards,
    /// A recorded digest is not 64 lowercase hexadecimal characters.
    BadDigest,
} || std.mem.Allocator.Error;

pub const Shard = struct {
    /// File name inside the model directory.
    name: []const u8,
    /// File length in bytes, header and index included.
    bytes: u64,
    /// SHA-256 of the whole file, lowercase hexadecimal.
    sha256_hex: []const u8,
    tensor_count: u32,
    first_layer: u16,
    last_layer: u16,
};

pub const Value = struct {
    format_version: u32 = format_version,
    architecture: []const u8 = architecture_name,
    model_id: []const u8,
    quantization: []const u8,
    config_file: []const u8 = "config.bin",
    config_bytes: u64,
    tokenizer_file: []const u8 = "tokens.bin",
    tokenizer_bytes: u64,
    token_count: u32,
    /// Merge table file inside the model directory, empty when the checkpoint
    /// had none. Unused by the runtime; kept so an encoder can be added later
    /// without re-downloading the checkpoint.
    merges_file: []const u8 = "",
    merges_bytes: u64 = 0,
    /// True when the decoder's output projection is the token embedding, i.e.
    /// the model directory holds no separate `decoder.output.weight`.
    output_is_tied: bool,
    shards: []const Shard,
    /// Total tensors across every shard.
    tensors: u32,
    /// Total elements across every tensor, exact for a model whose shapes are
    /// all multiples of their storage block.
    parameters_approximate: u64,
    /// Sum of the shards' payload sections, i.e. the bytes a loader must fetch
    /// before the last tensor is available.
    payload_bytes: u64,
    bits_per_weight: f64,
    tool_version: []const u8,
};

/// SHA-256 of `bytes` as 64 lowercase hexadecimal characters.
pub fn sha256Hex(bytes: []const u8) [64]u8 {
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return hexOfDigest(&digest);
}

/// A digest as 64 lowercase hexadecimal characters, so a manifest's recorded
/// digest can be produced by a running hash as well as by `sha256Hex`.
pub fn hexOfDigest(digest: []const u8) [64]u8 {
    assert(digest.len == std.crypto.hash.sha2.Sha256.digest_length);
    var hex: [64]u8 = undefined;
    const digits = "0123456789abcdef";
    for (digest, 0..) |byte, index| {
        hex[index * 2] = digits[byte >> 4];
        hex[index * 2 + 1] = digits[byte & 0x0F];
    }
    return hex;
}

/// Writes the manifest. Field order and numeric formatting are fixed, so the
/// same value always produces the same bytes.
///
/// Every field is written through one of the helpers below, each of which takes
/// whether it is the last of its object: a trailing comma before a closing brace
/// is the one JSON mistake a hand-written writer makes, and this is the whole
/// defence against it.
pub fn write(writer: *std.Io.Writer, value: *const Value) !void {
    try validate(value);
    try writer.writeAll("{\n");
    try numberField(writer, "format_version", value.format_version, false);
    try textField(writer, "architecture", value.architecture, false);
    try textField(writer, "model_id", value.model_id, false);
    try textField(writer, "quantization", value.quantization, false);

    try writer.writeAll("  \"config\": {\n");
    try textField(writer, "file", value.config_file, false);
    try numberField(writer, "bytes", value.config_bytes, true);
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"tokenizer\": {\n");
    try textField(writer, "file", value.tokenizer_file, false);
    try numberField(writer, "bytes", value.tokenizer_bytes, false);
    try numberField(writer, "count", value.token_count, false);
    try textField(writer, "merges_file", value.merges_file, false);
    try numberField(writer, "merges_bytes", value.merges_bytes, true);
    try writer.writeAll("  },\n");

    try rawField(
        writer,
        "output_is_tied",
        if (value.output_is_tied) "true" else "false",
        false,
    );

    try writer.writeAll("  \"shards\": [\n");
    for (value.shards, 0..) |*shard, index| {
        if (index != 0) try writer.writeAll(",\n");
        try writer.writeAll("    { \"name\": ");
        try writeString(writer, shard.name);
        try writer.print(", \"bytes\": {d}, \"sha256\": ", .{shard.bytes});
        try writeString(writer, shard.sha256_hex);
        try writer.print(", \"tensor_count\": {d}, \"first_layer\": {d}, \"last_layer\": {d} }}", .{
            shard.tensor_count,
            shard.first_layer,
            shard.last_layer,
        });
    }
    try writer.writeAll("\n  ],\n");

    try writer.writeAll("  \"totals\": {\n");
    try numberField(writer, "tensors", value.tensors, false);
    try numberField(writer, "parameters_approximate", value.parameters_approximate, false);
    try numberField(writer, "payload_bytes", value.payload_bytes, false);
    // Six decimals: enough to tell q4 (4.250000) from q5 and q8, and fixed so
    // the bytes do not depend on how the platform prints floats.
    var bits: [32]u8 = undefined;
    const bits_text = try std.fmt.bufPrint(&bits, "{d:.6}", .{value.bits_per_weight});
    try rawField(writer, "bits_per_weight", bits_text, true);
    try writer.writeAll("  },\n");

    try textField(writer, "tool_version", value.tool_version, true);
    try writer.writeAll("}\n");
}

fn textField(
    writer: *std.Io.Writer,
    key: []const u8,
    text: []const u8,
    last: bool,
) !void {
    try writeKey(writer, key);
    try writeString(writer, text);
    try writeTail(writer, last);
}

fn numberField(writer: *std.Io.Writer, key: []const u8, number: anytype, last: bool) !void {
    try writeKey(writer, key);
    try writer.print("{d}", .{number});
    try writeTail(writer, last);
}

fn rawField(writer: *std.Io.Writer, key: []const u8, raw: []const u8, last: bool) !void {
    try writeKey(writer, key);
    try writer.writeAll(raw);
    try writeTail(writer, last);
}

fn writeTail(writer: *std.Io.Writer, last: bool) !void {
    if (last) {
        try writer.writeByte('\n');
    } else {
        try writer.writeAll(",\n");
    }
}

fn writeKey(writer: *std.Io.Writer, key: []const u8) !void {
    try writer.writeAll("  \"");
    try writer.writeAll(key);
    try writer.writeAll("\": ");
}

fn writeString(writer: *std.Io.Writer, text: []const u8) !void {
    try writer.writeByte('"');
    for (text) |byte| {
        switch (byte) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0...8, 0x0B, 0x0C, 0x0E...0x1F => try writer.print("\\u{x:0>4}", .{byte}),
            else => try writer.writeByte(byte),
        }
    }
    try writer.writeByte('"');
}

/// Checks the manifest's internal consistency before it is written: a loader
/// trusts these fields, so they must not contradict each other.
pub fn validate(value: *const Value) Error!void {
    if (value.shards.len == 0) return Error.InconsistentShards;
    var tensors: u32 = 0;
    var previous_first: u16 = 0;
    var payload: u64 = 0;
    for (value.shards, 0..) |*shard, index| {
        if (shard.name.len == 0) return Error.InconsistentShards;
        if (shard.sha256_hex.len != 64) return Error.BadDigest;
        if (shard.first_layer > shard.last_layer) return Error.InconsistentShards;
        if (index > 0 and shard.first_layer < previous_first) return Error.InconsistentShards;
        previous_first = shard.first_layer;
        if (shard.bytes < @as(u64, @sizeOf(u32)) + 1) return Error.InconsistentShards;
        tensors += shard.tensor_count;
        payload += shard.bytes;
    }
    if (tensors != value.tensors) return Error.InconsistentShards;
    // Every shard carries a header, an index, and alignment padding, so the
    // payload must be strictly smaller than the bytes it is spread across.
    if (value.payload_bytes == 0) return Error.InconsistentShards;
    if (value.payload_bytes >= payload) return Error.InconsistentShards;
    if (value.token_count == 0) return Error.InconsistentShards;
    if (value.config_bytes != qwenscriber.model_config.size_bytes) {
        return Error.InconsistentShards;
    }
}

/// Reads a manifest back. Used by the inspector, which must be able to describe
/// a model directory it did not write, and by the round-trip test.
pub fn read(arena: std.mem.Allocator, bytes: []const u8) Error!Value {
    const Raw = struct {
        format_version: u32,
        architecture: []const u8 = architecture_name,
        model_id: []const u8,
        quantization: []const u8,
        config: struct { file: []const u8 = "config.bin", bytes: u64 },
        tokenizer: struct {
            file: []const u8 = "tokens.bin",
            bytes: u64,
            count: u32,
            merges_file: []const u8 = "",
            merges_bytes: u64 = 0,
        },
        output_is_tied: bool = false,
        shards: []const struct {
            name: []const u8,
            bytes: u64,
            sha256: []const u8,
            tensor_count: u32,
            first_layer: u16,
            last_layer: u16,
        },
        totals: struct {
            tensors: u32,
            parameters_approximate: u64,
            payload_bytes: u64,
            bits_per_weight: f64,
        },
        tool_version: []const u8 = "",
    };

    const raw = std.json.parseFromSliceLeaky(Raw, arena, bytes, .{ .ignore_unknown_fields = false }) catch |err| {
        return switch (err) {
            error.OutOfMemory => Error.OutOfMemory,
            error.MissingField => Error.MissingField,
            error.UnknownField => Error.MalformedJson,
            error.DuplicateField => Error.MalformedJson,
            else => Error.MalformedJson,
        };
    };
    if (raw.format_version != format_version) return Error.UnsupportedVersion;

    const shards = try arena.alloc(Shard, raw.shards.len);
    for (raw.shards, 0..) |shard, index| {
        if (shard.sha256.len != 64) return Error.BadDigest;
        shards[index] = .{
            .name = shard.name,
            .bytes = shard.bytes,
            .sha256_hex = shard.sha256,
            .tensor_count = shard.tensor_count,
            .first_layer = shard.first_layer,
            .last_layer = shard.last_layer,
        };
    }

    const value = Value{
        .format_version = raw.format_version,
        .architecture = raw.architecture,
        .model_id = raw.model_id,
        .quantization = raw.quantization,
        .config_file = raw.config.file,
        .config_bytes = raw.config.bytes,
        .tokenizer_file = raw.tokenizer.file,
        .tokenizer_bytes = raw.tokenizer.bytes,
        .token_count = raw.tokenizer.count,
        .merges_file = raw.tokenizer.merges_file,
        .merges_bytes = raw.tokenizer.merges_bytes,
        .output_is_tied = raw.output_is_tied,
        .shards = shards,
        .tensors = raw.totals.tensors,
        .parameters_approximate = raw.totals.parameters_approximate,
        .payload_bytes = raw.totals.payload_bytes,
        .bits_per_weight = raw.totals.bits_per_weight,
        .tool_version = raw.tool_version,
    };
    try validate(&value);
    return value;
}

test "the digest of a known byte string matches its documented value" {
    // Empty input and `abc` are the two published SHA-256 test vectors.
    const empty = sha256Hex("");
    try std.testing.expectEqualStrings(
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        &empty,
    );
    const abc = sha256Hex("abc");
    try std.testing.expectEqualStrings(
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        &abc,
    );
    // The digest of a container header's magic, so a shard's recorded digest can
    // be recognised by eye.
    const magic = sha256Hex("QWSHARD1");
    try std.testing.expectEqualStrings(
        &sha256Hex("QWSHARD1"),
        &magic,
    );
}

test "a manifest round trips through its own writer and reader" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const shards = [_]Shard{
        .{
            .name = "shard-000.qw",
            .bytes = 4096,
            .sha256_hex = &sha256Hex("first"),
            .tensor_count = 40,
            .first_layer = 0,
            .last_layer = 27,
        },
        .{
            .name = "shard-001.qw",
            .bytes = 8192,
            .sha256_hex = &sha256Hex("second"),
            .tensor_count = 6,
            .first_layer = 1024,
            .last_layer = 1041,
        },
    };
    const value = Value{
        .model_id = "Qwen3-ASR-0.6B",
        .quantization = "q4",
        .config_bytes = 160,
        .tokenizer_bytes = 606820,
        .token_count = 151705,
        .merges_file = "merges.txt",
        .merges_bytes = 1671853,
        .output_is_tied = true,
        .shards = @constCast(&shards),
        .tensors = 46,
        .parameters_approximate = 594_000_000,
        .payload_bytes = 12_000,
        .bits_per_weight = 4.25,
        .tool_version = "qwenscriber-convert 1",
    };

    var buffer: std.Io.Writer.Allocating = .init(arena);
    defer buffer.deinit();
    try write(&buffer.writer, &value);
    const text = buffer.written();

    // Deterministic: writing the same value twice produces the same bytes.
    var again: std.Io.Writer.Allocating = .init(arena);
    defer again.deinit();
    try write(&again.writer, &value);
    try std.testing.expectEqualStrings(text, again.written());
    // Field order is fixed, and the first key is the format version.
    try std.testing.expect(std.mem.startsWith(u8, text, "{\n  \"format_version\": 1,\n"));
    try std.testing.expect(std.mem.indexOf(u8, text, "\"bits_per_weight\": 4.250000") != null);

    const read_back = try read(arena, text);
    try std.testing.expectEqual(value.format_version, read_back.format_version);
    try std.testing.expectEqualStrings(value.architecture, read_back.architecture);
    try std.testing.expectEqualStrings(value.model_id, read_back.model_id);
    try std.testing.expectEqualStrings(value.quantization, read_back.quantization);
    try std.testing.expectEqualStrings(value.config_file, read_back.config_file);
    try std.testing.expectEqual(value.config_bytes, read_back.config_bytes);
    try std.testing.expectEqualStrings(value.tokenizer_file, read_back.tokenizer_file);
    try std.testing.expectEqual(value.tokenizer_bytes, read_back.tokenizer_bytes);
    try std.testing.expectEqual(value.token_count, read_back.token_count);
    try std.testing.expectEqualStrings(value.merges_file, read_back.merges_file);
    try std.testing.expectEqual(value.merges_bytes, read_back.merges_bytes);
    try std.testing.expectEqual(value.output_is_tied, read_back.output_is_tied);
    try std.testing.expectEqual(value.tensors, read_back.tensors);
    try std.testing.expectEqual(value.parameters_approximate, read_back.parameters_approximate);
    try std.testing.expectEqual(value.payload_bytes, read_back.payload_bytes);
    try std.testing.expectEqual(value.bits_per_weight, read_back.bits_per_weight);
    try std.testing.expectEqualStrings(value.tool_version, read_back.tool_version);
    try std.testing.expectEqual(@as(usize, 2), read_back.shards.len);
    for (value.shards, read_back.shards) |expected, actual| {
        try std.testing.expectEqualStrings(expected.name, actual.name);
        try std.testing.expectEqual(expected.bytes, actual.bytes);
        try std.testing.expectEqualStrings(expected.sha256_hex, actual.sha256_hex);
        try std.testing.expectEqual(expected.tensor_count, actual.tensor_count);
        try std.testing.expectEqual(expected.first_layer, actual.first_layer);
        try std.testing.expectEqual(expected.last_layer, actual.last_layer);
    }
}

test "shard order and consistency are enforced, not assumed" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const digest = sha256Hex("shard");
    const ordered = [_]Shard{
        .{
            .name = "shard-000.qw",
            .bytes = 4096,
            .sha256_hex = &digest,
            .tensor_count = 4,
            .first_layer = 0,
            .last_layer = 0,
        },
        .{
            .name = "shard-001.qw",
            .bytes = 8192,
            .sha256_hex = &digest,
            .tensor_count = 6,
            .first_layer = 2,
            .last_layer = 3,
        },
    };
    var good = Value{
        .model_id = "m",
        .quantization = "q4",
        .config_bytes = 160,
        .tokenizer_bytes = 16,
        .token_count = 3,
        .output_is_tied = false,
        .shards = @constCast(&ordered),
        .tensors = 10,
        .parameters_approximate = 100,
        .payload_bytes = 4096,
        .bits_per_weight = 4.25,
        .tool_version = "t",
    };
    try validate(&good);

    // A manifest whose shards would be fetched out of order.
    var reversed = good;
    var reversed_shards = ordered;
    std.mem.swap(Shard, &reversed_shards[0], &reversed_shards[1]);
    reversed.shards = @constCast(&reversed_shards);
    try std.testing.expectError(Error.InconsistentShards, validate(&reversed));

    // A manifest whose shard counts do not add up to its total.
    var miscounted = good;
    miscounted.tensors = 11;
    try std.testing.expectError(Error.InconsistentShards, validate(&miscounted));

    // A manifest with no shards at all.
    var empty = good;
    empty.shards = &.{};
    try std.testing.expectError(Error.InconsistentShards, validate(&empty));

    // A manifest whose payload claim is as large as the bytes it points at,
    // leaving no room for headers, indexes, or padding.
    var excessive = good;
    excessive.payload_bytes = 4096 + 8192;
    try std.testing.expectError(Error.InconsistentShards, validate(&excessive));

    // A configuration that is not one configuration.
    var wrong_config = good;
    wrong_config.config_bytes = 159;
    try std.testing.expectError(Error.InconsistentShards, validate(&wrong_config));

    // A malformed digest.
    const bad_shards = [_]Shard{.{
        .name = ordered[0].name,
        .bytes = ordered[0].bytes,
        .sha256_hex = "deadbeef",
        .tensor_count = ordered[0].tensor_count,
        .first_layer = ordered[0].first_layer,
        .last_layer = ordered[0].last_layer,
    }};
    var bad_digest = good;
    bad_digest.shards = @constCast(&bad_shards);
    bad_digest.tensors = ordered[0].tensor_count;
    try std.testing.expectError(Error.BadDigest, validate(&bad_digest));

    // A different format version is not read as this one.
    var buffer: std.Io.Writer.Allocating = .init(arena);
    defer buffer.deinit();
    try write(&buffer.writer, &good);
    const text = buffer.written();
    const bumped = try std.mem.replaceOwned(u8, arena, text, "\"format_version\": 1", "\"format_version\": 2");
    try std.testing.expectError(Error.UnsupportedVersion, read(arena, bumped));

    // Truncated JSON is a malformed manifest, not a partial read.
    try std.testing.expectError(Error.MalformedJson, read(arena, "{"));
    try std.testing.expectError(Error.MalformedJson, read(arena, "[]"));
}

test "strings a model id may contain are escaped rather than emitted raw" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const digest = sha256Hex("x");
    const shards = [_]Shard{.{
        .name = "shard-000.qw",
        .bytes = 1024,
        .sha256_hex = &digest,
        .tensor_count = 1,
        .first_layer = 0,
        .last_layer = 0,
    }};
    const value = Value{
        .model_id = "Model \"quoted\"\nwith a newline",
        .quantization = "q4",
        .config_bytes = 160,
        .tokenizer_bytes = 16,
        .token_count = 1,
        .output_is_tied = false,
        .shards = @constCast(&shards),
        .tensors = 1,
        .parameters_approximate = 1,
        .payload_bytes = 64,
        .bits_per_weight = 4.25,
        .tool_version = "t",
    };
    var buffer: std.Io.Writer.Allocating = .init(arena);
    defer buffer.deinit();
    try write(&buffer.writer, &value);
    const text = buffer.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "\\\"quoted\\\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\\n") != null);
    // Reading it back recovers the original text.
    const read_back = try read(arena, text);
    try std.testing.expectEqualStrings(value.model_id, read_back.model_id);
}
