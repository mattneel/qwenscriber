//! Byte-level BPE vocabulary and detokenization.
//!
//! Qwen3-ASR uses a Qwen2 tokenizer: GPT-2 style byte-level BPE over a
//! 151,643 entry vocabulary plus added special tokens. Token strings are stored
//! in the byte-to-unicode alphabet that keeps BPE merge rules printable, so
//! `Ġ` stands for a space and bytes above 0x7E become private code points.
//! Detokenization therefore has to undo that alphabet before the result is
//! valid UTF-8, which is what `detokenize` does.
//!
//! Decoding needs no merges, no regex, and no allocation: it is a table lookup
//! per code point followed by a byte append. Encoding text back to tokens needs
//! the pre-tokenization splitter and merge ranks, which arrive with the
//! container's merge table; this module deliberately exposes only decoding plus
//! the byte alphabet, so the runtime carries no dead weight for the streaming
//! path.

const std = @import("std");

/// The byte-to-unicode alphabet used by GPT-2 and every byte-level BPE
/// tokenizer since.
///
/// Printable bytes (0x21..=0x7E, 0xA1..=0xAC, 0xAE..=0xFF) map to themselves.
/// The remaining 68 bytes -- controls, space, DEL, and 0xAD -- map to code
/// points starting at 0x100, in ascending byte order, so the alphabet stays
/// printable and injective.
pub fn byteToCodepoint(byte: u8) u21 {
    if (isIdentityByte(byte)) return byte;
    var offset: u21 = 0;
    var candidate: u16 = 0;
    while (candidate < byte) : (candidate += 1) {
        if (!isIdentityByte(@intCast(candidate))) offset += 1;
    }
    return 0x100 + offset;
}

fn isIdentityByte(byte: u8) bool {
    return (byte >= 0x21 and byte <= 0x7E) or
        (byte >= 0xA1 and byte <= 0xAC) or
        (byte >= 0xAE and byte <= 0xFF);
}

/// Inverse of `byteToCodepoint`. Returns null for code points outside the
/// alphabet, which is how a corrupt vocabulary entry is detected.
pub fn codepointToByte(codepoint: u21) ?u8 {
    if (codepoint < 0x100) {
        const byte: u8 = @intCast(codepoint);
        return if (isIdentityByte(byte)) byte else null;
    }
    var offset = codepoint - 0x100;
    var candidate: u16 = 0;
    while (candidate < 256) : (candidate += 1) {
        const byte: u8 = @intCast(candidate);
        if (isIdentityByte(byte)) continue;
        if (offset == 0) return byte;
        offset -= 1;
    }
    return null;
}

pub const Error = error{
    /// A token string contained a code point outside the byte alphabet.
    InvalidTokenAlphabet,
    /// The token id is not present in the vocabulary.
    UnknownToken,
    /// The caller's output buffer is too small.
    OutputTooSmall,
};

/// Encodes raw bytes into the vocabulary's byte alphabet: the inverse of what
/// `appendToken` decodes.
///
/// A tokenizer table stores `\u0120` for a space and `\u010a` for a newline, so a
/// caller holding ordinary text ("assistant\n", "language ") has to encode it
/// before it can be compared against a token. Returns null when `out` cannot hold
/// the result.
pub fn encodeAlphabet(out: []u8, text: []const u8) ?usize {
    var length: usize = 0;
    for (text) |byte| {
        var codepoint_buffer: [4]u8 = undefined;
        const encoded = std.unicode.utf8Encode(byteToCodepoint(byte), &codepoint_buffer) catch
            return null;
        if (length + encoded > out.len) return null;
        @memcpy(out[length..][0..encoded], codepoint_buffer[0..encoded]);
        length += encoded;
    }
    return length;
}

/// A read-only view of a vocabulary: token id -> token string.
///
/// Both slices live in the model container, so a vocabulary costs no runtime
/// allocation and no parsing beyond the container's own index.
pub const TokenTable = struct {
    /// `offsets[id]..offsets[id + 1]` delimits token `id` inside `bytes`.
    /// Must have exactly `count + 1` entries.
    offsets: []const u32,
    bytes: []const u8,
    count: u32,

    pub fn validate(self: *const TokenTable) error{MalformedTable}!void {
        if (self.offsets.len != self.count + 1) return error.MalformedTable;
        var previous: u32 = 0;
        for (self.offsets) |offset| {
            if (offset < previous) return error.MalformedTable;
            if (offset > self.bytes.len) return error.MalformedTable;
            previous = offset;
        }
        return;
    }

    pub fn token(self: *const TokenTable, id: u32) Error![]const u8 {
        if (id >= self.count) return Error.UnknownToken;
        return self.bytes[self.offsets[id]..self.offsets[id + 1]];
    }
};

/// Appends the bytes of one token to `out`, undoing the byte alphabet.
///
/// Returns the number of bytes written.
pub fn appendToken(out: []u8, token_text: []const u8) Error!usize {
    var written: usize = 0;
    var view = std.unicode.Utf8View.init(token_text) catch return Error.InvalidTokenAlphabet;
    var iterator = view.iterator();
    while (iterator.nextCodepoint()) |codepoint| {
        const byte = codepointToByte(codepoint) orelse return Error.InvalidTokenAlphabet;
        if (written == out.len) return Error.OutputTooSmall;
        out[written] = byte;
        written += 1;
    }
    return written;
}

/// Decodes a token sequence into UTF-8 text.
///
/// Special tokens decode to their literal control strings, which is what the
/// reference tokenizer does when `skip_special_tokens=False`; callers that want
/// only the transcript filter them out first.
pub fn detokenize(table: *const TokenTable, ids: []const u32, out: []u8) Error!usize {
    var written: usize = 0;
    for (ids) |id| {
        const token_text = try table.token(id);
        written += try appendToken(out[written..], token_text);
    }
    return written;
}

/// Bytes a token sequence will occupy once decoded, before the byte alphabet is
/// undone (an upper bound: every decoded byte comes from one code point).
pub fn detokenizedLengthBound(table: *const TokenTable, ids: []const u32) Error!usize {
    var total: usize = 0;
    for (ids) |id| {
        const token_text = try table.token(id);
        total += std.unicode.utf8CountCodepoints(token_text) catch return Error.InvalidTokenAlphabet;
    }
    return total;
}

test "byte alphabet is injective and round trips" {
    var seen_offsets = std.mem.zeroes([256]bool);
    for (0..256) |value| {
        const byte: u8 = @intCast(value);
        const codepoint = byteToCodepoint(byte);
        const restored = codepointToByte(codepoint);
        try std.testing.expectEqual(byte, restored.?);
        // Distinct bytes must map to distinct code points, otherwise decoding
        // would be ambiguous.
        if (codepoint >= 0x100) {
            const offset: usize = codepoint - 0x100;
            try std.testing.expect(offset < 256);
            try std.testing.expect(!seen_offsets[offset]);
            seen_offsets[offset] = true;
        }
    }
    // All 68 offsets must be used exactly once.
    var used: usize = 0;
    for (seen_offsets) |seen| {
        if (seen) used += 1;
    }
    try std.testing.expectEqual(@as(usize, 68), used);
}

test "byte alphabet matches the published GPT-2 table" {
    // Spot checks against the alphabet every byte-level BPE implementation
    // uses. Printable bytes map to themselves...
    try std.testing.expectEqual(@as(u21, 0x21), byteToCodepoint('!'));
    try std.testing.expectEqual(@as(u21, 0x41), byteToCodepoint('A'));
    try std.testing.expectEqual(@as(u21, 0x7E), byteToCodepoint('~'));
    try std.testing.expectEqual(@as(u21, 0xA1), byteToCodepoint(0xA1));
    try std.testing.expectEqual(@as(u21, 0xAC), byteToCodepoint(0xAC));
    try std.testing.expectEqual(@as(u21, 0xAE), byteToCodepoint(0xAE));
    try std.testing.expectEqual(@as(u21, 0xFF), byteToCodepoint(0xFF));
    // ...and the 68 remaining bytes take ascending code points from 0x100.
    // 0x00 is the first of them, and space is the 33rd: 'Ā' and 'Ġ'.
    try std.testing.expectEqual(@as(u21, 0x100), byteToCodepoint(0x00));
    try std.testing.expectEqual(@as(u21, 0x101), byteToCodepoint(0x01));
    try std.testing.expectEqual(@as(u21, 0x11F), byteToCodepoint(0x1F));
    try std.testing.expectEqual(@as(u21, 0x120), byteToCodepoint(' '));
    // 0x7F (DEL) follows the 33 bytes below space.
    try std.testing.expectEqual(@as(u21, 0x121), byteToCodepoint(0x7F));
    // 0xA0 is the 67th excluded byte, and 0xAD the 68th and last.
    try std.testing.expectEqual(@as(u21, 0x142), byteToCodepoint(0xA0));
    try std.testing.expectEqual(@as(u21, 0x143), byteToCodepoint(0xAD));
}

test "code points outside the alphabet are rejected" {
    // Excluded bytes below 0x100 have no identity meaning.
    try std.testing.expectEqual(@as(?u8, null), codepointToByte(0x00));
    try std.testing.expectEqual(@as(?u8, null), codepointToByte(0x1F));
    try std.testing.expectEqual(@as(?u8, null), codepointToByte(0x7F));
    try std.testing.expectEqual(@as(?u8, null), codepointToByte(0xA0));
    try std.testing.expectEqual(@as(?u8, null), codepointToByte(0xAD));
    // 0x143 is the last code point the alphabet assigns.
    try std.testing.expectEqual(@as(?u8, 0xAD), codepointToByte(0x143));
    try std.testing.expectEqual(@as(?u8, null), codepointToByte(0x144));
    // Anything a real text tokenizer would never emit either.
    try std.testing.expectEqual(@as(?u8, null), codepointToByte(0x4E00));
    try std.testing.expectEqual(@as(?u8, 0x41), codepointToByte('A'));
}

test "a token decodes back to the bytes it was built from" {
    // Encode "Hello world" into the alphabet, then decode it.
    const source = "Hello, world";
    var encoded: [64]u8 = undefined;
    var encoded_len: usize = 0;
    for (source) |byte| {
        encoded_len += std.unicode.utf8Encode(byteToCodepoint(byte), encoded[encoded_len..]) catch unreachable;
    }
    var decoded: [64]u8 = undefined;
    const decoded_len = try appendToken(&decoded, encoded[0..encoded_len]);
    try std.testing.expectEqualStrings(source, decoded[0..decoded_len]);
}

test "multi-byte UTF-8 survives the round trip" {
    // The alphabet operates on bytes, so a CJK character becomes three
    // separately mapped bytes and must reassemble into valid UTF-8.
    const source = "你好世界";
    var encoded: [64]u8 = undefined;
    var encoded_len: usize = 0;
    for (source) |byte| {
        encoded_len += std.unicode.utf8Encode(byteToCodepoint(byte), encoded[encoded_len..]) catch unreachable;
    }
    var decoded: [64]u8 = undefined;
    const decoded_len = try appendToken(&decoded, encoded[0..encoded_len]);
    try std.testing.expectEqualStrings(source, decoded[0..decoded_len]);
    try std.testing.expect(std.unicode.utf8ValidateSlice(decoded[0..decoded_len]));
}

/// Builds a token table over token strings for tests.
fn testTable(strings: []const []const u8, offsets: []u32, storage: []u8) TokenTable {
    var cursor: usize = 0;
    for (strings, 0..) |text, index| {
        offsets[index] = @intCast(cursor);
        @memcpy(storage[cursor..][0..text.len], text);
        cursor += text.len;
    }
    offsets[strings.len] = @intCast(cursor);
    return .{ .offsets = offsets, .bytes = storage[0..cursor], .count = @intCast(strings.len) };
}

test "detokenize concatenates tokens and undoes the alphabet" {
    // "Hi" = 'H', 'i'; then a mapped space, then "there".
    var offsets: [5]u32 = undefined;
    var storage: [32]u8 = undefined;
    const table = testTable(
        &.{ "H", "i", "\xC4\xA0", "there" },
        &offsets,
        &storage,
    );
    try table.validate();

    var out: [32]u8 = undefined;
    const written = try detokenize(&table, &.{ 0, 1, 2, 3 }, &out);
    try std.testing.expectEqualStrings("Hi there", out[0..written]);
}

test "detokenize reports the size it needs" {
    var offsets: [3]u32 = undefined;
    var storage: [16]u8 = undefined;
    const table = testTable(&.{ "ab", "cd" }, &offsets, &storage);
    var small: [3]u8 = undefined;
    try std.testing.expectError(Error.OutputTooSmall, detokenize(&table, &.{ 0, 1 }, &small));
    try std.testing.expectEqual(@as(usize, 4), try detokenizedLengthBound(&table, &.{ 0, 1 }));
}

test "unknown token ids and malformed tables are rejected" {
    var offsets: [3]u32 = undefined;
    var storage: [16]u8 = undefined;
    const table = testTable(&.{ "ab", "cd" }, &offsets, &storage);
    var out: [16]u8 = undefined;
    try std.testing.expectError(Error.UnknownToken, detokenize(&table, &.{2}, &out));

    var bad_offsets: [3]u32 = .{ 0, 99, 4 };
    const bad = TokenTable{ .offsets = &bad_offsets, .bytes = storage[0..4], .count = 2 };
    try std.testing.expectError(error.MalformedTable, bad.validate());

    var short_offsets: [2]u32 = .{ 0, 4 };
    const short = TokenTable{ .offsets = &short_offsets, .bytes = storage[0..4], .count = 2 };
    try std.testing.expectError(error.MalformedTable, short.validate());
}

test "a token that is not in the alphabet is an error, not a silent drop" {
    var out: [8]u8 = undefined;
    // Control characters and CJK code points never appear in a byte-level BPE
    // vocabulary; seeing one means the table is corrupt, not that a byte should
    // be skipped.
    try std.testing.expectError(Error.InvalidTokenAlphabet, appendToken(&out, "\x00"));
    try std.testing.expectError(Error.InvalidTokenAlphabet, appendToken(&out, "\x7f"));
    try std.testing.expectError(Error.InvalidTokenAlphabet, appendToken(&out, "\u{4e00}"));
    // Ordinary printable ASCII is in the alphabet and decodes to itself.
    try std.testing.expectEqual(@as(usize, 2), try appendToken(&out, "ab"));
    try std.testing.expectEqualStrings("ab", out[0..2]);
}
