//! Writing the records this repository emits: `manifest.json` and the
//! transcription metrics.
//!
//! Both are hand-written rather than produced by a serializer, because both must
//! be byte-identical for the same input: the manifest doubles as a cache key, and
//! two metric records must diff cleanly. A serializer that omits a field or
//! reorders one would take that property away.
//!
//! The whole point of the shared module is the one mistake a hand-written JSON
//! writer makes: a trailing comma before a closing brace. Every field goes
//! through a helper that is told whether it is the last of its object, so the
//! comma is decided in one place instead of at every call site.

const std = @import("std");

const assert = std.debug.assert;

pub const top_indent = "  ";
pub const nested_indent = "    ";

pub fn textField(
    writer: *std.Io.Writer,
    indent: []const u8,
    key: []const u8,
    text: []const u8,
    last: bool,
) !void {
    try writeKey(writer, indent, key);
    try writeString(writer, text);
    try writeTail(writer, last);
}

pub fn numberField(
    writer: *std.Io.Writer,
    indent: []const u8,
    key: []const u8,
    number: anytype,
    last: bool,
) !void {
    try writeKey(writer, indent, key);
    try writer.print("{d}", .{number});
    try writeTail(writer, last);
}

/// A field whose text is already JSON: a boolean, or a number whose formatting
/// the caller fixed.
pub fn rawField(
    writer: *std.Io.Writer,
    indent: []const u8,
    key: []const u8,
    raw: []const u8,
    last: bool,
) !void {
    try writeKey(writer, indent, key);
    try writer.writeAll(raw);
    try writeTail(writer, last);
}

/// A number formatted the one way floats are written here: six decimals, so the
/// bytes do not depend on how the platform prints floats and a diff between two
/// records shows only what changed.
pub fn floatField(
    writer: *std.Io.Writer,
    indent: []const u8,
    key: []const u8,
    value: f64,
    last: bool,
) !void {
    var buffer: [32]u8 = undefined;
    const text = try std.fmt.bufPrint(&buffer, "{d:.6}", .{value});
    try rawField(writer, indent, key, text, last);
}

pub fn writeTail(writer: *std.Io.Writer, last: bool) !void {
    if (last) {
        try writer.writeByte('\n');
    } else {
        try writer.writeAll(",\n");
    }
}

pub fn writeKey(writer: *std.Io.Writer, indent: []const u8, key: []const u8) !void {
    try writer.writeAll(indent);
    try writer.writeByte('"');
    try writer.writeAll(key);
    try writer.writeAll("\": ");
}

/// A JSON string with every character that must be escaped escaped. Model
/// identifiers and tool versions reach this, and a model directory may be named
/// by anyone.
pub fn writeString(writer: *std.Io.Writer, text: []const u8) !void {
    assert(text.len <= 1 << 20);
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

test "strings are escaped so the record stays parseable" {
    var buffer: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try writeString(&writer, "a\"b\\c\nd\te\x01f");
    try std.testing.expectEqualStrings("\"a\\\"b\\\\c\\nd\\te\\u0001f\"", buffer[0..writer.end]);
}

test "the last field carries no comma and the others do" {
    var buffer: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try writer.writeAll("{\n");
    try numberField(&writer, top_indent, "a", 1, false);
    try textField(&writer, top_indent, "b", "two", true);
    try writer.writeAll("}");
    try std.testing.expectEqualStrings("{\n  \"a\": 1,\n  \"b\": \"two\"\n}", buffer[0..writer.end]);
}

test "floats are written with a fixed number of decimals" {
    var buffer: [64]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try floatField(&writer, top_indent, "bits", 5.32, true);
    // The same value printed on two platforms must produce the same bytes.
    try std.testing.expectEqualStrings("  \"bits\": 5.320000\n", buffer[0..writer.end]);
}
