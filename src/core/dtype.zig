//! Storage formats for tensor payloads.
//!
//! The numeric values in `Format` are part of the on-disk model container format
//! and the WASM ABI, so they must never be renumbered. New formats are appended.
//!
//! Two families exist:
//!
//!   * `f32`/`f16`/`bf16` — plain element formats, used for normalization
//!     weights, biases, and the convolutional frontend.
//!   * `q4`/`q5`/`q8` — group-quantized formats, used for every large matrix
//!     weight. Their byte layout lives in `quant.zig`; this file only records
//!     identity, block size, and total bytes per block.

const std = @import("std");
const quant = @import("quant.zig");

pub const Format = enum(u8) {
    f32 = 0,
    f16 = 1,
    bf16 = 2,
    q4 = 16,
    q5 = 17,
    q8 = 18,
    _,

    /// Every format this build understands, in a fixed order. `parse` and the
    /// tests iterate this table, so adding a format without teaching `name` and
    /// `parse` about it is caught by `format names round trip`.
    pub const all = [_]Format{ .f32, .f16, .bf16, .q4, .q5, .q8 };

    /// Human readable name used by tooling and manifest dumps.
    pub fn name(self: Format) []const u8 {
        return switch (self) {
            .f32 => "f32",
            .f16 => "f16",
            .bf16 => "bf16",
            .q4 => "q4",
            .q5 => "q5",
            .q8 => "q8",
            else => "unknown",
        };
    }

    /// Parses a format name as written in manifests and CLI flags. An explicit
    /// table rather than reflection: the names are user-facing text, and this
    /// keeps them next to the values they must never disagree with.
    pub fn parse(text: []const u8) ?Format {
        const entries = [_]struct { text: []const u8, format: Format }{
            .{ .text = "f32", .format = .f32 },
            .{ .text = "f16", .format = .f16 },
            .{ .text = "bf16", .format = .bf16 },
            .{ .text = "q4", .format = .q4 },
            .{ .text = "q5", .format = .q5 },
            .{ .text = "q8", .format = .q8 },
        };
        for (entries) |entry| {
            if (std.mem.eql(u8, text, entry.text)) return entry.format;
        }
        return null;
    }

    /// True when the payload is stored as quantized groups rather than elements.
    pub fn isQuantized(self: Format) bool {
        return switch (self) {
            .q4, .q5, .q8 => true,
            else => false,
        };
    }

    /// True when the payload is a plain element array.
    pub fn isElement(self: Format) bool {
        return !self.isQuantized();
    }

    pub fn isKnown(self: Format) bool {
        return switch (self) {
            .f32, .f16, .bf16, .q4, .q5, .q8 => true,
            else => false,
        };
    }

    /// Bytes per stored element for element formats; 0 for quantized formats.
    pub fn elementSizeBytes(self: Format) u32 {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            else => 0,
        };
    }

    /// Weights covered by one quantization block; 1 for element formats.
    pub fn blockSize(self: Format) u32 {
        return switch (self) {
            .q4, .q5, .q8 => quant.group_size,
            else => 1,
        };
    }

    /// Total bytes occupied by one quantization block, including its scale.
    /// For element formats this is the element size.
    pub fn blockBytes(self: Format) u32 {
        return switch (self) {
            .q4 => quant.q4_scale_bytes_per_group + quant.q4_data_bytes_per_group,
            .q5 => quant.q5_scale_bytes_per_group + quant.q5_data_bytes_per_group,
            .q8 => quant.q8_scale_bytes_per_group + quant.q8_data_bytes_per_group,
            else => self.elementSizeBytes(),
        };
    }
};

pub const Item = enum(u8) {
    f32 = 0,
    f16 = 1,
    u32 = 2,
    i32 = 3,
};

pub fn itemName(item: Item) []const u8 {
    return switch (item) {
        .f32 => "f32",
        .f16 => "f16",
        .u32 => "u32",
        .i32 => "i32",
    };
}

pub fn itemSizeBytes(item: Item) u32 {
    return switch (item) {
        .f32, .u32, .i32 => 4,
        .f16 => 2,
    };
}

test "format identifiers are stable" {
    // These values are written into model containers and read back by browsers.
    try std.testing.expectEqual(@as(u8, 0), @backingInt(Format.f32));
    try std.testing.expectEqual(@as(u8, 1), @backingInt(Format.f16));
    try std.testing.expectEqual(@as(u8, 2), @backingInt(Format.bf16));
    try std.testing.expectEqual(@as(u8, 16), @backingInt(Format.q4));
    try std.testing.expectEqual(@as(u8, 17), @backingInt(Format.q5));
    try std.testing.expectEqual(@as(u8, 18), @backingInt(Format.q8));
}

test "format classification and sizes" {
    try std.testing.expect(Format.q4.isQuantized());
    try std.testing.expect(!Format.f16.isQuantized());
    try std.testing.expect(Format.f16.isElement());
    try std.testing.expectEqual(@as(u32, 4), Format.f32.elementSizeBytes());
    try std.testing.expectEqual(@as(u32, 2), Format.f16.elementSizeBytes());
    try std.testing.expectEqual(@as(u32, 0), Format.q4.elementSizeBytes());
    try std.testing.expectEqual(@as(u32, 64), Format.q4.blockSize());
    try std.testing.expectEqual(@as(u32, 1), Format.f32.blockSize());
    try std.testing.expectEqual(@as(u32, 4), Format.f32.blockBytes());
}

test "format names round trip" {
    for (Format.all) |format| {
        try std.testing.expect(format.isKnown());
        const parsed = Format.parse(format.name());
        try std.testing.expectEqual(format, parsed orelse return error.TestUnexpectedResult);
    }
    try std.testing.expectEqual(@as(?Format, null), Format.parse("q6"));
    try std.testing.expectEqual(@as(?Format, null), Format.parse(""));
    try std.testing.expectEqual(@as(?Format, null), Format.parse("Q4"));
}

test "unknown format values are not silently accepted" {
    const bogus: Format = @fromBackingInt(@intCast(99));
    try std.testing.expect(!bogus.isKnown());
    try std.testing.expectEqualStrings("unknown", bogus.name());
}
