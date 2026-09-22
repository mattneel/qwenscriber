//! Writer for the `QWFIX001` container used to compare stages against the
//! reference implementation.
//!
//! The container is deliberately tiny: an eight-byte magic, `rank`, four
//! dimensions, then a row-major f32 payload. `tests/reference_check.zig` reads
//! the same format, and `tools/reference/*.py` writes it, so all three sides
//! agree on one layout instead of three ad hoc ones.
//!
//! Everything is stored as f32, including token ids: the released vocabulary is
//! far below 2^24, so ids round trip exactly and a second dtype field would buy
//! nothing.

const std = @import("std");
const Io = std.Io;

pub const magic = "QWFIX001";
pub const header_bytes: u32 = 28;

pub const Error = error{
    RankTooLarge,
    TooManyElements,
    WriteFailed,
};

/// Serialises `values` with shape `dims` and writes it to `path` in `dir`.
pub fn write(
    io: Io,
    dir: Io.Dir,
    path: []const u8,
    dims: []const u32,
    values: []const f32,
) !void {
    const layout = try encode(std.heap.page_allocator, dims, values);
    defer std.heap.page_allocator.free(layout);
    try dir.writeFile(io, .{ .sub_path = path, .data = layout });
}

/// Builds the container in memory. Caller owns the result.
pub fn encode(
    allocator: std.mem.Allocator,
    dims: []const u32,
    values: []const f32,
) Error![]u8 {
    if (dims.len > 4) return Error.RankTooLarge;
    var product: usize = 1;
    for (dims) |dim| {
        product = std.math.mul(usize, product, dim) catch return Error.TooManyElements;
    }
    if (product != values.len) return Error.TooManyElements;

    const total = header_bytes + values.len * @sizeOf(f32);
    const out = allocator.alloc(u8, total) catch return Error.TooManyElements;
    @memcpy(out[0..8], magic);
    std.mem.writeInt(u32, out[8..12], @intCast(dims.len), .little);
    var index: usize = 0;
    while (index < 4) : (index += 1) {
        const dim: u32 = if (index < dims.len) dims[index] else 1;
        std.mem.writeInt(u32, out[12 + index * 4 ..][0..4], dim, .little);
    }
    for (values, 0..) |value, position| {
        std.mem.writeInt(u32, out[header_bytes + position * 4 ..][0..4], @bitCast(value), .little);
    }
    return out;
}

test "the container round trips through the reader's layout" {
    const dims = [_]u32{ 2, 3 };
    const values = [_]f32{ 1.0, -2.0, 3.5, 0.0, 1e-8, -1e8 };
    const bytes = try encode(std.testing.allocator, &dims, &values);
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqual(@as(usize, header_bytes + 24), bytes.len);
    try std.testing.expectEqualStrings(magic, bytes[0..8]);
    try std.testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, bytes[8..12], .little));
    try std.testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, bytes[12..16], .little));
    try std.testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, bytes[16..20], .little));
    for (values, 0..) |value, index| {
        const bits = std.mem.readInt(u32, bytes[header_bytes + index * 4 ..][0..4], .little);
        try std.testing.expectEqual(value, @as(f32, @bitCast(bits)));
    }
}

test "mismatched shapes are refused" {
    const dims = [_]u32{ 2, 3 };
    const values = [_]f32{ 1.0, 2.0 };
    try std.testing.expectError(Error.TooManyElements, encode(std.testing.allocator, &dims, &values));
    const too_many_dims = [_]u32{ 1, 1, 1, 1, 1 };
    try std.testing.expectError(
        Error.RankTooLarge,
        encode(std.testing.allocator, &too_many_dims, &values),
    );
}
