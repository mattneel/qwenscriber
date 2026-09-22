//! Tensor shapes and descriptors.
//!
//! Shapes are at most four-dimensional because the only tensors above rank two
//! are the audio frontend's convolution kernels. Every other weight, activation,
//! and cache buffer is a matrix, and matrices are addressed as `rows x cols`
//! where `rows` is the leading (output) dimension: PyTorch stores `nn.Linear`
//! weights as `[out_features, in_features]`.
//!
//! `Shape` is larger than 16 bytes, so methods take `*const Shape`. That is not
//! only about copies: a slice taken from a by-value parameter points into the
//! callee's frame, and returning it is a use-after-return that the compiler does
//! not diagnose. Index-based accessors make that class of bug unrepresentable,
//! and `gpu`/`manifest` code that genuinely needs `dims[0..rank]` slices the
//! caller's own storage.

const std = @import("std");
const dtype = @import("dtype.zig");
const quant = @import("quant.zig");

pub const rank_max: u8 = 4;
pub const dims_max: u32 = 1 << 24;

pub const Error = quant.Error || error{
    RankTooLarge,
    DimensionTooLarge,
    DimensionZero,
    RankMismatch,
    ShapeMismatch,
    NotMatrix,
    ByteLengthMismatch,
    CountOverflow,
};

/// Up to four dimensions, ordered outermost first.
pub const Shape = struct {
    dims: [rank_max]u32 = .{ 1, 1, 1, 1 },
    rank: u8 = 0,

    pub fn vector(count: u32) Error!Shape {
        if (count == 0) return Error.DimensionZero;
        if (count > dims_max) return Error.DimensionTooLarge;
        return .{ .dims = .{ count, 1, 1, 1 }, .rank = 1 };
    }

    /// A matrix with `row_count` output features and `col_count` input features.
    pub fn matrix(row_count: u32, col_count: u32) Error!Shape {
        if (row_count == 0 or col_count == 0) return Error.DimensionZero;
        if (row_count > dims_max or col_count > dims_max) return Error.DimensionTooLarge;
        return .{ .dims = .{ row_count, col_count, 1, 1 }, .rank = 2 };
    }

    /// A convolution kernel in PyTorch's `[out, in, kh, kw]` order.
    pub fn conv2d(out_channels: u32, in_channels: u32, kernel_h: u32, kernel_w: u32) Error!Shape {
        if (out_channels == 0 or in_channels == 0 or kernel_h == 0 or kernel_w == 0) {
            return Error.DimensionZero;
        }
        if (out_channels > dims_max or in_channels > dims_max) return Error.DimensionTooLarge;
        return .{ .dims = .{ out_channels, in_channels, kernel_h, kernel_w }, .rank = 4 };
    }

    /// A shape from an explicit extent list, as recorded in a manifest.
    pub fn fromDims(extents: []const u32) Error!Shape {
        if (extents.len == 0) return Error.DimensionZero;
        if (extents.len > rank_max) return Error.RankTooLarge;
        var shape = Shape{};
        for (extents, 0..) |extent_value, index| {
            if (extent_value == 0) return Error.DimensionZero;
            if (extent_value > dims_max) return Error.DimensionTooLarge;
            shape.dims[index] = extent_value;
        }
        shape.rank = @intCast(extents.len);
        return shape;
    }

    pub fn extentCount(self: *const Shape) u8 {
        return self.rank;
    }

    /// Extent along one active dimension, counting from the outermost.
    pub fn extent(self: *const Shape, index: u8) Error!u32 {
        if (index >= self.rank) return Error.RankMismatch;
        return self.dims[index];
    }

    /// Product of every active extent.
    pub fn elementCount(self: *const Shape) Error!u64 {
        if (self.rank == 0) return Error.RankMismatch;
        var count: u64 = 1;
        var index: u8 = 0;
        while (index < self.rank) : (index += 1) {
            const product = std.math.mul(u64, count, self.dims[index]) catch {
                return Error.CountOverflow;
            };
            count = product;
        }
        return count;
    }

    pub fn isMatrix(self: *const Shape) bool {
        return self.rank == 2;
    }

    pub fn rows(self: *const Shape) Error!u32 {
        if (!self.isMatrix()) return Error.NotMatrix;
        return self.dims[0];
    }

    pub fn cols(self: *const Shape) Error!u32 {
        if (!self.isMatrix()) return Error.NotMatrix;
        return self.dims[1];
    }

    pub fn eql(self: *const Shape, other: *const Shape) bool {
        if (self.rank != other.rank) return false;
        var index: u8 = 0;
        while (index < self.rank) : (index += 1) {
            if (self.dims[index] != other.dims[index]) return false;
        }
        return true;
    }

    /// Validates that this shape can be stored in `format`.
    pub fn validateForFormat(self: *const Shape, format: dtype.Format) Error!void {
        if (!format.isKnown()) return Error.UnsupportedFormat;
        if (!format.isQuantized()) return;
        if (!self.isMatrix()) return Error.NotMatrix;
        if (self.dims[1] % quant.group_size != 0) return Error.RowNotGroupAligned;
    }

    /// Exact payload bytes for `format`, quantization padding included.
    pub fn byteLength(self: *const Shape, format: dtype.Format) Error!u64 {
        try self.validateForFormat(format);
        if (format.isQuantized()) {
            const layout = try quant.planeLayout(
                format,
                self.dims[0],
                self.dims[1],
                0,
                quant.tensor_alignment_bytes,
            );
            return layout.total_len_bytes;
        }
        const count = try self.elementCount();
        return count * format.elementSizeBytes();
    }
};

/// A tensor's location inside a container, as recorded in a manifest.
pub const Ref = struct {
    format: dtype.Format,
    shape: Shape,
    /// Offset from the start of the payload section of the owning shard.
    offset_bytes: u64,
    /// Exact payload length, quantization padding included.
    len_bytes: u64,

    pub fn validate(self: *const Ref) Error!void {
        try self.shape.validateForFormat(self.format);
        const expected = try self.shape.byteLength(self.format);
        if (expected != self.len_bytes) return Error.ByteLengthMismatch;
    }
};

test "matrix shapes report rows and columns" {
    const shape = try Shape.matrix(2048, 1024);
    try std.testing.expectEqual(@as(u8, 2), shape.rank);
    try std.testing.expectEqual(@as(u32, 2048), try shape.rows());
    try std.testing.expectEqual(@as(u32, 1024), try shape.cols());
    try std.testing.expectEqual(@as(u64, 2_097_152), try shape.elementCount());
}

test "shapes reject zero and oversized dimensions" {
    try std.testing.expectError(Error.DimensionZero, Shape.matrix(0, 8));
    try std.testing.expectError(Error.DimensionZero, Shape.matrix(8, 0));
    try std.testing.expectError(Error.DimensionTooLarge, Shape.matrix(dims_max + 1, 8));
    try std.testing.expectError(Error.RankTooLarge, Shape.fromDims(&.{ 1, 2, 3, 4, 5 }));
    try std.testing.expectError(Error.DimensionZero, Shape.fromDims(&.{}));
}

test "extents are reachable by index and bounds checked" {
    const shape = try Shape.conv2d(480, 1, 3, 3);
    try std.testing.expectEqual(@as(u8, 4), shape.extentCount());
    try std.testing.expectEqual(@as(u32, 480), try shape.extent(0));
    try std.testing.expectEqual(@as(u32, 1), try shape.extent(1));
    try std.testing.expectEqual(@as(u32, 3), try shape.extent(2));
    try std.testing.expectEqual(@as(u32, 3), try shape.extent(3));
    try std.testing.expectError(Error.RankMismatch, shape.extent(4));
}

test "convolution kernels are rank four and never quantized" {
    const shape = try Shape.conv2d(480, 480, 3, 3);
    try std.testing.expectEqual(@as(u8, 4), shape.rank);
    try std.testing.expectEqual(@as(u64, 480 * 480 * 9), try shape.elementCount());
    try std.testing.expectError(Error.NotMatrix, shape.rows());
    // Only quantized formats insist on a matrix; element formats are layout-free.
    try std.testing.expectError(Error.NotMatrix, shape.validateForFormat(.q4));
    try shape.validateForFormat(.f16);
    try std.testing.expectEqual(@as(u64, 480 * 480 * 9 * 2), try shape.byteLength(.f16));
}

test "quantized byte length matches the plane layout" {
    const shape = try Shape.matrix(896, 896);
    const layout = try quant.planeLayout(.q4, 896, 896, 0, quant.tensor_alignment_bytes);
    try std.testing.expectEqual(layout.total_len_bytes, try shape.byteLength(.q4));
    // 896*896 weights at 4.25 bits plus alignment padding.
    try std.testing.expect(try shape.byteLength(.q4) < 896 * 896);
}

test "quantization rejects row lengths that are not whole groups" {
    const shape = try Shape.matrix(480, 9);
    try std.testing.expectError(Error.RowNotGroupAligned, shape.validateForFormat(.q4));
    try std.testing.expectError(Error.RowNotGroupAligned, shape.byteLength(.q5));
}

test "tensor refs validate their recorded byte length" {
    const shape = try Shape.matrix(64, 64);
    const expected = try shape.byteLength(.q4);
    const good = Ref{ .format = .q4, .shape = shape, .offset_bytes = 0, .len_bytes = expected };
    try good.validate();

    const bad = Ref{ .format = .q4, .shape = shape, .offset_bytes = 0, .len_bytes = expected - 1 };
    try std.testing.expectError(Error.ByteLengthMismatch, bad.validate());
}

test "unknown formats are rejected rather than guessed" {
    const shape = try Shape.matrix(64, 64);
    const bogus: dtype.Format = @fromBackingInt(@intCast(200));
    try std.testing.expectError(Error.UnsupportedFormat, shape.validateForFormat(bogus));
}

test "shape equality compares extents and rank" {
    const matrix = try Shape.matrix(64, 64);
    const same = try Shape.matrix(64, 64);
    const transposed = try Shape.matrix(64, 32);
    const vector = try Shape.vector(64);
    try std.testing.expect(matrix.eql(&same));
    try std.testing.expect(!matrix.eql(&transposed));
    try std.testing.expect(!matrix.eql(&vector));
}
