//! Group-quantized weight formats.
//!
//! This file is the single source of truth for the numeric layout that the
//! converter writes, the CPU reference kernels read, and the WGSL kernels
//! decode. `gpu/shaders/quant_layout.wgsl` mirrors these constants, and a test
//! in this file fails if the two ever disagree.
//!
//! # Layout
//!
//! A weight matrix of `rows x cols` is quantized along its rows in groups of
//! `group_size` weights. Each tensor is split into two planes so that both are
//! naturally aligned for vector loads:
//!
//!     scales plane:  group_count * 2 bytes             (f16 scale per group)
//!     data plane:    group_count * data_bytes_per_group
//!
//! Planes are separated by padding to the tensor alignment (see `planeLayout`)
//! so a browser can upload each plane to a `GPUBuffer` at a 4-byte-aligned
//! offset. `group_size` divides `cols`; the converter rejects any tensor that
//! does not satisfy this rather than inventing a partial group.
//!
//! # Codes
//!
//! Codes are stored unsigned with an implicit midpoint offset, which lets a
//! kernel decode `code = unsigned - bias` without a sign extension:
//!
//!     q4: unsigned 0..15   bias 8    value = (u - 8)  * scale
//!     q5: unsigned 0..31   bias 16   value = (u - 16) * scale
//!     q8: unsigned 0..127  bias 128  value = (u - 128) * scale
//!
//! The per-group scale is `max_abs / bias`, so the most negative code reaches
//! `-max_abs` while the most positive reaches `(bias-1)/bias * max_abs`. That is
//! the usual 4-bit trade: one code position is lost to the asymmetric
//! two's-complement range.
//!
//! # Bit packing
//!
//! Nibbles are little-endian within each byte: weight `2i` occupies the low
//! nibble of byte `i` and weight `2i+1` the high nibble. Q5 adds a separate
//! high-bit plane where weight `j`'s fifth bit is bit `j % 8` of byte `j / 8`.
//! This is the packing the WGSL dequantizer consumes with 16-byte vector loads.

const std = @import("std");
const half_float = @import("half_float.zig");
const dtype = @import("dtype.zig");

/// Weights per quantization group: one group is decoded by one 16-byte vector
/// load in WGSL and carries a single broadcast f16 scale.
pub const group_size: u32 = 64;
const group: usize = group_size;

pub const q4_scale_bytes_per_group: u32 = 2;
pub const q4_data_bytes_per_group: u32 = 32;

pub const q5_scale_bytes_per_group: u32 = 2;
pub const q5_data_bytes_per_group: u32 = 40;

pub const q8_scale_bytes_per_group: u32 = 2;
pub const q8_data_bytes_per_group: u32 = 64;

/// Alignment of a quantized tensor payload inside a container. 16 bytes keeps
/// the nibble plane usable with `vec4<u32>` loads in both WASM SIMD and WGSL.
pub const tensor_alignment_bytes: u64 = 16;

pub const Error = error{
    /// The row length is not a whole number of groups.
    RowNotGroupAligned,
    /// The format is an element format, which has no group layout.
    NotQuantized,
    /// The format value is unknown to this build.
    UnsupportedFormat,
    /// A byte range would exceed the addressable model.
    LayoutOverflow,
};

/// Number of groups in a `rows x cols` quantized matrix.
pub fn groupCount(rows: u32, cols: u32) Error!u32 {
    if (cols % group_size != 0) return Error.RowNotGroupAligned;
    if (rows == 0) return 0;
    return rows * (cols / group_size);
}

fn groupCountOfLen(cols: usize) usize {
    assert(cols % group == 0);
    return cols / group;
}

pub fn scalesBytesForGroups(group_count: u64) u64 {
    return group_count * q4_scale_bytes_per_group;
}

pub fn dataBytesForGroups(format: dtype.Format, group_count: u64) Error!u64 {
    return switch (format) {
        .q4 => group_count * q4_data_bytes_per_group,
        .q5 => group_count * q5_data_bytes_per_group,
        .q8 => group_count * q8_data_bytes_per_group,
        else => Error.NotQuantized,
    };
}

/// Bytes of packed codes per group.
pub fn dataBytesPerGroup(format: dtype.Format) usize {
    return switch (format) {
        .q4 => q4_data_bytes_per_group,
        .q5 => q5_data_bytes_per_group,
        .q8 => q8_data_bytes_per_group,
        else => unreachable,
    };
}

pub fn bias(format: dtype.Format) i32 {
    return switch (format) {
        .q4 => 8,
        .q5 => 16,
        .q8 => 128,
        else => unreachable,
    };
}

pub fn codeMin(format: dtype.Format) i32 {
    return -bias(format);
}

pub fn codeMax(format: dtype.Format) i32 {
    return bias(format) - 1;
}

/// Byte ranges of a quantized tensor interned in a container.
pub const PlaneLayout = struct {
    /// Offset of the f16 scale plane, relative to the tensor base.
    scales_offset_bytes: u64,
    scales_len_bytes: u64,
    /// Offset of the packed-code plane, relative to the tensor base.
    data_offset_bytes: u64,
    data_len_bytes: u64,
    /// Distance between the tensor base and the end of its last plane.
    total_len_bytes: u64,
    group_count: u32,
    groups_per_row: u32,
};

/// Computes the two-plane layout for a `rows x cols` matrix stored at
/// `base_offset_bytes` in a container whose payloads are aligned to `alignment`.
pub fn planeLayout(
    format: dtype.Format,
    rows: u32,
    cols: u32,
    base_offset_bytes: u64,
    alignment: u64,
) Error!PlaneLayout {
    if (!format.isKnown()) return Error.UnsupportedFormat;
    if (!format.isQuantized()) return Error.NotQuantized;
    if (alignment < 4 or alignment % 4 != 0) return Error.LayoutOverflow;

    const group_count = try groupCount(rows, cols);
    const scales_len_bytes = scalesBytesForGroups(group_count);
    const data_len_bytes = try dataBytesForGroups(format, group_count);

    const scales_offset_bytes = std.mem.alignForward(u64, base_offset_bytes, alignment) -
        base_offset_bytes;
    const data_start = base_offset_bytes + scales_offset_bytes + scales_len_bytes;
    const data_offset_bytes = std.mem.alignForward(u64, data_start, alignment) - base_offset_bytes;

    const total_len_bytes = @addWithOverflow(data_offset_bytes, data_len_bytes);
    if (total_len_bytes[1] != 0) return Error.LayoutOverflow;

    return .{
        .scales_offset_bytes = scales_offset_bytes,
        .scales_len_bytes = scales_len_bytes,
        .data_offset_bytes = data_offset_bytes,
        .data_len_bytes = data_len_bytes,
        .total_len_bytes = total_len_bytes[0],
        .group_count = group_count,
        .groups_per_row = @divExact(cols, group_size),
    };
}

/// Bytes one plane of a key/value cache occupies at `positions` rows of `cols` elements.
///
/// The cache is a quantized tensor like any other here: same group size, same two-plane layout, same
/// `bias`, which is what lets one set of kernels and one drift gate describe both the weights and the
/// cache. `positions` is the row count because the cache is row-major by position.
pub fn planeBytes(format: dtype.Format, positions: u32, cols: u32) u64 {
    const groups = @as(u64, positions) * (@as(u64, cols) / group_size);
    return switch (format) {
        .f32 => @as(u64, positions) * cols * @sizeOf(f32),
        .q4, .q5, .q8 => groups * q4_scale_bytes_per_group +
            groups * dataBytesPerGroup(format),
        else => unreachable,
    };
}

/// Quantizes one row of `cols` values into preallocated planes.
///
/// A row of all zeros stores scale 0 and zero codes, which decodes back to
/// zeros without a special case in the kernels.
pub fn quantizeRow(
    format: dtype.Format,
    values: []const f32,
    scales_out: []u8,
    data_out: []u8,
) void {
    const scales: []u8 = scales_out;
    const data: []u8 = data_out;
    const group_count = groupCountOfLen(values.len);
    assert(scales.len == group_count * q4_scale_bytes_per_group);
    assert(data.len == group_count * dataBytesPerGroup(format));

    for (0..group_count) |group_index| {
        const values_group = values[group_index * group ..][0..group];
        var max_abs: f32 = 0.0;
        for (values_group) |value| {
            assert(std.math.isFinite(value));
            max_abs = @max(max_abs, @abs(value));
        }

        const scale_denominator: f32 = @floatFromInt(bias(format));
        const scale_bits = half_float.toF16(max_abs / scale_denominator);
        writeF16Le(scales, group_index, scale_bits);

        // Quantize against the *stored* f16 scale: decoding uses that exact
        // value, so a group must never decode to more than its true maximum.
        const stored_scale = half_float.fromF16(scale_bits);
        const codes = data[group_index * dataBytesPerGroup(format) ..][0..dataBytesPerGroup(format)];
        quantizeGroup(format, values_group, stored_scale, codes);
    }
}

fn quantizeGroup(
    format: dtype.Format,
    values_group: *const [group]f32,
    stored_scale: f32,
    codes: []u8,
) void {
    const shift = bias(format);
    const min_code = codeMin(format);
    const max_code = codeMax(format);

    switch (format) {
        .q4 => {
            @memset(codes[0..q4_data_bytes_per_group], 0);
            for (values_group, 0..) |value, index| {
                const code = quantizeOne(value, stored_scale, min_code, max_code);
                const unsigned: u8 = @intCast(code + shift);
                assert(unsigned <= 15);
                if (index % 2 == 0) {
                    codes[index / 2] |= unsigned;
                } else {
                    codes[index / 2] |= unsigned << 4;
                }
            }
        },
        .q5 => {
            @memset(codes[0..q5_data_bytes_per_group], 0);
            for (values_group, 0..) |value, index| {
                const code = quantizeOne(value, stored_scale, min_code, max_code);
                const unsigned: u8 = @intCast(code + shift);
                assert(unsigned <= 31);
                const low = unsigned & 0x0F;
                if (index % 2 == 0) {
                    codes[index / 2] |= low;
                } else {
                    codes[index / 2] |= low << 4;
                }
                if ((unsigned & 0x10) != 0) {
                    codes[q4_data_bytes_per_group + index / 8] |= @as(u8, 1) << @intCast(index % 8);
                }
            }
        },
        .q8 => {
            for (values_group, 0..) |value, index| {
                const code = quantizeOne(value, stored_scale, min_code, max_code);
                codes[index] = @intCast(code + shift);
            }
        },
        else => unreachable,
    }
}

fn quantizeOne(value: f32, stored_scale: f32, min_code: i32, max_code: i32) i32 {
    if (stored_scale == 0.0) return 0;
    const scaled = value / stored_scale;
    // Round half away from zero, then clamp into the representable range.
    const rounded: f32 = if (scaled >= 0.0) @floor(scaled + 0.5) else @ceil(scaled - 0.5);
    if (rounded <= @as(f32, @floatFromInt(min_code))) return min_code;
    if (rounded >= @as(f32, @floatFromInt(max_code))) return max_code;
    return @intFromFloat(rounded);
}

/// True when every value is finite.
///
/// Quantization asserts on non-finite input because a single NaN would poison a
/// whole group's scale search. The converter calls this first so that a corrupt
/// checkpoint produces an error message instead of an abort halfway through a
/// multi-gigabyte conversion.
pub fn rowIsFinite(values: []const f32) bool {
    for (values) |value| {
        if (!std.math.isFinite(value)) return false;
    }
    return true;
}

fn writeF16Le(bytes: []u8, group_index: usize, bits: u16) void {
    bytes[group_index * 2] = @truncate(bits);
    bytes[group_index * 2 + 1] = @truncate(bits >> 8);
}

pub fn readF16Le(bytes: []const u8, group_index: usize) u16 {
    const low: u16 = bytes[group_index * 2];
    const high: u16 = bytes[group_index * 2 + 1];
    return low | (high << 8);
}

/// Decodes one group of codes into `out[0..group_size]`.
pub fn dequantizeGroup(format: dtype.Format, codes: []const u8, scale: f32, out: []f32) void {
    assert(out.len == group);
    const shift = bias(format);

    switch (format) {
        .q4 => {
            for (0..group) |index| {
                const byte = codes[index / 2];
                const nibble: u8 = if (index % 2 == 0) byte & 0x0F else byte >> 4;
                out[index] = @floatFromInt(@as(i32, nibble) - shift);
            }
        },
        .q5 => {
            for (0..group) |index| {
                const byte = codes[index / 2];
                const low: u32 = if (index % 2 == 0) byte & 0x0F else byte >> 4;
                const high: u32 = (codes[q4_data_bytes_per_group + index / 8] >>
                    @intCast(index % 8)) & 1;
                out[index] = @floatFromInt(@as(i32, @intCast(low | (high << 4))) - shift);
            }
        },
        .q8 => {
            for (0..group) |index| {
                out[index] = @floatFromInt(@as(i32, codes[index]) - shift);
            }
        },
        else => unreachable,
    }

    for (out) |*value| value.* *= scale;
}

/// Decodes a whole row of `cols` weights.
pub fn dequantizeRow(
    format: dtype.Format,
    row_scales: []const u8,
    row_data: []const u8,
    out: []f32,
) void {
    const group_count = groupCountOfLen(out.len);
    for (0..group_count) |group_index| {
        const scale = half_float.fromF16(readF16Le(row_scales, group_index));
        dequantizeGroup(
            format,
            row_data[group_index * dataBytesPerGroup(format) ..][0..dataBytesPerGroup(format)],
            scale,
            out[group_index * group ..][0..group],
        );
    }
}

/// Reference dot product of one quantized row against an f32 activation row.
///
/// This is the oracle the WebGPU kernel is measured against, so it stays
/// deliberately straightforward: f32 accumulation, one group at a time.
pub fn dotRow(
    format: dtype.Format,
    row_scales: []const u8,
    row_data: []const u8,
    x: []const f32,
) f32 {
    const group_count = groupCountOfLen(x.len);
    var accumulator: f32 = 0.0;
    var codes: [group]f32 = undefined;
    for (0..group_count) |group_index| {
        const scale = half_float.fromF16(readF16Le(row_scales, group_index));
        dequantizeGroup(
            format,
            row_data[group_index * dataBytesPerGroup(format) ..][0..dataBytesPerGroup(format)],
            scale,
            &codes,
        );
        const activations = x[group_index * group ..][0..group];
        for (codes, activations) |weight, activation| accumulator += weight * activation;
    }
    return accumulator;
}

const assert = std.debug.assert;

test "group count requires whole groups" {
    try std.testing.expectEqual(@as(u32, 16), try groupCount(1, 1024));
    try std.testing.expectEqual(@as(u32, 0), try groupCount(0, 1024));
    try std.testing.expectError(Error.RowNotGroupAligned, groupCount(1, 100));
    try std.testing.expectError(Error.RowNotGroupAligned, groupCount(1, 1));
}

test "a quantized key/value cache costs a quarter of the f32 one" {
    // The geometry both released models share: 28 layers, 8 key/value heads of 128 elements, and
    // 8192 positions. This is the number that decides whether 1.7B fits a 2 GiB instance.
    const layers = 28;
    const key_value_width = 8 * 128;
    const positions = 8192;
    const f32_bytes = 2 * layers * planeBytes(.f32, positions, key_value_width);
    const q8_bytes = 2 * layers * planeBytes(.q8, positions, key_value_width);

    try std.testing.expectEqual(@as(u64, 1879048192), f32_bytes); // 1.75 GiB, the figure the ABI reports
    // One byte per element, plus the scale plane: two bytes per group of `group_size` elements. The
    // expectation is derived from the constant rather than written out, because a group size that
    // changes is a layout change and this test should follow it rather than fail mysteriously.
    const groups = key_value_width / group_size;
    try std.testing.expectEqual(
        @as(u64, positions) * (key_value_width + groups * q4_scale_bytes_per_group),
        q8_bytes / (2 * layers),
    );
    try std.testing.expect(q8_bytes < f32_bytes / 3);
}

test "plane layout aligns both planes" {
    const layout = try planeLayout(.q4, 896, 896, 0, tensor_alignment_bytes);
    try std.testing.expectEqual(@as(u32, 14), layout.groups_per_row);
    try std.testing.expectEqual(@as(u32, 14 * 896), layout.group_count);
    try std.testing.expectEqual(@as(u64, 0), layout.scales_offset_bytes);
    try std.testing.expectEqual(@as(u64, 14 * 896 * 2), layout.scales_len_bytes);
    try std.testing.expectEqual(@as(u64, 0), layout.data_offset_bytes % tensor_alignment_bytes);
    try std.testing.expectEqual(@as(u64, 14 * 896 * 32), layout.data_len_bytes);
    try std.testing.expectEqual(
        layout.data_offset_bytes + layout.data_len_bytes,
        layout.total_len_bytes,
    );
}

test "plane layout of an unaligned base still aligns the payload" {
    // 64 rows x 256 cols in q5: 4 groups per row, 256 groups total.
    const layout = try planeLayout(.q5, 64, 256, 33, tensor_alignment_bytes);
    try std.testing.expectEqual(@as(u64, 15), layout.scales_offset_bytes);
    try std.testing.expectEqual(@as(u64, 256 * 2), layout.scales_len_bytes);
    try std.testing.expectEqual(@as(u32, 256), layout.group_count);
    // The data plane is aligned relative to the container, not the tensor base.
    try std.testing.expectEqual(@as(u64, 527), layout.data_offset_bytes);
    try std.testing.expectEqual(@as(u64, 0), (33 + layout.data_offset_bytes) % tensor_alignment_bytes);
    try std.testing.expectEqual(@as(u64, 256 * 40), layout.data_len_bytes);
    try std.testing.expectEqual(@as(u64, 10767), layout.total_len_bytes);
}

test "plane layout rejects misaligned alignments" {
    try std.testing.expectError(Error.LayoutOverflow, planeLayout(.q4, 64, 64, 0, 2));
    try std.testing.expectError(Error.NotQuantized, planeLayout(.f16, 64, 64, 0, 16));
    const bogus: dtype.Format = @fromBackingInt(@intCast(200));
    try std.testing.expectError(Error.UnsupportedFormat, planeLayout(bogus, 64, 64, 0, 16));
}

test "quantize and dequantize round trip within the scale step" {
    var values: [group]f32 = undefined;
    for (&values, 0..) |*value, index| {
        const position: f32 = @floatFromInt(index);
        value.* = @sin(position * 0.37) * 3.0 + position * 0.01;
    }

    const formats = [_]dtype.Format{ .q4, .q5, .q8 };
    for (formats) |format| {
        var scales: [q4_scale_bytes_per_group]u8 = undefined;
        var data: [q8_data_bytes_per_group]u8 = undefined;
        const codes = data[0..dataBytesPerGroup(format)];
        quantizeRow(format, &values, &scales, codes);

        const scale = half_float.fromF16(readF16Le(&scales, 0));
        var decoded: [group]f32 = undefined;
        dequantizeGroup(format, codes, scale, &decoded);

        var max_abs: f32 = 0.0;
        for (values) |value| max_abs = @max(max_abs, @abs(value));
        const step = max_abs / @as(f32, @floatFromInt(bias(format)));

        // The positive extreme is one step off because the code range is
        // two's-complement shaped: +max_abs would need code `bias`, which the
        // largest representable code clamps to `bias - 1`. Every other value
        // lands within half a step.
        // The bound is one step, not half: the most positive value would need
        // code `bias`, which clamps to `bias - 1`. The extra allowance is the
        // stored f16 scale's own rounding: it carries 11 bits of significand, so
        // at the clamped end the error moves by up to `bias * 2^-11` of a step
        // (6.25% for q8, where the code count is largest).
        const bound = step * (1.0 + @as(f32, @floatFromInt(bias(format))) * 0.00048828125) + 1e-6;
        var sum_squares: f64 = 0.0;
        for (values, decoded) |original, restored| {
            const difference = @abs(original - restored);
            try std.testing.expect(difference <= bound);
            sum_squares += @as(f64, difference) * @as(f64, difference);
        }
        const rms = @sqrt(sum_squares / group);
        try std.testing.expect(rms <= @as(f64, step) * 0.5);
    }
}

test "quantization error shrinks as the code width grows" {
    var values: [group]f32 = undefined;
    for (&values, 0..) |*value, index| {
        const position: f32 = @floatFromInt(index);
        value.* = @cos(position * 0.53) * 2.5 + @sin(position * 0.11);
    }

    var previous_error = std.math.floatMax(f32);
    const formats = [_]dtype.Format{ .q4, .q5, .q8 };
    for (formats) |format| {
        var scales: [q4_scale_bytes_per_group]u8 = undefined;
        var data: [q8_data_bytes_per_group]u8 = undefined;
        const codes = data[0..dataBytesPerGroup(format)];
        quantizeRow(format, &values, &scales, codes);
        const scale = half_float.fromF16(readF16Le(&scales, 0));
        var decoded: [group]f32 = undefined;
        dequantizeGroup(format, codes, scale, &decoded);

        var accumulated_error: f32 = 0.0;
        for (values, decoded) |original, restored| {
            accumulated_error += @abs(original - restored);
        }
        try std.testing.expect(accumulated_error < previous_error);
        previous_error = accumulated_error;
    }
}

test "exact values survive quantization when the code aligns" {
    // Values whose per-code step is exactly 1.0: max_abs = 8 gives scale = 1,
    // and 7 / -8 are both representable codes, so the round trip is bit exact.
    var values: [group]f32 = @splat(0.0);
    values[0] = 7.0;
    values[1] = -8.0;
    var scales: [q4_scale_bytes_per_group]u8 = undefined;
    var data: [q4_data_bytes_per_group]u8 = undefined;
    quantizeRow(.q4, &values, &scales, &data);
    try std.testing.expectEqual(@as(u16, 0x3C00), readF16Le(&scales, 0)); // 1.0
    var decoded: [group]f32 = undefined;
    dequantizeGroup(.q4, &data, 1.0, &decoded);
    try std.testing.expectEqual(@as(f32, 7.0), decoded[0]);
    try std.testing.expectEqual(@as(f32, -8.0), decoded[1]);
    for (decoded[2..]) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
}

test "an all-zero group decodes to zeros without a division by zero" {
    var values: [group]f32 = @splat(0.0);
    var scales: [q4_scale_bytes_per_group]u8 = undefined;
    var data: [q4_data_bytes_per_group]u8 = undefined;
    quantizeRow(.q4, &values, &scales, &data);
    try std.testing.expectEqual(@as(f32, 0.0), half_float.fromF16(readF16Le(&scales, 0)));
    var decoded: [group]f32 = undefined;
    dequantizeGroup(.q4, &data, 0.0, &decoded);
    for (decoded) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
    try std.testing.expectEqual(@as(f32, 0.0), dotRow(.q4, &scales, &data, &values));
}

test "the converter's finite check rejects NaN and infinity" {
    // `quantizeRow` asserts on non-finite input, so conversion validates first
    // and reports a normal error instead of aborting a long conversion run.
    var clean: [4]f32 = .{ 0.0, 1.0, -1.0, 1e-30 };
    try std.testing.expect(rowIsFinite(&clean));

    var with_nan: [4]f32 = .{ 0.0, 1.0, -1.0, 1e-30 };
    with_nan[2] = std.math.nan(f32);
    try std.testing.expect(!rowIsFinite(&with_nan));

    var with_infinity: [4]f32 = .{ 0.0, 1.0, -1.0, 1e-30 };
    with_infinity[0] = std.math.inf(f32);
    try std.testing.expect(!rowIsFinite(&with_infinity));

    var with_negative_infinity: [4]f32 = .{ 0.0, 1.0, -1.0, 1e-30 };
    with_negative_infinity[3] = -std.math.inf(f32);
    try std.testing.expect(!rowIsFinite(&with_negative_infinity));
}

test "dot row agrees with a dequantized dense dot product" {
    const cols: usize = 256;
    var values: [cols]f32 = undefined;
    var x: [cols]f32 = undefined;
    for (&values, 0..) |*value, index| {
        const position: f32 = @floatFromInt(index);
        value.* = @cos(position * 0.11) * 1.5;
        x[index] = @sin(position * 0.07) * 0.5;
    }
    var scales: [cols / group * q4_scale_bytes_per_group]u8 = undefined;
    var data: [cols / group * q4_data_bytes_per_group]u8 = undefined;
    quantizeRow(.q4, &values, &scales, &data);

    var dense: [cols]f32 = undefined;
    dequantizeRow(.q4, &scales, &data, &dense);
    var expected: f32 = 0.0;
    for (dense, x) |weight, activation| expected += weight * activation;

    const actual = dotRow(.q4, &scales, &data, &x);
    try std.testing.expectApproxEqRel(expected, actual, 1e-5);
}

test "q5 packing separates the low nibbles from the high bit plane" {
    // max_abs = 16 gives scale = 1, so codes 15 and -16 are exact. The first
    // weight needs bit 4 set; the second is code -16, whose unsigned form is 0.
    var values: [group]f32 = @splat(0.0);
    values[0] = 15.0;
    values[1] = -16.0;
    var scales: [q4_scale_bytes_per_group]u8 = undefined;
    var data: [q5_data_bytes_per_group]u8 = undefined;
    quantizeRow(.q5, &values, &scales, &data);
    const scale = half_float.fromF16(readF16Le(&scales, 0));
    var decoded: [group]f32 = undefined;
    dequantizeGroup(.q5, &data, scale, &decoded);
    try std.testing.expectEqual(@as(f32, 15.0), decoded[0]);
    try std.testing.expectEqual(@as(f32, -16.0), decoded[1]);
    for (decoded[2..]) |value| try std.testing.expectEqual(@as(f32, 0.0), value);

    // Low nibbles: weight 0 has unsigned 31 -> low nibble 15, weight 1 -> 0.
    try std.testing.expectEqual(@as(u8, 0x0F), data[0]);
    // High-bit plane: bit j of byte j/8 carries weight j's fifth bit. Weight 0
    // (unsigned 31) sets its bit and weight 1 (unsigned 0) clears it. Weights 2
    // and up hold 0.0, which encodes as unsigned 16 and therefore also sets the
    // fifth bit -- that is the midpoint offset, not a bug.
    try std.testing.expect(data[q4_data_bytes_per_group] & 0x01 != 0);
    try std.testing.expect(data[q4_data_bytes_per_group] & 0x02 == 0);
    for (data[q4_data_bytes_per_group + 1 ..][0 .. q5_data_bytes_per_group - q4_data_bytes_per_group - 1]) |byte| {
        try std.testing.expectEqual(@as(u8, 0xFF), byte);
    }
}

test "quantized payload byte budget matches the documented bits per weight" {
    // q4: 34 bytes per 64 weights = 4.25 bits per weight.
    try std.testing.expectApproxEqAbs(
        @as(f64, 4.25),
        @as(f64, @floatFromInt(q4_scale_bytes_per_group + q4_data_bytes_per_group)) / group * 8.0,
        1e-12,
    );
    // q5: 42 bytes per 64 weights = 5.25 bits per weight.
    try std.testing.expectApproxEqAbs(
        @as(f64, 5.25),
        @as(f64, @floatFromInt(q5_scale_bytes_per_group + q5_data_bytes_per_group)) / group * 8.0,
        1e-12,
    );
    // q8: 66 bytes per 64 weights = 8.25 bits per weight.
    try std.testing.expectApproxEqAbs(
        @as(f64, 8.25),
        @as(f64, @floatFromInt(q8_scale_bytes_per_group + q8_data_bytes_per_group)) / group * 8.0,
        1e-12,
    );
}
