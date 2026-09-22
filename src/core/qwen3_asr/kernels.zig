//! CPU kernels for the Qwen3-ASR forward pass.
//!
//! These are the reference implementation: the WebGPU kernels are validated
//! against them, and the WASM backend that ships as the portable fallback *is*
//! them. They are written for clarity and determinism first, with `@Vector`
//! arithmetic where it maps directly onto the algorithm so that Zig can emit
//! WASM SIMD.
//!
//! Layouts, fixed here and relied on by callers:
//!
//!   * A weight matrix is `rows x cols`, row major, `rows` being output
//!     features. PyTorch stores `nn.Linear` weights this way already, so no
//!     transposes are needed anywhere in the pipeline.
//!   * An activation batch is `tokens x cols`, row major, so token `t`'s
//!     features occupy `x[t * cols ..][0..cols]`.
//!   * A product writes `tokens x rows` the same way.
//!   * Quantized weights arrive as the two planes `quant.planeLayout` describes.
//!
//! Every function here is allocation-free and takes plain slices. Nothing in
//! this file knows about shards, browsers, or the ABI.

const std = @import("std");
const math = @import("../math.zig");
const dtype = @import("../dtype.zig");
const quant = @import("../quant.zig");
const half_float = @import("../half_float.zig");

/// Lanes used for the vectorized reductions. Four f32 lanes is exactly one WASM
/// `v128` and lets the compiler keep a single accumulator live.
const lanes = 4;
const Vector = @Vector(lanes, f32);

pub const Error = error{
    /// A slice length disagrees with the declared shape.
    ShapeMismatch,
};

/// Dot product of two equal-length f32 rows.
pub fn dotF32(left: []const f32, right: []const f32) f32 {
    assert(left.len == right.len);
    var accumulator: Vector = @splat(0.0);
    var index: usize = 0;
    while (index + lanes <= left.len) : (index += lanes) {
        const a: Vector = left[index..][0..lanes].*;
        const b: Vector = right[index..][0..lanes].*;
        accumulator += a * b;
    }
    var total: f32 = @reduce(.Add, accumulator);
    while (index < left.len) : (index += 1) total += left[index] * right[index];
    return total;
}

/// `out[tokens][rows] = weights[rows][cols] * x[tokens][cols]`, f32 weights.
pub fn linearF32(
    out: []f32,
    x: []const f32,
    weights: []const f32,
    rows: u32,
    cols: u32,
    tokens: u32,
) Error!void {
    if (out.len != @as(usize, rows) * tokens) return Error.ShapeMismatch;
    if (x.len != @as(usize, cols) * tokens) return Error.ShapeMismatch;
    if (weights.len != @as(usize, rows) * cols) return Error.ShapeMismatch;

    var row: u32 = 0;
    while (row < rows) : (row += 1) {
        const weight_row = weights[@as(usize, row) * cols ..][0..cols];
        var token: u32 = 0;
        while (token < tokens) : (token += 1) {
            const activation_row = x[@as(usize, token) * cols ..][0..cols];
            out[@as(usize, token) * rows + row] = dotF32(weight_row, activation_row);
        }
    }
}

/// `out[tokens][rows] = weights[rows][cols] * x[tokens][cols]`, f16 weights.
///
/// `row_scratch` must hold at least `cols` values. Decoding one row at a time
/// into it means each weight row is read once no matter how many tokens are
/// being projected, and it keeps the kernel free of its own allocator.
pub fn linearF16(
    out: []f32,
    x: []const f32,
    weights: []const u16,
    rows: u32,
    cols: u32,
    tokens: u32,
    row_scratch: []f32,
) Error!void {
    if (out.len != @as(usize, rows) * tokens) return Error.ShapeMismatch;
    if (x.len != @as(usize, cols) * tokens) return Error.ShapeMismatch;
    if (weights.len != @as(usize, rows) * cols) return Error.ShapeMismatch;
    if (row_scratch.len < cols) return Error.ShapeMismatch;

    var row: u32 = 0;
    while (row < rows) : (row += 1) {
        const weight_row = weights[@as(usize, row) * cols ..][0..cols];
        // `row_scratch` is a shared reuse buffer sized for the widest layer in
        // the model, so every use of it is the `cols`-long prefix, never the
        // whole slice.
        const scratch_row = row_scratch[0..cols];
        for (weight_row, scratch_row) |bits, *value| value.* = half_float.fromF16(bits);
        var token: u32 = 0;
        while (token < tokens) : (token += 1) {
            const activation_row = x[@as(usize, token) * cols ..][0..cols];
            out[@as(usize, token) * rows + row] = dotF32(scratch_row, activation_row);
        }
    }
}

/// `out[tokens][rows] = weights[rows][cols] * x[tokens][cols]` for a quantized
/// weight matrix.
///
/// The dequantization is fused: a row's codes are decoded once into scratch
/// memory and reused for every token, so the packed bytes are read exactly once
/// and no dense copy of the matrix is ever materialized.
pub fn linearQuantized(
    out: []f32,
    x: []const f32,
    format: dtype.Format,
    scales: []const u8,
    data: []const u8,
    rows: u32,
    cols: u32,
    tokens: u32,
    row_scratch: []f32,
) Error!void {
    if (out.len != @as(usize, rows) * tokens) return Error.ShapeMismatch;
    if (x.len != @as(usize, cols) * tokens) return Error.ShapeMismatch;
    if (cols % quant.group_size != 0) return Error.ShapeMismatch;
    const groups_per_row = cols / quant.group_size;
    if (scales.len != @as(usize, rows) * groups_per_row * quant.q4_scale_bytes_per_group) {
        return Error.ShapeMismatch;
    }
    if (data.len != @as(usize, rows) * groups_per_row * quant.dataBytesPerGroup(format)) {
        return Error.ShapeMismatch;
    }
    if (row_scratch.len < cols) return Error.ShapeMismatch;

    var row: u32 = 0;
    while (row < rows) : (row += 1) {
        const row_scales = scales[@as(usize, row) * groups_per_row *
            quant.q4_scale_bytes_per_group ..];
        const row_data = data[@as(usize, row) * groups_per_row *
            quant.dataBytesPerGroup(format) ..];
        // See `linearF16`: the scratch is longer than `cols` by design, so the
        // dequantized prefix is what gets dotted against the activation row.
        const scratch_row = row_scratch[0..cols];
        quant.dequantizeRow(format, row_scales, row_data, scratch_row);

        var token: u32 = 0;
        while (token < tokens) : (token += 1) {
            const activation_row = x[@as(usize, token) * cols ..][0..cols];
            out[@as(usize, token) * rows + row] = dotF32(scratch_row, activation_row);
        }
    }
}

/// Dispatches a linear layer over whichever storage the tensor uses.
pub const Matrix = union(enum) {
    f32: []const f32,
    f16: []const u16,
    quantized: struct {
        format: dtype.Format,
        scales: []const u8,
        data: []const u8,
    },

    pub fn rows(self: Matrix, cols: u32) u32 {
        return switch (self) {
            .f32 => |weights| @intCast(weights.len / cols),
            .f16 => |weights| @intCast(weights.len / cols),
            .quantized => |planes| @intCast(
                planes.scales.len / (cols / quant.group_size * quant.q4_scale_bytes_per_group),
            ),
        };
    }

    /// `row_scratch` must hold at least `cols` values when the storage is not
    /// f32; it is unused for f32 weights, so pass an empty slice there.
    pub fn apply(
        self: Matrix,
        out: []f32,
        x: []const f32,
        cols: u32,
        tokens: u32,
        row_scratch: []f32,
    ) Error!void {
        const row_count = self.rows(cols);
        switch (self) {
            .f32 => |weights| try linearF32(out, x, weights, row_count, cols, tokens),
            .f16 => |weights| try linearF16(out, x, weights, row_count, cols, tokens, row_scratch),
            .quantized => |planes| try linearQuantized(
                out,
                x,
                planes.format,
                planes.scales,
                planes.data,
                row_count,
                cols,
                tokens,
                row_scratch,
            ),
        }
    }
};

/// Adds a bias vector to every token row, in place.
pub fn addBias(x: []f32, bias: []const f32, rows: u32, tokens: u32) Error!void {
    if (x.len != @as(usize, rows) * tokens) return Error.ShapeMismatch;
    if (bias.len != rows) return Error.ShapeMismatch;
    var token: u32 = 0;
    while (token < tokens) : (token += 1) {
        const row = x[@as(usize, token) * rows ..][0..rows];
        for (row, bias) |*value, offset| value.* += offset;
    }
}

/// `x += y` for equal-length slices.
pub fn addInPlace(x: []f32, y: []const f32) Error!void {
    if (x.len != y.len) return Error.ShapeMismatch;
    var index: usize = 0;
    while (index + lanes <= x.len) : (index += lanes) {
        const a: Vector = x[index..][0..lanes].*;
        const b: Vector = y[index..][0..lanes].*;
        x[index..][0..lanes].* = a + b;
    }
    while (index < x.len) : (index += 1) x[index] += y[index];
}

/// `out = silu(gate) * up`, the decoder MLP's gated activation.
pub fn siluMul(out: []f32, gate: []const f32, up: []const f32) Error!void {
    if (out.len != gate.len) return Error.ShapeMismatch;
    if (out.len != up.len) return Error.ShapeMismatch;
    for (out, gate, up) |*value, gate_value, up_value| {
        value.* = math.silu(gate_value) * up_value;
    }
}

/// LayerNorm over each token row, with the audio tower's affine parameters.
pub fn layerNorm(
    x: []f32,
    weight: []const f32,
    bias: []const f32,
    width: u32,
    tokens: u32,
    eps: f32,
) Error!void {
    if (x.len != @as(usize, width) * tokens) return Error.ShapeMismatch;
    var token: u32 = 0;
    while (token < tokens) : (token += 1) {
        math.layerNormInPlace(x[@as(usize, token) * width ..][0..width], weight, bias, eps);
    }
}

/// LayerNorm from `source` into `target`, leaving the source intact so that a
/// residual connection can still add it back.
pub fn layerNormInto(
    target: []f32,
    source: []const f32,
    weight: []const f32,
    bias: []const f32,
    width: u32,
    tokens: u32,
    eps: f32,
) Error!void {
    if (target.len != source.len) return Error.ShapeMismatch;
    @memcpy(target, source);
    try layerNorm(target, weight, bias, width, tokens, eps);
}

/// RMSNorm from `source` into `target`, leaving the source intact.
pub fn rmsNormInto(
    target: []f32,
    source: []const f32,
    weight: []const f32,
    width: u32,
    tokens: u32,
    eps: f32,
) Error!void {
    if (target.len != source.len) return Error.ShapeMismatch;
    @memcpy(target, source);
    try rmsNorm(target, weight, width, tokens, eps);
}

/// RMSNorm over each token row, with the decoder's weight vectors.
pub fn rmsNorm(
    x: []f32,
    weight: []const f32,
    width: u32,
    tokens: u32,
    eps: f32,
) Error!void {
    if (x.len != @as(usize, width) * tokens) return Error.ShapeMismatch;
    var token: u32 = 0;
    while (token < tokens) : (token += 1) {
        math.rmsNormInPlace(x[@as(usize, token) * width ..][0..width], weight, eps);
    }
}

/// Total, valid, and convolutional-padding offsets of one frame.
pub const ConvGeometry = struct {
    /// Output channels.
    out_channels: u32,
    /// Input channels. The audio tower's first convolution is single channel;
    /// the second and third take the previous stage's full width.
    in_channels: u32 = 1,
    /// Height (frequency) of the input.
    in_height: u32,
    /// Width (time) of the input.
    in_width: u32,
    kernel: u32,

    pub fn outHeight(self: ConvGeometry) u32 {
        return (self.in_height - 1) / 2 + 1;
    }

    pub fn outWidth(self: ConvGeometry) u32 {
        return (self.in_width - 1) / 2 + 1;
    }
};

/// Three-by-three, stride two, padding one convolution followed by a bias add
/// and the GELU activation.
///
/// The audio tower applies this three times with GELU in between, which is why
/// the activation is fused here rather than left to the caller. Weights are
/// `[out_channels][in_channels][3][3]`, input is `[in_channels][in_height][in_width]`,
/// and output is `[out_channels][out_height][out_width]` -- the layouts the
/// checkpoint and the previous stage already have, so nothing is repacked.
pub fn conv3x3Stride2Gelu(
    out: []f32,
    input: []const f32,
    weights: []const u16,
    bias: []const f32,
    geometry: ConvGeometry,
) Error!void {
    const out_height = geometry.outHeight();
    const out_width = geometry.outWidth();
    const in_plane = @as(usize, geometry.in_height) * geometry.in_width;
    const out_plane = @as(usize, out_height) * out_width;
    const kernel_values = @as(usize, geometry.kernel) * geometry.kernel;
    if (geometry.kernel != 3) return Error.ShapeMismatch;
    if (input.len != in_plane * geometry.in_channels) return Error.ShapeMismatch;
    if (weights.len != @as(usize, geometry.out_channels) *
        geometry.in_channels * kernel_values) return Error.ShapeMismatch;
    if (bias.len != geometry.out_channels) return Error.ShapeMismatch;
    if (out.len != out_plane * geometry.out_channels) return Error.ShapeMismatch;

    var channel: u32 = 0;
    while (channel < geometry.out_channels) : (channel += 1) {
        const out_slice = out[@as(usize, channel) * out_plane ..][0..out_plane];
        const weight_base = @as(usize, channel) * geometry.in_channels * kernel_values;
        try convolveChannel(out_slice, input, weights[weight_base..], bias[channel], geometry);
        for (out_slice) |*value| value.* = math.gelu(value.*);
    }
}

/// One output channel's reduction over every input channel.
///
/// The nine weights of each `(out, in)` pair are decoded once per input channel
/// rather than once per output position: at 480 channels the weight traffic would
/// otherwise dominate the arithmetic.
fn convolveChannel(
    out: []f32,
    input: []const f32,
    weights: []const u16,
    channel_bias: f32,
    geometry: ConvGeometry,
) Error!void {
    const out_height = geometry.outHeight();
    const out_width = geometry.outWidth();
    const in_plane = @as(usize, geometry.in_height) * geometry.in_width;
    const kernel_values = @as(usize, geometry.kernel) * geometry.kernel;
    if (out.len != @as(usize, out_height) * out_width) return Error.ShapeMismatch;

    @memset(out, channel_bias);
    var in_channel: u32 = 0;
    while (in_channel < geometry.in_channels) : (in_channel += 1) {
        const channel_weights = weights[@as(usize, in_channel) * kernel_values ..][0..kernel_values];
        const channel_input = input[@as(usize, in_channel) * in_plane ..][0..in_plane];
        try convolvePlane(out, channel_input, channel_weights, geometry);
    }
}

/// Adds one input channel's contribution into an output plane.
fn convolvePlane(
    out: []f32,
    input: []const f32,
    weights: []const u16,
    geometry: ConvGeometry,
) Error!void {
    const out_height = geometry.outHeight();
    const out_width = geometry.outWidth();
    var kernel_weights: [9]f32 = undefined;
    const kernel_values = @as(usize, geometry.kernel) * geometry.kernel;
    if (kernel_values > kernel_weights.len) return Error.ShapeMismatch;
    for (0..kernel_values) |index| kernel_weights[index] = half_float.fromF16(weights[index]);

    var out_row: u32 = 0;
    while (out_row < out_height) : (out_row += 1) {
        var out_column: u32 = 0;
        while (out_column < out_width) : (out_column += 1) {
            var accumulator: f32 = 0.0;
            var kernel_row: u32 = 0;
            while (kernel_row < geometry.kernel) : (kernel_row += 1) {
                const in_row = @as(isize, @intCast(out_row * 2)) +
                    @as(isize, @intCast(kernel_row)) - 1;
                if (in_row < 0 or in_row >= geometry.in_height) continue;
                var kernel_column: u32 = 0;
                while (kernel_column < geometry.kernel) : (kernel_column += 1) {
                    const in_column = @as(isize, @intCast(out_column * 2)) +
                        @as(isize, @intCast(kernel_column)) - 1;
                    if (in_column < 0 or in_column >= geometry.in_width) continue;
                    const sample = input[
                        @as(usize, @intCast(in_row)) * geometry.in_width +
                            @as(usize, @intCast(in_column))
                    ];
                    accumulator += kernel_weights[kernel_row * geometry.kernel + kernel_column] *
                        sample;
                }
            }
            out[@as(usize, out_row) * out_width + out_column] += accumulator;
        }
    }
}

/// Windowed, non-causal attention for the audio encoder.
///
/// `q`, `k`, and `v` hold one row per packed step, each `heads * head_dim` wide.
/// Attention runs independently inside each window, which is how the reference
/// splits long audio into `n_window_infer`-sized groups. `out` receives the
/// concatenated per-head results.
pub fn audioAttentionWindow(
    out: []f32,
    out_stride: u32,
    q: []const f32,
    q_stride: u32,
    k: []const f32,
    k_stride: u32,
    v: []const f32,
    v_stride: u32,
    window_start: u32,
    window_length: u32,
    heads: u32,
    head_dim: u32,
    scores: []f32,
) Error!void {
    // Rows are spaced `stride` elements apart inside their own projection
    // buffer, which is how the caller keeps one contiguous projection without
    // copying a window out of it.
    if (scores.len < window_length) return Error.ShapeMismatch;
    const scaling = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    var head: u32 = 0;
    while (head < heads) : (head += 1) {
        var query: u32 = 0;
        while (query < window_length) : (query += 1) {
            const query_row = q[@as(usize, window_start + query) * q_stride +
                head * head_dim ..][0..head_dim];
            var key: u32 = 0;
            while (key < window_length) : (key += 1) {
                const key_row = k[@as(usize, window_start + key) * k_stride +
                    head * head_dim ..][0..head_dim];
                scores[key] = dotF32(query_row, key_row) * scaling;
            }
            math.softmaxInPlace(scores[0..window_length]);

            const out_row = out[@as(usize, window_start + query) * out_stride +
                head * head_dim ..][0..head_dim];
            @memset(out_row, 0.0);
            key = 0;
            while (key < window_length) : (key += 1) {
                const value_row = v[@as(usize, window_start + key) * v_stride +
                    head * head_dim ..][0..head_dim];
                const weight = scores[key];
                var index: usize = 0;
                while (index + lanes <= head_dim) : (index += lanes) {
                    const current: Vector = out_row[index..][0..lanes].*;
                    const addend: Vector = value_row[index..][0..lanes].*;
                    out_row[index..][0..lanes].* = current + addend * @as(Vector, @splat(weight));
                }
                while (index < head_dim) : (index += 1) out_row[index] += value_row[index] * weight;
            }
        }
    }
}

/// Applies rotary position embeddings to one row per token, in place.
///
/// `rotated` holds `rows` rows of `heads * head_dim` values. The layout is the
/// reference's `rotate_half` pairing, not interleaved: for `i < head_dim / 2`,
/// `out[i] = x[i] * cos - x[i + half] * sin` and
/// `out[i + half] = x[i + half] * cos + x[i] * sin`.
pub fn ropeInPlace(
    rotated: []f32,
    positions: []const u32,
    heads: u32,
    head_dim: u32,
    cos_table: []const f32,
    sin_table: []const f32,
    table_stride: u32,
) Error!void {
    const width = heads * head_dim;
    const half = head_dim / 2;
    if (rotated.len != @as(usize, width) * positions.len) return Error.ShapeMismatch;
    if (table_stride < half) return Error.ShapeMismatch;

    for (positions, 0..) |position, token| {
        const table_base = @as(usize, position) * table_stride;
        const cos_row = cos_table[table_base..][0..half];
        const sin_row = sin_table[table_base..][0..half];
        var head: u32 = 0;
        while (head < heads) : (head += 1) {
            const row = rotated[@as(usize, token) * width + head * head_dim ..][0..head_dim];
            var index: u32 = 0;
            while (index < half) : (index += 1) {
                const low = row[index];
                const high = row[index + half];
                const cos_value = cos_row[index];
                const sin_value = sin_row[index];
                row[index] = low * cos_value - high * sin_value;
                row[index + half] = high * cos_value + low * sin_value;
            }
        }
    }
}

/// Fills the cosine and sine tables `ropeInPlace` reads.
///
/// One row per position, `stride` values per row, `half = head_dim / 2` of them
/// used. Precomputing is what keeps the rotation from calling the sine and
/// cosine of the same angle once per head.
pub fn buildRopeTables(
    cos_table: []f32,
    sin_table: []f32,
    half: u32,
    theta: f32,
    max_positions: u32,
    stride: u32,
) Error!void {
    if (stride < half) return Error.ShapeMismatch;
    if (cos_table.len < @as(usize, stride) * max_positions) return Error.ShapeMismatch;
    if (sin_table.len < @as(usize, stride) * max_positions) return Error.ShapeMismatch;

    const head_dim = half * 2;
    var position: u32 = 0;
    while (position < max_positions) : (position += 1) {
        var index: u32 = 0;
        while (index < half) : (index += 1) {
            const angle = @as(f32, @floatFromInt(position)) *
                inverseFrequency(index, head_dim, theta);
            const offset = @as(usize, position) * stride + index;
            cos_table[offset] = math.cos(angle);
            sin_table[offset] = math.sin(angle);
        }
    }
}

/// `inv_freq[i] = theta ^ (-2i / head_dim)`.
pub fn inverseFrequency(index: u32, head_dim: u32, theta: f32) f32 {
    const exponent = -2.0 * @as(f32, @floatFromInt(index)) / @as(f32, @floatFromInt(head_dim));
    return std.math.pow(f32, theta, exponent);
}

/// One decoder attention step over a key/value cache.
///
/// `query` is `heads * head_dim` wide. `cache_keys` and `cache_values` hold
/// `positions * key_value_heads * head_dim` values; group query heads share a
/// key/value head, which the reference calls `num_key_value_groups`. `scores`
/// must hold at least `positions` values.
pub fn decodeAttentionStep(
    out: []f32,
    query: []const f32,
    cache_keys: []const f32,
    cache_values: []const f32,
    cached: u32,
    heads: u32,
    key_value_heads: u32,
    head_dim: u32,
    scores: []f32,
) Error!void {
    const query_width = heads * head_dim;
    const key_value_width = key_value_heads * head_dim;
    if (query.len != query_width) return Error.ShapeMismatch;
    if (out.len != query_width) return Error.ShapeMismatch;
    if (cache_keys.len != @as(usize, cached) * key_value_width) return Error.ShapeMismatch;
    if (cache_values.len != cache_keys.len) return Error.ShapeMismatch;
    if (scores.len < cached) return Error.ShapeMismatch;
    if (heads % key_value_heads != 0) return Error.ShapeMismatch;

    const groups = heads / key_value_heads;
    const scaling = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    var head: u32 = 0;
    while (head < heads) : (head += 1) {
        const head_query = query[head * head_dim ..][0..head_dim];
        const key_value_head = head / groups;

        var position: u32 = 0;
        while (position < cached) : (position += 1) {
            const key_row = cache_keys[@as(usize, position) * key_value_width +
                key_value_head * head_dim ..][0..head_dim];
            scores[position] = dotF32(head_query, key_row) * scaling;
        }
        math.softmaxInPlace(scores[0..cached]);

        const head_out = out[head * head_dim ..][0..head_dim];
        @memset(head_out, 0.0);
        position = 0;
        while (position < cached) : (position += 1) {
            const value_row = cache_values[@as(usize, position) * key_value_width +
                key_value_head * head_dim ..][0..head_dim];
            const weight = scores[position];
            var index: usize = 0;
            while (index + lanes <= head_dim) : (index += lanes) {
                const current: Vector = head_out[index..][0..lanes].*;
                const addend: Vector = value_row[index..][0..lanes].*;
                head_out[index..][0..lanes].* = current + addend * @as(Vector, @splat(weight));
            }
            while (index < head_dim) : (index += 1) head_out[index] += value_row[index] * weight;
        }
    }
}

const assert = std.debug.assert;

test "a linear layer multiplies a batch of tokens" {
    // Two tokens, two rows, three columns.
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const weights = [_]f32{ 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
    var out: [4]f32 = undefined;
    try linearF32(&out, &x, &weights, 2, 3, 2);
    try std.testing.expectEqual(@as(f32, 1.0), out[0]);
    try std.testing.expectEqual(@as(f32, 5.0), out[1]);
    try std.testing.expectEqual(@as(f32, 4.0), out[2]);
    try std.testing.expectEqual(@as(f32, 11.0), out[3]);
}

test "linear rejects mismatched shapes instead of reading out of bounds" {
    var out: [4]f32 = undefined;
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const weights = [_]f32{ 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
    try std.testing.expectError(Error.ShapeMismatch, linearF32(&out, &x, &weights, 2, 3, 1));
    try std.testing.expectError(
        Error.ShapeMismatch,
        linearF32(&out, x[0..3], &weights, 2, 3, 2),
    );
}

test "f16 and f32 linear layers agree on exactly representable weights" {
    // Two tokens, four columns each.
    const x = [_]f32{ 0.5, -1.5, 2.0, 1.0, -0.5, 0.25, 1.0, -2.0 };
    const weights_f32 = [_]f32{ 2.0, 0.5, -1.0, 0.0, 1.0, 1.0, 0.5, -2.0 };
    var weights_f16: [8]u16 = undefined;
    for (weights_f32, &weights_f16) |value, *bits| bits.* = half_float.toF16(value);

    var out_f32: [4]f32 = undefined;
    var out_f16: [4]f32 = undefined;
    var row_scratch: [4]f32 = undefined;
    try linearF32(&out_f32, &x, &weights_f32, 2, 4, 2);
    try linearF16(&out_f16, &x, &weights_f16, 2, 4, 2, &row_scratch);
    for (out_f32, out_f16) |expected, actual| try std.testing.expectEqual(expected, actual);
}

test "quantized linear matches a dequantized dense linear" {
    const cols: u32 = 128;
    const tokens: u32 = 3;
    const rows: u32 = 4;
    var values: [rows * cols]f32 = undefined;
    for (&values, 0..) |*value, index| {
        const position: f32 = @floatFromInt(index);
        value.* = @cos(position * 0.07) * 1.25;
    }
    var x: [tokens * cols]f32 = undefined;
    for (&x, 0..) |*value, index| {
        const position: f32 = @floatFromInt(index);
        value.* = @sin(position * 0.11) * 0.75;
    }

    // Sizes are comptime arithmetic on the format constants rather than a
    // runtime layout call, so the buffers can be stack arrays.
    const groups_per_row = cols / quant.group_size;
    const scales_len = rows * groups_per_row * quant.q4_scale_bytes_per_group;
    const data_len = rows * groups_per_row * quant.q4_data_bytes_per_group;
    var scales: [scales_len]u8 = undefined;
    var data: [data_len]u8 = undefined;
    for (0..rows) |row| {
        const row_values = values[row * cols ..][0..cols];
        quant.quantizeRow(
            .q4,
            row_values,
            scales[row * groups_per_row * quant.q4_scale_bytes_per_group ..][0 .. groups_per_row * quant.q4_scale_bytes_per_group],
            data[row * groups_per_row * quant.dataBytesPerGroup(.q4) ..][0 .. groups_per_row * quant.dataBytesPerGroup(.q4)],
        );
    }

    var actual: [tokens * rows]f32 = undefined;
    var row_scratch: [cols]f32 = undefined;
    try linearQuantized(&actual, &x, .q4, &scales, &data, rows, cols, tokens, &row_scratch);

    var expected: [tokens * rows]f32 = undefined;
    var decoded: [cols]f32 = undefined;
    for (0..rows) |row| {
        quant.dequantizeRow(
            .q4,
            scales[row * groups_per_row * quant.q4_scale_bytes_per_group ..],
            data[row * groups_per_row * quant.dataBytesPerGroup(.q4) ..],
            &decoded,
        );
        for (0..tokens) |token| {
            expected[token * rows + row] = dotF32(&decoded, x[token * cols ..][0..cols]);
        }
    }
    for (expected, actual) |expected_value, actual_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual_value, 1e-5);
    }
}

test "the matrix dispatch covers every storage kind" {
    const cols: u32 = 64;
    const rows: u32 = 2;
    var x: [64]f32 = undefined;
    @memset(&x, 1.0);
    var weights: [rows * cols]f32 = undefined;
    @memset(&weights, 1.0);
    var out: [rows]f32 = undefined;

    var row_scratch: [64]f32 = undefined;
    try (Matrix{ .f32 = &weights }).apply(&out, &x, cols, 1, &row_scratch);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), out[1], 1e-4);

    var weights_f16: [rows * cols]u16 = undefined;
    for (&weights_f16) |*bits| bits.* = half_float.toF16(1.0);
    try (Matrix{ .f16 = &weights_f16 }).apply(&out, &x, cols, 1, &row_scratch);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), out[0], 1e-4);

    // Both rows are identical, so quantizing one row's worth of values and
    // using it for both rows keeps the plane sizes honest: one group each.
    var scales: [quant.q4_scale_bytes_per_group]u8 = undefined;
    var data: [quant.q4_data_bytes_per_group]u8 = undefined;
    quant.quantizeRow(.q4, weights[0..cols], &scales, &data);
    const matrix = Matrix{ .quantized = .{
        .format = .q4,
        .scales = &scales,
        .data = &data,
    } };
    // The dispatch reads the row count from the scale plane, so one group is
    // one row.
    try std.testing.expectEqual(@as(u32, 1), matrix.rows(cols));
    try std.testing.expectEqual(@as(u32, rows * cols), @as(u32, weights.len));
    var quantized_out: [1]f32 = undefined;
    try matrix.apply(&quantized_out, &x, cols, 1, &row_scratch);
    // A row of ones quantizes with scale 1/8 and code 7 -- the positive extreme
    // clamps, exactly as `quant` documents -- so each weight decodes to 0.875
    // and 64 identical columns sum to 56.
    try std.testing.expectEqual(@as(f32, 56.0), quantized_out[0]);
}

test "convolution matches a hand-computed three by three stride two result" {
    // A 3x3 input with a single output channel whose kernel picks the centre.
    var weights: [9]u16 = undefined;
    const kernel = [_]f32{ 0, 0, 0, 0, 1, 0, 0, 0, 0 };
    for (kernel, &weights) |value, *bits| bits.* = half_float.toF16(value);

    const input = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var out: [4]f32 = undefined;
    const bias = [_]f32{0.0};
    // Output is (3-1)/2+1 = 2 in each dimension.
    try conv3x3Stride2Gelu(&out, &input, &weights, &bias, .{
        .out_channels = 1,
        .in_channels = 1,
        .in_height = 3,
        .in_width = 3,
        .kernel = 3,
    });
    // Centre taps of a 3x3 stride-2 convolution with padding 1 are input
    // positions (0,0), (0,2), (2,0), (2,2) -> 1, 3, 7, 9. gelu is applied to
    // each, and gelu of a positive value keeps it near the identity.
    try std.testing.expectApproxEqAbs(math.gelu(1.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(math.gelu(3.0), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(math.gelu(7.0), out[2], 1e-5);
    try std.testing.expectApproxEqAbs(math.gelu(9.0), out[3], 1e-5);
}

test "a multi-channel convolution reduces over channels in weight order" {
    // The second and third convolutions of the audio tower are 480 input
    // channels wide, so the kernel has to reduce over channels *and* keep the
    // `[out][in][kh][kw]` weight layout the checkpoint uses. This case pins both:
    // channel zero contributes its centre tap, channel one its top-left tap, and
    // the four output positions differ only by which taps land inside the input.
    var weights: [2 * 9]u16 = undefined;
    const centre = [_]f32{ 0, 0, 0, 0, 1, 0, 0, 0, 0 };
    const top_left = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0, 0 };
    for (centre, 0..) |value, index| weights[index] = half_float.toF16(value);
    for (top_left, 0..) |value, index| weights[9 + index] = half_float.toF16(value);

    const input = [_]f32{
        1, 2, 3, 4, 5, 6, 7, 8, 9, // channel zero
        10, 20, 30, 40, 50, 60, 70, 80, 90, // channel one
    };
    var out: [4]f32 = undefined;
    const bias = [_]f32{0.0};
    try conv3x3Stride2Gelu(&out, &input, &weights, &bias, .{
        .out_channels = 1,
        .in_channels = 2,
        .in_height = 3,
        .in_width = 3,
        .kernel = 3,
    });
    // Centre taps of channel zero: 1, 3, 7, 9. Channel one's top-left tap lands
    // inside the input only for the last output position, where it reads the
    // middle of channel one's second row, 50.
    try std.testing.expectApproxEqAbs(math.gelu(1.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(math.gelu(3.0), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(math.gelu(7.0), out[2], 1e-5);
    try std.testing.expectApproxEqAbs(math.gelu(59.0), out[3], 1e-5);

    // A channel count that disagrees with the input is an error, not a partial
    // convolution: reading past the input plane would be a silent overrun.
    try std.testing.expectError(Error.ShapeMismatch, conv3x3Stride2Gelu(
        &out,
        &input,
        &weights,
        &bias,
        .{ .out_channels = 1, .in_channels = 3, .in_height = 3, .in_width = 3, .kernel = 3 },
    ));
}

test "windowed attention mixes only inside its window" {
    // Two steps, one head, head_dim 2, two separate windows of length one: each
    // output must equal its own value vector exactly.
    const width = 2;
    const q = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const k = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const v = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    // Zeroed so the assertions below also prove the kernel writes nothing
    // outside the window it was given.
    var out = std.mem.zeroes([4]f32);
    var scores: [2]f32 = undefined;

    try audioAttentionWindow(&out, width, &q, width, &k, width, &v, width, 0, 1, 1, width, &scores);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out[1], 1e-6);
    // Step one is outside this window and must be left alone.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[2], 1e-6);

    @memset(&out, 0.0);
    try audioAttentionWindow(&out, width, &q, width, &k, width, &v, width, 1, 1, 1, width, &scores);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), out[3], 1e-6);
}

test "attention over a window averages identical keys" {
    const q = [_]f32{ 1.0, 0.0, 1.0, 0.0 };
    const k = [_]f32{ 1.0, 0.0, 1.0, 0.0 };
    const v = [_]f32{ 2.0, 0.0, 4.0, 0.0 };
    var out = std.mem.zeroes([4]f32);
    var scores: [2]f32 = undefined;
    try audioAttentionWindow(&out, 2, &q, 2, &k, 2, &v, 2, 0, 2, 1, 2, &scores);
    // Equal scores mean equal weights, so the output is the mean of the values.
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[2], 1e-5);
}

test "rope rotates by position and leaves the norm invariant" {
    const head_dim: u32 = 4;
    const positions = [_]u32{ 0, 1, 7 };
    const max_positions = 8;
    var rotated: [12]f32 = undefined;
    for (0..3) |token| {
        @memcpy(rotated[token * 4 ..][0..4], &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    }
    var cos_table: [max_positions * 2]f32 = undefined;
    var sin_table: [max_positions * 2]f32 = undefined;
    try buildRopeTables(&cos_table, &sin_table, 2, 1000000.0, max_positions, 2);
    try ropeInPlace(&rotated, &positions, 1, head_dim, &cos_table, &sin_table, 2);

    // Position zero must be untouched, because every angle is zero.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rotated[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), rotated[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), rotated[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), rotated[3], 1e-6);

    // Rotation is norm preserving for every position.
    var token: usize = 0;
    while (token < positions.len) : (token += 1) {
        var squares: f32 = 0.0;
        for (rotated[token * head_dim ..][0..head_dim]) |value| squares += value * value;
        try std.testing.expectApproxEqAbs(@as(f32, 30.0), squares, 1e-4);
    }
}

test "inverse frequencies follow the theta convention" {
    // For index 0 the exponent is zero, so the frequency is exactly one.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), inverseFrequency(0, 128, 1e6), 1e-6);
    // For index head_dim/2 the exponent is -1, so the frequency is 1/theta.
    try std.testing.expectApproxEqAbs(
        @as(f32, 1e-6),
        inverseFrequency(64, 128, 1e6),
        1e-12,
    );
}

test "decode attention reads only the cached positions" {
    const heads: u32 = 2;
    const key_value_heads: u32 = 1;
    const head_dim: u32 = 2;
    const query = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    // Two cached positions, one key/value head.
    const keys = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const values = [_]f32{ 1.0, 1.0, 3.0, 3.0 };
    var out: [4]f32 = undefined;
    var scores: [2]f32 = undefined;
    try decodeAttentionStep(&out, &query, &keys, &values, 2, heads, key_value_heads, head_dim, &scores);

    // Query head 0 aligns with key 0, so it should weight value 0 more.
    try std.testing.expect(out[0] < 2.0);
    // Query head 1 aligns with key 1, so it should weight value 1 more.
    try std.testing.expect(out[2] > 2.0);
    // Both share the single key/value head, so their weights mirror.
    try std.testing.expectApproxEqAbs(out[0] + out[2], 4.0, 1e-5);
    try std.testing.expectApproxEqAbs(out[1] + out[3], 4.0, 1e-5);
}

test "decode attention refuses a cache that is not a whole number of rows" {
    const query = [_]f32{ 1.0, 0.0 };
    const keys = [_]f32{ 1.0, 0.0 };
    var out: [2]f32 = undefined;
    var scores: [2]f32 = undefined;
    // heads 1, key_value_heads 1, so a cached count of 2 needs four keys.
    try std.testing.expectError(
        Error.ShapeMismatch,
        decodeAttentionStep(&out, &query, &keys, &keys, 2, 1, 1, 2, &scores),
    );
}

test "silu mul is gated elementwise" {
    var out: [3]f32 = undefined;
    const gate = [_]f32{ 0.0, 1.0, -1.0 };
    const up = [_]f32{ 1.0, 2.0, 3.0 };
    try siluMul(&out, &gate, &up);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(math.silu(1.0) * 2.0, out[1], 1e-6);
    try std.testing.expectApproxEqAbs(math.silu(-1.0) * 3.0, out[2], 1e-6);
}

test "bias and residual additions are consistent" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const bias = [_]f32{ 10.0, 20.0 };
    try addBias(&x, &bias, 2, 2);
    try std.testing.expectEqualSlices(f32, &.{ 11.0, 22.0, 13.0, 24.0 }, &x);

    const residual = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    try addInPlace(&x, &residual);
    try std.testing.expectEqualSlices(f32, &.{ 12.0, 23.0, 14.0, 25.0 }, &x);
}
