//! Deterministic scalar math shared by the host and `wasm32-freestanding` builds.
//!
//! Everything here must produce the same bits on every target, because the WASM
//! backend doubles as the correctness oracle for the WebGPU kernels. Only the
//! error function is implemented in this file; the remaining transcendentals are
//! lowered by the Zig compiler into its own `compiler_rt` implementations for
//! freestanding targets, which keeps the freestanding module free of libm.
//!
//! Precision budget: every routine is accurate to better than 1e-6 relative over
//! the input range the model exercises. `erf` is the weakest link at ~1.2e-7
//! absolute, which is four orders of magnitude below the bf16 weight noise in the
//! released checkpoints.

const std = @import("std");

/// Natural exponential. Accuracy follows the target's `compiler_rt`/libm `expf`.
pub fn exp(x: f32) f32 {
    return @exp(x);
}

/// Base-10 logarithm, as required by the Whisper-style log-mel frontend.
pub fn log10(x: f32) f32 {
    return std.math.log10(x);
}

/// Reciprocal square root, the core of every normalization layer.
pub fn rsqrt(x: f32) f32 {
    return 1.0 / @sqrt(x);
}

/// Sine and cosine, used for rotary embeddings and the mel filterbank.
pub fn sin(x: f32) f32 {
    return @sin(x);
}

pub fn cos(x: f32) f32 {
    return @cos(x);
}

/// Error function by rational approximation (Numerical Recipes, `erfc` form).
///
/// The approximation is the published Chebyshev fit with |error| < 1.2e-7 over
/// the whole real line. We use it instead of a series because GELU feeds on
/// `x / sqrt(2)` for arguments that routinely exceed 3, where the Taylor series
/// loses all significant digits.
pub fn erf(x: f32) f32 {
    const sign: f32 = if (x < 0.0) -1.0 else 1.0;
    const magnitude = @abs(x);
    const t = 1.0 / (1.0 + 0.5 * magnitude);
    const tau = t * exp(-magnitude * magnitude - 1.26551223 +
        t * (1.00002368 +
            t * (0.37409196 +
                t * (0.09678418 +
                    t * (-0.18628806 +
                        t * (0.27886807 +
                            t * (-1.13520398 +
                                t * (1.48851587 +
                                    t * (-0.82215223 + t * 0.17087277)))))))));
    return sign * (1.0 - tau);
}

/// Exact (error-function) GELU, matching `torch.nn.functional.gelu` defaults.
pub fn gelu(x: f32) f32 {
    const inverse_sqrt_two: f32 = 0.70710678;
    return 0.5 * x * (1.0 + erf(x * inverse_sqrt_two));
}

/// Sigmoid linear unit, the decoder MLP gate activation.
pub fn silu(x: f32) f32 {
    return x / (1.0 + exp(-x));
}

/// Numerically stable softmax over a row, in place.
///
/// `max` is subtracted before exponentiation so that the largest logit maps to
/// 1.0; the attention path relies on that for long audio windows.
pub fn softmaxInPlace(row: []f32) void {
    assert(row.len > 0);
    var row_max: f32 = row[0];
    for (row[1..]) |value| {
        row_max = @max(row_max, value);
    }
    var total: f32 = 0.0;
    for (row) |*value| {
        value.* = exp(value.* - row_max);
        total += value.*;
    }
    assert(total > 0.0);
    const inverse_total: f32 = 1.0 / total;
    for (row) |*value| {
        value.* *= inverse_total;
    }
}

/// LayerNorm (mean/variance), used by the audio encoder.
///
/// The variance is biased (divided by `count`), matching `torch.nn.LayerNorm`.
/// Precision mirrors the reference implementation: statistics in f32.
pub fn layerNormInPlace(row: []f32, weight: []const f32, bias: []const f32, eps: f32) void {
    assert(row.len > 0);
    assert(row.len == weight.len);
    assert(row.len == bias.len);
    var sum: f32 = 0.0;
    for (row) |value| sum += value;
    const mean: f32 = sum / @as(f32, @floatFromInt(row.len));
    var sum_squares: f32 = 0.0;
    for (row) |value| {
        const centered = value - mean;
        sum_squares += centered * centered;
    }
    const variance: f32 = sum_squares / @as(f32, @floatFromInt(row.len));
    const inverse_std: f32 = rsqrt(variance + eps);
    for (row, weight, bias) |*value, scale, offset| {
        value.* = (value.* - mean) * inverse_std * scale + offset;
    }
}

/// Root-mean-square normalization without a mean subtraction, used by the
/// Qwen3 text decoder. `weight` is applied after normalization, in f32.
pub fn rmsNormInPlace(row: []f32, weight: []const f32, eps: f32) void {
    assert(row.len > 0);
    assert(row.len == weight.len);
    var sum_squares: f32 = 0.0;
    for (row) |value| sum_squares += value * value;
    const mean_square: f32 = sum_squares / @as(f32, @floatFromInt(row.len));
    const inverse_rms: f32 = rsqrt(mean_square + eps);
    for (row, weight) |*value, scale| {
        value.* = value.* * inverse_rms * scale;
    }
}

const assert = std.debug.assert;

test "erf matches high precision reference values" {
    // Reference values from the C standard library `erf` at double precision.
    const cases = [_]struct { x: f32, expected: f32 }{
        .{ .x = 0.0, .expected = 0.0 },
        .{ .x = 0.1, .expected = 0.1124629160182849 },
        .{ .x = 0.5, .expected = 0.5204998778130465 },
        .{ .x = 1.0, .expected = 0.8427007929497149 },
        .{ .x = 1.5, .expected = 0.9661051464753107 },
        .{ .x = 2.0, .expected = 0.9953222650189527 },
        .{ .x = 3.0, .expected = 0.9999779095030014 },
        .{ .x = 4.0, .expected = 0.9999999845827421 },
        .{ .x = -0.75, .expected = -0.7111556336535151 },
        .{ .x = -2.5, .expected = -0.999593047982555 },
    };
    for (cases) |case| {
        const actual = erf(case.x);
        try std.testing.expectApproxEqAbs(case.expected, actual, 2e-7);
    }
}

test "erf is odd and saturates" {
    try std.testing.expectEqual(@as(f32, 0.0), erf(0.0));
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), erf(-40.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), erf(40.0), 1e-6);
    var x: f32 = -6.0;
    while (x <= 6.0) : (x += 0.25) {
        try std.testing.expectApproxEqAbs(-erf(-x), erf(x), 1e-7);
    }
}

test "gelu matches the exact error-function definition" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gelu(0.0), 1e-7);
    // gelu(1) = 0.5 * (1 + erf(1/sqrt(2))) = 0.8413447460685429
    try std.testing.expectApproxEqAbs(@as(f32, 0.84134475), gelu(1.0), 1e-6);
    // gelu(-1) = -0.15865525393145707
    try std.testing.expectApproxEqAbs(@as(f32, -0.15865525), gelu(-1.0), 1e-6);
    // Large negative inputs must vanish rather than go negative.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gelu(-10.0), 1e-6);
}

test "silu agrees with its definition" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(0.0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), silu(1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.26894143), silu(-1.0), 1e-6);
    // Saturates to the identity for large positive input and to zero for large
    // negative input; the decoder's gate never leaves that range.
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), silu(30.0), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(-30.0), 1e-9);
}

test "softmax is a probability distribution and extreme inputs do not overflow" {
    var row = [_]f32{ 1.0, 2.0, 3.0 };
    softmaxInPlace(&row);
    var total: f32 = 0.0;
    for (row) |value| total += value;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.09003057), row[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.24472848), row[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.66524096), row[2], 1e-6);

    var extreme = [_]f32{ 1000.0, -1000.0, 0.0 };
    softmaxInPlace(&extreme);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), extreme[0], 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), extreme[1]);
}

test "rms norm scales a unit row to unit rms" {
    var row = [_]f32{ 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0 };
    rmsNormInPlace(&row, &weight, 0.0);
    // rms(3,4) = sqrt(12.5); the normalized row must have unit mean square.
    const mean_square = (row[0] * row[0] + row[1] * row[1]) / 2.0;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mean_square, 1e-6);
}

test "rms norm remains finite for an all-zero row" {
    var row = [_]f32{ 0.0, 0.0, 0.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0 };
    rmsNormInPlace(&row, &weight, 1e-6);
    for (row) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
}
