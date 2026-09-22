//! Half-precision and brain-float bit conversions.
//!
//! Conversions round to nearest, ties to even, matching what `torch` produces
//! when it casts a checkpoint's tensors. Quantized groups store their scale in
//! f16, so a single bit of difference here moves every decoded weight in a group.

const std = @import("std");

/// f32 -> f16 bits, round to nearest even.
///
/// The compiler's own `f32 -> f16` conversion is correctly rounded on every
/// target (including `wasm32-freestanding`, where `compiler_rt` supplies it), so
/// we defer to it rather than reimplementing the rounding logic.
pub fn toF16(value: f32) u16 {
    const half: f16 = @floatCast(value);
    return @bitCast(half);
}

/// f16 bits -> f32.
pub fn fromF16(bits: u16) f32 {
    const half: f16 = @bitCast(bits);
    return @floatCast(half);
}

/// f32 -> bfloat16 bits, round to nearest even.
///
/// bfloat16 is the top 16 bits of f32, so rounding is decided by the 16 bits
/// being discarded. A tie (exactly 0x8000 in the low half) rounds up only when
/// the kept mantissa is odd.
pub fn toBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    if ((bits & 0x7FFF_FFFF) > 0x7F80_0000) {
        // NaN: keep it a NaN rather than rounding the payload into infinity.
        return @truncate((bits >> 16) | 0x0040);
    }
    const round_bias: u32 = 0x7FFF + ((bits >> 16) & 1);
    const rounded = bits +% round_bias;
    return @truncate(rounded >> 16);
}

/// bfloat16 bits -> f32.
pub fn fromBf16(bits: u16) f32 {
    const widened: u32 = @as(u32, bits) << 16;
    return @bitCast(widened);
}

test "f16 round trip preserves exactly representable values" {
    for ([_]f32{ 0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1024.0, 0.0009765625 }) |value| {
        try std.testing.expectEqual(value, fromF16(toF16(value)));
    }
    try std.testing.expectEqual(@as(f32, 65504.0), fromF16(toF16(65504.0)));
}

test "f16 rounds to nearest even at the boundary" {
    // 1 + 2^-11 is exactly halfway between the two neighbouring f16 values.
    const halfway: f32 = 1.0 + 1.0 / 2048.0;
    // 1.0 has an even mantissa, so a tie must stay at 1.0.
    try std.testing.expectEqual(@as(f32, 1.0), fromF16(toF16(halfway)));
    // 1 + 3*2^-11 is halfway above 1 + 2^-10, whose mantissa is odd, so it
    // rounds to 1 + 2^-9.
    const above: f32 = 1.0 + 3.0 / 2048.0;
    try std.testing.expectEqual(@as(f32, 1.0 + 2.0 / 1024.0), fromF16(toF16(above)));
}

test "f16 overflow saturates to infinity" {
    try std.testing.expect(std.math.isPositiveInf(fromF16(toF16(1.0e30))));
    try std.testing.expect(std.math.isNegativeInf(fromF16(toF16(-1.0e30))));
}

test "f16 conversion preserves scale magnitudes used by quantization" {
    var value: f32 = 1e-4;
    while (value < 1.0) : (value *= 1.7) {
        const round_tripped = fromF16(toF16(value));
        const relative = @abs(round_tripped - value) / value;
        try std.testing.expect(relative < 0.0005);
    }
}

test "bf16 round trips values with eight mantissa bits" {
    for ([_]f32{ 0.0, 1.0, -1.0, 0.5, 1.5, -0.25, 256.0, 1024.0, 65536.0 }) |value| {
        try std.testing.expectEqual(value, fromBf16(toBf16(value)));
    }
}

test "bf16 truncates the low mantissa bits of f32" {
    const value: f32 = 1.0e10;
    const restored = fromBf16(toBf16(value));
    // bf16 keeps 8 significand bits, so the relative error is bounded by 2^-9.
    try std.testing.expect(@abs(restored - value) / value < 0.002);
    try std.testing.expect(restored != value);
}

test "bf16 rounds to nearest even" {
    // The low 16 bits are exactly halfway, so the kept mantissa decides.
    const even_bits: u32 = 0x3F80_8000; // 1.001953125
    const even_value: f32 = @bitCast(even_bits);
    try std.testing.expectEqual(@as(f32, 1.0), fromBf16(toBf16(even_value)));

    const odd_bits: u32 = 0x3F81_8000; // 1.00390625 * (1 + 2^-8)
    const odd_value: f32 = @bitCast(odd_bits);
    const expected: f32 = @bitCast(@as(u32, 0x3F82_0000));
    try std.testing.expectEqual(expected, fromBf16(toBf16(odd_value)));
}

test "bf16 keeps infinities and does not turn NaN into infinity" {
    try std.testing.expect(std.math.isPositiveInf(fromBf16(toBf16(std.math.inf(f32)))));
    const quiet_nan: f32 = @bitCast(@as(u32, 0x7FC0_1234));
    try std.testing.expect(std.math.isNan(fromBf16(toBf16(quiet_nan))));
    const loud_nan: f32 = @bitCast(@as(u32, 0x7F80_0001));
    try std.testing.expect(std.math.isNan(fromBf16(toBf16(loud_nan))));
}
