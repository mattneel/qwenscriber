//! Self-verification of the numerical core.
//!
//! The same routine runs in three places:
//!
//!   * `zig build test` asserts the hashes against golden constants on the host,
//!   * the native `qwenscriber-selftest` tool prints them,
//!   * the WASM export `qw_selftest` reports them to JavaScript.
//!
//! Because the WASM backend doubles as the correctness oracle for the WebGPU
//! kernels, "the host build and the freestanding build compute the same bits"
//! is a property worth testing rather than assuming. Divergence between the
//! golden constants and either build is a real failure, not noise: every routine
//! exercised here is designed to be deterministic across targets.

const std = @import("std");
const math = @import("math.zig");
const mel = @import("mel.zig");
const quant = @import("quant.zig");
const dtype = @import("dtype.zig");
const tokenizer = @import("tokenizer.zig");

pub const Report = struct {
    /// Bit mask of failed checks; zero means the core verified itself.
    failures: u32,
    /// FNV-1a over the packed bytes of the fixed quantization pattern.
    quant_hash: u64,
    /// FNV-1a over the bits of the fixed log-mel output.
    mel_hash: u64,
    /// Mel band that responds most strongly to a 1 kHz tone.
    loudest_band: u32,
    /// Log-mel value the frontend produces for digital silence.
    silence_value: f32,

    pub const check_quant = 1 << 0;
    pub const check_mel = 1 << 1;
    pub const check_tokenizer = 1 << 2;
    pub const check_simd = 1 << 3;
    pub const check_math = 1 << 4;
};

/// Samples used by the mel check: a 1 kHz tone, which must excite the band
/// covering 1 kHz and nothing else particularly.
pub const tone_sample_count: usize = 16000;
pub const tone_frames: usize = 100;

pub fn fillTone(waveform: []f32) void {
    for (waveform, 0..) |*sample, index| {
        const position: f32 = @floatFromInt(index);
        sample.* = @sin(2.0 * std.math.pi * 1000.0 * position / 16000.0);
    }
}

pub fn run() Report {
    var report = Report{
        .failures = 0,
        .quant_hash = 0,
        .mel_hash = 0,
        .loudest_band = 0,
        .silence_value = 0.0,
    };

    if (!checkQuantization(&report)) report.failures |= Report.check_quant;
    if (!checkMel(&report)) report.failures |= Report.check_mel;
    if (!checkTokenizer()) report.failures |= Report.check_tokenizer;
    if (!checkSimd()) report.failures |= Report.check_simd;
    if (!checkMath()) report.failures |= Report.check_math;
    return report;
}

/// Quantizes a fixed pattern, hashes the packed bytes, and verifies that the
/// packed dot product agrees with a dense f32 dot product over the decoded
/// weights.
fn checkQuantization(report: *Report) bool {
    const cols: usize = 128;
    var values: [cols]f32 = undefined;
    var activations: [cols]f32 = undefined;
    for (&values, 0..) |*value, index| {
        const position: f32 = @floatFromInt(index);
        value.* = @cos(position * 0.19) * 1.75;
        activations[index] = @sin(position * 0.07) * 0.5;
    }

    var hasher = std.hash.Fnv1a_64.init();
    var ok = true;

    const formats = [_]dtype.Format{ .q4, .q5, .q8 };
    for (formats) |format| {
        var scales_buffer: [cols / quant.group_size * quant.q4_scale_bytes_per_group]u8 =
            undefined;
        var data_buffer: [cols / quant.group_size * quant.q8_data_bytes_per_group]u8 =
            undefined;
        const scales = scales_buffer[0 .. cols / quant.group_size *
            quant.q4_scale_bytes_per_group];
        const data = data_buffer[0 .. cols / quant.group_size * quant.dataBytesPerGroup(format)];
        quant.quantizeRow(format, &values, scales, data);
        hasher.update(scales);
        hasher.update(data);

        var decoded: [cols]f32 = undefined;
        quant.dequantizeRow(format, scales, data, &decoded);
        var dense: f32 = 0.0;
        for (decoded, activations) |weight, activation| dense += weight * activation;
        const packed_dot = quant.dotRow(format, scales, data, &activations);

        // The packed path must reproduce the dense path: same decoded weights,
        // same order of accumulation per group.
        if (@abs(dense - packed_dot) > 1e-4 * @max(1.0, @abs(dense))) ok = false;

        // And the decoded weights must be close to the originals.
        // The bound is one step, not half: the most positive value would need
        // code `bias`, which clamps to `bias - 1`. The extra allowance is the
        // stored f16 scale's own rounding: it carries 11 bits of significand, so
        // at the clamped end the error moves by up to `bias * 2^-11` of a step
        // (6.25% for q8, where the code count is largest).
        var max_abs: f32 = 0.0;
        for (values) |value| max_abs = @max(max_abs, @abs(value));
        const step = max_abs / @as(f32, @floatFromInt(quant.bias(format)));
        const bound = step * (1.0 + @as(f32, @floatFromInt(quant.bias(format))) * 0.00048828125) + 1e-6;
        for (values, decoded) |original, restored| {
            if (@abs(original - restored) > bound) ok = false;
        }
    }

    report.quant_hash = hasher.final();
    return ok;
}

/// Runs the log-mel frontend over a fixed tone and silence, hashing the tone's
/// features and recording where the tone lands.
fn checkMel(report: *Report) bool {
    var waveform: [tone_sample_count]f32 = undefined;
    fillTone(&waveform);

    var features: [mel.mel_bins * tone_frames]f32 = undefined;
    _ = mel.compute(&waveform, &features, tone_frames) catch return false;

    var hasher = std.hash.Fnv1a_64.init();
    var loudest_band: usize = 0;
    var loudest_value: f32 = -std.math.floatMax(f32);
    // Frame 40 covers samples well inside the clip, so no padding rule applies.
    for (0..mel.mel_bins) |bin| {
        const value = features[bin * tone_frames + 40];
        if (value > loudest_value) {
            loudest_value = value;
            loudest_band = bin;
        }
    }
    report.loudest_band = @intCast(loudest_band);

    for (features) |value| {
        if (!std.math.isFinite(value)) return false;
        hasher.update(std.mem.asBytes(&value));
    }
    report.mel_hash = hasher.final();

    // The band that responds most must be centred on 1 kHz (bin 25 of 201).
    var peak_bin: usize = 0;
    var peak_weight: f32 = 0.0;
    for (0..mel.frequency_bins) |bin| {
        const weight = mel.tables.filters[bin * mel.mel_bins + loudest_band];
        if (weight > peak_weight) {
            peak_weight = weight;
            peak_bin = bin;
        }
    }
    if (peak_bin < 24 or peak_bin > 26) return false;

    var silence: [8000]f32 = @splat(0.0);
    var silence_features: [mel.mel_bins * 50]f32 = undefined;
    _ = mel.compute(&silence, &silence_features, 50) catch return false;
    report.silence_value = silence_features[0];
    if (@abs(report.silence_value - -1.5) > 1e-6) return false;

    return true;
}

/// Verifies the byte alphabet round trips and that a decoded token sequence is
/// valid UTF-8.
fn checkTokenizer() bool {
    var encoded: [256]u8 = undefined;
    var encoded_len: usize = 0;
    const source = "Qwenscriber \xE4\xBD\xA0\xE5\xA5\xBD\xF0\x9F\x8E\x99";
    for (source) |byte| {
        encoded_len += std.unicode.utf8Encode(
            tokenizer.byteToCodepoint(byte),
            encoded[encoded_len..],
        ) catch return false;
    }

    var decoded: [256]u8 = undefined;
    const decoded_len = tokenizer.appendToken(&decoded, encoded[0..encoded_len]) catch return false;
    if (!std.mem.eql(u8, source, decoded[0..decoded_len])) return false;
    if (!std.unicode.utf8ValidateSlice(decoded[0..decoded_len])) return false;

    // Every byte must survive the alphabet individually.
    for (0..256) |value| {
        const byte: u8 = @intCast(value);
        const codepoint = tokenizer.byteToCodepoint(byte);
        const restored = tokenizer.codepointToByte(codepoint) orelse return false;
        if (restored != byte) return false;
    }
    return true;
}

/// Confirms the vectorized dot product path agrees with a scalar loop. This is
/// the check that actually exercises WASM SIMD: if `@Vector` code miscompiles or
/// is lowered differently on the freestanding target, this catches it.
fn checkSimd() bool {
    const count: usize = 4096;
    var left: [count]f32 = undefined;
    var right: [count]f32 = undefined;
    for (&left, &right, 0..) |*a, *b, index| {
        const position: f32 = @floatFromInt(index);
        a.* = @sin(position * 0.013) * 3.0;
        b.* = @cos(position * 0.017) * 2.0;
    }

    var scalar: f64 = 0.0;
    for (left, right) |a, b| scalar += @as(f64, a) * @as(f64, b);

    const vectorized = vectorDot(&left, &right);
    const tolerance: f32 = @floatCast(@abs(scalar) * 1e-4 + 1e-3);
    if (@abs(vectorized - @as(f32, @floatCast(scalar))) > tolerance) return false;

    // Also verify the vector path handles a length that is not a multiple of the
    // vector width, which is where tail handling goes wrong.
    const tail_count: usize = count + 3;
    var left_tail: [tail_count]f32 = undefined;
    var right_tail: [tail_count]f32 = undefined;
    @memcpy(left_tail[0..count], &left);
    @memcpy(right_tail[0..count], &right);
    left_tail[count] = 1.0;
    left_tail[count + 1] = -2.0;
    left_tail[count + 2] = 0.5;
    right_tail[count] = 4.0;
    right_tail[count + 1] = 4.0;
    right_tail[count + 2] = 4.0;
    const expected_tail = vectorized + 4.0 - 8.0 + 2.0;
    const actual_tail = vectorDot(&left_tail, &right_tail);
    if (@abs(expected_tail - actual_tail) > 1e-3) return false;

    return true;
}

/// Dot product written with `@Vector`, which is what the preprocessing and
/// fallback kernels rely on for WASM SIMD generation.
pub fn vectorDot(left: []const f32, right: []const f32) f32 {
    assert(left.len == right.len);
    const lanes = 4;
    const Vector = @Vector(lanes, f32);

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

/// Checks the numerically delicate helpers that the encoders depend on.
fn checkMath() bool {
    if (@abs(math.erf(1.0) - 0.8427007929497149) > 2e-7) return false;
    if (@abs(math.erf(-2.0) + 0.9953222650189527) > 2e-7) return false;
    if (@abs(math.gelu(1.0) - 0.84134475) > 1e-6) return false;
    if (@abs(math.silu(1.0) - 0.7310586) > 1e-6) return false;
    if (@abs(math.log10(1000.0) - 3.0) > 1e-6) return false;
    if (@abs(math.rsqrt(4.0) - 0.5) > 1e-7) return false;

    var row = [_]f32{ 1.0, 2.0, 3.0 };
    math.softmaxInPlace(&row);
    var total: f32 = 0.0;
    for (row) |value| total += value;
    if (@abs(total - 1.0) > 1e-6) return false;
    return true;
}

const assert = std.debug.assert;

test "the core verifies itself" {
    const report = run();
    try std.testing.expectEqual(@as(u32, 0), report.failures);
}

test "the fixed patterns hash to the values the freestanding build must match" {
    // These constants are the cross-target contract. If a change to a numeric
    // routine alters them, the WASM self-test in tools/wasm_selftest.mjs has to
    // be updated in the same commit, which is exactly the review signal we want.
    const report = run();
    try std.testing.expectEqual(@as(u32, 0), report.failures);
    try std.testing.expectEqual(@as(u64, 0xf247917b05691fd1), report.quant_hash);
    try std.testing.expectEqual(@as(u64, 0xfec7aacfeea768d7), report.mel_hash);
    try std.testing.expectEqual(@as(u32, 42), report.loudest_band);
    try std.testing.expectEqual(@as(f32, -1.5), report.silence_value);
}
