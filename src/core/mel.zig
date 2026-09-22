//! Whisper-style log-mel frontend for Qwen3-ASR.
//!
//! The released pipeline feeds the audio encoder a 128-bin log-mel spectrogram
//! with a 400-sample FFT, a 160-sample hop, and Slaney-spaced, Slaney-normalized
//! mel filters between 0 Hz and 8 kHz. This module reproduces that transform
//! closely enough that the encoder sees the same numbers a `transformers`
//! pipeline would produce; `src/host/reference_check.zig` compares it against
//! fixtures generated from that pipeline.
//!
//! # Matching the reference exactly
//!
//!   * `torch.stft(center=True)` reflect-pads by `n_fft / 2 = 200` samples at
//!     both ends of its input buffer and frames at `hop_length`, then the
//!     reference discards the final frame.
//!   * The window is `torch.hann_window(400)`, which is periodic: it is one
//!     sample shorter than a symmetric Hann window of the same length.
//!   * The reference computes `stft.abs() ** 2`. We compute `re^2 + im^2`
//!     directly, which differs only by the rounding a square root and square
//!     introduce in f32.
//!   * The tail is `log10(max(energy, 1e-10))`, then `max(log, global_max - 8)`,
//!     then `(x + 4) / 4`, with `global_max` taken over the whole spectrogram.
//!
//! # The 30-second buffer matters
//!
//! The reference does not transform the clip as supplied. It zero-pads the clip
//! to a whole 30 seconds (`capacity_samples`), transforms *that* buffer, and
//! keeps only the frames whose first sample is inside the real audio. Two
//! consequences shape this implementation:
//!
//!   * Samples past the end of the clip are zeros, not a reflection of the real
//!     audio. Reflecting there would silently change the last few frames of
//!     every clip, so `sampleAt` returns zero for that whole region and only
//!     mirrors at the very end of the 30-second capacity.
//!   * Frames that survive are `floor(sample_count / hop_length)`: a final partial
//!     hop is dropped, because that is what the reference processor marks valid
//!     and therefore what the model is fed. The feature extractor called on its
//!     own keeps that frame, so the two disagree by one and the processor wins.
//!
//! Frames that fall in the zero-padded region hold samples of pure digital
//! silence. Their log-mel value is not zero either: silence clamps to `1e-10`,
//! whose base-10 logarithm is -10, which the dynamic-range clamp then lifts to
//! `global_max - 8`. `compute` therefore returns the global maximum it measured
//! so callers can fill those frames with the same value the reference would.

const std = @import("std");
const math = @import("math.zig");

pub const n_fft: u32 = 400;
pub const hop_length: u32 = 160;
pub const mel_bins: u32 = 128;
/// Frequency bins kept from a real transform of `n_fft` samples.
pub const frequency_bins: u32 = n_fft / 2 + 1;
/// Samples of padding on each side introduced by `center = true`.
pub const center_padding: u32 = n_fft / 2;
/// Audio capacity of the reference buffer: 30 seconds at 16 kHz.
pub const capacity_samples: usize = 480000;
/// Frames that capacity holds at `hop_length`.
pub const capacity_frames: usize = capacity_samples / hop_length;
/// The reference zero-pads clips below this length before transforming.
pub const min_samples: usize = 8000;
pub const sample_rate_hz: u32 = 16000;

/// Lowest mel energy the reference clamps to.
const clamp_floor: f32 = 1e-10;
/// Dynamic range kept below the global maximum, in log10 units.
const dynamic_range: f32 = 8.0;
/// Offset and scale of the final normalization.
const normalization_offset: f32 = 4.0;
const normalization_scale: f32 = 4.0;
/// Log10 magnitude of a frame of pure digital silence, before the range clamp.
const silence_log: f32 = -10.0;

pub const Error = error{
    /// The output slice is not exactly `mel_bins * frame_count` elements.
    OutputShapeMismatch,
    /// More frames were requested than the audio can supply.
    FrameCountExceedsAudio,
    /// More frames were requested than the reference buffer can hold.
    FrameCountExceedsCapacity,
    /// An empty waveform has no defined transform.
    EmptyWaveform,
};

/// Precomputed tables. Every entry is derived at compile time from the fixed
/// geometry above, so constructing the frontend costs nothing at runtime, needs
/// no allocation, and cannot fail.
pub const Tables = struct {
    /// Periodic Hann window.
    window: [n_fft]f32,
    /// Mel projection matrix, `frequency_bins x mel_bins`, Slaney-normalized.
    filters: [frequency_bins * mel_bins]f32,
};

pub const tables: Tables = computeTables();

/// Frames the reference frontend produces for `sample_count` samples.
///
/// A final partial hop is dropped. That is what the reference *processor* marks
/// valid, and the processor is what the model is fed: measured against the
/// installed implementation at 12345 -> 77, 67200 -> 420, 67263 -> 420,
/// 67264 -> 420, 80000 -> 500, and 480000 -> 3000 samples. Calling the feature
/// extractor directly instead keeps the partial frame (12345 -> 78,
/// 67263 -> 421), so the two disagree by one and this is the rule that matters.
pub fn framesForSamples(sample_count: usize) usize {
    return sample_count / hop_length;
}

/// Clip length the reference effectively transforms, after its minimum-length
/// zero padding.
pub fn paddedSampleCount(sample_count: usize) usize {
    return @max(sample_count, min_samples);
}

/// Computes log-mel features for a complete clip.
///
/// `out` must be exactly `mel_bins * frame_count` elements and is written
/// row-major as `[mel_bin][frame]`, matching the reference's `(bins, frames)`
/// layout. `frame_count` must not exceed `framesForSamples(waveform.len)`.
///
/// Returns the largest log10 mel magnitude seen before clamping, which callers
/// need in order to fill later frames of a partial encoder chunk with
/// `fillPaddingFrames`.
pub fn compute(waveform: []const f32, out: []f32, frame_count: usize) Error!f32 {
    if (waveform.len == 0) return Error.EmptyWaveform;
    if (out.len != @as(usize, mel_bins) * frame_count) return Error.OutputShapeMismatch;
    if (frame_count > framesForSamples(waveform.len)) return Error.FrameCountExceedsAudio;
    if (frame_count > capacity_frames) return Error.FrameCountExceedsCapacity;
    if (frame_count == 0) return silence_log;

    // Frames beyond `frame_count` hold pure silence in the reference, whose
    // logarithm is `silence_log`, so the global maximum can never be below it.
    var global_max_log: f32 = silence_log;

    for (0..frame_count) |frame_index| {
        var windowed: [n_fft]f32 = undefined;
        frameWindowed(waveform, frame_index, &windowed);
        var spectrum: [frequency_bins]f32 = undefined;
        powerSpectrum(&windowed, &spectrum);
        const column = out[frame_index..];
        projectMel(&spectrum, column, frame_count);
        global_max_log = @max(global_max_log, columnMax(column, frame_count));
    }

    // Clamping and normalization need the global maximum, so they run in a
    // second pass over the frames.
    const floor = @max(global_max_log - dynamic_range, silence_log);
    for (0..frame_count) |frame_index| {
        const column = out[frame_index..];
        var bin: usize = 0;
        while (bin < mel_bins) : (bin += 1) {
            const clamped = @max(column[bin * frame_count], floor);
            column[bin * frame_count] = (clamped + normalization_offset) / normalization_scale;
        }
    }
    return global_max_log;
}

/// Value the reference produces for a frame of zero-padded audio that lies
/// inside a partially filled encoder chunk.
///
/// `global_max_log` is the value `compute` returned for the same clip.
pub fn paddingFrameValue(global_max_log: f32) f32 {
    const clamped = @max(global_max_log - dynamic_range, silence_log);
    return (clamped + normalization_offset) / normalization_scale;
}

/// Fills `count` frames starting at `start_frame` with the padding value, so a
/// partially filled encoder chunk matches the reference's mel-domain padding
/// (`np.pad(features, ...)`, which pads with zeros, is only used *past* the
/// audio; within the audio region the zeros live in the audio itself).
pub fn fillPaddingFrames(
    out: []f32,
    frames: usize,
    start_frame: usize,
    count: usize,
    global_max_log: f32,
) void {
    assert(out.len == @as(usize, mel_bins) * frames);
    assert(start_frame + count <= frames);
    const value = paddingFrameValue(global_max_log);
    var frame_index = start_frame;
    while (frame_index < start_frame + count) : (frame_index += 1) {
        var bin: usize = 0;
        while (bin < mel_bins) : (bin += 1) {
            out[bin * frames + frame_index] = value;
        }
    }
}

const assert = std.debug.assert;

fn columnMax(column: []const f32, stride: usize) f32 {
    var maximum: f32 = -std.math.floatMax(f32);
    var bin: usize = 0;
    while (bin < mel_bins) : (bin += 1) {
        maximum = @max(maximum, column[bin * stride]);
    }
    return maximum;
}

/// Multiplies the power spectrum by the mel filterbank and takes the logarithm.
fn projectMel(spectrum: *const [frequency_bins]f32, column: []f32, stride: usize) void {
    var filter_index: usize = 0;
    while (filter_index < mel_bins) : (filter_index += 1) {
        var accumulator: f32 = 0.0;
        var bin: usize = 0;
        while (bin < frequency_bins) : (bin += 1) {
            accumulator += tables.filters[bin * mel_bins + filter_index] * spectrum[bin];
        }
        // `clamp(mel_spec, min=1e-10).log10()`: the clamp applies to the raw
        // energy, before the logarithm.
        column[filter_index * stride] = math.log10(@max(accumulator, clamp_floor));
    }
}

/// |X[k]|^2 for k in 0..frequency_bins of one framed, windowed frame.
fn powerSpectrum(windowed: *const [n_fft]f32, spectrum: *[frequency_bins]f32) void {
    var stage_real: [n_fft]f32 = undefined;
    var stage_imag: [n_fft]f32 = undefined;

    // Stage one: 16 transforms of length 25 over the strided input.
    var fast_index: usize = 0;
    while (fast_index < fast_size) : (fast_index += 1) {
        var slow_bin: usize = 0;
        while (slow_bin < slow_size) : (slow_bin += 1) {
            var sum_real: f32 = 0.0;
            var sum_imag: f32 = 0.0;
            var slow_index: usize = 0;
            while (slow_index < slow_size) : (slow_index += 1) {
                const twiddle = slow_twiddles[(slow_index * slow_bin) % slow_size];
                const sample = windowed[slow_index * fast_size + fast_index];
                sum_real += sample * twiddle[0];
                sum_imag += sample * twiddle[1];
            }
            stage_real[slow_bin * fast_size + fast_index] = sum_real;
            stage_imag[slow_bin * fast_size + fast_index] = sum_imag;
        }
    }

    // Stage two: the twiddle rotation between the two transforms, W_N^(n2 * k1)
    // where the flat index is `k1 * fast_size + n2`.
    for (0..n_fft) |index| {
        const fast_bin = index % fast_size;
        const slow_bin = index / fast_size;
        const twiddle = combined_twiddles[(fast_bin * slow_bin) % n_fft];
        const real = stage_real[index];
        const imag = stage_imag[index];
        stage_real[index] = real * twiddle[0] - imag * twiddle[1];
        stage_imag[index] = real * twiddle[1] + imag * twiddle[0];
    }

    // Stage three: transforms of length 16 over the fast index. Only bins up to
    // Nyquist are needed, so the loop over `fast_bin` stops at the last useful
    // bin of each slow bin.
    var slow_bin: usize = 0;
    while (slow_bin < slow_size) : (slow_bin += 1) {
        const bin_limit = (frequency_bins - 1 - slow_bin) / slow_size;
        var fast_bin: usize = 0;
        while (fast_bin <= bin_limit) : (fast_bin += 1) {
            var sum_real: f32 = 0.0;
            var sum_imag: f32 = 0.0;
            var inner_index: usize = 0;
            while (inner_index < fast_size) : (inner_index += 1) {
                const twiddle = fast_twiddles[(inner_index * fast_bin) % fast_size];
                const real = stage_real[slow_bin * fast_size + inner_index];
                const imag = stage_imag[slow_bin * fast_size + inner_index];
                sum_real += real * twiddle[0] - imag * twiddle[1];
                sum_imag += real * twiddle[1] + imag * twiddle[0];
            }
            const bin = slow_bin + slow_size * fast_bin;
            assert(bin < frequency_bins);
            spectrum[bin] = sum_real * sum_real + sum_imag * sum_imag;
        }
    }
}

/// Extracts and windows one frame, matching `torch.stft(center=True)`.
fn frameWindowed(waveform: []const f32, frame_index: usize, windowed: *[n_fft]f32) void {
    const origin = @as(isize, @intCast(frame_index * hop_length)) -
        @as(isize, @intCast(center_padding));
    for (windowed, 0..) |*sample, offset| {
        sample.* = sampleAt(waveform, origin + @as(isize, @intCast(offset))) *
            tables.window[offset];
    }
}

/// Reads one sample of the reference's zero-padded, reflect-padded buffer.
///
/// The reference buffer is `[real audio][zeros up to 30 s]`, reflect-padded by
/// `center_padding` at both ends, so:
///
///   * before the start, mirror the real audio;
///   * inside the clip, read it;
///   * after the clip but inside the 30-second capacity, read zero;
///   * past the capacity, mirror the end of the buffer.
///
/// Mirroring is clamped rather than wrapped so that audio shorter than
/// `center_padding` still produces a defined result; the released pipeline
/// zero-pads every clip to at least 8000 samples, so the clamp never triggers
/// on the real path.
fn sampleAt(waveform: []const f32, position: isize) f32 {
    if (position < 0) {
        const mirrored = -position;
        if (mirrored > @as(isize, @intCast(waveform.len - 1))) return waveform[waveform.len - 1];
        return waveform[@intCast(mirrored)];
    }
    const plain: usize = @intCast(position);
    if (plain < waveform.len) return waveform[plain];
    if (plain < capacity_samples) return 0.0;

    const last: isize = @intCast(capacity_samples - 1);
    const mirrored = 2 * last - position;
    if (mirrored < 0) return 0.0;
    const mirrored_index: usize = @intCast(mirrored);
    if (mirrored_index < waveform.len) return waveform[mirrored_index];
    return 0.0;
}

// The transform is a Cooley-Tukey split of the 400-point DFT into
// `slow_size = 25` transforms of length `fast_size = 16` and 16 transforms of
// length 25, since 400 = 25 * 16. Both sub-transforms use precomputed twiddle
// tables, which keeps the code obviously correct; measurement (`zig build
// bench`) shows the cost is far below real time, so the more intricate
// radix-5/radix-2 formulation is not yet justified.
const slow_size = 25;
const fast_size = 16;

const slow_twiddles = computeTwiddles(slow_size);
const fast_twiddles = computeTwiddles(fast_size);
const combined_twiddles = computeTwiddles(n_fft);

fn computeTwiddles(comptime size: usize) [size][2]f32 {
    var result: [size][2]f32 = undefined;
    for (&result, 0..) |*entry, index| {
        const angle = -2.0 * std.math.pi * @as(f64, @floatFromInt(index)) /
            @as(f64, @floatFromInt(size));
        entry.* = .{ @floatCast(@cos(angle)), @floatCast(@sin(angle)) };
    }
    return result;
}

fn computeTables() Tables {
    // Building the filterbank at compile time costs about 26k inner iterations
    // plus the twiddle tables; the default evaluation quota is far below that.
    @setEvalBranchQuota(1_000_000);
    var result: Tables = undefined;
    computeWindow(&result.window);
    computeFilters(&result.filters);
    return result;
}

fn computeWindow(window: *[n_fft]f32) void {
    // Periodic Hann: w[i] = 0.5 - 0.5 * cos(2*pi*i/n_fft). numpy builds the
    // same sequence as np.hanning(n_fft + 1)[:-1].
    const length: f64 = n_fft;
    for (window, 0..) |*value, index| {
        const phase = 2.0 * std.math.pi * @as(f64, @floatFromInt(index)) / length;
        value.* = @floatCast(0.5 - 0.5 * @cos(phase));
    }
}

/// Slaney-scale mel filterbank, matching `transformers.audio_utils.mel_filter_bank`
/// with `norm="slaney"`, `mel_scale="slaney"`.
fn computeFilters(filters: *[frequency_bins * mel_bins]f32) void {
    const filter_count = mel_bins + 2;
    var mel_frequencies: [filter_count]f64 = undefined;
    const mel_min = hertzToMel(0.0);
    const mel_max = hertzToMel(8000.0);
    for (&mel_frequencies, 0..) |*value, index| {
        const fraction = @as(f64, @floatFromInt(index)) /
            @as(f64, @floatFromInt(filter_count - 1));
        value.* = mel_min + (mel_max - mel_min) * fraction;
    }

    var filter_frequencies: [filter_count]f64 = undefined;
    for (&filter_frequencies, mel_frequencies) |*value, mel| value.* = melToHertz(mel);

    // Slaney area normalization: each band is scaled to roughly constant energy.
    var enorm: [mel_bins]f64 = undefined;
    for (&enorm, 0..) |*value, index| {
        value.* = 2.0 / (filter_frequencies[index + 2] - filter_frequencies[index]);
    }

    // FFT bin centre frequencies, from 0 Hz to sample_rate / 2 inclusive.
    var fft_frequencies: [frequency_bins]f64 = undefined;
    for (&fft_frequencies, 0..) |*value, index| {
        const fraction = @as(f64, @floatFromInt(index)) /
            @as(f64, @floatFromInt(frequency_bins - 1));
        value.* = fraction * 8000.0;
    }

    for (0..mel_bins) |filter_index| {
        const lower = filter_frequencies[filter_index];
        const center = filter_frequencies[filter_index + 1];
        const upper = filter_frequencies[filter_index + 2];
        for (0..frequency_bins) |bin_index| {
            const frequency = fft_frequencies[bin_index];
            const rising = (frequency - lower) / (center - lower);
            const falling = (upper - frequency) / (upper - center);
            const response = @max(0.0, @min(rising, falling));
            filters[bin_index * mel_bins + filter_index] = @floatCast(
                response * enorm[filter_index],
            );
        }
    }
}

/// Slaney mel scale: linear below 1 kHz, logarithmic above.
fn hertzToMel(frequency: f64) f64 {
    const min_log_hertz: f64 = 1000.0;
    const min_log_mel: f64 = 15.0;
    const logstep: f64 = 27.0 / @log(6.4);
    if (frequency >= min_log_hertz) {
        return min_log_mel + @log(frequency / min_log_hertz) * logstep;
    }
    return 3.0 * frequency / 200.0;
}

fn melToHertz(mel: f64) f64 {
    const min_log_hertz: f64 = 1000.0;
    const min_log_mel: f64 = 15.0;
    const logstep: f64 = @log(6.4) / 27.0;
    if (mel >= min_log_mel) {
        return min_log_hertz * @exp(logstep * (mel - min_log_mel));
    }
    return 200.0 * mel / 3.0;
}

test "table geometry is consistent" {
    try std.testing.expectEqual(@as(u32, 201), frequency_bins);
    try std.testing.expectEqual(@as(u32, 200), center_padding);
    try std.testing.expectEqual(@as(usize, slow_size * fast_size), @as(usize, n_fft));
    try std.testing.expectEqual(@as(usize, 3000), capacity_frames);
    try std.testing.expectEqual(@as(usize, n_fft), tables.window.len);
    try std.testing.expectEqual(@as(usize, frequency_bins * mel_bins), tables.filters.len);
}

test "hann window is periodic and symmetric" {
    try std.testing.expectEqual(@as(f32, 0.0), tables.window[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tables.window[n_fft / 2], 1e-6);
    // Periodic means the value at index n_fft would repeat index 0, so the last
    // sample is still positive.
    try std.testing.expect(tables.window[n_fft - 1] > 0.0);
    for (1..n_fft / 2) |index| {
        try std.testing.expectApproxEqAbs(tables.window[index], tables.window[n_fft - index], 1e-6);
    }
}

test "mel filters are non-negative and Slaney-normalized" {
    // Triangle areas are tiny by design: Slaney normalization divides each band
    // by its width, and band widths grow with frequency. The property worth
    // pinning is that band gains stay within a small factor of one another --
    // without the normalization the ratio would exceed 50, because the highest
    // band is roughly twenty times wider than the lowest.
    var sum_min: f64 = std.math.floatMax(f64);
    var sum_max: f64 = 0.0;
    for (0..mel_bins) |filter_index| {
        var sum: f64 = 0.0;
        var peak: f32 = 0.0;
        for (0..frequency_bins) |bin| {
            const weight = tables.filters[bin * mel_bins + filter_index];
            try std.testing.expect(weight >= 0.0);
            try std.testing.expect(std.math.isFinite(weight));
            sum += weight;
            peak = @max(peak, weight);
        }
        try std.testing.expect(peak > 0.0);
        try std.testing.expect(sum >= @as(f64, peak));
        sum_min = @min(sum_min, sum);
        sum_max = @max(sum_max, sum);
    }
    try std.testing.expect(sum_min > 0.005);
    try std.testing.expect(sum_max < 0.1);
    try std.testing.expect(sum_max / sum_min < 6.0);
}

test "mel filters are shaped like triangles" {
    // Each filter must rise then fall, with its peak strictly inside the band.
    for ([_]usize{ 10, 40, 80, 120 }) |filter_index| {
        var previous: f32 = 0.0;
        var peaked = false;
        var saw_peak = false;
        for (0..frequency_bins) |bin| {
            const weight = tables.filters[bin * mel_bins + filter_index];
            if (weight > previous) {
                try std.testing.expect(!peaked);
            } else if (weight < previous) {
                peaked = true;
                saw_peak = true;
            }
            previous = weight;
        }
        try std.testing.expect(saw_peak);
    }
}

test "sample access mirrors at the start and zeroes past the clip" {
    const waveform = [_]f32{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    try std.testing.expectEqual(@as(f32, 1.0), sampleAt(&waveform, -1));
    try std.testing.expectEqual(@as(f32, 4.0), sampleAt(&waveform, -4));
    try std.testing.expectEqual(@as(f32, 0.0), sampleAt(&waveform, 0));
    try std.testing.expectEqual(@as(f32, 4.0), sampleAt(&waveform, 4));
    // Past the clip, but inside the 30-second capacity: silence.
    try std.testing.expectEqual(@as(f32, 0.0), sampleAt(&waveform, 5));
    try std.testing.expectEqual(@as(f32, 0.0), sampleAt(&waveform, 100000));
    // Past the capacity: a mirror of the buffer end.
    try std.testing.expectEqual(@as(f32, 0.0), sampleAt(&waveform, capacity_samples));

    const short = [_]f32{ 7.0, 8.0 };
    try std.testing.expectEqual(@as(f32, 8.0), sampleAt(&short, -9));
}

test "a pure tone lands in the mel band that covers its frequency" {
    // 1 kHz at 16 kHz. Frame 40 covers samples well inside the clip, so no
    // padding rule is involved.
    const sample_count: usize = 16000;
    var waveform: [sample_count]f32 = undefined;
    for (&waveform, 0..) |*sample, index| {
        const position: f32 = @floatFromInt(index);
        sample.* = @sin(2.0 * std.math.pi * 1000.0 * position / 16000.0);
    }
    var features: [mel_bins * 100]f32 = undefined;
    _ = try compute(&waveform, &features, 100);

    var loudest: usize = 0;
    var loudest_value: f32 = -std.math.floatMax(f32);
    for (0..mel_bins) |bin| {
        const value = features[bin * 100 + 40];
        if (value > loudest_value) {
            loudest_value = value;
            loudest = bin;
        }
    }

    // The band that responds most must be the band whose passband contains the
    // tone, so its peak weight has to sit at the 1 kHz frequency bin (bin 25 of
    // 201 covering 0..8000 Hz).
    var peak_bin: usize = 0;
    var peak_weight: f32 = 0.0;
    for (0..frequency_bins) |bin| {
        const weight = tables.filters[bin * mel_bins + loudest];
        if (weight > peak_weight) {
            peak_weight = weight;
            peak_bin = bin;
        }
    }
    try std.testing.expect(peak_bin >= 24);
    try std.testing.expect(peak_bin <= 26);
}

test "silence produces the documented constant floor" {
    var waveform: [8000]f32 = @splat(0.0);
    var features: [mel_bins * 50]f32 = undefined;
    const global_max_log = try compute(&waveform, &features, 50);
    try std.testing.expectEqual(silence_log, global_max_log);
    for (features) |value| {
        try std.testing.expect(std.math.isFinite(value));
        // max(-10, -10 - 8) == -10, so (-10 + 4) / 4 == -1.5
        try std.testing.expectApproxEqAbs(@as(f32, -1.5), value, 1e-6);
    }
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), paddingFrameValue(global_max_log), 1e-6);
}

test "padding frames carry the clamped silence value" {
    var waveform: [8000]f32 = @splat(0.0);
    for (&waveform, 0..) |*sample, index| {
        const position: f32 = @floatFromInt(index);
        sample.* = @sin(2.0 * std.math.pi * 440.0 * position / 16000.0);
    }
    var features: [mel_bins * 50]f32 = undefined;
    const global_max_log = try compute(&waveform, &features, 50);
    fillPaddingFrames(&features, 50, 45, 5, global_max_log);
    const expected = (global_max_log - dynamic_range + normalization_offset) / normalization_scale;
    for (0..mel_bins) |bin| {
        for (45..50) |frame| {
            try std.testing.expectApproxEqAbs(expected, features[bin * 50 + frame], 1e-6);
        }
    }
}

test "output and frame bounds are enforced" {
    var waveform: [8001]f32 = @splat(0.0);
    var features: [mel_bins * 52]f32 = undefined;

    // The output slice must be exactly mel_bins * frame_count.
    try std.testing.expectError(
        Error.OutputShapeMismatch,
        compute(&waveform, features[0 .. mel_bins * 51], 50),
    );
    // 8001 samples supply 51 valid frames, so 52 is one too many.
    try std.testing.expectError(
        Error.FrameCountExceedsAudio,
        compute(&waveform, &features, 52),
    );
    // The reference buffer caps the frame count at a whole 30 seconds, even
    // when the clip itself is longer.
    const long = try std.testing.allocator.alloc(f32, capacity_samples + hop_length);
    defer std.testing.allocator.free(long);
    @memset(long, 0.0);
    const wide = try std.testing.allocator.alloc(f32, mel_bins * (capacity_frames + 1));
    defer std.testing.allocator.free(wide);
    try std.testing.expectError(
        Error.FrameCountExceedsCapacity,
        compute(long, wide, capacity_frames + 1),
    );
    try std.testing.expectError(Error.EmptyWaveform, compute(&.{}, features[0..0], 0));
}

test "frame counts follow the reference's masking rule" {
    // ceil(samples / hop): the last frame is kept while its first sample is real.
    try std.testing.expectEqual(@as(usize, 77), framesForSamples(12345));
    try std.testing.expectEqual(@as(usize, 78), framesForSamples(12480));
    try std.testing.expectEqual(@as(usize, 78), framesForSamples(12481));
    try std.testing.expectEqual(@as(usize, 0), framesForSamples(1));
    try std.testing.expectEqual(@as(usize, 800), framesForSamples(128000));
    try std.testing.expectEqual(@as(usize, 50), framesForSamples(min_samples));
    try std.testing.expectEqual(@as(usize, 3000), framesForSamples(capacity_samples));
}
