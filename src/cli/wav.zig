//! Minimal RIFF/WAVE reader for the transcription tool.
//!
//! Deliberately narrow: uncompressed PCM and IEEE float, any channel count
//! (mixed down to mono), 16 kHz required. Anything else is reported as an
//! unsupported input rather than silently reinterpreted, because a misread
//! sample rate produces plausible-looking garbage rather than an error.
//!
//! The browser side does not use this: `AudioContext.decodeAudioData` handles
//! every container a browser supports. This exists so the native bring-up runner
//! can read the fixtures the reference pipeline reads.

const std = @import("std");

pub const Error = error{
    NotRiffWave,
    MissingFormatChunk,
    MissingDataChunk,
    UnsupportedFormat,
    UnsupportedSampleRate,
    UnsupportedChannels,
    TruncatedChunk,
    EmptyAudio,
};

pub const expected_sample_rate: u32 = 16000;

pub const Audio = struct {
    /// Mono samples in [-1, 1]. Owned by the caller's buffer.
    samples: []const f32,
    source_sample_rate: u32,
    source_channels: u16,
};

/// Decodes a WAVE file into mono f32 samples.
///
/// `scratch` receives the decoded samples; it must be large enough for
/// `frame_count`. Returns a slice of it.
pub fn decode(bytes: []const u8, scratch: []f32) Error!Audio {
    if (bytes.len < 12) return Error.NotRiffWave;
    if (!std.mem.eql(u8, bytes[0..4], "RIFF")) return Error.NotRiffWave;
    if (!std.mem.eql(u8, bytes[8..12], "WAVE")) return Error.NotRiffWave;

    var format: ?Format = null;
    var data: ?[]const u8 = null;

    var cursor: usize = 12;
    while (cursor + 8 <= bytes.len) {
        const chunk_id = bytes[cursor..][0..4];
        const chunk_size = std.mem.readInt(u32, bytes[cursor + 4 ..][0..4], .little);
        const body_start = cursor + 8;
        const body_end = body_start + chunk_size;
        if (body_end > bytes.len) return Error.TruncatedChunk;
        const body = bytes[body_start..body_end];

        if (std.mem.eql(u8, chunk_id, "fmt ")) {
            format = try parseFormat(body);
        } else if (std.mem.eql(u8, chunk_id, "data")) {
            data = body;
        }
        // Chunks are word aligned: an odd size is followed by a pad byte.
        cursor = body_end + (chunk_size % 2);
    }

    const fmt = format orelse return Error.MissingFormatChunk;
    const payload = data orelse return Error.MissingDataChunk;
    if (fmt.sample_rate != expected_sample_rate) return Error.UnsupportedSampleRate;
    if (fmt.channels == 0 or fmt.channels > 8) return Error.UnsupportedChannels;

    const frames = payload.len / fmt.bytesPerFrame();
    if (frames == 0) return Error.EmptyAudio;
    if (scratch.len < frames) return Error.UnsupportedFormat;

    const samples = scratch[0..frames];
    var frame: usize = 0;
    while (frame < frames) : (frame += 1) {
        var sum: f32 = 0.0;
        var channel: u16 = 0;
        while (channel < fmt.channels) : (channel += 1) {
            const offset = frame * fmt.bytesPerFrame() + channel * fmt.bytesPerSample();
            sum += fmt.readSample(payload[offset..]);
        }
        samples[frame] = sum / @as(f32, @floatFromInt(fmt.channels));
    }
    return .{
        .samples = samples,
        .source_sample_rate = fmt.sample_rate,
        .source_channels = fmt.channels,
    };
}

/// Number of frames a file holds, for sizing the decode buffer.
pub fn frameCount(bytes: []const u8) Error!usize {
    const format = try scanFormat(bytes);
    var cursor: usize = 12;
    while (cursor + 8 <= bytes.len) {
        const chunk_id = bytes[cursor..][0..4];
        const chunk_size = std.mem.readInt(u32, bytes[cursor + 4 ..][0..4], .little);
        if (std.mem.eql(u8, chunk_id, "data")) {
            return chunk_size / format.bytesPerFrame();
        }
        cursor += 8 + chunk_size + (chunk_size % 2);
    }
    return Error.MissingDataChunk;
}

const Format = struct {
    /// 1 = PCM integer, 3 = IEEE float.
    tag: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,

    fn bytesPerSample(self: Format) usize {
        return self.bits_per_sample / 8;
    }

    fn bytesPerFrame(self: Format) usize {
        return self.bytesPerSample() * self.channels;
    }

    fn readSample(self: Format, bytes: []const u8) f32 {
        if (self.tag == 3) {
            // IEEE float, 32 bits.
            return @bitCast(std.mem.readInt(u32, bytes[0..4], .little));
        }
        return switch (self.bits_per_sample) {
            16 => @as(f32, @floatFromInt(std.mem.readInt(i16, bytes[0..2], .little))) / 32768.0,
            24 => blk: {
                const raw = std.mem.readInt(u24, bytes[0..3], .little);
                // Sign extend 24 bits: widen, shift the value into the top of a
                // 32-bit word, then shift it back down arithmetically.
                const widened: i32 = @bitCast(@as(u32, raw) << 8);
                break :blk @as(f32, @floatFromInt(widened >> 8)) / 8388608.0;
            },
            32 => @as(f32, @floatFromInt(std.mem.readInt(i32, bytes[0..4], .little))) /
                2147483648.0,
            8 => @as(f32, @floatFromInt(bytes[0])) / 128.0 - 1.0,
            else => 0.0,
        };
    }
};

fn parseFormat(body: []const u8) Error!Format {
    if (body.len < 16) return Error.TruncatedChunk;
    const tag = std.mem.readInt(u16, body[0..2], .little);
    const format = Format{
        .tag = tag,
        .channels = std.mem.readInt(u16, body[2..4], .little),
        .sample_rate = std.mem.readInt(u32, body[4..8], .little),
        .bits_per_sample = std.mem.readInt(u16, body[14..16], .little),
    };
    if (format.tag == 0xFFFE) return Error.UnsupportedFormat; // WAVE_FORMAT_EXTENSIBLE
    if (format.tag != 1 and format.tag != 3) return Error.UnsupportedFormat;
    if (format.tag == 3 and format.bits_per_sample != 32) return Error.UnsupportedFormat;
    if (format.tag == 1) {
        switch (format.bits_per_sample) {
            8, 16, 24, 32 => {},
            else => return Error.UnsupportedFormat,
        }
    }
    return format;
}

fn scanFormat(bytes: []const u8) Error!Format {
    if (bytes.len < 12) return Error.NotRiffWave;
    var cursor: usize = 12;
    while (cursor + 8 <= bytes.len) {
        const chunk_size = std.mem.readInt(u32, bytes[cursor + 4 ..][0..4], .little);
        if (std.mem.eql(u8, bytes[cursor..][0..4], "fmt ")) {
            return parseFormat(bytes[cursor + 8 ..][0..chunk_size]);
        }
        cursor += 8 + chunk_size + (chunk_size % 2);
    }
    return Error.MissingFormatChunk;
}

/// Builds a WAVE file in memory, so the reader can be tested without fixtures.
fn buildWav(
    allocator: std.mem.Allocator,
    samples: []const i16,
    channels: u16,
    sample_rate: u32,
) ![]u8 {
    const data_bytes = samples.len * 2;
    const total = 44 + data_bytes;
    const out = try allocator.alloc(u8, total);
    @memcpy(out[0..4], "RIFF");
    std.mem.writeInt(u32, out[4..8], @intCast(total - 8), .little);
    @memcpy(out[8..12], "WAVE");
    @memcpy(out[12..16], "fmt ");
    std.mem.writeInt(u32, out[16..20], 16, .little);
    std.mem.writeInt(u16, out[20..22], 1, .little);
    std.mem.writeInt(u16, out[22..24], channels, .little);
    std.mem.writeInt(u32, out[24..28], sample_rate, .little);
    std.mem.writeInt(u32, out[28..32], sample_rate * channels * 2, .little);
    std.mem.writeInt(u16, out[32..34], channels * 2, .little);
    std.mem.writeInt(u16, out[34..36], 16, .little);
    @memcpy(out[36..40], "data");
    std.mem.writeInt(u32, out[40..44], @intCast(data_bytes), .little);
    for (samples, 0..) |sample, index| {
        std.mem.writeInt(i16, out[44 + index * 2 ..][0..2], sample, .little);
    }
    return out;
}

test "a mono file decodes to normalized samples" {
    const samples = [_]i16{ 0, 16384, -16384, 32767 };
    const bytes = try buildWav(std.testing.allocator, &samples, 1, 16000);
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqual(@as(usize, 4), try frameCount(bytes));
    var scratch: [8]f32 = undefined;
    const audio = try decode(bytes, &scratch);
    try std.testing.expectEqual(@as(usize, 4), audio.samples.len);
    try std.testing.expectEqual(@as(f32, 0.0), audio.samples[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), audio.samples[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), audio.samples[2], 1e-6);
    try std.testing.expect(audio.samples[3] > 0.999);
}

test "stereo is mixed down to mono" {
    const samples = [_]i16{ 32767, -32767, 0, 0 };
    const bytes = try buildWav(std.testing.allocator, &samples, 2, 16000);
    defer std.testing.allocator.free(bytes);
    var scratch: [4]f32 = undefined;
    const audio = try decode(bytes, &scratch);
    try std.testing.expectEqual(@as(usize, 2), audio.samples.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), audio.samples[0], 1e-3);
    try std.testing.expectEqual(@as(f32, 0.0), audio.samples[1]);
    try std.testing.expectEqual(@as(u16, 2), audio.source_channels);
}

test "unsupported inputs are named, not misread" {
    const samples = [_]i16{ 1, 2, 3, 4 };
    const wrong_rate = try buildWav(std.testing.allocator, &samples, 1, 8000);
    defer std.testing.allocator.free(wrong_rate);
    var scratch: [8]f32 = undefined;
    try std.testing.expectError(Error.UnsupportedSampleRate, decode(wrong_rate, &scratch));

    const not_wave = "this is not a wave file at all";
    try std.testing.expectError(Error.NotRiffWave, decode(not_wave, &scratch));

    const headers_only = try buildWav(std.testing.allocator, &[_]i16{}, 1, 16000);
    defer std.testing.allocator.free(headers_only);
    try std.testing.expectError(Error.EmptyAudio, decode(headers_only, &scratch));

    // A truncated data chunk must be caught by the size check.
    const truncated = try buildWav(std.testing.allocator, &samples, 1, 16000);
    defer std.testing.allocator.free(truncated);
    try std.testing.expectError(Error.TruncatedChunk, decode(truncated[0 .. truncated.len - 4], &scratch));
}
