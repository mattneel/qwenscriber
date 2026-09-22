//! Numeric comparison against fixtures generated from the reference
//! `transformers` pipeline.
//!
//! Fixtures are embedded rather than read from disk so the tests cannot depend
//! on the working directory, and so a missing fixture is a compile error instead
//! of a silently skipped check. Regenerate them with
//! `tools/reference/gen_fixtures.py`.

const std = @import("std");
const qw = @import("qwenscriber");

/// The `QWFIX001` container: magic, rank, four dimensions, then f32 payload.
const Fixture = struct {
    dims: [4]u32,
    rank: u8,
    payload: []const f32,
    storage: []align(4) const u8,

    const Error = error{
        BadMagic,
        UnsupportedRank,
        LengthMismatch,
        OutOfMemory,
    };

    /// Decodes a fixture, copying the payload into freshly allocated,
    /// correctly-aligned storage. Values are read as little-endian f32 halves of
    /// a u32 so the decode does not depend on the host's alignment rules.
    fn decode(allocator: std.mem.Allocator, bytes: []const u8) Error!Fixture {
        if (bytes.len < 28) return Error.LengthMismatch;
        if (!std.mem.eql(u8, bytes[0..8], "QWFIX001")) return Error.BadMagic;
        const rank = std.mem.readInt(u32, bytes[8..12], .little);
        if (rank == 0 or rank > 4) return Error.UnsupportedRank;

        var dims: [4]u32 = undefined;
        var count: usize = 1;
        for (0..4) |index| {
            dims[index] = std.mem.readInt(u32, bytes[12 + index * 4 ..][0..4], .little);
            if (index < rank) count *= dims[index];
        }
        if (bytes.len != 28 + count * 4) return Error.LengthMismatch;

        const storage = try allocator.alignedAlloc(u8, .of(f32), count * 4);
        const payload: []f32 = @alignCast(std.mem.bytesAsSlice(f32, storage));
        for (0..count) |index| {
            const bits = std.mem.readInt(u32, bytes[28 + index * 4 ..][0..4], .little);
            payload[index] = @bitCast(bits);
        }
        return .{ .dims = dims, .rank = @intCast(rank), .payload = payload, .storage = storage };
    }

    fn expectShape(self: *const Fixture, rank: u8, dims: []const u32) !void {
        try std.testing.expectEqual(rank, self.rank);
        for (dims, 0..) |dim, index| {
            try std.testing.expectEqual(dim, self.dims[index]);
        }
    }
};

const mel_filter_bytes = @embedFile("fixtures/mel_filters.f32");
const mel_waveform_bytes = @embedFile("fixtures/mel_waveform.f32");
const mel_expected_bytes = @embedFile("fixtures/mel_expected.f32");
const mel_expected_unpadded_bytes = @embedFile("fixtures/mel_expected_unpadded.f32");

test "mel filterbank matches the reference matrix element for element" {
    const fixture = try Fixture.decode(std.testing.allocator, mel_filter_bytes);
    defer std.testing.allocator.free(fixture.storage);
    try fixture.expectShape(2, &.{ qw.mel.frequency_bins, qw.mel.mel_bins });

    var worst: f32 = 0.0;
    for (qw.mel.tables.filters, 0..) |ours, index| {
        worst = @max(worst, @abs(ours - fixture.payload[index]));
    }
    // The reference builds this matrix in float64 and stores float32; we do the
    // same at compile time, so the only difference is the final rounding.
    // Measured: exactly zero across all 25,728 entries, i.e. bit-identical.
    try std.testing.expectEqual(@as(f32, 0.0), worst);
}

test "log-mel matches the reference pipeline output" {
    const waveform = try Fixture.decode(std.testing.allocator, mel_waveform_bytes);
    defer std.testing.allocator.free(waveform.storage);
    const expected = try Fixture.decode(std.testing.allocator, mel_expected_bytes);
    defer std.testing.allocator.free(expected.storage);

    try waveform.expectShape(1, &.{12345});
    const frames = qw.mel.framesForSamples(waveform.payload.len);
    try std.testing.expectEqual(@as(usize, 77), frames);
    try expected.expectShape(2, &.{ qw.mel.mel_bins, @as(u32, @intCast(frames)) });

    const features = try std.testing.allocator.alloc(f32, qw.mel.mel_bins * frames);
    defer std.testing.allocator.free(features);
    const global_max_log = try qw.mel.compute(waveform.payload, features, frames);
    try std.testing.expectApproxEqAbs(@as(f32, 1.9120302), global_max_log, 1e-6);

    var worst: f32 = 0.0;
    var total: f64 = 0.0;
    for (features, expected.payload) |ours, reference| {
        const difference = @abs(ours - reference);
        total += difference;
        worst = @max(worst, difference);
    }
    const mean = total / @as(f64, @floatFromInt(features.len));
    // The reference computes magnitudes with `abs()` (a square root) before
    // squaring, in float32; we use `re^2 + im^2`. That, plus a different FFT
    // summation order, is the whole difference. Measured on this fixture:
    // max 8.6e-5 and mean 1.7e-6, both in normalized log10 units, at the single
    // loudest bin (bin 32, frame 14) -- every other entry is far tighter
    // because most of the fixture sits in the dynamic-range clamp.
    try std.testing.expect(worst < 1e-4);
    try std.testing.expect(mean < 1e-5);
}

test "our global maximum matches the one the reference used" {
    // The reference normalizes with `max(log_spec, global_max - 8)`, so the
    // largest entry of a finished fixture is exactly the unclamped maximum. That
    // makes the fixture an independent check of our two-pass clamp: we must have
    // measured the same global maximum before applying it.
    const waveform = try Fixture.decode(std.testing.allocator, mel_waveform_bytes);
    defer std.testing.allocator.free(waveform.storage);
    const expected = try Fixture.decode(std.testing.allocator, mel_expected_bytes);
    defer std.testing.allocator.free(expected.storage);

    var reference_max: f32 = -std.math.floatMax(f32);
    for (expected.payload) |value| reference_max = @max(reference_max, value);
    const implied_global_max = 4.0 * reference_max - 4.0;

    const frames = qw.mel.framesForSamples(waveform.payload.len);
    const features = try std.testing.allocator.alloc(f32, qw.mel.mel_bins * frames);
    defer std.testing.allocator.free(features);
    const our_global_max = try qw.mel.compute(waveform.payload, features, frames);

    // Measured: identical to seven decimal places (1.9120302), so the two-pass
    // clamp is reproducing the reference's global maximum exactly.
    try std.testing.expectApproxEqAbs(implied_global_max, our_global_max, 1e-6);

    // Only a handful of entries escape the clamp, so the fixture mostly tests
    // the clamp path; assert that this stays true so the test keeps meaning what
    // it says.
    var unclamped: usize = 0;
    for (expected.payload) |value| {
        if (value > reference_max - 1e-6) unclamped += 1;
    }
    try std.testing.expect(unclamped > 0);
    try std.testing.expect(unclamped < 8);
}

test "the clipped and padded reference paths agree away from the clip end" {
    // Two fixtures describe the same waveform: one transformed directly, one
    // through the padded 30-second buffer. They must agree exactly for every
    // frame that is not influenced by end-of-audio padding, which is what lets
    // the runtime compute only the valid prefix and treat the rest as the
    // reference's zero-padded capacity.
    const direct = try Fixture.decode(std.testing.allocator, mel_expected_unpadded_bytes);
    defer std.testing.allocator.free(direct.storage);
    const padded = try Fixture.decode(std.testing.allocator, mel_expected_bytes);
    defer std.testing.allocator.free(padded.storage);

    // Both fixtures now end at the same frame: the runtime, the fixture, and the
    // reference processor all drop a final partial hop.
    try direct.expectShape(2, &.{ qw.mel.mel_bins, 77 });
    try padded.expectShape(2, &.{ qw.mel.mel_bins, 77 });

    var worst_shared: f32 = 0.0;
    for (0..qw.mel.mel_bins) |bin| {
        for (0..76) |frame| {
            worst_shared = @max(worst_shared, @abs(
                direct.payload[bin * 77 + frame] - padded.payload[bin * 77 + frame],
            ));
        }
    }
    // The last frame is deliberately not compared: it is the frame the two
    // reference paths disagree about (0.142 apart, because the padded path
    // computes it against end-of-buffer zeros), and dropping it is exactly what
    // `framesForSamples` and the fixture generator now do.
    try std.testing.expectEqual(@as(f32, 0.0), worst_shared);
}
