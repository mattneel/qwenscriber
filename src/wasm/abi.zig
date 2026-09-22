//! WASM ABI v1: the complete contract between the freestanding core and
//! JavaScript.
//!
//! The ABI is deliberately C-like and boring: fixed-width integers, linear
//! memory offsets, explicit lengths, and machine-readable error codes. No Zig
//! slice, error union, optional, or layout-sensitive struct ever crosses the
//! boundary. Every struct that does cross is declared here with its byte layout
//! documented, and `src/wasm/exports.zig` is the only file that may touch a raw
//! offset.
//!
//! Versioning: `qw_abi_version` returns `major << 16 | minor`. JavaScript must
//! reject a module whose major differs from the one it was built against.
//! Additive changes bump minor; any layout or semantic break bumps major.

const std = @import("std");

pub const major: u32 = 1;
pub const minor: u32 = 0;

pub fn packVersion(comptime version_major: u32, comptime version_minor: u32) u32 {
    return (version_major << 16) | version_minor;
}

pub const version: u32 = packVersion(major, minor);

/// Status codes. Zero is success; every failure is a distinct negative value so
/// JavaScript can map them to typed errors without parsing text.
pub const Status = enum(i32) {
    ok = 0,
    /// A pointer, length, or alignment argument was not usable.
    invalid_argument = -1,
    /// The allocator could not satisfy a request.
    out_of_memory = -2,
    /// The requested capability does not exist in this build.
    unsupported = -3,
    /// A container's magic bytes did not match.
    bad_magic = -4,
    /// A container or manifest version this build cannot read.
    bad_version = -5,
    /// An integrity check failed.
    checksum_mismatch = -6,
    /// A tensor or buffer shape violated the model's contract.
    shape_mismatch = -7,
    /// A call arrived in the wrong state (for example decode before begin).
    invalid_state = -8,
    /// A fixed capacity would have been exceeded.
    limit_exceeded = -9,
    /// A referenced object does not exist in the container.
    not_found = -10,
    /// Input ended before the declared structure did.
    truncated = -11,
    /// Text was not valid UTF-8, or a token was outside the byte alphabet.
    invalid_encoding = -12,
    /// The model configuration is not one this build supports.
    unsupported_model = -13,
    /// More audio was supplied than the runtime accepts in one call.
    audio_too_long = -14,

    pub fn code(self: Status) i32 {
        return @backingInt(self);
    }
};

/// Maps a core error to a status code.
///
/// Errors that indicate a violated invariant rather than bad input map to
/// `invalid_state`, which is the closest thing to "the runtime is confused"
/// that the ABI can express without trapping.
pub fn statusFromError(err: anyerror) Status {
    return switch (err) {
        error.OutOfMemory => .out_of_memory,
        error.Unsupported, error.UnsupportedFormat, error.UnsupportedModel => .unsupported_model,
        error.UnsupportedSampleRate,
        error.UnsupportedFftSize,
        error.UnsupportedMelBins,
        error.UnsupportedHopLength,
        => .unsupported,
        error.BadMagic => .bad_magic,
        error.BadVersion => .bad_version,
        error.ChecksumMismatch => .checksum_mismatch,
        error.ShapeMismatch,
        error.RowNotGroupAligned,
        error.NotMatrix,
        error.RankMismatch,
        error.ByteLengthMismatch,
        error.DimensionZero,
        error.DimensionTooLarge,
        error.RankTooLarge,
        error.CountOverflow,
        error.LayoutOverflow,
        error.OutputShapeMismatch,
        => .shape_mismatch,
        error.InvalidState => .invalid_state,
        error.LimitExceeded, error.FrameCountExceedsCapacity => .limit_exceeded,
        error.NotFound, error.UnknownToken => .not_found,
        error.Truncated,
        error.LengthMismatch,
        error.CutShort,
        => .truncated,
        error.UnknownError => .invalid_state,
        error.InvalidEncoding,
        error.InvalidTokenAlphabet,
        error.MalformedTable,
        => .invalid_encoding,
        error.AudioTooLong, error.FrameCountExceedsAudio => .audio_too_long,
        else => .invalid_argument,
    };
}

/// Human readable description of a status code.
///
/// The text lives in the module's read-only data, so JavaScript reads it through
/// linear memory with `qw_error_message_ptr`/`qw_error_message_len`.
pub fn statusMessage(status: Status) []const u8 {
    return switch (status) {
        .ok => "ok",
        .invalid_argument => "invalid argument",
        .out_of_memory => "out of memory",
        .unsupported => "unsupported",
        .bad_magic => "bad magic",
        .bad_version => "bad version",
        .checksum_mismatch => "checksum mismatch",
        .shape_mismatch => "shape mismatch",
        .invalid_state => "invalid state",
        .limit_exceeded => "limit exceeded",
        .not_found => "not found",
        .truncated => "truncated input",
        .invalid_encoding => "invalid encoding",
        .unsupported_model => "unsupported model",
        .audio_too_long => "audio too long",
    };
}

/// `qw_mel_result`: written by `qw_mel_compute`.
///
///     offset  0  u32  frames_written
///     offset  4  u32  bytes_written
///     offset  8  f32  global_max_log   (the value `qw_mel_padding_value` needs)
///     offset 12  u32  reserved, written as zero
pub const MelResult = extern struct {
    frames_written: u32,
    bytes_written: u32,
    global_max_log: f32,
    reserved: u32 = 0,
};

/// `qw_selftest_result`: written by `qw_selftest`.
///
///     offset  0  u32  failures (bit mask of failed checks)
///     offset  4  u32  loudest_band
///     offset  8  f32  silence_value
///     offset 12  u32  reserved
///     offset 16  u64  quant_hash
///     offset 24  u64  mel_hash
pub const SelfTestResult = extern struct {
    failures: u32,
    loudest_band: u32,
    silence_value: f32,
    reserved: u32 = 0,
    quant_hash: u64,
    mel_hash: u64,
};

/// `qw_tokenizer`: the vocabulary the runtime decodes with.
///
///     offset 0  u32  offsets_ptr   (u32 per token, count + 1 entries)
///     offset 4  u32  offsets_len
///     offset 8  u32  bytes_ptr
///     offset 12 u32  bytes_len
///     offset 16 u32  token_count
///     offset 20 u32  reserved
pub const TokenizerDescriptor = extern struct {
    offsets_ptr: u32,
    offsets_len: u32,
    bytes_ptr: u32,
    bytes_len: u32,
    token_count: u32,
    reserved: u32 = 0,
};

pub const handle_max: u32 = 1;

comptime {
    // These sizes are part of the ABI; JavaScript allocates exactly them.
    std.debug.assert(@sizeOf(MelResult) == 16);
    std.debug.assert(@sizeOf(SelfTestResult) == 32);
    std.debug.assert(@sizeOf(TokenizerDescriptor) == 24);
    std.debug.assert(@sizeOf(Status) == 4);
}

test "version packing is major-then-minor" {
    try std.testing.expectEqual(@as(u32, 0x0001_0000), version);
    try std.testing.expectEqual(@as(u32, 0x0002_0003), packVersion(2, 3));
}

test "status codes are distinct, negative, and documented" {
    var seen = std.mem.zeroes([16]bool);
    inline for (@typeInfo(Status).@"enum".field_names) |field_name| {
        const status: Status = @field(Status, field_name);
        if (status == .ok) continue;
        try std.testing.expect(status.code() < 0);
        const index: usize = @intCast(-status.code() - 1);
        try std.testing.expect(index < seen.len);
        try std.testing.expect(!seen[index]);
        seen[index] = true;
        try std.testing.expect(statusMessage(status).len > 0);
    }
}

test "errors map to the status a caller can act on" {
    try std.testing.expectEqual(Status.out_of_memory, statusFromError(error.OutOfMemory));
    try std.testing.expectEqual(Status.bad_magic, statusFromError(error.BadMagic));
    try std.testing.expectEqual(Status.bad_version, statusFromError(error.BadVersion));
    try std.testing.expectEqual(Status.shape_mismatch, statusFromError(error.ShapeMismatch));
    try std.testing.expectEqual(Status.unsupported_model, statusFromError(error.UnsupportedFormat));
    try std.testing.expectEqual(Status.truncated, statusFromError(error.Truncated));
    try std.testing.expectEqual(
        Status.invalid_argument,
        statusFromError(error.SomethingNobodyDeclared),
    );
}
