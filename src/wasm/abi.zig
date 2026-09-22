//! WASM ABI v1: the complete contract between the core and JavaScript.
//!
//! The contract belongs to the core's source, not to a toolchain: the
//! `wasm32-freestanding` build and the thread-enabled `wasm32-emscripten` build
//! (ADR-0005) export the same entry points with the same semantics. What differs
//! between them is instantiation — imports and memory — not this file.
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

/// Key/value cache widths a caller may ask for, as `qw_model_set_cache_format` takes them.
///
/// The cache is read in full on every decoded token, so its width decides whether a model fits a
/// bounded instance: `q8` is a quarter of the f32 size and decodes at the same speed. The variants
/// are not named `f32` and `q8`: a declaration may not shadow a primitive type name, which Zig
/// refuses outright.
pub const CacheFormat = enum(u32) {
    /// One f32 per element. The reference path, and what a model gets by default.
    full_precision = 0,
    /// One signed 8-bit code per element, with an f16 scale per quantization group.
    q8 = 1,
    _,

    pub fn code(self: CacheFormat) u32 {
        return @backingInt(self);
    }
};

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
        // The model family: a tensor or prompt whose shape contradicts the
        // configuration is the same class of failure as a mel buffer that is
        // the wrong size.
        error.UnexpectedShape,
        error.PromptMismatch,
        => .shape_mismatch,
        error.InvalidState => .invalid_state,
        error.LimitExceeded,
        error.FrameCountExceedsCapacity,
        // The model's own capacity budgets: positions the decoder may address,
        // and the packed encoder sequence a clip may need.
        error.CapacityExceeded,
        error.PositionExceeded,
        => .limit_exceeded,
        error.NotFound, error.UnknownToken => .not_found,
        // A required tensor that no supplied shard carries is the model-family
        // spelling of `not_found`: the caller is missing a file.
        error.MissingTensor, error.MissingShard => .not_found,
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
        // A duplicate shard or tensor means the caller's inputs are ambiguous,
        // which is an argument problem rather than a state problem.
        error.DuplicateShard, error.DuplicateTensor => .invalid_argument,
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

/// `qw_features`: the capability families compiled into this build.
///
/// The bits exist so JavaScript negotiates instead of guessing. A module built
/// from an older source of the same ABI version simply has the model and decode
/// bits clear, and the SDK reports the missing stage as `not_implemented`
/// rather than calling an export that does not exist.
pub const Feature = enum(u32) {
    /// `qw_mel_*`: the log-mel frontend.
    mel = 1 << 0,
    /// `qw_tokenizer_set` and `qw_detokenize`.
    tokenizer = 1 << 1,
    /// `qw_selftest`.
    selftest = 1 << 2,
    /// `qw_model_*`: container parsing and model loading.
    model = 1 << 3,
    /// `qw_decode_*`: audio tower, projector, and greedy decoding.
    decode = 1 << 4,

    pub fn bit(self: Feature) u32 {
        return @backingInt(self);
    }
};

/// Every feature family `src/wasm/exports.zig` publishes.
///
/// The mask is written out rather than derived from the enum's field list,
/// because clearing a bit has to be a deliberate edit: a build that drops a
/// family must also drop its exports, and this constant is what tells
/// JavaScript which of the two happened.
pub const features: u32 =
    Feature.mel.bit() |
    Feature.tokenizer.bit() |
    Feature.selftest.bit() |
    Feature.model.bit() |
    Feature.decode.bit();

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

/// `qw_model_requirements`: what a loaded model keeps resident, and the limits
/// it decodes within. Written by `qw_model_requirements`.
///
///     offset  0  u64  weight_bytes        shard bytes the caller supplied
///     offset  8  u64  cache_bytes         key and value cache, all layers
///     offset 16  u64  scratch_bytes       activations, rotary tables, logits
///     offset 24  u64  total_bytes         weight_bytes + cache_bytes + scratch_bytes
///     offset 32  u32  max_positions       decoder positions the cache addresses
///     offset 36  u32  max_audio_frames    mel frames one clip may hold
///     offset 40  u32  max_decode_tokens   token budget of one utterance
///     offset 44  u32  reserved, written as zero
///
/// `weight_bytes` counts the shard bytes exactly as the caller supplied them,
/// container padding included, because the model borrows those buffers instead
/// of copying them and they must therefore stay resident. The numbers are
/// measured from the loaded model, not predicted: a caller that needs a
/// prediction before loading reads the manifest, which carries the same
/// per-shard byte counts.
///
/// `total_bytes` deliberately excludes the small bookkeeping the runtime holds
/// (tensor bindings, layer tables, the prompt and token buffers, which together
/// stay under 64 KiB for a 0.6B model) and the caller's own buffers: the
/// configuration, the vocabulary, the log-mel features, and the audio.
pub const ModelRequirements = extern struct {
    weight_bytes: u64,
    cache_bytes: u64,
    scratch_bytes: u64,
    total_bytes: u64,
    max_positions: u32,
    max_audio_frames: u32,
    max_decode_tokens: u32,
    reserved: u32 = 0,
};

/// `qw_model_audio_config`: the audio tower's geometry. Written by
/// `qw_model_audio_config`.
///
///     offset  0  u32  d_model                  tower width, in and out of every block
///     offset  4  u32  layers                   encoder blocks
///     offset  8  u32  attention_heads          heads per block
///     offset 12  u32  head_dim                 d_model / attention_heads
///     offset 16  u32  ffn_dim                  feed-forward width inside a block
///     offset 20  u32  downsample_hidden_size   the convolution stack's output channels
///     offset 24  u32  n_window                 mel frames per half-chunk
///     offset 28  u32  n_window_infer           mel frames one attention window spans
///     offset 32  u32  chunk_frames             mel frames one convolution chunk consumes
///     offset 36  u32  frequency_bins           mel bins left after the convolution stack
///     offset 40  u32  conv_out_input_features  downsample_hidden_size * frequency_bins
///     offset 44  u32  chunk_steps              time steps the stack emits per chunk
///     offset 48  u32  max_position_steps       rows in the sinusoidal position table
///     offset 52  u32  output_dim               projector output width, the decoder's hidden size
///     offset 56  u32  mel_bins                 mel bins the frontend produces
///     offset 60  f32  layer_norm_eps           epsilon of every tower LayerNorm
///     offset 64  u32  reserved_0, written as zero
///     offset 68  u32  reserved_1, written as zero
///
/// Every derived value is resolved by the core's own configuration helpers, not
/// left for a caller to recompute: `head_dim`, `chunk_frames`, `frequency_bins`,
/// `conv_out_input_features`, and `chunk_steps` are the same numbers the core's
/// forward pass uses, so a caller that dispatches the tower itself cannot drift
/// from the path the reference transcript came out of. A caller that never asks
/// for this sees the behavior it already had, which is why the model family grew
/// this export in place.
pub const AudioConfig = extern struct {
    d_model: u32,
    layers: u32,
    attention_heads: u32,
    head_dim: u32,
    ffn_dim: u32,
    downsample_hidden_size: u32,
    n_window: u32,
    n_window_infer: u32,
    chunk_frames: u32,
    frequency_bins: u32,
    conv_out_input_features: u32,
    chunk_steps: u32,
    max_position_steps: u32,
    output_dim: u32,
    mel_bins: u32,
    layer_norm_eps: f32,
    reserved_0: u32 = 0,
    reserved_1: u32 = 0,
};

/// `qw_model_tensor_descriptor`: where one tensor's bytes are and what they mean. Written by
/// `qw_model_tensor_descriptor`.
///
///     offset  0  u64  offset_bytes   payload offset from the start of the shard file
///     offset  8  u64  len_bytes      exact payload length, quantization padding included
///     offset 16  u32  kind           TensorKind value from `src/core/container.zig`
///     offset 20  u32  layer          encoder block index; 0 for tensors outside a block
///     offset 24  u32  format         dtype.Format value; quantized formats carry their plane layout
///     offset 28  u32  rank           1..4, the number of live dimensions
///     offset 32  u32  dims[0]        outermost dimension, most significant first
///     offset 36  u32  dims[1]
///     offset 40  u32  dims[2]
///     offset 44  u32  dims[3]        written as zero beyond `rank`
///     offset 48  u32  shard_index    which shard of the model holds the tensor
///     offset 52  u32  reserved_0, written as zero
///     offset 56  u32  reserved_1, written as zero
///     offset 60  u32  reserved_2, written as zero
///
/// The offset is absolute within the shard file rather than relative to the payload section, so a
/// caller that holds the shard bytes -- fetched, cached, or mapped -- can slice the tensor without
/// knowing the container's header layout. Tensors are enumerated in shard order, and within a shard
/// in index order, which is also payload order: the container writes them in that order precisely so
/// a shard moves to a GPU in one piece and each tensor is addressed without further work.
///
/// `kind` is the stable identity a caller matches on. The names the container gives those kinds
/// (`audio.conv1.weight`, `audio.layer.attention.q.weight`, and so on) are not carried here: an
/// integer stays a fixed-width ABI field, and the mirror of the enum on the JavaScript side is
/// checked against this file by `tests/gpu/layout_drift.mjs` the way the WGSL constants are.
pub const TensorDescriptor = extern struct {
    offset_bytes: u64,
    len_bytes: u64,
    kind: u32,
    layer: u32,
    format: u32,
    rank: u32,
    dims: [4]u32,
    shard_index: u32,
    reserved_0: u32 = 0,
    reserved_1: u32 = 0,
    reserved_2: u32 = 0,
};

comptime {
    // These sizes are part of the ABI; JavaScript allocates exactly them.
    std.debug.assert(@sizeOf(MelResult) == 16);
    std.debug.assert(@sizeOf(SelfTestResult) == 32);
    std.debug.assert(@sizeOf(TokenizerDescriptor) == 24);
    std.debug.assert(@sizeOf(ModelRequirements) == 48);
    std.debug.assert(@sizeOf(AudioConfig) == 72);
    std.debug.assert(@sizeOf(TensorDescriptor) == 64);
    std.debug.assert(@sizeOf(Status) == 4);

    // Field offsets are documented above and asserted here, so a reordering is
    // a compile error rather than a silent memory corruption in JavaScript.
    std.debug.assert(@offsetOf(ModelRequirements, "weight_bytes") == 0);
    std.debug.assert(@offsetOf(ModelRequirements, "cache_bytes") == 8);
    std.debug.assert(@offsetOf(ModelRequirements, "scratch_bytes") == 16);
    std.debug.assert(@offsetOf(ModelRequirements, "total_bytes") == 24);
    std.debug.assert(@offsetOf(ModelRequirements, "max_positions") == 32);
    std.debug.assert(@offsetOf(ModelRequirements, "max_audio_frames") == 36);
    std.debug.assert(@offsetOf(ModelRequirements, "max_decode_tokens") == 40);
    std.debug.assert(@offsetOf(ModelRequirements, "reserved") == 44);
    std.debug.assert(@offsetOf(AudioConfig, "d_model") == 0);
    std.debug.assert(@offsetOf(AudioConfig, "layers") == 4);
    std.debug.assert(@offsetOf(AudioConfig, "attention_heads") == 8);
    std.debug.assert(@offsetOf(AudioConfig, "head_dim") == 12);
    std.debug.assert(@offsetOf(AudioConfig, "ffn_dim") == 16);
    std.debug.assert(@offsetOf(AudioConfig, "downsample_hidden_size") == 20);
    std.debug.assert(@offsetOf(AudioConfig, "n_window") == 24);
    std.debug.assert(@offsetOf(AudioConfig, "n_window_infer") == 28);
    std.debug.assert(@offsetOf(AudioConfig, "chunk_frames") == 32);
    std.debug.assert(@offsetOf(AudioConfig, "frequency_bins") == 36);
    std.debug.assert(@offsetOf(AudioConfig, "conv_out_input_features") == 40);
    std.debug.assert(@offsetOf(AudioConfig, "chunk_steps") == 44);
    std.debug.assert(@offsetOf(AudioConfig, "max_position_steps") == 48);
    std.debug.assert(@offsetOf(AudioConfig, "output_dim") == 52);
    std.debug.assert(@offsetOf(AudioConfig, "mel_bins") == 56);
    std.debug.assert(@offsetOf(AudioConfig, "layer_norm_eps") == 60);
    std.debug.assert(@offsetOf(AudioConfig, "reserved_0") == 64);
    std.debug.assert(@offsetOf(AudioConfig, "reserved_1") == 68);
    std.debug.assert(@offsetOf(TensorDescriptor, "offset_bytes") == 0);
    std.debug.assert(@offsetOf(TensorDescriptor, "len_bytes") == 8);
    std.debug.assert(@offsetOf(TensorDescriptor, "kind") == 16);
    std.debug.assert(@offsetOf(TensorDescriptor, "layer") == 20);
    std.debug.assert(@offsetOf(TensorDescriptor, "format") == 24);
    std.debug.assert(@offsetOf(TensorDescriptor, "rank") == 28);
    std.debug.assert(@offsetOf(TensorDescriptor, "dims") == 32);
    std.debug.assert(@offsetOf(TensorDescriptor, "shard_index") == 48);
    std.debug.assert(@offsetOf(TensorDescriptor, "reserved_0") == 52);
    std.debug.assert(@offsetOf(TensorDescriptor, "reserved_1") == 56);
    std.debug.assert(@offsetOf(TensorDescriptor, "reserved_2") == 60);

    // Feature bits are a mask, so every family must own exactly one bit.
    std.debug.assert(@popCount(features) == 5);
    std.debug.assert(features == @as(u32, 0b1_1111));
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

test "model failures map to the status a caller can act on" {
    try std.testing.expectEqual(Status.not_found, statusFromError(error.MissingTensor));
    try std.testing.expectEqual(Status.not_found, statusFromError(error.MissingShard));
    try std.testing.expectEqual(Status.shape_mismatch, statusFromError(error.UnexpectedShape));
    try std.testing.expectEqual(Status.shape_mismatch, statusFromError(error.PromptMismatch));
    try std.testing.expectEqual(Status.limit_exceeded, statusFromError(error.CapacityExceeded));
    try std.testing.expectEqual(Status.limit_exceeded, statusFromError(error.PositionExceeded));
    try std.testing.expectEqual(Status.invalid_argument, statusFromError(error.DuplicateShard));
    try std.testing.expectEqual(Status.invalid_argument, statusFromError(error.DuplicateTensor));
    try std.testing.expectEqual(Status.invalid_state, statusFromError(error.InvalidState));
}

test "the feature mask names every family exactly once" {
    var seen = std.mem.zeroes([32]bool);
    inline for (@typeInfo(Feature).@"enum".field_names) |field_name| {
        const feature: Feature = @field(Feature, field_name);
        const bit = feature.bit();
        try std.testing.expect(bit != 0);
        try std.testing.expectEqual(@as(u32, 1), @popCount(bit));
        // Every declared family must be published by the mask, or JavaScript
        // would negotiate a family this build actually exports.
        try std.testing.expect(features & bit == bit);
        const index: usize = @intCast(@ctz(bit));
        try std.testing.expect(index < seen.len);
        try std.testing.expect(!seen[index]);
        seen[index] = true;
    }
}
