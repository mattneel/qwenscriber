//! Architecture configuration: the bridge from an official checkpoint to the
//! runtime.
//!
//! The converter reads the checkpoint's JSON configuration once, validates it,
//! and writes this fixed-layout structure into the model directory. The browser
//! runtime never parses JSON and never reads a checkpoint: it reads these fields,
//! validates them, and derives every tensor shape from them.
//!
//! That matters for the 1.7B target. Nothing in the runtime may assume 0.6B
//! dimensions: `audio_d_model`, `audio_layers`, `text_hidden_size`,
//! `text_layers`, `text_key_value_heads`, `vocab_size`, and the rest are all
//! data here, and `layout.zig` derives the tensor inventory from them. Both
//! 0.6B and 1.7B carry the same *set* of tensors per layer, so a single
//! config-driven model description covers both.

const std = @import("std");
const mel = @import("mel.zig");

pub const magic = "QWCFG001";
pub const magic_bytes: [8]u8 = magic[0..8].*;
pub const format_version: u32 = 1;

/// Largest value any token id, dimension, or count may take. Keeps a corrupt
/// configuration from producing arithmetic that overflows a buffer or index.
pub const field_max: u32 = 1 << 24;
pub const layer_max: u32 = 512;
pub const vocab_max: u32 = 1 << 21;
pub const decode_tokens_max: u32 = 8192;

pub const Architecture = enum(u32) {
    /// Qwen3-ASR: a Whisper-style audio tower, a two-layer projector, and a
    /// Qwen3 decoder with audio embeddings spliced in at placeholder positions.
    qwen3_asr = 1,
    _,
};

/// Post-convolution steps produced by running the audio tower's convolution
/// stack over `frames` mel frames *as a single input*.
///
/// Each of the three convolutions is kernel 3, stride 2, padding 1, so a length
/// `n` becomes `(n - 1) / 2 + 1`. Zero-length input stays zero, matching the
/// reference's `_post_cnn_length`: a chunk of pure padding contributes no
/// post-convolution positions.
///
/// This is the per-chunk count. Do not use it on a whole clip: the released
/// pipeline convolves each chunk independently, and applying the same
/// downsampling to the concatenated frames would lose a step at every chunk
/// boundary (375 instead of 390 for a 30-second clip). Use `packedStepCount`
/// for a clip.
pub fn postConvolutionSteps(frames: u32) u32 {
    if (frames == 0) return 0;
    var steps = frames;
    var step: u32 = 0;
    while (step < 3) : (step += 1) steps = (steps - 1) / 2 + 1;
    return steps;
}

pub const Error = error{
    BadMagic,
    BadVersion,
    Truncated,
    ReservedNotEmpty,
    UnknownArchitecture,
    InvalidDimension,
    InvalidHeadGeometry,
    InvalidMelGeometry,
    InvalidTokenId,
    InvalidEpsilon,
    InvalidRopeTheta,
    InvalidChunkGeometry,
    Misaligned,
};

/// `extern` so the byte layout is exactly the struct, with no Zig-specific
/// padding or reordering. Field order is the on-disk order.
pub const Config = extern struct {
    magic: [8]u8,
    format_version: u32,
    architecture: u32,

    // Audio tower.
    audio_d_model: u32,
    audio_layers: u32,
    audio_attention_heads: u32,
    audio_ffn_dim: u32,
    audio_downsample_hidden_size: u32,
    /// Mel frames per half-chunk; the convolution stack consumes `2 * this`.
    audio_n_window: u32,
    /// Mel frames spanned by one encoder attention window.
    audio_n_window_infer: u32,
    /// Sinusoidal positional embedding rows, one per post-convolution step.
    audio_max_position_steps: u32,
    /// Projector output width, i.e. the decoder's hidden size.
    audio_output_dim: u32,
    mel_bins: u32,
    audio_layer_norm_eps: f32,

    // Text decoder.
    text_hidden_size: u32,
    text_layers: u32,
    text_attention_heads: u32,
    text_key_value_heads: u32,
    text_head_dim: u32,
    text_ffn_dim: u32,
    vocab_size: u32,
    text_rms_norm_eps: f32,
    rope_theta: f32,

    // Reserved positions the decoder can address. A hard limit on the KV cache.
    max_positions: u32,
    max_decode_tokens: u32,

    // Special tokens.
    token_audio_start: u32,
    token_audio_end: u32,
    token_audio_pad: u32,
    token_im_start: u32,
    token_im_end: u32,
    token_endoftext: u32,
    token_asr_text: u32,
    token_eos_primary: u32,
    token_eos_secondary: u32,
    token_pad: u32,

    /// Optional model facts, one bit each. Bit 0 is `output_projection_tied`.
    ///
    /// The runtime reads only `config.bin` and the shards — it does not consult
    /// the manifest — so a fact that changes how weights are *resolved* has to
    /// live here. Tying is exactly that: a tied checkpoint ships no output
    /// projection, and without this bit that absence is indistinguishable from a
    /// missing tensor, which the loader rightly rejects.
    ///
    /// Unknown bits are rejected rather than ignored: a bit written by a newer
    /// converter could mean anything, and guessing would silently change what the
    /// model computes.
    flags: u32 = 0,
    reserved1: u32 = 0,
    reserved2: u32 = 0,
    reserved3: u32 = 0,

    /// The checkpoint has no separate output projection; the token embedding
    /// doubles as the unembedding. The released 0.6B checkpoint is this case.
    pub const flag_output_projection_tied: u32 = 1 << 0;
    const known_flags: u32 = flag_output_projection_tied;

    pub fn outputProjectionTied(self: *const Config) bool {
        return self.flags & flag_output_projection_tied != 0;
    }

    pub fn architectureKind(self: *const Config) Error!Architecture {
        const architecture: Architecture = @fromBackingInt(@intCast(self.architecture));
        if (architecture != .qwen3_asr) return Error.UnknownArchitecture;
        return architecture;
    }

    /// Head width of the audio tower's attention.
    pub fn audioHeadDim(self: *const Config) u32 {
        return self.audio_d_model / self.audio_attention_heads;
    }

    /// Mel frames the convolution stack consumes per chunk.
    pub fn audioChunkFrames(self: *const Config) u32 {
        return self.audio_n_window * 2;
    }

    /// Frequency bins remaining after the three stride-2 convolutions.
    ///
    /// Each convolution is kernel 3, stride 2, padding 1, so a length `n`
    /// becomes `(n - 1) / 2 + 1`. The released configuration spells the same
    /// three-step expression out inline; deriving it here keeps the runtime and
    /// the converter from transcribing it twice.
    pub fn audioFrequencyBinsAfterConvolutions(self: *const Config) u32 {
        var bins = self.mel_bins;
        var step: u32 = 0;
        while (step < 3) : (step += 1) bins = (bins - 1) / 2 + 1;
        return bins;
    }

    /// Width of the tensor that flattens the convolution stack's output.
    pub fn audioConvOutInputFeatures(self: *const Config) u32 {
        return self.audio_downsample_hidden_size * self.audioFrequencyBinsAfterConvolutions();
    }

    /// Time steps the convolution stack emits for one full chunk.
    pub fn audioChunkSteps(self: *const Config) u32 {
        var steps = self.audioChunkFrames();
        var step: u32 = 0;
        while (step < 3) : (step += 1) steps = (steps - 1) / 2 + 1;
        return steps;
    }

    /// Total packed encoder positions for a clip of `frames` valid mel frames.
    ///
    /// The audio tower convolves each whole chunk of `audioChunkFrames`
    /// independently and keeps only the positions belonging to valid frames, so
    /// the total is the whole chunks' contribution plus the partial chunk's.
    /// This is both the number of audio placeholder tokens the prompt carries
    /// and the number of rows the encoder produces. It is *not* the same as
    /// running the convolution over the concatenated frames: that loses one step
    /// at every chunk boundary.
    pub fn packedStepCount(self: *const Config, frames: u32) u32 {
        const chunk = self.audioChunkFrames();
        return (frames / chunk) * postConvolutionSteps(chunk) +
            postConvolutionSteps(frames % chunk);
    }

    /// Width of the decoder's query and output projections, and of one
    /// attention head's concatenation. May exceed `text_hidden_size`.
    pub fn textQueryWidth(self: *const Config) u32 {
        return self.textQueryElements();
    }

    /// Decoder attention query heads sharing one key/value head.
    pub fn textQueryGroups(self: *const Config) u32 {
        return self.text_attention_heads / self.text_key_value_heads;
    }

    pub fn textQueryElements(self: *const Config) u32 {
        return self.text_attention_heads * self.text_head_dim;
    }

    pub fn textKeyValueElements(self: *const Config) u32 {
        return self.text_key_value_heads * self.text_head_dim;
    }

    /// Bytes one layer's key or value cache needs for `positions` positions.
    pub fn kvBytesPerLayer(self: *const Config, positions: u32, item_bytes: u32) u64 {
        return @as(u64, self.textKeyValueElements()) * positions * item_bytes;
    }

    pub fn validate(self: *const Config) Error!void {
        if (!std.mem.eql(u8, &self.magic, &magic_bytes)) return Error.BadMagic;
        if (self.format_version != format_version) return Error.BadVersion;
        if (!self.flagsAreKnown()) return Error.ReservedNotEmpty;
        _ = try self.architectureKind();

        try requireRange(self.audio_d_model);
        try requireRange(self.audio_ffn_dim);
        try requireRange(self.audio_downsample_hidden_size);
        try requireRange(self.audio_output_dim);
        try requireRange(self.text_hidden_size);
        try requireRange(self.text_ffn_dim);
        try requireRange(self.text_head_dim);
        try requireLayers(self.audio_layers);
        try requireLayers(self.text_layers);

        if (self.mel_bins != mel.mel_bins) return Error.InvalidMelGeometry;
        if (self.audio_attention_heads == 0) return Error.InvalidHeadGeometry;
        if (self.audio_d_model % self.audio_attention_heads != 0) return Error.InvalidHeadGeometry;
        if (self.audioHeadDim() < 2 or self.audioHeadDim() % 2 != 0) return Error.InvalidHeadGeometry;

        if (self.text_attention_heads == 0) return Error.InvalidHeadGeometry;
        if (self.text_key_value_heads == 0) return Error.InvalidHeadGeometry;
        if (self.text_attention_heads % self.text_key_value_heads != 0) {
            return Error.InvalidHeadGeometry;
        }
        // Note that the query projection's width is `heads * head_dim`, which
        // need not equal the hidden size: the released 0.6B checkpoint projects
        // 1024 hidden units up to 16 * 128 = 2048 query units, and `o_proj`
        // projects back down. Requiring equality here would have rejected the
        // model this runtime exists to run.
        if (self.text_head_dim < 2 or self.text_head_dim % 2 != 0) return Error.InvalidHeadGeometry;
        if (self.text_hidden_size != self.audio_output_dim) return Error.InvalidDimension;

        // The mel frontend pads to a multiple of the chunk length, and encoder
        // attention windows must align with whole chunks.
        if (self.audio_n_window == 0) return Error.InvalidChunkGeometry;
        if (self.audioChunkFrames() > mel.capacity_frames) return Error.InvalidChunkGeometry;
        if (self.audio_n_window_infer % self.audioChunkFrames() != 0) {
            return Error.InvalidChunkGeometry;
        }
        if (self.audio_max_position_steps < self.audioChunkSteps()) {
            return Error.InvalidChunkGeometry;
        }
        if (self.mel_bins % 8 != 0) return Error.InvalidMelGeometry;

        if (self.vocab_size == 0 or self.vocab_size > vocab_max) return Error.InvalidDimension;
        try requireRange(self.max_positions);
        if (self.max_decode_tokens == 0 or self.max_decode_tokens > decode_tokens_max) {
            return Error.InvalidDimension;
        }

        if (!std.math.isFinite(self.audio_layer_norm_eps)) return Error.InvalidEpsilon;
        if (!std.math.isFinite(self.text_rms_norm_eps)) return Error.InvalidEpsilon;
        if (!(self.audio_layer_norm_eps > 0.0)) return Error.InvalidEpsilon;
        if (!(self.text_rms_norm_eps > 0.0)) return Error.InvalidEpsilon;
        if (!std.math.isFinite(self.rope_theta)) return Error.InvalidRopeTheta;
        if (!(self.rope_theta > 1.0)) return Error.InvalidRopeTheta;

        const tokens = [_]u32{
            self.token_audio_start, self.token_audio_end,   self.token_audio_pad,
            self.token_im_start,    self.token_im_end,      self.token_endoftext,
            self.token_asr_text,    self.token_eos_primary, self.token_eos_secondary,
            self.token_pad,
        };
        for (tokens) |token| {
            if (token >= self.vocab_size) return Error.InvalidTokenId;
        }
    }

    /// True when every set flag bit is one this build understands and the
    /// reserved words are untouched.
    fn flagsAreKnown(self: *const Config) bool {
        return self.flags & ~known_flags == 0 and
            self.reserved1 == 0 and
            self.reserved2 == 0 and
            self.reserved3 == 0;
    }

    fn requireRange(value: u32) Error!void {
        if (value == 0 or value > field_max) return Error.InvalidDimension;
    }

    fn requireLayers(value: u32) Error!void {
        if (value == 0 or value > layer_max) return Error.InvalidDimension;
    }
};

/// Reads a configuration from a model directory's `config.bin`.
///
/// `bytes` must be exactly one configuration and 4-byte aligned, because the
/// returned pointer aliases it rather than copying. JavaScript allocates the
/// buffer with `qw_alloc(size, 4)`.
pub fn parse(bytes: []const u8) Error!*const Config {
    if (bytes.len != @sizeOf(Config)) return Error.Truncated;
    if (@intFromPtr(bytes.ptr) % @alignOf(Config) != 0) return Error.Misaligned;
    const config: *const Config = @ptrCast(@alignCast(bytes.ptr));
    try config.validate();
    return config;
}

/// Exact on-disk size of `Config`, in bytes. JavaScript allocates this much and
/// the runtime refuses anything else, so the size is part of the format: adding
/// a field is a format version bump, not an incidental change.
pub const size_bytes: u32 = 160;

comptime {
    std.debug.assert(@sizeOf(Config) == size_bytes);
    std.debug.assert(@alignOf(Config) == 4);
    std.debug.assert(@offsetOf(Config, "magic") == 0);
    std.debug.assert(@offsetOf(Config, "format_version") == 8);
    std.debug.assert(@offsetOf(Config, "audio_d_model") == 16);
    std.debug.assert(@offsetOf(Config, "text_hidden_size") == 60);
    std.debug.assert(@offsetOf(Config, "token_audio_start") == 104);
    std.debug.assert(@offsetOf(Config, "flags") == 144);
}

test "configuration geometry derives from the released 0.6B checkpoint" {
    const config = Config{
        .magic = magic_bytes,
        .format_version = format_version,
        .architecture = @backingInt(Architecture.qwen3_asr),
        .audio_d_model = 896,
        .audio_layers = 18,
        .audio_attention_heads = 14,
        .audio_ffn_dim = 3584,
        .audio_downsample_hidden_size = 480,
        .audio_n_window = 50,
        .audio_n_window_infer = 800,
        .audio_max_position_steps = 13,
        .audio_output_dim = 1024,
        .mel_bins = 128,
        .audio_layer_norm_eps = 1e-5,
        .text_hidden_size = 1024,
        .text_layers = 28,
        .text_attention_heads = 16,
        .text_key_value_heads = 8,
        .text_head_dim = 128,
        .text_ffn_dim = 3072,
        .vocab_size = 151936,
        .text_rms_norm_eps = 1e-6,
        .rope_theta = 1000000.0,
        .max_positions = 16384,
        .max_decode_tokens = 256,
        .token_audio_start = 151669,
        .token_audio_end = 151670,
        .token_audio_pad = 151676,
        .token_im_start = 151644,
        .token_im_end = 151645,
        .token_endoftext = 151643,
        .token_asr_text = 151704,
        .token_eos_primary = 151643,
        .token_eos_secondary = 151645,
        .token_pad = 151643,
    };
    try config.validate();

    try std.testing.expectEqual(@as(u32, 64), config.audioHeadDim());
    try std.testing.expectEqual(@as(u32, 100), config.audioChunkFrames());
    // 128 mel bins downsample by 8 across three stride-2 convolutions.
    try std.testing.expectEqual(@as(u32, 16), config.audioFrequencyBinsAfterConvolutions());
    // 480 channels x 16 bins is the 7680-wide `conv_out` input.
    try std.testing.expectEqual(@as(u32, 7680), config.audioConvOutInputFeatures());
    try std.testing.expectEqual(@as(u32, 13), config.audioChunkSteps());
    try std.testing.expectEqual(@as(u32, 2), config.textQueryGroups());
    try std.testing.expectEqual(@as(u32, 2048), config.textQueryElements());
    try std.testing.expectEqual(@as(u32, 1024), config.textKeyValueElements());
    // A 30-second clip is 30 chunks of 13 steps each. The per-chunk formula is
    // the only one that yields 390, and it is what the audio placeholder count
    // in the prompt depends on.
    try std.testing.expectEqual(@as(u32, 390), config.packedStepCount(3000));
    try std.testing.expectEqual(@as(u32, 375), postConvolutionSteps(3000));
    // An 8-second clip is 8 whole chunks plus nothing left over.
    try std.testing.expectEqual(@as(u32, 104), config.packedStepCount(800));
    // A partial chunk contributes its own downsampled remainder.
    try std.testing.expectEqual(
        @as(u32, 104 + 7),
        config.packedStepCount(850),
    );
}

test "configuration geometry derives from the released 1.7B checkpoint" {
    // The 1.7B target differs only in dimensions. If any of these derivations
    // needed a branch, the runtime would have baked in 0.6B assumptions.
    const config = Config{
        .magic = magic_bytes,
        .format_version = format_version,
        .architecture = @backingInt(Architecture.qwen3_asr),
        .audio_d_model = 1024,
        .audio_layers = 24,
        .audio_attention_heads = 16,
        .audio_ffn_dim = 4096,
        .audio_downsample_hidden_size = 480,
        .audio_n_window = 50,
        .audio_n_window_infer = 800,
        .audio_max_position_steps = 13,
        .audio_output_dim = 2048,
        .mel_bins = 128,
        .audio_layer_norm_eps = 1e-5,
        .text_hidden_size = 2048,
        .text_layers = 28,
        .text_attention_heads = 16,
        .text_key_value_heads = 8,
        .text_head_dim = 128,
        .text_ffn_dim = 6144,
        .vocab_size = 151936,
        .text_rms_norm_eps = 1e-6,
        .rope_theta = 1000000.0,
        .max_positions = 16384,
        .max_decode_tokens = 256,
        .token_audio_start = 151669,
        .token_audio_end = 151670,
        .token_audio_pad = 151676,
        .token_im_start = 151644,
        .token_im_end = 151645,
        .token_endoftext = 151643,
        .token_asr_text = 151704,
        .token_eos_primary = 151643,
        .token_eos_secondary = 151645,
        .token_pad = 151643,
    };
    try config.validate();
    try std.testing.expectEqual(@as(u32, 64), config.audioHeadDim());
    try std.testing.expectEqual(@as(u32, 7680), config.audioConvOutInputFeatures());
    try std.testing.expectEqual(@as(u32, 2048), config.textQueryElements());
}

test "invalid configurations are rejected with a specific reason" {
    const base = Config{
        .magic = magic_bytes,
        .format_version = format_version,
        .architecture = @backingInt(Architecture.qwen3_asr),
        .audio_d_model = 896,
        .audio_layers = 18,
        .audio_attention_heads = 14,
        .audio_ffn_dim = 3584,
        .audio_downsample_hidden_size = 480,
        .audio_n_window = 50,
        .audio_n_window_infer = 800,
        .audio_max_position_steps = 13,
        .audio_output_dim = 1024,
        .mel_bins = 128,
        .audio_layer_norm_eps = 1e-5,
        .text_hidden_size = 1024,
        .text_layers = 28,
        .text_attention_heads = 16,
        .text_key_value_heads = 8,
        .text_head_dim = 128,
        .text_ffn_dim = 3072,
        .vocab_size = 151936,
        .text_rms_norm_eps = 1e-6,
        .rope_theta = 1000000.0,
        .max_positions = 16384,
        .max_decode_tokens = 256,
        .token_audio_start = 151669,
        .token_audio_end = 151670,
        .token_audio_pad = 151676,
        .token_im_start = 151644,
        .token_im_end = 151645,
        .token_endoftext = 151643,
        .token_asr_text = 151704,
        .token_eos_primary = 151643,
        .token_eos_secondary = 151645,
        .token_pad = 151643,
    };
    try base.validate();

    var bad_magic = base;
    bad_magic.magic[0] = 'X';
    try std.testing.expectError(Error.BadMagic, bad_magic.validate());

    var bad_version = base;
    bad_version.format_version = format_version + 1;
    try std.testing.expectError(Error.BadVersion, bad_version.validate());

    // Bit 0 is a known flag: the tied output projection. An unknown bit is a
    // configuration written by a newer converter, and guessing its meaning would
    // silently change what the model computes.
    var tied = base;
    tied.flags = Config.flag_output_projection_tied;
    try tied.validate();
    try std.testing.expect(tied.outputProjectionTied());

    var bad_flags = base;
    bad_flags.flags = 1 << 1;
    try std.testing.expectError(Error.ReservedNotEmpty, bad_flags.validate());

    var bad_reserved = base;
    bad_reserved.reserved1 = 1;
    try std.testing.expectError(Error.ReservedNotEmpty, bad_reserved.validate());

    var untied = base;
    untied.flags = 0;
    try untied.validate();
    try std.testing.expect(!untied.outputProjectionTied());

    var bad_arch = base;
    bad_arch.architecture = 99;
    try std.testing.expectError(Error.UnknownArchitecture, bad_arch.validate());

    var bad_heads = base;
    bad_heads.audio_attention_heads = 13;
    try std.testing.expectError(Error.InvalidHeadGeometry, bad_heads.validate());

    var bad_text_heads = base;
    bad_text_heads.text_attention_heads = 12;
    try std.testing.expectError(Error.InvalidHeadGeometry, bad_text_heads.validate());

    var bad_groups = base;
    bad_groups.text_key_value_heads = 5;
    try std.testing.expectError(Error.InvalidHeadGeometry, bad_groups.validate());

    // The query width is allowed to differ from the hidden size (that is how
    // the released 0.6B checkpoint is built), but the projector must land on
    // the decoder's hidden size.
    const wider_queries = base;
    try wider_queries.validate();

    var bad_projector = base;
    bad_projector.audio_output_dim = 2048;
    try std.testing.expectError(Error.InvalidDimension, bad_projector.validate());

    var bad_mel = base;
    bad_mel.mel_bins = 80;
    try std.testing.expectError(Error.InvalidMelGeometry, bad_mel.validate());

    var bad_window = base;
    bad_window.audio_n_window_infer = 850;
    try std.testing.expectError(Error.InvalidChunkGeometry, bad_window.validate());

    var bad_positions = base;
    bad_positions.audio_max_position_steps = 12;
    try std.testing.expectError(Error.InvalidChunkGeometry, bad_positions.validate());

    var bad_token = base;
    bad_token.token_audio_pad = 151936;
    try std.testing.expectError(Error.InvalidTokenId, bad_token.validate());

    var bad_eps = base;
    bad_eps.text_rms_norm_eps = 0.0;
    try std.testing.expectError(Error.InvalidEpsilon, bad_eps.validate());

    var bad_theta = base;
    bad_theta.rope_theta = 0.5;
    try std.testing.expectError(Error.InvalidRopeTheta, bad_theta.validate());

    var bad_layers = base;
    bad_layers.text_layers = 0;
    try std.testing.expectError(Error.InvalidDimension, bad_layers.validate());
}

test "parse enforces the exact configuration size" {
    var storage align(@alignOf(Config)) = std.mem.zeroes([@sizeOf(Config)]u8);
    try std.testing.expectError(Error.BadMagic, parse(&storage));
    try std.testing.expectError(
        Error.Truncated,
        parse(storage[0 .. storage.len - 1]),
    );
    const misaligned = try std.testing.allocator.alignedAlloc(u8, .@"4", storage.len + 4);
    defer std.testing.allocator.free(misaligned);
    try std.testing.expectError(Error.Misaligned, parse(misaligned[1..][0..storage.len]));

    const config: *Config = @ptrCast(&storage);
    config.* = .{
        .magic = magic_bytes,
        .format_version = format_version,
        .architecture = @backingInt(Architecture.qwen3_asr),
        .audio_d_model = 896,
        .audio_layers = 18,
        .audio_attention_heads = 14,
        .audio_ffn_dim = 3584,
        .audio_downsample_hidden_size = 480,
        .audio_n_window = 50,
        .audio_n_window_infer = 800,
        .audio_max_position_steps = 13,
        .audio_output_dim = 1024,
        .mel_bins = 128,
        .audio_layer_norm_eps = 1e-5,
        .text_hidden_size = 1024,
        .text_layers = 28,
        .text_attention_heads = 16,
        .text_key_value_heads = 8,
        .text_head_dim = 128,
        .text_ffn_dim = 3072,
        .vocab_size = 151936,
        .text_rms_norm_eps = 1e-6,
        .rope_theta = 1000000.0,
        .max_positions = 16384,
        .max_decode_tokens = 256,
        .token_audio_start = 151669,
        .token_audio_end = 151670,
        .token_audio_pad = 151676,
        .token_im_start = 151644,
        .token_im_end = 151645,
        .token_endoftext = 151643,
        .token_asr_text = 151704,
        .token_eos_primary = 151643,
        .token_eos_secondary = 151645,
        .token_pad = 151643,
    };
    const parsed = try parse(&storage);
    try std.testing.expectEqual(@as(u32, 896), parsed.audio_d_model);
}
