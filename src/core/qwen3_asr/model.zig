//! The Qwen3-ASR model: audio tower, projector, decoder, and greedy decoding.
//!
//! # Loading
//!
//! `Model.load` resolves every tensor `layout.Iterator` requires out of the
//! parsed shards and stores the resulting views in named fields. Nothing is
//! copied: each view points into the shard bytes the caller owns. Loading is
//! also where a model is *validated*: a missing tensor, the wrong shape, or a
//! format the build does not understand is reported here, once, instead of
//! surfacing as silent garbage during decoding.
//!
//! # Execution
//!
//! The forward pass follows the reference implementation step for step, because
//! every departure from it is a difference that has to be debugged later:
//!
//!   1. The log-mel frontend (`mel.compute`, done by the caller) yields 128 bins
//!      by `frames`.
//!   2. The convolution stack runs **per chunk** of `audio_n_window * 2` mel
//!      frames, stride 2 each with GELU in between, then a linear projection
//!      from `downsample_hidden_size * frequency_bins` to `audio_d_model`, then
//!      a sinusoidal position embedding added per time step.
//!   3. Chunks contribute only the steps that belong to valid mel frames, and
//!      those steps are packed into one flat sequence.
//!   4. The audio tower's blocks run over the packed sequence, with attention
//!      confined to windows of `audio_n_window_infer` mel frames' worth of
//!      steps. Each block is pre-norm: LayerNorm, attention, residual,
//!      LayerNorm, feed-forward, residual.
//!   5. `ln_post`, then the projector: linear, GELU, linear.
//!   6. The decoder is a Qwen3 stack with RMSNorm, per-head query/key norms,
//!      rotary embeddings, grouped-query attention with a key/value cache, and a
//!      gated SiLU feed-forward. Audio features are spliced into the embedding
//!      at the positions of the audio placeholder tokens.

const std = @import("std");
const container = @import("../container.zig");
const dtype = @import("../dtype.zig");
const math = @import("../math.zig");
const mel = @import("../mel.zig");
const model_config = @import("../model_config.zig");
const quant = @import("../quant.zig");
const tensor = @import("../tensor.zig");
const layout = @import("layout.zig");
const kernels = @import("kernels.zig");

/// Every failure loading or running a model can produce: container parse
/// errors, shape and binding errors, and runtime capacity limits.
pub const Error = container.Error || error{
    MissingTensor,
    UnexpectedShape,
    UnsupportedFormat,
    MissingShard,
    OutOfMemory,
    CapacityExceeded,
    DuplicateTensor,
    AudioTooLong,
    PromptMismatch,
};

/// Largest number of decoder positions the runtime will allocate a key/value
/// cache for. The configuration's `max_positions` is checked against this.
pub const position_limit: u32 = 8192;

/// A tensor as it was found in a shard.
pub const Binding = struct {
    kind: container.TensorKind,
    layer: u16,
    shape: tensor.Shape,
    format: dtype.Format,
    /// The tensor's payload, borrowed from the shard.
    bytes: []const u8,

    pub fn elementCount(self: Binding) u32 {
        return @intCast(self.shape.elementCount() catch 0);
    }

    /// The container guarantees every tensor payload starts on a 16-byte
    /// boundary inside a 16-byte-aligned shard, so these casts are sound; the
    /// assertion documents the invariant they rely on.
    pub fn f32Values(self: Binding) []const f32 {
        assert(self.format == .f32);
        assert(@intFromPtr(self.bytes.ptr) % 4 == 0);
        return std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(self.bytes)));
    }

    pub fn f16Values(self: Binding) []const u16 {
        assert(self.format == .f16);
        assert(@intFromPtr(self.bytes.ptr) % 2 == 0);
        return std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(self.bytes)));
    }

    pub fn matrix(self: Binding) kernels.Matrix {
        return switch (self.format) {
            .f32 => .{ .f32 = self.f32Values() },
            .f16 => .{ .f16 = self.f16Values() },
            .q4, .q5, .q8 => blk: {
                const rows = self.shape.dims[0];
                const cols = self.shape.dims[1];
                const plane = quant.planeLayout(
                    self.format,
                    rows,
                    cols,
                    0,
                    quant.tensor_alignment_bytes,
                ) catch unreachable;
                break :blk .{ .quantized = .{
                    .format = self.format,
                    .scales = self.bytes[@intCast(plane.scales_offset_bytes)..][0..@intCast(plane.scales_len_bytes)],
                    .data = self.bytes[@intCast(plane.data_offset_bytes)..][0..@intCast(plane.data_len_bytes)],
                } };
            },
            else => unreachable,
        };
    }
};

/// One audio tower block's weights.
pub const AudioLayer = struct {
    q_weight: kernels.Matrix,
    q_bias: []const f32,
    k_weight: kernels.Matrix,
    k_bias: []const f32,
    v_weight: kernels.Matrix,
    v_bias: []const f32,
    out_weight: kernels.Matrix,
    out_bias: []const f32,
    attention_norm_weight: []const f32,
    attention_norm_bias: []const f32,
    ffn_in_weight: kernels.Matrix,
    ffn_in_bias: []const f32,
    ffn_out_weight: kernels.Matrix,
    ffn_out_bias: []const f32,
    final_norm_weight: []const f32,
    final_norm_bias: []const f32,
};

/// One decoder block's weights.
pub const DecoderLayer = struct {
    q_weight: kernels.Matrix,
    k_weight: kernels.Matrix,
    v_weight: kernels.Matrix,
    out_weight: kernels.Matrix,
    q_norm_weight: []const f32,
    k_norm_weight: []const f32,
    attention_norm_weight: []const f32,
    ffn_norm_weight: []const f32,
    gate_weight: kernels.Matrix,
    up_weight: kernels.Matrix,
    down_weight: kernels.Matrix,
};

/// Scratch buffers, allocated once. Every buffer here is sized from the
/// configuration, so the forward pass performs no allocation.
pub const Scratch = struct {
    /// One weight row, for the row-at-a-time decoding kernels.
    row: []f32,
    /// Audio tower: mel frames of one chunk, `mel_bins x chunk_frames`.
    mel_chunk: []f32,
    /// Convolution stack outputs for one chunk.
    conv1: []f32,
    conv2: []f32,
    conv3: []f32,
    /// Packed encoder sequence, `steps x audio_d_model`.
    encoder: []f32,
    /// Flattened (channel, frequency) row feeding `conv_out`.
    conv_flat: []f32,
    /// Packed attention projections, `steps x (3 * audio_d_model)`.
    encoder_qkv: []f32,
    /// Normalization target, so the residual in `encoder` survives.
    encoder_normed: []f32,
    /// One block's output (attention or feed-forward), added back into `encoder`.
    encoder_block: []f32,
    encoder_ffn: []f32,
    /// Encoder attention scores, one window.
    window_scores: []f32,
    /// Projector output, `steps x text_hidden_size`.
    projected: []f32,
    /// Decoder: one token's hidden state, and its projections.
    hidden: []f32,
    normed: []f32,
    query: []f32,
    key: []f32,
    value: []f32,
    attention_out: []f32,
    gate: []f32,
    up: []f32,
    /// Decoder attention scores, one position.
    position_scores: []f32,
    /// Logits for one position.
    logits: []f32,
    /// Rotary tables, `max_positions x head_dim/2`.
    rope_cos: []f32,
    rope_sin: []f32,
    /// Key/value cache, `positions x key_value_width`, per layer. Exactly one of the two pairs is
    /// allocated: the f32 pair, or the quantized pair, whichever the model was loaded for. The
    /// unused pair is empty rather than absent, so nothing has to be nullable.
    cache_keys: []f32,
    cache_values: []f32,
    /// The same cache in the model's own quantized layout: one plane per layer, scales then codes,
    /// as `quant.planeBytes` describes.
    cache_keys_q8: []u8,
    cache_values_q8: []u8,
};

pub const Model = struct {
    config: model_config.Config,

    /// Width of the key/value cache this model holds. The cache is read in full on every decoded
    /// token, so this decides whether a model fits a bounded instance at all; `cacheBytes` reports
    /// what it costs, and the ABI carries that figure to the caller so a budget is computed from
    /// the format actually in use rather than from the widest one possible.
    cache_format: model_config.CacheFormat = .f32,

    conv1_weight: []const u16,
    conv1_bias: []const f32,
    conv2_weight: []const u16,
    conv2_bias: []const f32,
    conv3_weight: []const u16,
    conv3_bias: []const f32,
    conv_out_weight: kernels.Matrix,
    audio_final_norm_weight: []const f32,
    audio_final_norm_bias: []const f32,

    projector_in_weight: kernels.Matrix,
    projector_in_bias: []const f32,
    projector_out_weight: kernels.Matrix,
    projector_out_bias: []const f32,

    embed_tokens: kernels.Matrix,
    output_weight: kernels.Matrix,
    decoder_final_norm_weight: []const f32,

    audio_layers: []const AudioLayer,
    decoder_layers: []const DecoderLayer,

    scratch: Scratch,

    /// Steps the packed encoder sequence can hold, i.e. a whole 30 seconds.
    pub fn maxSteps(self: *const Model) u32 {
        return self.config.packedStepCount(mel.capacity_frames);
    }

    pub fn maxWindow(self: *const Model) u32 {
        return (self.config.audio_n_window_infer / self.config.audioChunkFrames()) *
            self.config.audioChunkSteps();
    }

    pub fn kvWidth(self: *const Model) u32 {
        return self.config.textKeyValueElements();
    }

    /// Bytes of key/value cache the model holds, in the format it was loaded for.
    pub fn cacheBytes(self: *const Model) u64 {
        const per_layer = quant.planeBytes(
            self.cache_format.toDtype(),
            self.config.max_positions,
            self.kvWidth(),
        );
        return 2 * @as(u64, self.config.text_layers) * per_layer;
    }

    /// Bytes of scratch the model holds, *excluding* the key/value cache, which
    /// `cacheBytes` reports on its own: a caller that budgets by adding the two
    /// must not count the cache twice.
    ///
    /// The cache is identified by field name rather than by subtracting
    /// `cacheBytes`, because the subtraction only holds for a model whose cache
    /// buffers are already populated — and a caller asking what scratch it must
    /// provide is asking about exactly the state where they are not.
    pub fn scratchBytes(self: *const Model) u64 {
        const info = @typeInfo(Scratch).@"struct";
        var total: u64 = 0;
        inline for (info.field_names, info.field_types) |field_name, field_type| {
            // Every scratch buffer is a slice, but not every one is f32: the quantized cache planes
            // are bytes. The element size therefore comes from the field's own type rather than from
            // an assumption, which is what keeps a byte plane from being counted as f32 words.
            comptime std.debug.assert(@typeInfo(field_type) == .pointer);
            if (comptime isCacheBuffer(field_name)) continue;
            const buffer = @field(self.scratch, field_name);
            total += @as(u64, buffer.len) * @sizeOf(std.meta.Elem(field_type));
        }
        return total;
    }

    /// The scratch buffers that belong to the key/value cache, by name.
    ///
    /// Renaming one is a compile error here rather than a silently inflated
    /// scratch report, which is the failure the old subtraction-based version was
    /// trying to catch from the other end.
    fn isCacheBuffer(comptime field_name: []const u8) bool {
        const cache_fields = [_][]const u8{ "cache_keys", "cache_values", "cache_keys_q8", "cache_values_q8" };
        inline for (cache_fields) |name| {
            if (!@hasField(Scratch, name)) {
                @compileError("a cache buffer was renamed: update scratchBytes' exclusion list");
            }
            if (std.mem.eql(u8, field_name, name)) return true;
        }
        return false;
    }

    /// Resolves every required tensor out of the parsed shards.
    ///
    /// `shards` are expected to be the model's shards in any order; a tensor may
    /// live in any of them. Duplicates are an error rather than a silent
    /// last-wins, because they mean the converter emitted an ambiguous model.
    /// Loads a model with the reference-width f32 key/value cache.
    pub fn load(
        arena: std.mem.Allocator,
        config: model_config.Config,
        shards: []const *const container.File,
    ) Error!Model {
        return loadWithCache(arena, config, shards, .f32);
    }

    /// Loads a model whose key/value cache stores elements in `cache_format`.
    pub fn loadWithCache(
        arena: std.mem.Allocator,
        config: model_config.Config,
        shards: []const *const container.File,
        cache_format: model_config.CacheFormat,
    ) Error!Model {
        try validateCapacity(&config);

        var bindings = try arena.alloc(Binding, layout.Iterator.count(&config));
        var iterator = layout.Iterator.init(&config);
        const tied_output = config.outputProjectionTied();
        var index: usize = 0;
        while (iterator.next()) |required| {
            bindings[index] = try resolveRequired(&config, required, shards, tied_output);
            index += 1;
        }

        var model = Model{
            .config = config,
            .conv1_weight = undefined,
            .conv1_bias = undefined,
            .conv2_weight = undefined,
            .conv2_bias = undefined,
            .conv3_weight = undefined,
            .conv3_bias = undefined,
            .conv_out_weight = undefined,
            .audio_final_norm_weight = undefined,
            .audio_final_norm_bias = undefined,
            .projector_in_weight = undefined,
            .projector_in_bias = undefined,
            .projector_out_weight = undefined,
            .projector_out_bias = undefined,
            .embed_tokens = undefined,
            .output_weight = undefined,
            .decoder_final_norm_weight = undefined,
            .audio_layers = undefined,
            .decoder_layers = undefined,
            .scratch = undefined,
        };

        model.conv1_weight = findBinding(&config, bindings, .audio_conv1_weight, 0).f16Values();
        model.conv1_bias = findBinding(&config, bindings, .audio_conv1_bias, 0).f32Values();
        model.conv2_weight = findBinding(&config, bindings, .audio_conv2_weight, 0).f16Values();
        model.conv2_bias = findBinding(&config, bindings, .audio_conv2_bias, 0).f32Values();
        model.conv3_weight = findBinding(&config, bindings, .audio_conv3_weight, 0).f16Values();
        model.conv3_bias = findBinding(&config, bindings, .audio_conv3_bias, 0).f32Values();
        model.conv_out_weight = findBinding(&config, bindings, .audio_conv_out_weight, 0).matrix();
        model.audio_final_norm_weight = findBinding(
            &config,
            bindings,
            .audio_final_norm_weight,
            0,
        ).f32Values();
        model.audio_final_norm_bias = findBinding(
            &config,
            bindings,
            .audio_final_norm_bias,
            0,
        ).f32Values();

        model.projector_in_weight = findBinding(&config, bindings, .projector_in_weight, 0).matrix();
        model.projector_in_bias = findBinding(&config, bindings, .projector_in_bias, 0).f32Values();
        model.projector_out_weight = findBinding(&config, bindings, .projector_out_weight, 0).matrix();
        model.projector_out_bias = findBinding(&config, bindings, .projector_out_bias, 0).f32Values();

        model.embed_tokens = findBinding(&config, bindings, .decoder_embed_tokens_weight, 0).matrix();
        model.decoder_final_norm_weight = findBinding(
            &config,
            bindings,
            .decoder_final_norm_weight,
            0,
        ).f32Values();
        // A tied checkpoint has no separate output projection, in which case the
        // embedding matrix is also the unembedding, exactly as the reference's
        // `tie_word_embeddings` intends.
        model.output_weight = if (findOptional(&config, bindings, .decoder_output_weight, 0)) |binding|
            binding.matrix()
        else
            model.embed_tokens;

        model.cache_format = cache_format;
        model.audio_layers = try loadAudioLayers(arena, &config, bindings);
        model.decoder_layers = try loadDecoderLayers(arena, &config, bindings);
        model.scratch = try allocateScratch(arena, &config, cache_format);
        try kernels.buildRopeTables(
            model.scratch.rope_cos,
            model.scratch.rope_sin,
            config.text_head_dim / 2,
            config.rope_theta,
            config.max_positions,
            config.text_head_dim / 2,
        );
        return model;
    }

    /// Parses a set of shard files and loads the model from them.
    ///
    /// Shard byte slices are borrowed for the lifetime of the model: the weight
    /// views point straight into them, so the caller must keep them alive and
    /// unmodified.
    pub fn loadFromShardBytes(
        arena: std.mem.Allocator,
        config: model_config.Config,
        shard_bytes: []const []align(16) const u8,
    ) Error!Model {
        return loadFromShardBytesWithCache(arena, config, shard_bytes, .f32);
    }

    /// Parses shards and loads them into a model with the named cache format.
    pub fn loadFromShardBytesWithCache(
        arena: std.mem.Allocator,
        config: model_config.Config,
        shard_bytes: []const []align(16) const u8,
        cache_format: model_config.CacheFormat,
    ) Error!Model {
        const files = arena.alloc(container.File, shard_bytes.len) catch return Error.OutOfMemory;
        const pointers = arena.alloc(*const container.File, shard_bytes.len) catch
            return Error.OutOfMemory;
        for (shard_bytes, 0..) |bytes, index| {
            files[index] = container.File.parse(bytes) catch |err| switch (err) {
                error.ChecksumMismatch => return Error.UnsupportedFormat,
                else => return Error.MissingShard,
            };
            try files[index].verifyChecksum();
            pointers[index] = &files[index];
        }
        return loadWithCache(arena, config, pointers, cache_format);
    }

    /// Runs the audio tower and projector over a log-mel spectrogram.
    ///
    /// `log_mel` is `mel_bins x frames` row major, as `mel.compute` writes it.
    /// `out` receives `steps x text_hidden_size`, where `steps` is what
    /// `config.packedStepCount(frames)` says. Returns the step count.
    pub fn projectAudio(
        self: *Model,
        log_mel: []const f32,
        frames: u32,
        out: []f32,
    ) Error!u32 {
        const steps = try self.encodeAudio(log_mel, frames);
        try self.projectSteps(steps, out);
        return steps;
    }

    /// Runs the convolution stack and the audio tower, leaving the packed
    /// sequence in scratch memory. Returns the step count.
    ///
    /// Kept separate from `projectSteps` so the bring-up runner can dump the
    /// audio tower's output on its own and compare it against the reference
    /// implementation stage by stage.
    pub fn encodeAudio(self: *Model, log_mel: []const f32, frames: u32) Error!u32 {
        const steps = try self.encodeConvStage(log_mel, frames);
        try self.runAudioTower(steps);
        return steps;
    }

    /// The convolution stack and its projection, packed, without the tower's
    /// blocks. Returns the step count.
    ///
    /// Split from `encodeAudio` because the reference implementation dumps this
    /// stage on its own (`audio_conv_out`), and comparing it is what separates a
    /// convolution defect from a block defect. Nothing else calls it.
    pub fn encodeConvStage(self: *Model, log_mel: []const f32, frames: u32) Error!u32 {
        const config = &self.config;
        const steps = config.packedStepCount(frames);
        if (steps > self.maxSteps()) return Error.AudioTooLong;
        if (log_mel.len != @as(usize, config.mel_bins) * frames) return Error.UnexpectedShape;
        try self.convolveAndPack(log_mel, frames, steps);
        return steps;
    }

    /// The tower's blocks, over a packed sequence produced by `encodeConvStage`.
    /// Split for the same reason: the two stages are dumped and compared
    /// separately against the reference.
    pub fn encodeTowerStage(self: *Model, steps: u32) Error!void {
        return self.runAudioTower(steps);
    }

    /// The packed audio tower output for the most recent `encodeAudio`.
    pub fn encodedSteps(self: *Model, steps: u32) []const f32 {
        return self.scratch.encoder[0 .. @as(usize, steps) * self.config.audio_d_model];
    }

    /// Projects the packed sequence into the decoder's hidden size.
    pub fn projectSteps(self: *Model, steps: u32, out: []f32) Error!void {
        if (out.len < @as(usize, steps) * self.config.text_hidden_size) {
            return Error.CapacityExceeded;
        }
        try self.runProjector(steps, out);
    }

    /// Convolution stack, per chunk, packed into `scratch.encoder`.
    fn convolveAndPack(self: *Model, log_mel: []const f32, frames: u32, steps: u32) Error!void {
        const config = &self.config;
        const chunk_frames = config.audioChunkFrames();
        const chunk_steps = config.audioChunkSteps();
        const frequency_bins = config.audioFrequencyBinsAfterConvolutions();
        const downsample = config.audio_downsample_hidden_size;
        const d_model = config.audio_d_model;
        const conv_out_features = config.audioConvOutInputFeatures();

        const last_height = frequency_bins;

        // Each stage halves the frequency and time extent of the one before it.
        // Writing them as a chain rather than as three literals is what keeps the
        // third stage from being handed the second stage's input geometry, which
        // is exactly the mistake the kernel's shape check caught.
        const stages = [3]kernels.ConvGeometry{
            .{
                .out_channels = downsample,
                .in_channels = 1,
                .in_height = config.mel_bins,
                .in_width = chunk_frames,
                .kernel = 3,
            },
            .{
                .out_channels = downsample,
                .in_channels = downsample,
                .in_height = (config.mel_bins - 1) / 2 + 1,
                .in_width = (chunk_frames - 1) / 2 + 1,
                .kernel = 3,
            },
            .{
                .out_channels = downsample,
                .in_channels = downsample,
                .in_height = ((config.mel_bins - 1) / 2 + 1 - 1) / 2 + 1,
                .in_width = ((chunk_frames - 1) / 2 + 1 - 1) / 2 + 1,
                .kernel = 3,
            },
        };

        const chunk_count = (frames + chunk_frames - 1) / chunk_frames;
        var packed_count: u32 = 0;

        var chunk: u32 = 0;
        while (chunk < chunk_count) : (chunk += 1) {
            const chunk_start = chunk * chunk_frames;
            const available = if (frames > chunk_start) frames - chunk_start else 0;
            const valid_frames = @min(available, chunk_frames);

            // Assemble this chunk's mel frames, zero-filling the tail. The
            // reference pads in the mel domain, so absent frames are zeros.
            @memset(self.scratch.mel_chunk, 0.0);
            var bin: u32 = 0;
            while (bin < config.mel_bins) : (bin += 1) {
                const source = log_mel[@as(usize, bin) * frames + chunk_start ..][0..valid_frames];
                @memcpy(self.scratch.mel_chunk[bin * chunk_frames ..][0..valid_frames], source);
            }

            try kernels.conv3x3Stride2Gelu(
                self.scratch.conv1,
                self.scratch.mel_chunk,
                self.conv1_weight,
                self.conv1_bias,
                stages[0],
            );
            try kernels.conv3x3Stride2Gelu(
                self.scratch.conv2,
                self.scratch.conv1,
                self.conv2_weight,
                self.conv2_bias,
                stages[1],
            );
            try kernels.conv3x3Stride2Gelu(
                self.scratch.conv3,
                self.scratch.conv2,
                self.conv3_weight,
                self.conv3_bias,
                stages[2],
            );

            // The reference permutes the convolution output to
            // (time, channel, frequency) before flattening, so the projection
            // sees channel-major features with the time step leading.
            const valid_steps = model_config.postConvolutionSteps(valid_frames);
            var step: u32 = 0;
            while (step < chunk_steps) : (step += 1) {
                if (step >= valid_steps) break;
                const row = self.scratch.encoder[@as(usize, packed_count) * d_model ..][0..d_model];
                try self.projectConvStep(step, downsample, last_height, conv_out_features, row);
                packed_count += 1;
            }
        }
        assert(packed_count == steps);
    }

    /// One time step of the convolution stack through `conv_out` and the
    /// sinusoidal position embedding.
    fn projectConvStep(
        self: *Model,
        step: u32,
        downsample: u32,
        frequency_bins: u32,
        conv_out_features: u32,
        out: []f32,
    ) Error!void {
        const d_model = self.config.audio_d_model;
        assert(out.len == d_model);
        assert(downsample * frequency_bins == conv_out_features);

        // Flatten (channel, frequency) as the reference's permute-then-view does.
        // The projection's column count is fixed, so the row is assembled into
        // the row scratch buffer and multiplied directly.
        if (self.scratch.conv_flat.len < conv_out_features) return Error.CapacityExceeded;
        const flat = self.scratch.conv_flat[0..conv_out_features];
        var channel: u32 = 0;
        while (channel < downsample) : (channel += 1) {
            var frequency: u32 = 0;
            while (frequency < frequency_bins) : (frequency += 1) {
                flat[channel * frequency_bins + frequency] =
                    self.scratch.conv3[
                        (@as(usize, channel) * frequency_bins + frequency) *
                            self.config.audioChunkSteps() + step
                    ];
            }
        }
        try self.conv_out_weight.apply(out, flat, conv_out_features, 1, self.scratch.row);
        addSinusoidalPosition(out, step, d_model);
    }

    /// The audio tower's blocks over the packed sequence.
    ///
    /// Each block is the reference's pre-norm shape:
    ///
    ///     x = x + attention(norm(x))
    ///     x = x + feed_forward(norm(x))
    ///
    /// so the normalization must write to a separate buffer; normalizing in
    /// place would destroy the residual the block has to add back.
    fn runAudioTower(self: *Model, steps: u32) Error!void {
        const config = &self.config;
        const d_model = config.audio_d_model;
        const window_steps = self.maxWindow();
        const total = @as(usize, steps) * d_model;
        const encoder = self.scratch.encoder[0..total];
        const normed = self.scratch.encoder_normed[0..total];
        const block = self.scratch.encoder_block[0..total];
        const qkv = self.scratch.encoder_qkv[0 .. @as(usize, steps) * 3 * d_model];

        for (self.audio_layers) |*layer| {
            try kernels.layerNormInto(
                normed,
                encoder,
                layer.attention_norm_weight,
                layer.attention_norm_bias,
                d_model,
                steps,
                config.audio_layer_norm_eps,
            );
            try projectQkv(self, layer, normed, qkv, steps);
            try runWindowedAttention(self, qkv, block, steps, window_steps);
            try projectAttentionOutput(self, layer, block, steps);
            try kernels.addInPlace(encoder, block);

            try kernels.layerNormInto(
                normed,
                encoder,
                layer.final_norm_weight,
                layer.final_norm_bias,
                d_model,
                steps,
                config.audio_layer_norm_eps,
            );
            try activateFfn(
                self,
                layer.ffn_in_weight,
                layer.ffn_in_bias,
                layer.ffn_out_weight,
                layer.ffn_out_bias,
                normed,
                self.scratch.encoder_ffn,
                block,
                steps,
                d_model,
                config.audio_ffn_dim,
            );
            try kernels.addInPlace(encoder, block);
        }

        try kernels.layerNorm(
            encoder,
            self.audio_final_norm_weight,
            self.audio_final_norm_bias,
            d_model,
            steps,
            config.audio_layer_norm_eps,
        );
    }

    /// Projector: linear, GELU, linear.
    fn runProjector(self: *Model, steps: u32, out: []f32) Error!void {
        const config = &self.config;
        const d_model = config.audio_d_model;
        const hidden = config.text_hidden_size;
        const encoder = self.scratch.encoder[0 .. @as(usize, steps) * d_model];
        // `encoder_ffn` is the only scratch wide enough to hold the
        // intermediate; the projection cannot write in place because a row's
        // output would overwrite input columns the later rows still need.
        const intermediate = self.scratch.encoder_ffn[0 .. @as(usize, steps) * d_model];

        try self.projector_in_weight.apply(intermediate, encoder, d_model, steps, self.scratch.row);
        try kernels.addBias(intermediate, self.projector_in_bias, d_model, steps);
        for (intermediate) |*value| value.* = math.gelu(value.*);

        try self.projector_out_weight.apply(
            out[0 .. @as(usize, steps) * hidden],
            intermediate,
            d_model,
            steps,
            self.scratch.row,
        );
        try kernels.addBias(
            out[0 .. @as(usize, steps) * hidden],
            self.projector_out_bias,
            hidden,
            steps,
        );
    }
};

const assert = std.debug.assert;

/// Adds the sinusoidal position embedding for `step` onto `row`, in place.
///
/// `log_timescale_increment = ln(10000) / (channels / 2 - 1)`, and the row is
/// `concat(sin(scaled), cos(scaled))`, exactly as the reference builds it.
///
/// The addition is the whole content of this function: the reference does
/// `conv_out += positional_embedding[:time_steps]`, so the embedding is a term
/// *added* to the projected convolution output. It wrote the embedding over the
/// row instead, and because it covers every index of the row, that discarded the
/// projected convolution output completely — the tower saw a pure position
/// signal with no audio in it, which is why a 1-second tone and a speech clip
/// decoded the same tokens.
fn addSinusoidalPosition(row: []f32, step: u32, channels: u32) void {
    const half = channels / 2;
    const log_timescale_increment = @log(10000.0) /
        (@as(f64, @floatFromInt(half)) - 1.0);
    const position: f64 = @floatFromInt(step);
    var index: u32 = 0;
    while (index < half) : (index += 1) {
        const inverse_timescale = @exp(-log_timescale_increment * @as(f64, @floatFromInt(index)));
        const angle = position * inverse_timescale;
        row[index] += @floatCast(@sin(angle));
        row[half + index] += @floatCast(@cos(angle));
    }
}

fn projectQkv(
    model: *Model,
    layer: *const AudioLayer,
    input_sequence: []const f32,
    qkv: []f32,
    steps: u32,
) Error!void {
    const d_model = model.config.audio_d_model;
    const stride = 3 * d_model;
    var step: u32 = 0;
    while (step < steps) : (step += 1) {
        const input = input_sequence[@as(usize, step) * d_model ..][0..d_model];
        const row = qkv[@as(usize, step) * stride ..][0..stride];
        try layer.q_weight.apply(row[0..d_model], input, d_model, 1, model.scratch.row);
        try layer.k_weight.apply(row[d_model..][0..d_model], input, d_model, 1, model.scratch.row);
        try layer.v_weight.apply(row[2 * d_model ..][0..d_model], input, d_model, 1, model.scratch.row);
        try kernels.addBias(row[0..d_model], layer.q_bias, d_model, 1);
        try kernels.addBias(row[d_model..][0..d_model], layer.k_bias, d_model, 1);
        try kernels.addBias(row[2 * d_model ..][0..d_model], layer.v_bias, d_model, 1);
    }
}

/// Attention inside each window, written straight into `block`.
///
/// The query, key, and value projections for the whole packed sequence live in
/// one buffer, separated by `audio_d_model`; passing the three starting offsets
/// with the shared row stride lets the kernel walk a window without copying it.
fn runWindowedAttention(
    model: *Model,
    qkv: []const f32,
    block: []f32,
    steps: u32,
    window_steps: u32,
) Error!void {
    const d_model = model.config.audio_d_model;
    const stride = 3 * d_model;
    var window_start: u32 = 0;
    while (window_start < steps) {
        const window_length = @min(window_steps, steps - window_start);
        try kernels.audioAttentionWindow(
            block,
            d_model,
            qkv,
            stride,
            qkv[d_model..],
            stride,
            qkv[2 * d_model ..],
            stride,
            window_start,
            window_length,
            model.config.audio_attention_heads,
            model.config.audioHeadDim(),
            model.scratch.window_scores,
        );
        window_start += window_length;
    }
}

/// The audio block's output projection, applied per window.
fn projectAttentionOutput(
    model: *Model,
    layer: *const AudioLayer,
    block: []f32,
    steps: u32,
) Error!void {
    const d_model = model.config.audio_d_model;
    const normed = model.scratch.encoder_normed;
    try layer.out_weight.apply(
        normed[0 .. @as(usize, steps) * d_model],
        block[0 .. @as(usize, steps) * d_model],
        d_model,
        steps,
        model.scratch.row,
    );
    try kernels.addBias(
        normed[0 .. @as(usize, steps) * d_model],
        layer.out_bias,
        d_model,
        steps,
    );
    // Move the projected result back into the block buffer, which the caller
    // then adds to the residual. `normed` is free at this point in the block.
    @memcpy(block[0 .. @as(usize, steps) * d_model], normed[0 .. @as(usize, steps) * d_model]);
}

/// One feed-forward sub-block: `gelu(x * in_weight + in_bias) * out_weight +
/// out_bias`, written to `target`. `x` is left intact for the residual.
fn activateFfn(
    model: *Model,
    in_weight: kernels.Matrix,
    in_bias: []const f32,
    out_weight: kernels.Matrix,
    out_bias: []const f32,
    x: []const f32,
    ffn: []f32,
    target: []f32,
    steps: u32,
    width: u32,
    ffn_dim: u32,
) Error!void {
    const wide = ffn[0 .. @as(usize, steps) * ffn_dim];
    try in_weight.apply(wide, x, width, steps, model.scratch.row);
    try kernels.addBias(wide, in_bias, ffn_dim, steps);
    for (wide) |*value| value.* = math.gelu(value.*);

    try out_weight.apply(target, wide, ffn_dim, steps, model.scratch.row);
    try kernels.addBias(target, out_bias, width, steps);
}

const d_model_max = 4096;

/// Resolves one tensor the inventory requires, tolerating the one absence a tied
/// model is allowed to have.
///
/// A tied checkpoint ships no output projection: the embedding matrix *is* the
/// unembedding, so the tensor legitimately does not exist. Substituting the
/// embedding's binding keeps `bindings` fully initialized — later lookups scan
/// every element — and leaves `load`'s output-projection fallback to do its job.
/// A checkpoint that does ship a projection is used as-is even when the flag is
/// set: bytes beat declarations, and the shape check inside `resolve` still
/// rejects an embedding that does not match the projection's shape, which is what
/// a broken tie looks like.
fn resolveRequired(
    config: *const model_config.Config,
    required: layout.Required,
    shards: []const *const container.File,
    tied_output: bool,
) Error!Binding {
    if (!tied_output or required.kind != .decoder_output_weight) {
        return resolve(required, shards);
    }
    return resolve(required, shards) catch |err| switch (err) {
        // The embedding's own requirement, not a hand-built one: the layout
        // already asserts that the two share a shape and a storage class.
        Error.MissingTensor => {
            const embedding = layout.find(config, .decoder_embed_tokens_weight, 0) orelse
                return Error.MissingTensor;
            return resolve(embedding, shards);
        },
        else => err,
    };
}

fn resolve(required: layout.Required, shards: []const *const container.File) Error!Binding {
    var found: ?Binding = null;
    for (shards) |shard| {
        const entry = shard.find(@backingInt(required.kind), required.layer) orelse continue;
        if (found != null) return Error.DuplicateTensor;
        const format = entry.storageFormat() catch return Error.UnsupportedFormat;
        const shape = entry.shape() catch return Error.UnexpectedShape;
        if (!shape.eql(&required.shape)) return Error.UnexpectedShape;
        const bytes = shard.tensorBytes(@backingInt(required.kind), required.layer) orelse
            return Error.MissingTensor;
        found = .{
            .kind = required.kind,
            .layer = required.layer,
            .shape = shape,
            .format = format,
            .bytes = bytes,
        };
    }
    return found orelse Error.MissingTensor;
}

fn findBinding(
    config: *const model_config.Config,
    bindings: []const Binding,
    kind: container.TensorKind,
    layer: u16,
) Binding {
    return findOptional(config, bindings, kind, layer) orelse unreachable;
}

fn findOptional(
    config: *const model_config.Config,
    bindings: []const Binding,
    kind: container.TensorKind,
    layer: u16,
) ?Binding {
    _ = config;
    for (bindings) |binding| {
        if (binding.kind == kind and binding.layer == layer) return binding;
    }
    return null;
}

fn loadAudioLayers(
    arena: std.mem.Allocator,
    config: *const model_config.Config,
    bindings: []const Binding,
) Error![]const AudioLayer {
    const layers = arena.alloc(AudioLayer, config.audio_layers) catch return Error.OutOfMemory;
    for (layers, 0..) |*layer, index| {
        const base: u16 = @intCast(container.audio_layer_base + index);
        layer.* = .{
            .q_weight = findBinding(config, bindings, .audio_layer_attention_q_weight, base).matrix(),
            .q_bias = findBinding(config, bindings, .audio_layer_attention_q_bias, base).f32Values(),
            .k_weight = findBinding(config, bindings, .audio_layer_attention_k_weight, base).matrix(),
            .k_bias = findBinding(config, bindings, .audio_layer_attention_k_bias, base).f32Values(),
            .v_weight = findBinding(config, bindings, .audio_layer_attention_v_weight, base).matrix(),
            .v_bias = findBinding(config, bindings, .audio_layer_attention_v_bias, base).f32Values(),
            .out_weight = findBinding(config, bindings, .audio_layer_attention_out_weight, base).matrix(),
            .out_bias = findBinding(config, bindings, .audio_layer_attention_out_bias, base).f32Values(),
            .attention_norm_weight = findBinding(config, bindings, .audio_layer_attention_norm_weight, base).f32Values(),
            .attention_norm_bias = findBinding(config, bindings, .audio_layer_attention_norm_bias, base).f32Values(),
            .ffn_in_weight = findBinding(config, bindings, .audio_layer_ffn_in_weight, base).matrix(),
            .ffn_in_bias = findBinding(config, bindings, .audio_layer_ffn_in_bias, base).f32Values(),
            .ffn_out_weight = findBinding(config, bindings, .audio_layer_ffn_out_weight, base).matrix(),
            .ffn_out_bias = findBinding(config, bindings, .audio_layer_ffn_out_bias, base).f32Values(),
            .final_norm_weight = findBinding(config, bindings, .audio_layer_final_norm_weight, base).f32Values(),
            .final_norm_bias = findBinding(config, bindings, .audio_layer_final_norm_bias, base).f32Values(),
        };
    }
    return layers;
}

fn loadDecoderLayers(
    arena: std.mem.Allocator,
    config: *const model_config.Config,
    bindings: []const Binding,
) Error![]const DecoderLayer {
    const layers = arena.alloc(DecoderLayer, config.text_layers) catch return Error.OutOfMemory;
    for (layers, 0..) |*layer, index| {
        const base: u16 = @intCast(layout.decoder_layer_base + index);
        layer.* = .{
            .q_weight = findBinding(config, bindings, .decoder_layer_attention_q_weight, base).matrix(),
            .k_weight = findBinding(config, bindings, .decoder_layer_attention_k_weight, base).matrix(),
            .v_weight = findBinding(config, bindings, .decoder_layer_attention_v_weight, base).matrix(),
            .out_weight = findBinding(config, bindings, .decoder_layer_attention_out_weight, base).matrix(),
            .q_norm_weight = findBinding(config, bindings, .decoder_layer_attention_q_norm_weight, base).f32Values(),
            .k_norm_weight = findBinding(config, bindings, .decoder_layer_attention_k_norm_weight, base).f32Values(),
            .attention_norm_weight = findBinding(config, bindings, .decoder_layer_attention_norm_weight, base).f32Values(),
            .ffn_norm_weight = findBinding(config, bindings, .decoder_layer_ffn_norm_weight, base).f32Values(),
            .gate_weight = findBinding(config, bindings, .decoder_layer_ffn_gate_weight, base).matrix(),
            .up_weight = findBinding(config, bindings, .decoder_layer_ffn_up_weight, base).matrix(),
            .down_weight = findBinding(config, bindings, .decoder_layer_ffn_down_weight, base).matrix(),
        };
    }
    return layers;
}

fn validateCapacity(config: *const model_config.Config) Error!void {
    if (config.max_positions > position_limit) return Error.CapacityExceeded;
    if (config.max_positions == 0) return Error.CapacityExceeded;
}

fn allocateScratch(
    arena: std.mem.Allocator,
    config: *const model_config.Config,
    cache_format: model_config.CacheFormat,
) Error!Scratch {
    const d_model = config.audio_d_model;
    const downsample = config.audio_downsample_hidden_size;
    const ffn_dim = config.audio_ffn_dim;
    // The f32 cache is counted in elements and the quantized cache in plane bytes, and each pair is
    // zero-length when the other format is in use.
    const cache_elements = @as(usize, config.text_layers) * config.max_positions *
        config.textKeyValueElements();
    const cache_plane_bytes = @as(u64, config.text_layers) * quant.planeBytes(
        cache_format.toDtype(),
        config.max_positions,
        config.textKeyValueElements(),
    );
    const hidden = config.text_hidden_size;
    const max_steps = config.packedStepCount(mel.capacity_frames);
    const window_steps = (config.audio_n_window_infer / config.audioChunkFrames()) *
        config.audioChunkSteps();
    const chunk_frames = config.audioChunkFrames();
    const mid_height = (config.mel_bins - 1) / 2 + 1;
    const mid_width = (chunk_frames - 1) / 2 + 1;
    const frequency_bins = config.audioFrequencyBinsAfterConvolutions();
    const half_head = config.text_head_dim / 2;
    const max_columns = @max(
        config.audioConvOutInputFeatures(),
        @max(config.text_ffn_dim, config.text_hidden_size),
    );

    return .{
        .row = try allocate(arena, f32, max_columns),
        .conv_flat = try allocate(arena, f32, config.audioConvOutInputFeatures()),
        .mel_chunk = try allocate(arena, f32, config.mel_bins * chunk_frames),
        .conv1 = try allocate(arena, f32, downsample * mid_height * mid_width),
        // Each convolution halves its input extent, and the kernel refuses a
        // buffer that is not exactly its output size: conv2's buffer is sized for
        // the second stage, not the first.
        .conv2 = try allocate(
            arena,
            f32,
            downsample * ((mid_height - 1) / 2 + 1) * ((mid_width - 1) / 2 + 1),
        ),
        .conv3 = try allocate(arena, f32, downsample * frequency_bins * config.audioChunkSteps()),
        .encoder = try allocate(arena, f32, @as(usize, max_steps) * d_model),
        .encoder_qkv = try allocate(arena, f32, @as(usize, max_steps) * 3 * d_model),
        .encoder_normed = try allocate(arena, f32, @as(usize, max_steps) * d_model),
        .encoder_block = try allocate(arena, f32, @as(usize, max_steps) * d_model),
        .encoder_ffn = try allocate(arena, f32, @as(usize, max_steps) * ffn_dim),
        .window_scores = try allocate(arena, f32, window_steps),
        .projected = try allocate(arena, f32, @as(usize, max_steps) * hidden),
        .hidden = try allocate(arena, f32, hidden),
        .normed = try allocate(arena, f32, hidden),
        .query = try allocate(arena, f32, config.textQueryElements()),
        .key = try allocate(arena, f32, config.textKeyValueElements()),
        .value = try allocate(arena, f32, config.textKeyValueElements()),
        .attention_out = try allocate(arena, f32, config.textQueryElements()),
        .gate = try allocate(arena, f32, config.text_ffn_dim),
        .up = try allocate(arena, f32, config.text_ffn_dim),
        .position_scores = try allocate(arena, f32, config.max_positions),
        .logits = try allocate(arena, f32, config.vocab_size),
        .rope_cos = try allocate(arena, f32, @as(usize, config.max_positions) * half_head),
        .rope_sin = try allocate(arena, f32, @as(usize, config.max_positions) * half_head),
        // Exactly one cache pair is allocated: the other stays empty, so a reader cannot mistake
        // a format for "unset" and the byte budget is the one `cacheBytes` reports.
        .cache_keys = if (cache_format == .f32) try allocate(arena, f32, cache_elements) else &.{},
        .cache_values = if (cache_format == .f32) try allocate(arena, f32, cache_elements) else &.{},
        .cache_keys_q8 = if (cache_format == .f32)
            &.{}
        else
            try allocate(arena, u8, cache_plane_bytes),
        .cache_values_q8 = if (cache_format == .f32)
            &.{}
        else
            try allocate(arena, u8, cache_plane_bytes),
    };
}

fn allocate(arena: std.mem.Allocator, comptime T: type, count: anytype) Error![]T {
    return arena.alloc(T, @intCast(count)) catch Error.OutOfMemory;
}

test "size helper rejects a cache beyond the hard position limit" {
    var config = testConfig();
    config.max_positions = position_limit + 1;
    try std.testing.expectError(Error.CapacityExceeded, validateCapacity(&config));
    config.max_positions = 0;
    try std.testing.expectError(Error.CapacityExceeded, validateCapacity(&config));
    config.max_positions = 512;
    try validateCapacity(&config);
}

test "scratch bytes account for every populated buffer, without the cache" {
    var config = testConfig();
    // Chosen so the cache the configuration allows is exactly the two cache
    // buffers this test fills: 2 * layers * heads * head_dim * positions * 4.
    config.text_layers = 1;
    config.text_key_value_heads = 1;
    config.text_head_dim = 2;
    config.max_positions = 16;
    var model: Model = undefined;
    model.config = config;
    model.scratch = std.mem.zeroes(Scratch);
    try std.testing.expectEqual(@as(u64, 0), model.scratchBytes());

    var first: [4]f32 = undefined;
    var second: [7]f32 = undefined;
    model.scratch.logits = &first;
    try std.testing.expectEqual(@as(u64, 16), model.scratchBytes());
    model.scratch.hidden = &second;
    try std.testing.expectEqual(@as(u64, 44), model.scratchBytes());

    // The cache lives in the scratch too, and `cacheBytes` reports it on its
    // own: filling both cache buffers must not change `scratchBytes`, or a
    // caller budgeting by the sum would count the cache twice.
    var cache: [32]f32 = undefined;
    model.scratch.cache_keys = &cache;
    model.scratch.cache_values = &cache;
    // A bare `Model` is all `undefined`, and `cacheBytes` reads the format: leaving it unset is a
    // programmer error the format's `unreachable` arm reports rather than a silent answer.
    model.cache_format = .f32;
    try std.testing.expectEqual(@as(u64, 44), model.scratchBytes());
    try std.testing.expectEqual(@as(u64, 256), model.cacheBytes());
}

test "sinusoidal positions match the reference construction" {
    var row = std.mem.zeroes([64]f32);
    addSinusoidalPosition(&row, 0, 64);
    // Position zero gives sin(0) = 0 and cos(0) = 1 in both halves.
    for (row[0..32]) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
    for (row[32..]) |value| try std.testing.expectApproxEqAbs(@as(f32, 1.0), value, 1e-6);

    // At position one the first pair rotates by exactly one radian, because
    // its inverse timescale is exp(0) = 1. The row is cleared first so these
    // checks read position one's embedding alone.
    @memset(&row, 0.0);
    addSinusoidalPosition(&row, 1, 64);
    for (row) |value| {
        try std.testing.expect(@abs(value) <= 1.0);
        try std.testing.expect(std.math.isFinite(value));
    }
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1.0)), row[0], 1e-6);
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1.0)), row[32], 1e-6);
    // Later pairs have smaller inverse timescales, so they rotate less: the
    // sine half shrinks toward zero and the cosine half grows toward one.
    try std.testing.expect(@abs(row[1]) > @abs(row[30]));
    try std.testing.expect(@abs(row[62]) > @abs(row[32]));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row[62], 1e-4);

    // The embedding is a term *added* to the projected convolution output, the
    // reference's `conv_out += positional[...]`, not a value written over it.
    // Both rows above start at zero, so an implementation that assigned would
    // pass every check in them while discarding the audio entirely.
    var seeded: [64]f32 = undefined;
    @memset(&seeded, 0.25);
    addSinusoidalPosition(&seeded, 0, 64);
    for (seeded[0..32]) |value| try std.testing.expectApproxEqAbs(@as(f32, 0.25), value, 1e-6);
    for (seeded[32..]) |value| try std.testing.expectApproxEqAbs(@as(f32, 1.25), value, 1e-6);
}

test "binding accessors interpret the two planes correctly" {
    const shape = try tensor.Shape.matrix(64, 64);
    const plane = try quant.planeLayout(.q4, 64, 64, 0, quant.tensor_alignment_bytes);
    const storage = try std.testing.allocator.alloc(u8, @intCast(plane.total_len_bytes));
    defer std.testing.allocator.free(storage);
    @memset(storage, 0);
    storage[@intCast(plane.data_offset_bytes)] = 0x0F;

    const binding = Binding{
        .kind = .audio_conv1_weight,
        .layer = 0,
        .shape = shape,
        .format = .q4,
        .bytes = storage,
    };
    const matrix = binding.matrix();
    try std.testing.expectEqual(@as(u32, 64), matrix.rows(64));
    switch (matrix) {
        .quantized => |planes| {
            try std.testing.expectEqual(@as(u8, 0x0F), planes.data[0]);
        },
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqual(@as(u32, 64 * 64), binding.elementCount());
}

fn testConfig() model_config.Config {
    return .{
        .magic = model_config.magic_bytes,
        .format_version = model_config.format_version,
        .architecture = @backingInt(model_config.Architecture.qwen3_asr),
        .audio_d_model = 896,
        .audio_layers = 2,
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
        .text_layers = 2,
        .text_attention_heads = 16,
        .text_key_value_heads = 8,
        .text_head_dim = 128,
        .text_ffn_dim = 3072,
        .vocab_size = 151936,
        .text_rms_norm_eps = 1e-6,
        .rope_theta = 1000000.0,
        .max_positions = 512,
        .max_decode_tokens = 64,
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
}
