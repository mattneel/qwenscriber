//! The tensor inventory of a Qwen3-ASR model, derived from its configuration.
//!
//! This is the contract between the converter and the runtime. The converter
//! walks the iterator, finds each tensor in the checkpoint, validates it against
//! the shape declared here, and writes it into a shard. The runtime walks the
//! same iterator, looks each tensor up in the shards it was given, and binds it.
//! A missing or mis-shaped tensor therefore fails on both sides with the same
//! expected shape in hand, and neither side can drift from the other silently.
//!
//! Every shape is a function of `model_config.Config`. Both released
//! checkpoints -- 0.6B and 1.7B -- carry the same *set* of tensors; only the
//! dimensions differ, so nothing here belongs to one variant.
//!
//! Layer numbering follows `container`: zero for tensors that belong to no
//! layer, `1..=text_layers` for decoder layers, and `audio_layer_base + i` for
//! audio tower layers.

const std = @import("std");
const container = @import("../container.zig");
const dtype = @import("../dtype.zig");
const model_config = @import("../model_config.zig");
const tensor = @import("../tensor.zig");

const TensorKind = container.TensorKind;

/// Decoder layer numbering starts at one; layer zero is reserved for tensors
/// that belong to no layer.
pub const decoder_layer_base: u16 = 1;

pub const fixed_tensor_count: u32 = 9;
pub const projector_tensor_count: u32 = 4;
pub const decoder_fixed_tensor_count: u32 = 3;
pub const decoder_layer_tensor_count: u32 = 11;
pub const audio_layer_tensor_count: u32 = 16;

/// How a tensor should be stored. The converter applies this policy; the runtime
/// validates whatever it reads against the same shapes.
pub const Storage = enum {
    /// Normalization weights and every bias. Kept in f32: they are tiny, and
    /// they scale or offset whole rows, so their precision matters more than
    /// their size.
    bias_f32,
    /// Element weights that quantization cannot reach, because the shape is not
    /// a matrix (the 4-D convolutions).
    element_f16,
    /// Large matrix weights: quantized, or f16 when quantization is disabled.
    matrix,

    pub fn format(self: Storage, quantization: dtype.Format) dtype.Format {
        return switch (self) {
            .bias_f32 => .f32,
            .element_f16 => .f16,
            .matrix => if (quantization.isQuantized()) quantization else .f16,
        };
    }
};

/// One tensor the model needs.
pub const Required = struct {
    kind: TensorKind,
    layer: u16,
    shape: tensor.Shape,
    storage: Storage,

    pub fn format(self: Required, quantization: dtype.Format) dtype.Format {
        return self.storage.format(quantization);
    }

    pub fn key(self: Required) u32 {
        return container.sortKey(self.layer, @intFromEnum(self.kind));
    }
};

/// Walks the model's tensors in `(layer, kind)` order: the order the converter
/// writes, and the order a shard index is sorted by.
pub const Iterator = struct {
    config: *const model_config.Config,
    position: u32 = 0,

    pub fn init(config: *const model_config.Config) Iterator {
        return .{ .config = config };
    }

    /// Number of tensors the model needs, excluding a tied output projection.
    pub fn count(config: *const model_config.Config) u32 {
        return fixed_tensor_count + projector_tensor_count + decoder_fixed_tensor_count +
            config.text_layers * decoder_layer_tensor_count +
            config.audio_layers * audio_layer_tensor_count;
    }

    pub fn next(self: *Iterator) ?Required {
        if (self.position >= Iterator.count(self.config)) return null;
        const required = at(self.config, self.position);
        self.position += 1;
        return required;
    }
};

/// The tensor at a flat position in `(layer, kind)` order. Returns null only for
/// a position past the end.
pub fn at(config: *const model_config.Config, position: u32) ?Required {
    var remaining = position;

    if (remaining < fixed_tensor_count) {
        return fixedEntry(config, remaining);
    }
    remaining -= fixed_tensor_count;

    if (remaining < projector_tensor_count) {
        return projectorEntry(config, remaining);
    }
    remaining -= projector_tensor_count;

    if (remaining < decoder_fixed_tensor_count) {
        return decoderFixedEntry(config, remaining);
    }
    remaining -= decoder_fixed_tensor_count;

    const decoder_total = config.text_layers * decoder_layer_tensor_count;
    if (remaining < decoder_total) {
        return decoderLayerEntry(
            config,
            remaining / decoder_layer_tensor_count,
            @intCast(remaining % decoder_layer_tensor_count),
        );
    }
    remaining -= decoder_total;

    const audio_total = config.audio_layers * audio_layer_tensor_count;
    if (remaining < audio_total) {
        return audioLayerEntry(
            config,
            remaining / audio_layer_tensor_count,
            @intCast(remaining % audio_layer_tensor_count),
        );
    }
    return null;
}

/// Looks up a single tensor description, for callers that need one specific
/// tensor rather than the whole inventory.
pub fn find(config: *const model_config.Config, kind: TensorKind, layer: u16) ?Required {
    var iterator = Iterator.init(config);
    while (iterator.next()) |required| {
        if (required.kind == kind and required.layer == layer) return required;
    }
    return null;
}

fn fixedEntry(config: *const model_config.Config, index: u32) ?Required {
    const downsample = config.audio_downsample_hidden_size;
    const d_model = config.audio_d_model;
    switch (index) {
        0 => return entry(
            .audio_conv1_weight,
            0,
            tryShape(tensor.Shape.conv2d(downsample, 1, 3, 3)),
            .element_f16,
        ),
        1 => return vector(.audio_conv1_bias, 0, downsample),
        2 => return entry(
            .audio_conv2_weight,
            0,
            tryShape(tensor.Shape.conv2d(downsample, downsample, 3, 3)),
            .element_f16,
        ),
        3 => return vector(.audio_conv2_bias, 0, downsample),
        4 => return entry(
            .audio_conv3_weight,
            0,
            tryShape(tensor.Shape.conv2d(downsample, downsample, 3, 3)),
            .element_f16,
        ),
        5 => return vector(.audio_conv3_bias, 0, downsample),
        6 => return entry(
            .audio_conv_out_weight,
            0,
            tryShape(tensor.Shape.matrix(d_model, config.audioConvOutInputFeatures())),
            .matrix,
        ),
        7 => return vector(.audio_final_norm_weight, 0, d_model),
        8 => return vector(.audio_final_norm_bias, 0, d_model),
        else => return null,
    }
}

fn projectorEntry(config: *const model_config.Config, index: u32) ?Required {
    const d_model = config.audio_d_model;
    switch (index) {
        0 => return entry(
            .projector_in_weight,
            0,
            tryShape(tensor.Shape.matrix(d_model, d_model)),
            .matrix,
        ),
        1 => return vector(.projector_in_bias, 0, d_model),
        2 => return entry(
            .projector_out_weight,
            0,
            tryShape(tensor.Shape.matrix(config.audio_output_dim, d_model)),
            .matrix,
        ),
        3 => return vector(.projector_out_bias, 0, config.audio_output_dim),
        else => return null,
    }
}

fn decoderFixedEntry(config: *const model_config.Config, index: u32) ?Required {
    const hidden = config.text_hidden_size;
    switch (index) {
        0 => return entry(
            .decoder_embed_tokens_weight,
            0,
            tryShape(tensor.Shape.matrix(config.vocab_size, hidden)),
            .matrix,
        ),
        1 => return vector(.decoder_final_norm_weight, 0, hidden),
        2 => return entry(
            .decoder_output_weight,
            0,
            tryShape(tensor.Shape.matrix(config.vocab_size, hidden)),
            .matrix,
        ),
        else => return null,
    }
}

fn audioLayerEntry(config: *const model_config.Config, layer_index: u32, within: u8) ?Required {
    const layer: u16 = @intCast(container.audio_layer_base + layer_index);
    const d_model = config.audio_d_model;
    const ffn = config.audio_ffn_dim;
    const square = tryShape(tensor.Shape.matrix(d_model, d_model));
    switch (within) {
        0 => return entry(.audio_layer_attention_q_weight, layer, square, .matrix),
        1 => return vector(.audio_layer_attention_q_bias, layer, d_model),
        2 => return entry(.audio_layer_attention_k_weight, layer, square, .matrix),
        3 => return vector(.audio_layer_attention_k_bias, layer, d_model),
        4 => return entry(.audio_layer_attention_v_weight, layer, square, .matrix),
        5 => return vector(.audio_layer_attention_v_bias, layer, d_model),
        6 => return entry(.audio_layer_attention_out_weight, layer, square, .matrix),
        7 => return vector(.audio_layer_attention_out_bias, layer, d_model),
        8 => return vector(.audio_layer_attention_norm_weight, layer, d_model),
        9 => return vector(.audio_layer_attention_norm_bias, layer, d_model),
        10 => return entry(
            .audio_layer_ffn_in_weight,
            layer,
            tryShape(tensor.Shape.matrix(ffn, d_model)),
            .matrix,
        ),
        11 => return vector(.audio_layer_ffn_in_bias, layer, ffn),
        12 => return entry(
            .audio_layer_ffn_out_weight,
            layer,
            tryShape(tensor.Shape.matrix(d_model, ffn)),
            .matrix,
        ),
        13 => return vector(.audio_layer_ffn_out_bias, layer, d_model),
        14 => return vector(.audio_layer_final_norm_weight, layer, d_model),
        15 => return vector(.audio_layer_final_norm_bias, layer, d_model),
        else => return null,
    }
}

fn decoderLayerEntry(config: *const model_config.Config, layer_index: u32, within: u8) ?Required {
    const layer: u16 = @intCast(decoder_layer_base + layer_index);
    const hidden = config.text_hidden_size;
    const query_width = config.textQueryElements();
    const kv_width = config.textKeyValueElements();
    const ffn = config.text_ffn_dim;
    switch (within) {
        0 => return entry(
            .decoder_layer_attention_q_weight,
            layer,
            tryShape(tensor.Shape.matrix(query_width, hidden)),
            .matrix,
        ),
        1 => return entry(
            .decoder_layer_attention_k_weight,
            layer,
            tryShape(tensor.Shape.matrix(kv_width, hidden)),
            .matrix,
        ),
        2 => return entry(
            .decoder_layer_attention_v_weight,
            layer,
            tryShape(tensor.Shape.matrix(kv_width, hidden)),
            .matrix,
        ),
        3 => return entry(
            .decoder_layer_attention_out_weight,
            layer,
            tryShape(tensor.Shape.matrix(hidden, query_width)),
            .matrix,
        ),
        4 => return vector(.decoder_layer_attention_q_norm_weight, layer, config.text_head_dim),
        5 => return vector(.decoder_layer_attention_k_norm_weight, layer, config.text_head_dim),
        6 => return vector(.decoder_layer_attention_norm_weight, layer, hidden),
        7 => return vector(.decoder_layer_ffn_norm_weight, layer, hidden),
        8 => return entry(
            .decoder_layer_ffn_gate_weight,
            layer,
            tryShape(tensor.Shape.matrix(ffn, hidden)),
            .matrix,
        ),
        9 => return entry(
            .decoder_layer_ffn_up_weight,
            layer,
            tryShape(tensor.Shape.matrix(ffn, hidden)),
            .matrix,
        ),
        10 => return entry(
            .decoder_layer_ffn_down_weight,
            layer,
            tryShape(tensor.Shape.matrix(hidden, ffn)),
            .matrix,
        ),
        else => return null,
    }
}

/// Shapes here are constructed from validated configuration values, so a
/// failure would mean `validate` let something through. Returning null turns
/// that into a missing tensor rather than a crash.
fn tryShape(shape: tensor.Error!tensor.Shape) tensor.Shape {
    return shape catch unreachable;
}

fn entry(kind: TensorKind, layer: u16, shape: tensor.Shape, storage: Storage) Required {
    return .{ .kind = kind, .layer = layer, .shape = shape, .storage = storage };
}

fn vector(kind: TensorKind, layer: u16, count: u32) ?Required {
    const shape = tensor.Shape.vector(count) catch return null;
    return .{ .kind = kind, .layer = layer, .shape = shape, .storage = .bias_f32 };
}

test "the iterator emits every tensor exactly once, in sorted order" {
    const config = testConfig();
    var iterator = Iterator.init(&config);
    var seen: u32 = 0;
    var previous_key: u32 = 0;
    while (iterator.next()) |required| {
        if (seen > 0) try std.testing.expect(required.key() > previous_key);
        previous_key = required.key();
        seen += 1;
    }
    try std.testing.expectEqual(Iterator.count(&config), seen);
    // The released 0.6B checkpoint contains exactly 612 tensors (plus a
    // metadata entry): 9 fixed, 4 projector, 3 decoder fixed, 28 decoder layers
    // of 11, and 18 audio layers of 16. The inventory matches it tensor for
    // tensor, which is the cheapest possible check that no tensor was forgotten.
    try std.testing.expectEqual(@as(u32, 612), seen);
    try std.testing.expectEqual(@as(u32, 9 + 4 + 3 + 28 * 11 + 18 * 16), seen);
}

test "a null description is impossible for a valid configuration" {
    const config = testConfig();
    var position: u32 = 0;
    while (position < Iterator.count(&config)) : (position += 1) {
        try std.testing.expect(at(&config, position) != null);
    }
    try std.testing.expect(at(&config, Iterator.count(&config)) == null);
    try std.testing.expect(at(&config, Iterator.count(&config) + 100) == null);
}

test "shapes follow the configuration rather than a fixed variant" {
    const small = testConfig();
    var large = testConfig();
    large.audio_d_model = 1024;
    large.audio_layers = 24;
    large.audio_attention_heads = 16;
    large.audio_ffn_dim = 4096;
    large.audio_output_dim = 2048;
    large.text_hidden_size = 2048;
    large.text_ffn_dim = 6144;
    try large.validate();

    try std.testing.expect(Iterator.count(&small) < Iterator.count(&large));
    // 0.6B: 18 audio layers. 1.7B: 24.
    try std.testing.expectEqual(
        Iterator.count(&small) + 6 * audio_layer_tensor_count,
        Iterator.count(&large),
    );

    // The convolution output projection stays 7680 wide in both variants even
    // though d_model differs, because it is derived from the mel geometry.
    for ([_]model_config.Config{ small, large }) |config| {
        const conv_out = find(&config, .audio_conv_out_weight, 0).?;
        try std.testing.expectEqual(@as(u32, 7680), try conv_out.shape.cols());
        try std.testing.expectEqual(config.audio_d_model, try conv_out.shape.rows());
    }
}

test "the embedding and the output projection share a shape" {
    const config = testConfig();
    const embedding = find(&config, .decoder_embed_tokens_weight, 0).?;
    const output = find(&config, .decoder_output_weight, 0).?;
    try std.testing.expect(embedding.shape.eql(&output.shape));
    try std.testing.expectEqual(Storage.matrix, embedding.storage);
    // The vocabulary is a whole number of quantization groups.
    try std.testing.expectEqual(@as(u32, 0), embedding.shape.dims[0] % 64);
}

test "storage policy picks the format the runtime validates" {
    try std.testing.expectEqual(dtype.Format.f32, Storage.bias_f32.format(.q4));
    try std.testing.expectEqual(dtype.Format.f16, Storage.element_f16.format(.q4));
    try std.testing.expectEqual(dtype.Format.q4, Storage.matrix.format(.q4));
    try std.testing.expectEqual(dtype.Format.q5, Storage.matrix.format(.q5));
    try std.testing.expectEqual(dtype.Format.f16, Storage.matrix.format(.f16));
}

test "convolutions keep their four-dimensional shape and cannot be quantized" {
    const config = testConfig();
    const conv = find(&config, .audio_conv2_weight, 0).?;
    try std.testing.expectEqual(@as(u8, 4), conv.shape.rank);
    try std.testing.expectEqual(Storage.element_f16, conv.storage);
    try std.testing.expectError(tensor.Error.NotMatrix, conv.shape.validateForFormat(.q4));
}

test "every audio layer exposes thirteen tensors and every decoder layer eleven" {
    const config = testConfig();
    var audio_counts = std.mem.zeroes([64]u32);
    var decoder_counts = std.mem.zeroes([64]u32);
    var iterator = Iterator.init(&config);
    while (iterator.next()) |required| {
        if (required.layer >= container.audio_layer_base) {
            audio_counts[required.layer - container.audio_layer_base] += 1;
        } else if (required.layer >= decoder_layer_base) {
            decoder_counts[required.layer - decoder_layer_base] += 1;
        }
    }
    for (audio_counts[0..config.audio_layers]) |count| {
        try std.testing.expectEqual(audio_layer_tensor_count, count);
    }
    for (decoder_counts[0..config.text_layers]) |count| {
        try std.testing.expectEqual(decoder_layer_tensor_count, count);
    }
}

fn testConfig() model_config.Config {
    return .{
        .magic = model_config.magic_bytes,
        .format_version = model_config.format_version,
        .architecture = @intFromEnum(model_config.Architecture.qwen3_asr),
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
}
