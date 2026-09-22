//! Mapping from an official checkpoint's tensor names to the runtime's
//! `(TensorKind, layer)` identity.
//!
//! Two spellings of the same model are in circulation, and both must convert:
//!
//!     Qwen/Qwen3-ASR-0.6B      vLLM-style: `thinker.audio_tower.*`,
//!                              `thinker.model.*`, `thinker.lm_head.weight`,
//!                              the projector inside the audio tower as
//!                              `thinker.audio_tower.proj1/proj2.*`
//!     Qwen/Qwen3-ASR-0.6B-hf   transformers-native: `model.audio_tower.*`,
//!                              `model.language_model.*`, and the projector as
//!                              `model.multi_modal_projector.linear_1/2.*`,
//!                              with no `lm_head` at all (weight-tied)
//!
//! They name the same tensors with the same shapes, so one table serves both:
//! the area prefix decides which part of the model a name belongs to, and the
//! suffix decides which kind.
//!
//! Layer numbering follows `container`: decoder layer `i` becomes `1 + i` and
//! audio layer `i` becomes `audio_layer_base + i`, so a name is mapped once and
//! the result is the key the runtime looks up. Everything else is layer zero.
//!
//! `officialName` is the inverse for diagnostics: when the converter reports
//! that a tensor is missing, the operator needs the checkpoint's own name for
//! it, not the runtime's.

const std = @import("std");
const qwenscriber = @import("qwenscriber");

const assert = std.debug.assert;
const container = qwenscriber.container;
const layout = qwenscriber.qwen3_asr.layout;
const model_config = qwenscriber.model_config;
const safetensors = @import("safetensors.zig");

const TensorKind = container.TensorKind;

pub const Error = error{
    /// The kind has no place in the released naming schemes.
    UnknownKind,
    /// The caller's buffer cannot hold the name.
    NameBufferTooSmall,
} || std.mem.Allocator.Error;

/// Where a checkpoint tensor belongs before its suffix is read.
const Area = enum { audio, decoder, projector, output };

/// A checkpoint tensor's place in the runtime's inventory.
pub const Mapping = struct {
    kind: TensorKind,
    layer: u16,

    pub fn key(self: Mapping) u32 {
        return container.sortKey(self.layer, @backingInt(self.kind));
    }
};

/// The `(kind, layer)` a checkpoint tensor name denotes, or null when the name
/// is not part of the released model.
pub fn mapName(name: []const u8) ?Mapping {
    const split = splitArea(name) orelse return null;
    return mapWithin(split.area, split.rest);
}

/// Strips the export's area prefix. Order matters: the longer prefixes are
/// tested first, so `model.language_model.` never falls through to the
/// decoder's own `model.` names.
fn splitArea(name: []const u8) ?Split {
    const stripped = if (std.mem.startsWith(u8, name, "thinker.")) name["thinker.".len..] else name;
    if (std.mem.eql(u8, stripped, "lm_head.weight")) return .{ .area = .output, .rest = "weight" };

    const prefixes = [_]struct { prefix: []const u8, area: Area }{
        .{ .prefix = "model.audio_tower.", .area = .audio },
        .{ .prefix = "model.language_model.", .area = .decoder },
        .{ .prefix = "model.multi_modal_projector.", .area = .projector },
        .{ .prefix = "audio_tower.", .area = .audio },
        .{ .prefix = "model.", .area = .decoder },
    };
    for (prefixes) |entry| {
        if (std.mem.startsWith(u8, stripped, entry.prefix)) {
            return .{ .area = entry.area, .rest = stripped[entry.prefix.len..] };
        }
    }
    return null;
}

const Split = struct { area: Area, rest: []const u8 };

fn mapWithin(area: Area, rest: []const u8) ?Mapping {
    switch (area) {
        .audio => {
            if (findSuffix(&audio_fixed_names, rest)) |kind| return .{ .kind = kind, .layer = 0 };
            const split = splitLayer(rest) orelse return null;
            const kind = findSuffix(&audio_layer_names, split.rest) orelse return null;
            const layer = std.math.add(
                u16,
                container.audio_layer_base,
                split.index,
            ) catch return null;
            return .{ .kind = kind, .layer = layer };
        },
        .decoder => {
            if (findSuffix(&decoder_fixed_names, rest)) |kind| return .{ .kind = kind, .layer = 0 };
            const split = splitLayer(rest) orelse return null;
            const kind = findSuffix(&decoder_layer_names, split.rest) orelse return null;
            const layer = std.math.add(
                u16,
                layout.decoder_layer_base,
                split.index,
            ) catch return null;
            return .{ .kind = kind, .layer = layer };
        },
        .projector => {
            const kind = findSuffix(&projector_names, rest) orelse return null;
            return .{ .kind = kind, .layer = 0 };
        },
        .output => {
            if (!std.mem.eql(u8, rest, "weight")) return null;
            return .{ .kind = .decoder_output_weight, .layer = 0 };
        },
    }
}

const NameEntry = struct { suffix: []const u8, kind: TensorKind };

const LayerSplit = struct { index: u16, rest: []const u8 };

/// Splits `layers.<index>.<suffix>`. The index is bounded to four digits so a
/// name that merely looks like a layer index cannot overflow the parse, and so
/// the scan stays bounded.
fn splitLayer(text: []const u8) ?LayerSplit {
    const prefix = "layers.";
    if (!std.mem.startsWith(u8, text, prefix)) return null;
    const tail = text[prefix.len..];
    const search = tail[0..@min(tail.len, 5)];
    const digits_len = std.mem.indexOfScalar(u8, search, '.') orelse return null;
    if (digits_len == 0) return null;
    const index = std.fmt.parseInt(u16, tail[0..digits_len], 10) catch return null;
    assert(digits_len < tail.len);
    return .{ .index = index, .rest = tail[digits_len + 1 ..] };
}

fn findSuffix(table: []const NameEntry, suffix: []const u8) ?TensorKind {
    for (table) |entry| {
        if (std.mem.eql(u8, entry.suffix, suffix)) return entry.kind;
    }
    return null;
}

fn suffixOfKind(table: []const NameEntry, kind: TensorKind) ?[]const u8 {
    for (table) |entry| {
        if (entry.kind == kind) return entry.suffix;
    }
    return null;
}

/// Audio tower tensors that carry no layer index. `conv_out` has no bias: the
/// released checkpoints build it without one.
const audio_fixed_names = [_]NameEntry{
    .{ .suffix = "conv2d1.bias", .kind = .audio_conv1_bias },
    .{ .suffix = "conv2d1.weight", .kind = .audio_conv1_weight },
    .{ .suffix = "conv2d2.bias", .kind = .audio_conv2_bias },
    .{ .suffix = "conv2d2.weight", .kind = .audio_conv2_weight },
    .{ .suffix = "conv2d3.bias", .kind = .audio_conv3_bias },
    .{ .suffix = "conv2d3.weight", .kind = .audio_conv3_weight },
    .{ .suffix = "conv_out.weight", .kind = .audio_conv_out_weight },
    .{ .suffix = "ln_post.bias", .kind = .audio_final_norm_bias },
    .{ .suffix = "ln_post.weight", .kind = .audio_final_norm_weight },
    // The vLLM export keeps the two-layer projector inside the audio tower.
    .{ .suffix = "proj1.bias", .kind = .projector_in_bias },
    .{ .suffix = "proj1.weight", .kind = .projector_in_weight },
    .{ .suffix = "proj2.bias", .kind = .projector_out_bias },
    .{ .suffix = "proj2.weight", .kind = .projector_out_weight },
};

const audio_layer_names = [_]NameEntry{
    .{ .suffix = "fc1.bias", .kind = .audio_layer_ffn_in_bias },
    .{ .suffix = "fc1.weight", .kind = .audio_layer_ffn_in_weight },
    .{ .suffix = "fc2.bias", .kind = .audio_layer_ffn_out_bias },
    .{ .suffix = "fc2.weight", .kind = .audio_layer_ffn_out_weight },
    .{ .suffix = "final_layer_norm.bias", .kind = .audio_layer_final_norm_bias },
    .{ .suffix = "final_layer_norm.weight", .kind = .audio_layer_final_norm_weight },
    .{ .suffix = "self_attn.k_proj.bias", .kind = .audio_layer_attention_k_bias },
    .{ .suffix = "self_attn.k_proj.weight", .kind = .audio_layer_attention_k_weight },
    .{ .suffix = "self_attn.out_proj.bias", .kind = .audio_layer_attention_out_bias },
    .{ .suffix = "self_attn.out_proj.weight", .kind = .audio_layer_attention_out_weight },
    .{ .suffix = "self_attn.q_proj.bias", .kind = .audio_layer_attention_q_bias },
    .{ .suffix = "self_attn.q_proj.weight", .kind = .audio_layer_attention_q_weight },
    .{ .suffix = "self_attn.v_proj.bias", .kind = .audio_layer_attention_v_bias },
    .{ .suffix = "self_attn.v_proj.weight", .kind = .audio_layer_attention_v_weight },
    .{ .suffix = "self_attn_layer_norm.bias", .kind = .audio_layer_attention_norm_bias },
    .{ .suffix = "self_attn_layer_norm.weight", .kind = .audio_layer_attention_norm_weight },
};

/// The transformers-native export spells the projector as two linear layers
/// outside the audio tower.
const projector_names = [_]NameEntry{
    .{ .suffix = "linear_1.bias", .kind = .projector_in_bias },
    .{ .suffix = "linear_1.weight", .kind = .projector_in_weight },
    .{ .suffix = "linear_2.bias", .kind = .projector_out_bias },
    .{ .suffix = "linear_2.weight", .kind = .projector_out_weight },
};

const decoder_fixed_names = [_]NameEntry{
    .{ .suffix = "embed_tokens.weight", .kind = .decoder_embed_tokens_weight },
    .{ .suffix = "norm.weight", .kind = .decoder_final_norm_weight },
};

/// The decoder carries no attention biases: `attention_bias` is false in both
/// released configurations, and the runtime has no kinds for them.
const decoder_layer_names = [_]NameEntry{
    .{ .suffix = "input_layernorm.weight", .kind = .decoder_layer_attention_norm_weight },
    .{ .suffix = "mlp.down_proj.weight", .kind = .decoder_layer_ffn_down_weight },
    .{ .suffix = "mlp.gate_proj.weight", .kind = .decoder_layer_ffn_gate_weight },
    .{ .suffix = "mlp.up_proj.weight", .kind = .decoder_layer_ffn_up_weight },
    .{ .suffix = "post_attention_layernorm.weight", .kind = .decoder_layer_ffn_norm_weight },
    .{ .suffix = "self_attn.k_norm.weight", .kind = .decoder_layer_attention_k_norm_weight },
    .{ .suffix = "self_attn.k_proj.weight", .kind = .decoder_layer_attention_k_weight },
    .{ .suffix = "self_attn.o_proj.weight", .kind = .decoder_layer_attention_out_weight },
    .{ .suffix = "self_attn.q_norm.weight", .kind = .decoder_layer_attention_q_norm_weight },
    .{ .suffix = "self_attn.q_proj.weight", .kind = .decoder_layer_attention_q_weight },
    .{ .suffix = "self_attn.v_proj.weight", .kind = .decoder_layer_attention_v_weight },
};

/// Tensor kinds that belong to no layer, with the vLLM-style name each one has
/// in the released checkpoint. Used by `officialName` for diagnostics, and kept
/// in one table so a kind cannot be given two different names.
const fixed_official_names = [_]struct { kind: TensorKind, name: []const u8 }{
    .{ .kind = .audio_conv1_weight, .name = "thinker.audio_tower.conv2d1.weight" },
    .{ .kind = .audio_conv1_bias, .name = "thinker.audio_tower.conv2d1.bias" },
    .{ .kind = .audio_conv2_weight, .name = "thinker.audio_tower.conv2d2.weight" },
    .{ .kind = .audio_conv2_bias, .name = "thinker.audio_tower.conv2d2.bias" },
    .{ .kind = .audio_conv3_weight, .name = "thinker.audio_tower.conv2d3.weight" },
    .{ .kind = .audio_conv3_bias, .name = "thinker.audio_tower.conv2d3.bias" },
    .{ .kind = .audio_conv_out_weight, .name = "thinker.audio_tower.conv_out.weight" },
    .{ .kind = .audio_final_norm_weight, .name = "thinker.audio_tower.ln_post.weight" },
    .{ .kind = .audio_final_norm_bias, .name = "thinker.audio_tower.ln_post.bias" },
    .{ .kind = .projector_in_weight, .name = "thinker.audio_tower.proj1.weight" },
    .{ .kind = .projector_in_bias, .name = "thinker.audio_tower.proj1.bias" },
    .{ .kind = .projector_out_weight, .name = "thinker.audio_tower.proj2.weight" },
    .{ .kind = .projector_out_bias, .name = "thinker.audio_tower.proj2.bias" },
    .{ .kind = .decoder_embed_tokens_weight, .name = "thinker.model.embed_tokens.weight" },
    .{ .kind = .decoder_final_norm_weight, .name = "thinker.model.norm.weight" },
    .{ .kind = .decoder_output_weight, .name = "thinker.lm_head.weight" },
};

/// The checkpoint name of a tensor in the vLLM-style export, written into
/// `buffer`. Returns `UnknownKind` for a kind the released naming schemes have
/// no place for.
pub fn officialName(kind: TensorKind, layer: u16, buffer: []u8) Error![]const u8 {
    assert(buffer.len > 0);
    if (layer >= container.audio_layer_base) {
        const suffix = suffixOfKind(&audio_layer_names, kind) orelse return Error.UnknownKind;
        const index = layer - container.audio_layer_base;
        return std.fmt.bufPrint(
            buffer,
            "thinker.audio_tower.layers.{d}.{s}",
            .{ index, suffix },
        ) catch Error.NameBufferTooSmall;
    }
    if (layer >= layout.decoder_layer_base) {
        const suffix = suffixOfKind(&decoder_layer_names, kind) orelse return Error.UnknownKind;
        const index = layer - layout.decoder_layer_base;
        return std.fmt.bufPrint(
            buffer,
            "thinker.model.layers.{d}.{s}",
            .{ index, suffix },
        ) catch Error.NameBufferTooSmall;
    }
    if (layer != 0) return Error.UnknownKind;
    const name = fixedName(kind) orelse return Error.UnknownKind;
    if (name.len > buffer.len) return Error.NameBufferTooSmall;
    @memcpy(buffer[0..name.len], name);
    return buffer[0..name.len];
}

fn fixedName(kind: TensorKind) ?[]const u8 {
    for (fixed_official_names) |entry| {
        if (entry.kind == kind) return entry.name;
    }
    return null;
}

/// One inventory position and the checkpoint tensor that filled it.
pub const Slot = struct {
    required: layout.Required,
    /// Index into the checkpoint's tensor list, or null when the checkpoint has
    /// nothing for this position.
    source_index: ?u32 = null,
};

/// The result of matching a checkpoint's tensor names against the inventory.
pub const Classification = struct {
    /// One entry per inventory position, in `(layer, kind)` order.
    slots: []Slot,
    /// Names no rule recognized at all: a tensor from another architecture, or
    /// something a conversion does not need.
    unknown: []const []const u8,
    /// Names a rule recognized but that this configuration has no place for --
    /// a layer beyond `num_hidden_layers`, say. Unlike `unknown`, this means the
    /// checkpoint and the configuration disagree about the model.
    unexpected: []const []const u8,
    /// Names that mapped onto a position another tensor had already filled.
    duplicate: []const []const u8,
    /// The configuration the slots were derived from, so `slotFor` can search.
    config: *const model_config.Config,

    pub fn missingCount(self: *const Classification) u32 {
        var missing: u32 = 0;
        for (self.slots) |slot| {
            if (slot.source_index == null) missing += 1;
        }
        return missing;
    }

    /// True when every inventory position was filled and nothing was left over.
    pub fn isComplete(self: *const Classification) bool {
        if (self.unknown.len != 0) return false;
        if (self.unexpected.len != 0) return false;
        if (self.duplicate.len != 0) return false;
        return self.missingCount() == 0;
    }

    pub fn slotFor(self: *const Classification, kind: TensorKind, layer: u16) ?*const Slot {
        const position = positionOf(self.config, kind, layer) orelse return null;
        return &self.slots[position];
    }
};

/// Maps every name against the inventory derived from `config`.
pub fn classifyNames(
    arena: std.mem.Allocator,
    config: *const model_config.Config,
    names: []const []const u8,
) Error!Classification {
    const count = layout.Iterator.count(config);
    const slots = try arena.alloc(Slot, count);
    for (slots, 0..) |*slot, position| {
        const required = layout.at(config, @intCast(position)) orelse unreachable;
        slot.* = .{ .required = required };
    }
    assert(slots.len == layout.Iterator.count(config));

    var unknown: std.ArrayList([]const u8) = .empty;
    var unexpected: std.ArrayList([]const u8) = .empty;
    var duplicate: std.ArrayList([]const u8) = .empty;
    for (names, 0..) |name, index| {
        const mapping = mapName(name) orelse {
            try unknown.append(arena, name);
            continue;
        };
        const position = positionOf(config, mapping.kind, mapping.layer) orelse {
            // A recognized name for a tensor this configuration does not have:
            // an extra layer, or a kind the size of the model excludes.
            try unexpected.append(arena, name);
            continue;
        };
        if (slots[position].source_index != null) {
            try duplicate.append(arena, name);
            continue;
        }
        slots[position].source_index = @intCast(index);
    }

    return .{
        .slots = slots,
        .unknown = try unknown.toOwnedSlice(arena),
        .unexpected = try unexpected.toOwnedSlice(arena),
        .duplicate = try duplicate.toOwnedSlice(arena),
        .config = config,
    };
}

/// Maps every tensor of a checkpoint file against the inventory.
pub fn classifyAll(
    arena: std.mem.Allocator,
    config: *const model_config.Config,
    file: *const safetensors.File,
) Error!Classification {
    const names = try arena.alloc([]const u8, file.tensors.len);
    for (file.tensors, 0..) |entry, index| names[index] = entry.name;
    assert(names.len == file.count());
    return classifyNames(arena, config, names);
}

/// Position of a `(kind, layer)` in the inventory. The inventory is in
/// `(layer, kind)` order, so this is a binary search over positions rather than
/// a linear scan through 612 descriptions.
pub fn positionOf(config: *const model_config.Config, kind: TensorKind, layer: u16) ?u32 {
    const key = container.sortKey(layer, @backingInt(kind));
    var low: u32 = 0;
    var high: u32 = layout.Iterator.count(config);
    assert(low <= high);
    while (low < high) {
        const middle = low + (high - low) / 2;
        const required = layout.at(config, middle) orelse unreachable;
        const middle_key = required.key();
        if (middle_key == key) return middle;
        if (middle_key < key) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return null;
}

/// The released 0.6B geometry. Written out here so the naming tests do not
/// depend on a downloaded checkpoint; `checkpoint_config` parses the same values
/// out of `config.json` and a test there compares the two.
fn testConfig() model_config.Config {
    return .{
        .magic = model_config.magic_bytes,
        .format_version = model_config.format_version,
        .architecture = @backingInt(model_config.Architecture.qwen3_asr),
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

/// Reads a fixture file from the repository, searching a bounded number of
/// parent directories so the test does not depend on the process's working
/// directory being the repository root.
fn readFixture(arena: std.mem.Allocator, path: []const u8) ![]u8 {
    const io = std.testing.io;
    const prefixes = [_][]const u8{ "", "../", "../../", "../../../" };
    for (prefixes) |prefix| {
        const full = try std.fmt.allocPrint(arena, "{s}{s}", .{ prefix, path });
        const bytes = std.Io.Dir.cwd().readFileAlloc(
            io,
            full,
            arena,
            .limited(1 << 20),
        ) catch |err| {
            switch (err) {
                error.FileNotFound, error.NotDir => continue,
                else => return err,
            }
        };
        return bytes;
    }
    return error.TestFixtureNotFound;
}

/// Splits a fixture's contents into non-empty lines, in file order.
fn fixtureLines(arena: std.mem.Allocator, path: []const u8) ![][]const u8 {
    const bytes = try readFixture(arena, path);
    var lines: std.ArrayList([]const u8) = .empty;
    var iterator = std.mem.splitScalar(u8, bytes, '\n');
    while (iterator.next()) |line| {
        if (line.len == 0) continue;
        try lines.append(arena, std.mem.trimEnd(u8, line, "\r"));
    }
    return lines.toOwnedSlice(arena);
}

test "the two released naming schemes map onto the same tensors" {
    const names = [_]struct { name: []const u8, kind: TensorKind, layer: u16 }{
        .{ .name = "thinker.audio_tower.conv2d1.weight", .kind = .audio_conv1_weight, .layer = 0 },
        .{ .name = "thinker.audio_tower.conv2d1.bias", .kind = .audio_conv1_bias, .layer = 0 },
        .{
            .name = "thinker.audio_tower.conv_out.weight",
            .kind = .audio_conv_out_weight,
            .layer = 0,
        },
        .{ .name = "thinker.audio_tower.ln_post.bias", .kind = .audio_final_norm_bias, .layer = 0 },
        .{ .name = "thinker.audio_tower.proj1.weight", .kind = .projector_in_weight, .layer = 0 },
        .{ .name = "thinker.audio_tower.proj2.weight", .kind = .projector_out_weight, .layer = 0 },
        .{
            .name = "thinker.model.embed_tokens.weight",
            .kind = .decoder_embed_tokens_weight,
            .layer = 0,
        },
        .{ .name = "thinker.model.norm.weight", .kind = .decoder_final_norm_weight, .layer = 0 },
        .{ .name = "thinker.lm_head.weight", .kind = .decoder_output_weight, .layer = 0 },
        .{
            .name = "thinker.audio_tower.layers.0.self_attn.q_proj.weight",
            .kind = .audio_layer_attention_q_weight,
            .layer = container.audio_layer_base,
        },
        .{
            .name = "thinker.audio_tower.layers.17.self_attn.out_proj.bias",
            .kind = .audio_layer_attention_out_bias,
            .layer = container.audio_layer_base + 17,
        },
        .{
            .name = "thinker.audio_tower.layers.7.self_attn_layer_norm.bias",
            .kind = .audio_layer_attention_norm_bias,
            .layer = container.audio_layer_base + 7,
        },
        .{
            .name = "thinker.audio_tower.layers.7.final_layer_norm.weight",
            .kind = .audio_layer_final_norm_weight,
            .layer = container.audio_layer_base + 7,
        },
        .{
            .name = "thinker.audio_tower.layers.7.fc2.bias",
            .kind = .audio_layer_ffn_out_bias,
            .layer = container.audio_layer_base + 7,
        },
        .{
            .name = "thinker.model.layers.0.input_layernorm.weight",
            .kind = .decoder_layer_attention_norm_weight,
            .layer = 1,
        },
        .{
            .name = "thinker.model.layers.27.mlp.down_proj.weight",
            .kind = .decoder_layer_ffn_down_weight,
            .layer = 28,
        },
        .{
            .name = "thinker.model.layers.27.self_attn.q_norm.weight",
            .kind = .decoder_layer_attention_q_norm_weight,
            .layer = 28,
        },
        // The transformers-native spellings of the same tensors.
        .{ .name = "model.audio_tower.conv2d1.weight", .kind = .audio_conv1_weight, .layer = 0 },
        .{
            .name = "model.audio_tower.ln_post.weight",
            .kind = .audio_final_norm_weight,
            .layer = 0,
        },
        .{
            .name = "model.multi_modal_projector.linear_1.weight",
            .kind = .projector_in_weight,
            .layer = 0,
        },
        .{
            .name = "model.multi_modal_projector.linear_2.bias",
            .kind = .projector_out_bias,
            .layer = 0,
        },
        .{ .name = "model.embed_tokens.weight", .kind = .decoder_embed_tokens_weight, .layer = 0 },
        .{
            .name = "model.language_model.embed_tokens.weight",
            .kind = .decoder_embed_tokens_weight,
            .layer = 0,
        },
        .{
            .name = "model.language_model.norm.weight",
            .kind = .decoder_final_norm_weight,
            .layer = 0,
        },
        .{
            .name = "model.language_model.layers.3.self_attn.v_proj.weight",
            .kind = .decoder_layer_attention_v_weight,
            .layer = 4,
        },
        .{
            .name = "model.audio_tower.layers.11.fc1.weight",
            .kind = .audio_layer_ffn_in_weight,
            .layer = container.audio_layer_base + 11,
        },
        .{
            .name = "model.language_model.layers.2.post_attention_layernorm.weight",
            .kind = .decoder_layer_ffn_norm_weight,
            .layer = 3,
        },
    };

    for (names) |expected| {
        const mapping = mapName(expected.name) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(expected.kind, mapping.kind);
        try std.testing.expectEqual(expected.layer, mapping.layer);

        // The inverse must name the same tensor again.
        var buffer: [128]u8 = undefined;
        const official = try officialName(mapping.kind, mapping.layer, &buffer);
        const round_trip = mapName(official) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(mapping.kind, round_trip.kind);
        try std.testing.expectEqual(mapping.layer, round_trip.layer);
    }
}

test "names the released model does not contain are rejected" {
    const rejected = [_][]const u8{
        "",
        "thinker.audio_tower.layers.0.conv1.weight",
        "thinker.audio_tower.layers.0.self_attn.q_proj.bias.weight",
        // The decoder has no attention biases.
        "thinker.model.layers.0.self_attn.q_proj.bias",
        // No such areas.
        "thinker.vision_model.layers.0.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        // A layer index that does not fit, or is not a number.
        "thinker.model.layers.99999.mlp.up_proj.weight",
        "thinker.model.layers..mlp.up_proj.weight",
        "thinker.model.layers.one.mlp.up_proj.weight",
        "thinker.model.layers.",
        // The suffix must match exactly.
        "thinker.model.embed.weight",
        "thinker.lm_head.bias",
        "thinker.model.layers.0.mlp.gate_proj.bias",
        "model.multi_modal_projector.linear_3.weight",
    };
    for (rejected) |name| {
        try std.testing.expectEqual(@as(?Mapping, null), mapName(name));
    }
}

test "every tensor of the released 0.6B checkpoint maps to the inventory" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // The list is the header of models/Qwen3-ASR-0.6B/model.safetensors, which
    // is where the 612-tensor count in `layout` comes from.
    const names = try fixtureLines(arena, "tests/fixtures/checkpoint_names_0p6b.txt");
    try std.testing.expectEqual(@as(usize, 612), names.len);

    const config = testConfig();
    const classification = try classifyNames(arena, &config, names);

    try std.testing.expectEqual(@as(usize, 0), classification.unknown.len);
    try std.testing.expectEqual(@as(usize, 0), classification.unexpected.len);
    try std.testing.expectEqual(@as(usize, 0), classification.duplicate.len);
    try std.testing.expectEqual(@as(u32, 0), classification.missingCount());
    try std.testing.expect(classification.isComplete());
    try std.testing.expectEqual(layout.Iterator.count(&config), @as(u32, 612));
    try std.testing.expectEqual(@as(usize, 612), classification.slots.len);

    // Every inventory description is reachable from a name, so the two
    // directions of the mapping cover the inventory exactly once each.
    for (classification.slots) |slot| {
        try std.testing.expect(slot.source_index != null);
        var buffer: [128]u8 = undefined;
        const official = try officialName(slot.required.kind, slot.required.layer, &buffer);
        const mapping = mapName(official) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(slot.required.kind, mapping.kind);
        try std.testing.expectEqual(slot.required.layer, mapping.layer);
    }
}

test "the transformers-native checkpoint lacks only the tied output projection" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const config = testConfig();
    const names = try fixtureLines(arena, "tests/fixtures/checkpoint_names_0p6b_hf.txt");
    // The native export is weight-tied: it ships 612 tensors minus `lm_head`.
    try std.testing.expectEqual(@as(usize, 611), names.len);

    const classification = try classifyNames(arena, &config, names);
    try std.testing.expectEqual(@as(usize, 0), classification.unknown.len);
    try std.testing.expectEqual(@as(usize, 0), classification.unexpected.len);
    try std.testing.expectEqual(@as(usize, 0), classification.duplicate.len);
    try std.testing.expectEqual(@as(u32, 1), classification.missingCount());
    try std.testing.expect(!classification.isComplete());

    const output = classification.slotFor(.decoder_output_weight, 0).?;
    try std.testing.expect(output.source_index == null);
    const embedding = classification.slotFor(.decoder_embed_tokens_weight, 0).?;
    try std.testing.expect(embedding.source_index != null);

    // Both lists covered the same inventory positions except that one.
    const other_names = try fixtureLines(arena, "tests/fixtures/checkpoint_names_0p6b.txt");
    const other = try classifyNames(arena, &config, other_names);
    for (classification.slots, other.slots) |left, right| {
        try std.testing.expectEqual(left.required.kind, right.required.kind);
        try std.testing.expectEqual(left.required.layer, right.required.layer);
        if (left.required.kind != .decoder_output_weight) {
            try std.testing.expect(left.source_index != null);
        }
    }
}

test "classifying a checkpoint file returns every tensor of it in file order" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // A two-tensor image standing in for a checkpoint, so the file-based entry
    // point is exercised rather than only the name list.
    const payload: [4]u8 = @splat(0);
    const stored = [_]safetensors.StoredTensor{
        .{
            .name = "thinker.model.embed_tokens.weight",
            .dtype_name = "F32",
            .dims = &.{ 1, 1 },
            .payload = &payload,
        },
        .{
            .name = "thinker.model.layers.0.mlp.down_proj.weight",
            .dtype_name = "F32",
            .dims = &.{ 1, 1 },
            .payload = &payload,
        },
    };
    const storage = try arena.alloc(u8, @intCast(safetensors.imageLength(&stored, 0)));
    const used = try safetensors.writeImage(storage, &stored, 0);
    const file = try safetensors.File.parse(arena, storage[0..used], .{});

    const config = testConfig();
    const classification = try classifyAll(arena, &config, &file);
    try std.testing.expectEqual(@as(u32, 2), file.count());
    // The embedding filled its own slot, and the file's second tensor filled a
    // decoder layer slot rather than the embedding's.
    const embedding = classification.slotFor(.decoder_embed_tokens_weight, 0).?;
    try std.testing.expectEqual(@as(u32, 0), embedding.source_index.?);
    const down = classification.slotFor(.decoder_layer_ffn_down_weight, 1).?;
    try std.testing.expectEqual(@as(u32, 1), down.source_index.?);
    try std.testing.expectEqual(@as(u32, 610), classification.missingCount());
    try std.testing.expectEqual(@as(usize, 0), classification.unknown.len);
}

test "a downloaded checkpoint's own configuration accepts its own tensor names" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    // Two released spellings of one model, each with the name list taken from
    // its own safetensors header. The configurations are read from disk, so this
    // binds the fixture names to the dimensions a checkpoint really declares --
    // 28 decoder layers and 18 audio layers -- rather than to the test's copy of
    // them. The checkpoints are downloads, so this test skips what is absent.
    const cases = [_]struct { dir: []const u8, fixture: []const u8, tied: bool }{
        .{
            .dir = "models/Qwen3-ASR-0.6B",
            .fixture = "tests/fixtures/checkpoint_names_0p6b.txt",
            .tied = false,
        },
        .{
            .dir = "models/Qwen3-ASR-0.6B-hf",
            .fixture = "tests/fixtures/checkpoint_names_0p6b_hf.txt",
            .tied = true,
        },
    };
    const checkpoint_config = @import("checkpoint_config.zig");
    const tokenizer_file = @import("tokenizer_file.zig");

    for (cases) |case| {
        var dir = std.Io.Dir.cwd().openDir(io, case.dir, .{}) catch continue;
        defer dir.close(io);
        const config_bytes = dir.readFileAlloc(
            io,
            "config.json",
            arena,
            .limited(1 << 20),
        ) catch continue;

        var token_diagnostics: tokenizer_file.Diagnostics = .{};
        const tokens = try tokenizer_file.read(arena, io, dir, &token_diagnostics);
        var config_diagnostics: checkpoint_config.Diagnostics = .{};
        const parsed = try checkpoint_config.parse(arena, .{
            .config_json = config_bytes,
            .tokenizer_config_json = dir.readFileAlloc(
                io,
                "tokenizer_config.json",
                arena,
                .limited(1 << 20),
            ) catch null,
            .generation_json = dir.readFileAlloc(
                io,
                "generation_config.json",
                arena,
                .limited(1 << 20),
            ) catch null,
            .specials = tokens.specials,
        }, &config_diagnostics);

        const names = try fixtureLines(arena, case.fixture);
        const classification = try classifyNames(arena, &parsed.config, names);
        try std.testing.expectEqual(@as(usize, 0), classification.unknown.len);
        try std.testing.expectEqual(@as(usize, 0), classification.unexpected.len);
        try std.testing.expectEqual(@as(usize, 0), classification.duplicate.len);
        try std.testing.expectEqual(@as(u32, 612), layout.Iterator.count(&parsed.config));
        const missing: u32 = if (case.tied) 1 else 0;
        try std.testing.expectEqual(missing, classification.missingCount());
        try std.testing.expectEqual(missing == 0, classification.isComplete());
    }
}

test "duplicate and unrecognized names are reported rather than dropped" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const config = testConfig();
    const names = [_][]const u8{
        "thinker.audio_tower.conv2d1.weight",
        "thinker.audio_tower.conv2d1.weight",
        "thinker.audio_tower.conv2d1.bias",
        "thinker.mystery.weight",
        // A real name, but for a layer this configuration does not have.
        "thinker.model.layers.28.mlp.up_proj.weight",
    };
    const classification = try classifyNames(arena, &config, &names);

    try std.testing.expectEqual(@as(usize, 1), classification.duplicate.len);
    try std.testing.expectEqualStrings(
        "thinker.audio_tower.conv2d1.weight",
        classification.duplicate[0],
    );
    try std.testing.expectEqual(@as(usize, 1), classification.unknown.len);
    try std.testing.expectEqualStrings("thinker.mystery.weight", classification.unknown[0]);
    // A name the mapping recognizes for a layer this configuration does not
    // have is a disagreement between the checkpoint and its own configuration.
    try std.testing.expectEqual(@as(usize, 1), classification.unexpected.len);
    try std.testing.expectEqualStrings(
        "thinker.model.layers.28.mlp.up_proj.weight",
        classification.unexpected[0],
    );
    // The second definition of the first name filled nothing.
    const first = classification.slotFor(.audio_conv1_weight, 0).?;
    try std.testing.expect(first.source_index != null);
    try std.testing.expectEqual(@as(u32, 0), first.source_index.?);
    try std.testing.expect(!classification.isComplete());

    // The reverse lookup refuses kinds with no released spelling.
    var buffer: [128]u8 = undefined;
    const bogus: TensorKind = @fromBackingInt(@intCast(60_000));
    try std.testing.expectError(Error.UnknownKind, officialName(bogus, 0, &buffer));
    try std.testing.expectError(
        Error.UnknownKind,
        officialName(.decoder_output_weight, 3, &buffer),
    );
    var small: [8]u8 = undefined;
    try std.testing.expectError(
        Error.NameBufferTooSmall,
        officialName(.decoder_embed_tokens_weight, 0, &small),
    );
}
