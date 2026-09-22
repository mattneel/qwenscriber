//! The decoder: a Qwen3 language model stack with audio features spliced in.
//!
//! One token at a time, with a key/value cache. Prompt tokens are fed through
//! the same path as generated tokens, which keeps the prompt and the decode loop
//! on one code path and therefore comparable to the reference by construction.
//!
//! `forwardToken` computes logits only when asked: the language modelling head
//! is the most expensive matrix in the model (`vocab_size x hidden_size`), and no
//! prompt position except the last needs it.

const std = @import("std");
const half_float = @import("../half_float.zig");
const model_config = @import("../model_config.zig");
const quant = @import("../quant.zig");

const assert = std.debug.assert;
const kernels = @import("kernels.zig");
const model_mod = @import("model.zig");

/// Every failure the model hierarchy can produce, plus the two conditions that
/// only arise while decoding.
pub const Error = model_mod.Error || error{
    PositionExceeded,
    ShapeMismatch,
};

pub const Model = model_mod.Model;

/// A sampled token and whether it ends the sequence.
pub const Token = struct {
    id: u32,
    /// True when the model produced an end-of-sequence token.
    finished: bool,
};

pub const Decoder = struct {
    model: *Model,
    /// Number of cached positions.
    position: u32 = 0,

    pub fn init(model: *Model) Decoder {
        return .{ .model = model };
    }

    /// Where one layer's quantized cache lives, in bytes.
    ///
    /// Both formats are written by `appendCacheRow` and read by the attention kernels, so the offsets are
    /// computed here once: a second copy of this arithmetic is how a layout and its readers drift apart.
    const CachePlane = struct {
        /// Bytes of scale plane per layer, which precede the codes.
        scales: usize,
        /// Bytes of code plane per layer.
        codes: usize,
        /// Bytes from one layer's planes to the next.
        stride: usize,
    };

    fn cachePlane(model: *const Model) CachePlane {
        const config = model.config;
        const groups_per_row = model.kvWidth() / quant.group_size;
        const scales = @as(usize, config.max_positions) * groups_per_row *
            quant.q4_scale_bytes_per_group;
        const codes = @as(usize, config.max_positions) * model.kvWidth();
        return .{ .scales = scales, .codes = codes, .stride = scales + codes };
    }

    /// Appends one key and value row to their layer's cache, quantizing when the cache holds codes.
    fn appendCacheRow(
        model: *Model,
        layer_index: u32,
        position: u32,
        key: []const f32,
        value: []const f32,
    ) void {
        const row_width = model.kvWidth();
        assert(key.len == row_width);
        assert(value.len == row_width);
        assert(position < model.config.max_positions);

        if (model.cache_format == .f32) {
            const stride = @as(usize, model.config.max_positions) * row_width;
            const offset = @as(usize, layer_index) * stride + @as(usize, position) * row_width;
            @memcpy(model.scratch.cache_keys[offset..][0..row_width], key);
            @memcpy(model.scratch.cache_values[offset..][0..row_width], value);
            return;
        }

        const plane = cachePlane(model);
        const layer_offset = @as(usize, layer_index) * plane.stride;
        const groups_per_row = row_width / quant.group_size;
        const scale_bytes = groups_per_row * quant.q4_scale_bytes_per_group;
        const scale_offset = layer_offset + @as(usize, position) * scale_bytes;
        const code_offset = layer_offset + plane.scales + @as(usize, position) * row_width;
        quant.quantizeRow(
            .q8,
            key,
            model.scratch.cache_keys_q8[scale_offset..][0..scale_bytes],
            model.scratch.cache_keys_q8[code_offset..][0..row_width],
        );
        quant.quantizeRow(
            .q8,
            value,
            model.scratch.cache_values_q8[scale_offset..][0..scale_bytes],
            model.scratch.cache_values_q8[code_offset..][0..row_width],
        );
    }

    /// Drops the key/value cache and returns to an empty context.
    pub fn reset(self: *Decoder) void {
        self.position = 0;
        // Whichever pair the model was loaded for, and only that one: the other is empty.
        if (self.model.cache_format == .f32) {
            const cache_length = @as(usize, self.model.config.text_layers) *
                self.model.config.max_positions * self.model.kvWidth();
            @memset(self.model.scratch.cache_keys[0..cache_length], 0.0);
            @memset(self.model.scratch.cache_values[0..cache_length], 0.0);
            return;
        }
        @memset(self.model.scratch.cache_keys_q8, 0);
        @memset(self.model.scratch.cache_values_q8, 0);
    }

    /// Position in the sequence the next token will occupy.
    pub fn nextPosition(self: *const Decoder) u32 {
        return self.position;
    }

    /// One forward pass.
    ///
    /// `audio_row`, when present, replaces the token embedding: that is how the
    /// reference splices the projected audio features into the placeholder
    /// positions instead of embedding a placeholder token.
    pub fn forwardToken(
        self: *Decoder,
        token: u32,
        audio_row: ?[]const f32,
        want_logits: bool,
    ) Error!void {
        const model = self.model;
        const config = &model.config;
        const hidden_size = config.text_hidden_size;
        if (self.position >= config.max_positions) return Error.PositionExceeded;

        const hidden = model.scratch.hidden;
        if (audio_row) |row| {
            if (row.len != hidden_size) return Error.UnexpectedShape;
            @memcpy(hidden, row);
        } else {
            try embeddingRow(model.embed_tokens, token, hidden_size, hidden);
        }

        const normed = model.scratch.normed;
        const query = model.scratch.query;
        const key = model.scratch.key;
        const value = model.scratch.value;
        const attention_out = model.scratch.attention_out;
        const gate = model.scratch.gate;
        const up = model.scratch.up;
        const query_width = config.textQueryElements();
        const key_value_width = config.textKeyValueElements();
        const ffn_dim = config.text_ffn_dim;
        const cache_layer_stride = @as(usize, config.max_positions) * key_value_width;

        for (model.decoder_layers, 0..) |*layer, layer_index| {
            try kernels.rmsNormInto(
                normed,
                hidden,
                layer.attention_norm_weight,
                hidden_size,
                1,
                config.text_rms_norm_eps,
            );

            try layer.q_weight.apply(query, normed, hidden_size, 1, model.scratch.row);
            try layer.k_weight.apply(key, normed, hidden_size, 1, model.scratch.row);
            try layer.v_weight.apply(value, normed, hidden_size, 1, model.scratch.row);

            // Per-head normalisation comes before the rotation, exactly as the
            // reference orders it.
            try kernels.rmsNorm(
                query,
                layer.q_norm_weight,
                config.text_head_dim,
                config.text_attention_heads,
                config.text_rms_norm_eps,
            );
            try kernels.rmsNorm(
                key,
                layer.k_norm_weight,
                config.text_head_dim,
                config.text_key_value_heads,
                config.text_rms_norm_eps,
            );

            const single_position = [_]u32{self.position};
            const half_head = config.text_head_dim / 2;
            try kernels.ropeInPlace(
                query,
                &single_position,
                config.text_attention_heads,
                config.text_head_dim,
                model.scratch.rope_cos,
                model.scratch.rope_sin,
                half_head,
            );
            try kernels.ropeInPlace(
                key,
                &single_position,
                config.text_key_value_heads,
                config.text_head_dim,
                model.scratch.rope_cos,
                model.scratch.rope_sin,
                half_head,
            );

            appendCacheRow(model, @intCast(layer_index), self.position, key, value);

            // Exactly the positions this layer has cached: the cache is
            // layer-major, and the kernel checks the slice length against the
            // number of positions it is told about, so handing it the layer's
            // whole stride fails for every position but the last.
            if (model.cache_format == .f32) {
                const cached_width = @as(usize, self.position + 1) * key_value_width;
                const layer_offset = @as(usize, layer_index) * cache_layer_stride;
                const layer_cache = model.scratch.cache_keys[layer_offset..][0..cached_width];
                const layer_values = model.scratch.cache_values[layer_offset..][0..cached_width];
                try kernels.decodeAttentionStep(
                    attention_out,
                    query,
                    layer_cache,
                    layer_values,
                    self.position + 1,
                    config.text_attention_heads,
                    config.text_key_value_heads,
                    config.text_head_dim,
                    model.scratch.position_scores,
                );
            } else {
                const plane = cachePlane(model);
                const layer_offset = @as(usize, layer_index) * plane.stride;
                try kernels.decodeAttentionStepQ8(
                    attention_out,
                    query,
                    model.scratch.cache_keys_q8[layer_offset..][0..plane.scales],
                    model.scratch.cache_keys_q8[layer_offset + plane.scales ..][0..plane.codes],
                    model.scratch.cache_values_q8[layer_offset..][0..plane.scales],
                    model.scratch.cache_values_q8[layer_offset + plane.scales ..][0..plane.codes],
                    self.position + 1,
                    config.text_attention_heads,
                    config.text_key_value_heads,
                    config.text_head_dim,
                    model.scratch.position_scores,
                );
            }

            try layer.out_weight.apply(normed, attention_out, query_width, 1, model.scratch.row);
            try kernels.addInPlace(hidden, normed[0..hidden_size]);

            try kernels.rmsNormInto(
                normed,
                hidden,
                layer.ffn_norm_weight,
                hidden_size,
                1,
                config.text_rms_norm_eps,
            );
            try layer.gate_weight.apply(gate, normed, hidden_size, 1, model.scratch.row);
            try layer.up_weight.apply(up, normed, hidden_size, 1, model.scratch.row);
            try kernels.siluMul(gate[0..ffn_dim], gate[0..ffn_dim], up[0..ffn_dim]);
            try layer.down_weight.apply(normed, gate[0..ffn_dim], ffn_dim, 1, model.scratch.row);
            try kernels.addInPlace(hidden, normed[0..hidden_size]);
        }

        self.position += 1;
        if (!want_logits) return;

        try kernels.rmsNormInto(
            normed,
            hidden,
            model.decoder_final_norm_weight,
            hidden_size,
            1,
            config.text_rms_norm_eps,
        );
        const logits = model.scratch.logits;
        try model.output_weight.apply(logits, normed, hidden_size, 1, model.scratch.row);
    }

    /// Argmax over the logits computed by the last `forwardToken`.
    pub fn argmaxLogits(self: *Decoder) u32 {
        const logits = self.model.scratch.logits;
        var best: u32 = 0;
        var best_value = logits[0];
        for (logits[1..], 1..) |value, index| {
            if (value > best_value) {
                best_value = value;
                best = @intCast(index);
            }
        }
        return best;
    }

    /// Feeds the prompt, splicing `audio` into the placeholder positions.
    ///
    /// `audio` holds `audio_steps` rows of `text_hidden_size`, as `projectAudio`
    /// writes them. Every placeholder in `tokens` consumes one row, in order.
    /// Returns the token the model predicts after the prompt.
    pub fn prefill(
        self: *Decoder,
        tokens: []const u32,
        audio: []const f32,
        audio_steps: u32,
    ) Error!u32 {
        const config = &self.model.config;
        const hidden_size = config.text_hidden_size;
        if (audio.len < @as(usize, audio_steps) * hidden_size) return Error.UnexpectedShape;
        if (tokens.len == 0) return Error.UnexpectedShape;

        var audio_index: u32 = 0;
        for (tokens, 0..) |token, index| {
            const is_last = index + 1 == tokens.len;
            if (token == config.token_audio_pad) {
                if (audio_index >= audio_steps) return Error.PromptMismatch;
                const row = audio[@as(usize, audio_index) * hidden_size ..][0..hidden_size];
                try self.forwardToken(token, row, is_last);
                audio_index += 1;
            } else {
                try self.forwardToken(token, null, is_last);
            }
        }
        if (audio_index != audio_steps) return Error.PromptMismatch;
        return self.argmaxLogits();
    }

    /// Feeds one generated token and returns the next.
    pub fn step(self: *Decoder, token: u32) Error!u32 {
        try self.forwardToken(token, null, true);
        return self.argmaxLogits();
    }
};

/// Reads one row of an embedding matrix into `out`.
///
/// The embedding is the largest tensor in the model and is normally quantized,
/// so it is decoded a row at a time rather than expanded.
pub fn embeddingRow(matrix: kernels.Matrix, row: u32, cols: u32, out: []f32) Error!void {
    if (out.len != cols) return Error.UnexpectedShape;
    switch (matrix) {
        .f32 => |weights| {
            const offset = @as(usize, row) * cols;
            if (offset + cols > weights.len) return Error.UnexpectedShape;
            @memcpy(out, weights[offset..][0..cols]);
        },
        .f16 => |weights| {
            const offset = @as(usize, row) * cols;
            if (offset + cols > weights.len) return Error.UnexpectedShape;
            for (weights[offset..][0..cols], out) |bits, *target| {
                target.* = half_float.fromF16(bits);
            }
        },
        .quantized => |planes| {
            const groups_per_row = cols / quant.group_size;
            const scale_stride = groups_per_row * quant.q4_scale_bytes_per_group;
            const data_stride = groups_per_row * quant.dataBytesPerGroup(planes.format);
            if (@as(usize, row + 1) * data_stride > planes.data.len) {
                return Error.UnexpectedShape;
            }
            quant.dequantizeRow(
                planes.format,
                planes.scales[@as(usize, row) * scale_stride ..],
                planes.data[@as(usize, row) * data_stride ..],
                out,
            );
        },
    }
}

test "embedding rows decode for every storage kind" {
    const cols: u32 = 64;
    const rows: u32 = 3;
    var values: [rows * cols]f32 = undefined;
    for (&values, 0..) |*value, index| {
        const position: f32 = @floatFromInt(index);
        value.* = @sin(position * 0.03);
    }

    // f32: exact.
    var out: [cols]f32 = undefined;
    try embeddingRow(.{ .f32 = &values }, 1, cols, &out);
    for (values[cols..][0..cols], out) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }

    // f16: lossy but close.
    var values_f16: [rows * cols]u16 = undefined;
    for (values, &values_f16) |value, *bits| {
        bits.* = half_float.toF16(value);
    }
    try embeddingRow(.{ .f16 = &values_f16 }, 2, cols, &out);
    for (values[2 * cols ..][0..cols], out) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-3);
    }

    // Quantized: within a step of the original.
    var scales: [rows * quant.q4_scale_bytes_per_group]u8 = undefined;
    var data: [rows * quant.q4_data_bytes_per_group]u8 = undefined;
    var row_index: u32 = 0;
    while (row_index < rows) : (row_index += 1) {
        quant.quantizeRow(
            .q4,
            values[row_index * cols ..][0..cols],
            scales[row_index * quant.q4_scale_bytes_per_group ..][0..quant.q4_scale_bytes_per_group],
            data[row_index * quant.q4_data_bytes_per_group ..][0..quant.q4_data_bytes_per_group],
        );
    }
    const matrix = kernels.Matrix{ .quantized = .{
        .format = .q4,
        .scales = &scales,
        .data = &data,
    } };
    try embeddingRow(matrix, 1, cols, &out);
    for (values[cols..][0..cols], out) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 0.2);
    }
}

test "a row past the end is refused" {
    const cols: u32 = 64;
    var values: [64]f32 = undefined;
    @memset(&values, 0.0);
    var out: [cols]f32 = undefined;
    try std.testing.expectError(
        Error.UnexpectedShape,
        embeddingRow(.{ .f32 = &values }, 1, cols, &out),
    );
    try std.testing.expectError(
        Error.UnexpectedShape,
        embeddingRow(.{ .f32 = &values }, 0, cols, out[0 .. cols - 1]),
    );
}
