//! Reads an official checkpoint's JSON configuration into the runtime's
//! `model_config.Config`.
//!
//! Two shapes are in circulation and both must parse:
//!
//!     Qwen/Qwen3-ASR-0.6B      everything under `thinker_config`, with
//!                              `audio_config` and `text_config` inside it
//!     Qwen/Qwen3-ASR-0.6B-hf   `audio_config` and `text_config` at the top
//!                              level, and `tie_word_embeddings` beside them
//!
//! They carry the same dimensions under different aliases, so this module names
//! each logical field, lists the spellings the released configurations use, and
//! reads the first that is present. A missing or mistyped field is reported
//! with its logical name in `Diagnostics` rather than as a bare failure: an
//! operator fixing a checkpoint needs to know *which* field.
//!
//! # Values that are derived, not read
//!
//! Three fields have no counterpart in either configuration:
//!
//!   * `audio_max_position_steps` is the number of post-convolution steps in
//!     one chunk, which `audioChunkSteps` derives from `n_window`. The released
//!     audio configuration advertises `max_source_positions: 1500`, which is a
//!     *different* quantity (the reference implementation's crop length, not
//!     the sinusoidal table's height), and the native configuration advertises
//!     `max_position_embeddings: 13`, which matches. So the derived value is
//!     used, and a declared `max_position_embeddings` is validated against it
//!     rather than trusted.
//!   * `audio_layer_norm_eps` is 1e-5, `torch.nn.LayerNorm`'s default, because
//!     the audio tower builds its layers without `eps=` and the value therefore
//!     never reaches the configuration file.
//!   * `max_positions` and `max_decode_tokens` are the runtime's own budgets
//!     (see the constants below), not the checkpoint's.

const std = @import("std");
const qwenscriber = @import("qwenscriber");

const assert = std.debug.assert;
const model_config = qwenscriber.model_config;
const tokenizer_file = @import("tokenizer_file.zig");

/// Positions the decoder may address, i.e. the hard limit on the KV cache.
///
/// The 0.6B checkpoint advertises `max_position_embeddings: 65536`, which at
/// 1024 key/value units per layer, 28 layers, and two bytes per unit would be a
/// 3.7 GiB cache -- not something a browser can hold. A 30-second clip needs 390
/// audio positions plus at most `max_decode_tokens` text positions, so 16384 is
/// far more than the runtime can use and small enough to allocate. A checkpoint
/// advertising less is honoured rather than overridden.
pub const max_positions_default: u32 = 16384;

/// Decoded tokens per request. A 30-second transcript is a few hundred tokens of
/// text; the reference pipeline's own `max_new_tokens` (512 in the native
/// export, absent in the other) describes its capacity, not this runtime's
/// budget, so the runtime sets its own and truncates rather than overruns.
pub const max_decode_tokens_default: u32 = 256;

/// `torch.nn.LayerNorm`'s default epsilon. The audio tower's layer norms are
/// constructed without an `eps` argument, so this value exists only in the
/// reference implementation's source.
pub const audio_layer_norm_eps_default: f32 = 1e-5;

/// The model type every released Qwen3-ASR configuration declares.
pub const model_type = "qwen3_asr";

/// dtypes a conversion source may have. Anything else is already quantized or
/// not a float, and the converter would have to guess.
const supported_dtypes = [_][]const u8{
    "bfloat16", "bf16", "float16", "fp16", "half", "float32", "fp32", "float",
};

pub const Error = error{
    /// A configuration file is not JSON, or the shape of it is not one of the
    /// two documented forms.
    MalformedJson,
    /// `model_type` is absent.
    MissingModelType,
    /// `model_type` names something other than Qwen3-ASR.
    UnknownModelType,
    /// A required field is absent or has the wrong JSON type. `Diagnostics`
    /// names it.
    MissingField,
    /// The configuration has no `text_config`.
    MissingTextConfig,
    /// The configuration has no `audio_config`.
    MissingAudioConfig,
    /// An attention head count is zero.
    ZeroHeadCount,
    /// The declared head geometry does not describe this model: the audio
    /// tower's width is not a whole number of heads, or the audio tower
    /// declares fewer key/value heads than query heads.
    HeadGeometryMismatch,
    /// A declared sinusoidal position table is shorter than one chunk's
    /// post-convolution steps.
    PositionTableTooShort,
    /// The configuration's dtype is not a 16- or 32-bit float.
    UnsupportedDtype,
    /// The audio start, end, or pad token id could not be resolved.
    MissingAudioStartToken,
    MissingAudioEndToken,
    MissingAudioPadToken,
    /// A prompt or transcript marker's id could not be resolved.
    MissingImStartToken,
    MissingImEndToken,
    MissingEndOfTextToken,
    MissingAsrTextToken,
    /// The end-of-sequence or padding id could not be resolved.
    MissingEosToken,
    MissingPadToken,
} || std.mem.Allocator.Error || model_config.Error;

/// Names the field that failed, so a checkpoint can be fixed without reading
/// this module.
pub const Diagnostics = struct {
    /// Logical name of the field that failed, e.g. `text_config.head_dim`.
    /// Empty when the failure was not about one field. Static storage.
    field: []const u8 = "",
    /// True when the named field was present but had the wrong JSON type.
    mistyped: bool = false,
};

/// Everything the converter needs from a checkpoint's configuration files.
pub const Source = struct {
    /// `config.json`.
    config_json: []const u8,
    /// `generation_config.json`, whose `eos_token_id` and `pad_token_id` are
    /// authoritative for sampling.
    generation_json: ?[]const u8 = null,
    /// `tokenizer_config.json`, which names the audio markers for the native
    /// export, where `config.json` has ids for the audio pad token only.
    tokenizer_config_json: ?[]const u8 = null,
    /// Added tokens (name and id) from the tokenizer files.
    specials: []const tokenizer_file.TokenSpecials = &.{},
};

pub const Parsed = struct {
    config: model_config.Config,
    /// The checkpoint's `tie_word_embeddings`. The converter decides tiedness
    /// from the tensors themselves and cross-checks this flag; a configuration
    /// that claims tiedness while shipping a different `lm_head` would produce
    /// a model that reuses the embedding where the checkpoint did not.
    output_is_tied: bool,
    /// dtype as written in `config.json` (`"bfloat16"`), for reporting.
    source_dtype: []const u8,
    /// Position budget the runtime will use, before the runtime's own clamp.
    /// Reported by the inspector; the runtime reads `config.max_positions`.
    config_positions_advertised: u32,
};

/// Parses a checkpoint's configuration. `arena` must outlive the result,
/// because `Parsed.source_dtype` and the JSON strings alias the arena.
pub fn parse(
    arena: std.mem.Allocator,
    source: Source,
    diagnostics: *Diagnostics,
) Error!Parsed {
    const root = try parseJson(arena, source.config_json);
    const root_object = switch (root) {
        .object => |object| object,
        else => return Error.MalformedJson,
    };

    // The vLLM export wraps the model's own configuration in `thinker_config`;
    // the native export puts it at the top level. Everything below reads the
    // chosen object, so both forms take the same path.
    const model = switch (root_object.get("thinker_config") orelse root) {
        .object => |object| object,
        else => return Error.MalformedJson,
    };

    try requireModelType(model);
    const audio = try subObject(model, "audio_config", Error.MissingAudioConfig, diagnostics);
    const text = try subObject(model, "text_config", Error.MissingTextConfig, diagnostics);
    const generation = try optionalObject(arena, source.generation_json);
    const token_config = try optionalObject(arena, source.tokenizer_config_json);

    var reader = FieldReader{ .diagnostics = diagnostics };
    // Every field is written below except the reserved words, which must stay
    // zero for `validate` to accept the result: a field left at zero by mistake
    // fails there rather than reaching a shard.
    var config = std.mem.zeroes(model_config.Config);
    config.magic = model_config.magic_bytes;
    config.format_version = model_config.format_version;
    config.architecture = @backingInt(model_config.Architecture.qwen3_asr);
    // Not in any configuration file; see the module comment.
    config.audio_layer_norm_eps = audio_layer_norm_eps_default;
    try readAudioFields(&reader, audio, &config);
    try readTextFields(&reader, text, &config);

    // Derived, not read: one chunk of audio is what the sinusoidal table has to
    // cover, and the runtime never indexes past that.
    config.audio_max_position_steps = config.audioChunkSteps();
    try checkPositionTable(&reader, audio, config.audio_max_position_steps);

    // The checkpoint advertises as much as 65536 positions. The runtime honours
    // a smaller claim but never allocates for a larger one.
    const advertised = reader.optionalU32(
        text,
        "text_config.max_position_embeddings",
        &.{"max_position_embeddings"},
    ) orelse max_positions_default;
    config.max_positions = @min(advertised, max_positions_default);
    config.max_decode_tokens = max_decode_tokens_default;
    const dtype = try readDtype(&reader, model, diagnostics);

    // Everything `config.json` alone decides is checked before the tokenizer
    // files are consulted: a model whose shape the runtime cannot represent
    // should fail on the field that says so, not on a token id.
    try config.validate();
    try readTokenFields(source.specials, model, generation, token_config, &config, diagnostics);
    try requireTokensInVocabulary(&config);

    return .{
        .config = config,
        .output_is_tied = readTiedFlag(model, text, audio),
        .source_dtype = dtype,
        .config_positions_advertised = advertised,
    };
}

fn parseJson(arena: std.mem.Allocator, bytes: []const u8) Error!std.json.Value {
    return std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{}) catch |err| switch (err) {
        error.OutOfMemory => Error.OutOfMemory,
        else => Error.MalformedJson,
    };
}

fn requireModelType(model: std.json.ObjectMap) Error!void {
    const declared = switch (model.get("model_type") orelse return Error.MissingModelType) {
        .string => |text| text,
        else => return Error.MissingModelType,
    };
    if (std.mem.eql(u8, declared, model_type)) return;
    return Error.UnknownModelType;
}

fn subObject(
    model: std.json.ObjectMap,
    name: []const u8,
    missing: Error,
    diagnostics: *Diagnostics,
) Error!std.json.ObjectMap {
    const value = model.get(name) orelse {
        diagnostics.field = name;
        return missing;
    };
    return switch (value) {
        .object => |object| object,
        else => Error.MalformedJson,
    };
}

fn optionalObject(arena: std.mem.Allocator, bytes: ?[]const u8) Error!?std.json.ObjectMap {
    const json = bytes orelse return null;
    const value = try parseJson(arena, json);
    return switch (value) {
        .object => |object| object,
        else => Error.MalformedJson,
    };
}

/// Reads a field through a list of spellings. The logical name is what a
/// diagnostic reports, and the alias list is what the released files actually
/// contain, so a checkpoint written by a future transformers version fails with
/// the name an operator can search for.
const FieldReader = struct {
    diagnostics: *Diagnostics,

    fn u32Field(
        self: *FieldReader,
        object: std.json.ObjectMap,
        field: []const u8,
        names: []const []const u8,
    ) Error!u32 {
        const value = self.optionalU32(object, field, names) orelse {
            self.diagnostics.field = field;
            return Error.MissingField;
        };
        return value;
    }

    fn optionalU32(
        self: *FieldReader,
        object: std.json.ObjectMap,
        field: []const u8,
        names: []const []const u8,
    ) ?u32 {
        for (names) |name| {
            const value = object.get(name) orelse continue;
            switch (value) {
                .integer => |number| {
                    if (number >= 0 and number <= std.math.maxInt(u32)) {
                        return @intCast(number);
                    }
                },
                else => {},
            }
            self.diagnostics.field = field;
            self.diagnostics.mistyped = true;
            return null;
        }
        return null;
    }

    fn f64Field(
        self: *FieldReader,
        object: std.json.ObjectMap,
        field: []const u8,
        names: []const []const u8,
    ) Error!f64 {
        for (names) |name| {
            const value = object.get(name) orelse continue;
            switch (value) {
                .float => |number| return number,
                .integer => |number| return @floatFromInt(number),
                else => {},
            }
            self.diagnostics.field = field;
            self.diagnostics.mistyped = true;
            return Error.MissingField;
        }
        self.diagnostics.field = field;
        return Error.MissingField;
    }

    fn stringField(
        self: *FieldReader,
        object: std.json.ObjectMap,
        field: []const u8,
        names: []const []const u8,
    ) ?[]const u8 {
        for (names) |name| {
            const value = object.get(name) orelse continue;
            switch (value) {
                // A null in the configuration means "unset", which is how the
                // released files spell an unused token.
                .null => continue,
                .string => |text| return text,
                else => {},
            }
            self.diagnostics.field = field;
            self.diagnostics.mistyped = true;
            return null;
        }
        return null;
    }
};

fn readAudioFields(
    reader: *FieldReader,
    audio: std.json.ObjectMap,
    config: *model_config.Config,
) Error!void {
    config.audio_d_model = try reader.u32Field(audio, "audio_config.d_model", &.{"d_model"});
    config.audio_layers = try reader.u32Field(
        audio,
        "audio_config.encoder_layers",
        &.{ "encoder_layers", "num_hidden_layers" },
    );
    const heads = try reader.u32Field(
        audio,
        "audio_config.encoder_attention_heads",
        &.{ "encoder_attention_heads", "num_attention_heads" },
    );
    if (heads == 0) return Error.ZeroHeadCount;
    config.audio_attention_heads = heads;
    config.audio_ffn_dim = try reader.u32Field(
        audio,
        "audio_config.encoder_ffn_dim",
        &.{ "encoder_ffn_dim", "intermediate_size" },
    );
    config.audio_downsample_hidden_size = try reader.u32Field(
        audio,
        "audio_config.downsample_hidden_size",
        &.{"downsample_hidden_size"},
    );
    config.audio_n_window = try reader.u32Field(audio, "audio_config.n_window", &.{"n_window"});
    config.audio_n_window_infer = try reader.u32Field(
        audio,
        "audio_config.n_window_infer",
        &.{"n_window_infer"},
    );
    config.audio_output_dim = try reader.u32Field(
        audio,
        "audio_config.output_dim",
        &.{"output_dim"},
    );
    config.mel_bins = try reader.u32Field(
        audio,
        "audio_config.num_mel_bins",
        &.{"num_mel_bins"},
    );

    // The audio tower is plain multi-head attention. A configuration that
    // declared fewer key/value heads would need a layout the runtime does not
    // have, so it is rejected instead of silently reading the query heads for
    // all of them.
    if (reader.optionalU32(audio, "audio_config.num_key_value_heads", &.{"num_key_value_heads"})) |kv| {
        if (kv != heads) return Error.HeadGeometryMismatch;
    }
    if (config.audio_d_model % config.audio_attention_heads != 0) {
        return Error.HeadGeometryMismatch;
    }
}

fn readTextFields(
    reader: *FieldReader,
    text: std.json.ObjectMap,
    config: *model_config.Config,
) Error!void {
    config.text_hidden_size = try reader.u32Field(text, "text_config.hidden_size", &.{"hidden_size"});
    config.text_layers = try reader.u32Field(
        text,
        "text_config.num_hidden_layers",
        &.{"num_hidden_layers"},
    );
    const heads = try reader.u32Field(
        text,
        "text_config.num_attention_heads",
        &.{"num_attention_heads"},
    );
    const kv_heads = try reader.u32Field(
        text,
        "text_config.num_key_value_heads",
        &.{"num_key_value_heads"},
    );
    if (heads == 0 or kv_heads == 0) return Error.ZeroHeadCount;
    config.text_attention_heads = heads;
    config.text_key_value_heads = kv_heads;
    config.text_head_dim = try reader.u32Field(text, "text_config.head_dim", &.{"head_dim"});
    config.text_ffn_dim = try reader.u32Field(
        text,
        "text_config.intermediate_size",
        &.{"intermediate_size"},
    );
    config.vocab_size = try reader.u32Field(text, "text_config.vocab_size", &.{"vocab_size"});
    config.text_rms_norm_eps = @floatCast(try reader.f64Field(
        text,
        "text_config.rms_norm_eps",
        &.{"rms_norm_eps"},
    ));
    config.rope_theta = @floatCast(try readRopeTheta(reader, text));
}

/// `rope_theta` sits at the top level of the text configuration in the vLLM
/// export and inside `rope_parameters` (or `rope_scaling`) in the native one.
fn readRopeTheta(reader: *FieldReader, text: std.json.ObjectMap) Error!f64 {
    if (nestedF64(text, "rope_parameters", "rope_theta")) |value| return value;
    if (nestedF64(text, "rope_scaling", "rope_theta")) |value| return value;
    return reader.f64Field(text, "text_config.rope_theta", &.{"rope_theta"});
}

fn nestedF64(object: std.json.ObjectMap, section: []const u8, field: []const u8) ?f64 {
    const inner = switch (object.get(section) orelse return null) {
        .object => |inner| inner,
        else => return null,
    };
    return switch (inner.get(field) orelse return null) {
        .float => |number| number,
        .integer => |number| @floatFromInt(number),
        else => null,
    };
}

/// The sinusoidal position table has one row per post-convolution step, so a
/// declared `max_position_embeddings` below the chunk's step count would index
/// past the table the checkpoint ships. The value is *not* copied into the
/// runtime's configuration: `max_source_positions` in the same file means
/// something else, and reading it as the table height would be wrong.
fn checkPositionTable(
    reader: *FieldReader,
    audio: std.json.ObjectMap,
    chunk_steps: u32,
) Error!void {
    const declared = reader.optionalU32(
        audio,
        "audio_config.max_position_embeddings",
        &.{"max_position_embeddings"},
    ) orelse return;
    if (declared < chunk_steps) return Error.PositionTableTooShort;
}

fn readTokenFields(
    specials: []const tokenizer_file.TokenSpecials,
    model: std.json.ObjectMap,
    generation: ?std.json.ObjectMap,
    token_config: ?std.json.ObjectMap,
    config: *model_config.Config,
    diagnostics: *Diagnostics,
) Error!void {
    var reader = FieldReader{ .diagnostics = diagnostics };
    const context = TokenContext{
        .reader = &reader,
        .specials = specials,
        .model = model,
        .token_config = token_config,
    };

    config.token_audio_start = try resolveToken(context, .{
        .role = .audio_start,
        .id_fields = &.{"audio_start_token_id"},
        .name_field = "audio_bos_token",
    });
    config.token_audio_end = try resolveToken(context, .{
        .role = .audio_end,
        .id_fields = &.{"audio_end_token_id"},
        .name_field = "audio_eos_token",
    });
    config.token_audio_pad = try resolveToken(context, .{
        .role = .audio_pad,
        .id_fields = &.{"audio_token_id"},
        .name_field = "audio_token",
    });
    config.token_im_start = try resolveToken(context, .{
        .role = .im_start,
        .name_field = "additional_special_tokens",
    });
    config.token_im_end = try resolveToken(context, .{ .role = .im_end });
    config.token_endoftext = try resolveToken(context, .{ .role = .endoftext });
    config.token_asr_text = try resolveToken(context, .{ .role = .asr_text });

    const eos = try readEos(context, generation, diagnostics);
    config.token_eos_primary = eos[0];
    config.token_eos_secondary = eos[1];
    config.token_pad = readPad(context, generation) orelse config.token_endoftext;
}

/// The special token ids that came from the tokenizer files are checked against
/// the vocabulary here, because `config.validate` ran before they were known.
/// The list is the same ten fields `model_config.validate` range-checks, and
/// this is the second of that pair: a token id past the vocabulary would index
/// outside the token table at decode time.
fn requireTokensInVocabulary(config: *const model_config.Config) Error!void {
    const tokens = [_]u32{
        config.token_audio_start, config.token_audio_end,   config.token_audio_pad,
        config.token_im_start,    config.token_im_end,      config.token_endoftext,
        config.token_asr_text,    config.token_eos_primary, config.token_eos_secondary,
        config.token_pad,
    };
    for (tokens) |token| {
        if (token >= config.vocab_size) return model_config.Error.InvalidTokenId;
    }
}

/// The special tokens the prompt and the transcript markers are built from.
const TokenRole = enum { audio_start, audio_end, audio_pad, im_start, im_end, endoftext, asr_text };

const TokenQuery = struct {
    role: TokenRole,
    /// Spellings of an explicit id field in `config.json`.
    id_fields: []const []const u8 = &.{},
    /// Spelling of a token *name* field in `tokenizer_config.json`, whose
    /// content is looked up in the added tokens.
    name_field: []const u8 = "",
};

const TokenContext = struct {
    reader: *FieldReader,
    specials: []const tokenizer_file.TokenSpecials,
    model: std.json.ObjectMap,
    token_config: ?std.json.ObjectMap,

    fn fromName(self: TokenContext, name: []const u8) ?u32 {
        for (self.specials) |entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry.id;
        }
        return null;
    }
};

fn canonicalName(role: TokenRole) []const u8 {
    return switch (role) {
        .audio_start => "<|audio_start|>",
        .audio_end => "<|audio_end|>",
        .audio_pad => "<|audio_pad|>",
        .im_start => "<|im_start|>",
        .im_end => "<|im_end|>",
        .endoftext => "<|endoftext|>",
        .asr_text => "<asr_text>",
    };
}

fn missingTokenError(role: TokenRole) Error {
    return switch (role) {
        .audio_start => Error.MissingAudioStartToken,
        .audio_end => Error.MissingAudioEndToken,
        .audio_pad => Error.MissingAudioPadToken,
        .im_start => Error.MissingImStartToken,
        .im_end => Error.MissingImEndToken,
        .endoftext => Error.MissingEndOfTextToken,
        .asr_text => Error.MissingAsrTextToken,
    };
}

/// Resolves a special token id from, in order: the configuration's own id
/// field, the tokenizer configuration's name for it, and the name the Qwen2
/// tokenizer gives it. The native export has ids for the audio pad token only,
/// so the name path is what makes it convertible at all.
fn resolveToken(context: TokenContext, query: TokenQuery) Error!u32 {
    if (query.id_fields.len > 0) {
        assert(query.id_fields.len > 0);
        const id = context.reader.optionalU32(context.model, query.id_fields[0], query.id_fields);
        if (id) |value| return value;
    }
    if (query.name_field.len > 0) {
        if (context.token_config) |token_config| {
            if (context.reader.stringField(token_config, query.name_field, &.{query.name_field})) |name| {
                if (context.fromName(name)) |id| return id;
            }
        }
    }
    if (context.fromName(canonicalName(query.role))) |id| return id;
    return missingTokenError(query.role);
}

/// The end-of-sequence ids. `generation_config.json` is authoritative: it is
/// what the reference implementation's sampler reads.
fn readEos(
    context: TokenContext,
    generation: ?std.json.ObjectMap,
    diagnostics: *Diagnostics,
) Error![2]u32 {
    if (generation) |object| {
        if (try readTokenIdArray(object, "eos_token_id", diagnostics)) |ids| {
            return ids;
        }
    }
    if (try readTokenIdArray(context.model, "eos_token_id", diagnostics)) |ids| {
        return ids;
    }
    // No declared end-of-sequence: the Qwen2 default is `endoftext`, with
    // `im_end` as the secondary, which is what the released generation
    // configurations spell out explicitly.
    const primary = try resolveToken(context, .{ .role = .endoftext });
    const secondary = try resolveToken(context, .{ .role = .im_end });
    return .{ primary, secondary };
}

fn readPad(context: TokenContext, generation: ?std.json.ObjectMap) ?u32 {
    if (generation) |object| {
        if (context.reader.optionalU32(object, "pad_token_id", &.{"pad_token_id"})) |id| return id;
    }
    return context.reader.optionalU32(context.model, "pad_token_id", &.{"pad_token_id"});
}

/// Reads `eos_token_id`, which the released files spell as either an array or a
/// single integer. The runtime carries two ids; a single one doubles.
fn readTokenIdArray(
    object: std.json.ObjectMap,
    field: []const u8,
    diagnostics: *Diagnostics,
) Error!?[2]u32 {
    const value = object.get(field) orelse return null;
    switch (value) {
        .integer => |number| {
            if (number < 0 or number > std.math.maxInt(u32)) {
                diagnostics.field = field;
                diagnostics.mistyped = true;
                return Error.MissingEosToken;
            }
            const id: u32 = @intCast(number);
            return .{ id, id };
        },
        .array => |array| {
            if (array.items.len == 0) return Error.MissingEosToken;
            var ids: [2]u32 = undefined;
            var count: usize = 0;
            while (count < array.items.len and count < 2) : (count += 1) {
                const element = switch (array.items[count]) {
                    .integer => |number| number,
                    else => {
                        diagnostics.field = field;
                        diagnostics.mistyped = true;
                        return Error.MissingEosToken;
                    },
                };
                if (element < 0 or element > std.math.maxInt(u32)) return Error.MissingEosToken;
                ids[count] = @intCast(element);
            }
            if (count == 1) ids[1] = ids[0];
            return ids;
        },
        else => {
            diagnostics.field = field;
            diagnostics.mistyped = true;
            return Error.MissingEosToken;
        },
    }
}

/// The checkpoint's storage dtype. Optional: transformers defaults to float32,
/// and a missing field means "whatever the tensors are", which the converter
/// checks tensor by tensor anyway.
fn readDtype(
    reader: *FieldReader,
    model: std.json.ObjectMap,
    diagnostics: *Diagnostics,
) Error![]const u8 {
    const declared = reader.stringField(model, "dtype", &.{ "dtype", "torch_dtype" }) orelse
        return "unknown";
    for (supported_dtypes) |supported| {
        if (std.mem.eql(u8, declared, supported)) return declared;
    }
    diagnostics.field = "dtype";
    return Error.UnsupportedDtype;
}

/// `tie_word_embeddings` appears at the top level of the native export and
/// inside `text_config` and `audio_config` of the vLLM export, where the audio
/// tower's copy is meaningless for the decoder. The first that is present wins,
/// and an absent flag means "not tied", which the converter then verifies
/// against the tensors.
fn readTiedFlag(
    model: std.json.ObjectMap,
    text: std.json.ObjectMap,
    audio: std.json.ObjectMap,
) bool {
    const sections = [_]std.json.ObjectMap{ model, text, audio };
    for (sections) |section| {
        const value = section.get("tie_word_embeddings") orelse continue;
        switch (value) {
            .bool => |flag| return flag,
            else => continue,
        }
    }
    return false;
}

test "both released configuration shapes parse to the same runtime configuration" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // The vLLM export: everything under `thinker_config`, ids beside it.
    const vllm =
        \\{"model_type":"qwen3_asr","thinker_config":{
        \\  "model_type":"qwen3_asr",
        \\  "audio_config":{"d_model":896,"encoder_layers":18,"encoder_attention_heads":14,
        \\    "encoder_ffn_dim":3584,"downsample_hidden_size":480,"n_window":50,
        \\    "n_window_infer":800,"output_dim":1024,"num_mel_bins":128,
        \\    "max_source_positions":1500},
        \\  "text_config":{"hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":16,
        \\    "num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,
        \\    "vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000,
        \\    "max_position_embeddings":65536,"tie_word_embeddings":true},
        \\  "audio_start_token_id":151669,"audio_end_token_id":151670,
        \\  "audio_token_id":151676,"dtype":"bfloat16"}}
    ;
    // The native export: the same values at the top level, `rope_parameters`
    // instead of `rope_theta`, and no audio marker ids at all.
    const native =
        \\{"model_type":"qwen3_asr","dtype":"bfloat16","tie_word_embeddings":true,
        \\ "audio_token_id":151676,"eos_token_id":[151643,151645],"pad_token_id":151645,
        \\ "audio_config":{"d_model":896,"encoder_layers":18,"encoder_attention_heads":14,
        \\   "encoder_ffn_dim":3584,"downsample_hidden_size":480,"n_window":50,
        \\   "n_window_infer":800,"output_dim":1024,"num_mel_bins":128,
        \\   "num_key_value_heads":14,"max_position_embeddings":13},
        \\ "text_config":{"hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":16,
        \\   "num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,
        \\   "vocab_size":151936,"rms_norm_eps":1e-06,
        \\   "rope_parameters":{"rope_theta":1000000,"rope_type":"default"},
        \\   "max_position_embeddings":65536,"tie_word_embeddings":true}}
    ;
    const specials = [_]tokenizer_file.TokenSpecials{
        .{ .name = "<|endoftext|>", .id = 151643 },
        .{ .name = "<|im_start|>", .id = 151644 },
        .{ .name = "<|im_end|>", .id = 151645 },
        .{ .name = "<|audio_start|>", .id = 151669 },
        .{ .name = "<|audio_end|>", .id = 151670 },
        .{ .name = "<|audio_pad|>", .id = 151676 },
        .{ .name = "<asr_text>", .id = 151704 },
    };
    const generation = "{\"eos_token_id\":[151643,151645],\"pad_token_id\":151643}";
    const token_config =
        \\{"audio_bos_token":"<|audio_start|>","audio_eos_token":"<|audio_end|>",
        \\ "audio_token":"<|audio_pad|>"}
    ;

    var diagnostics: Diagnostics = .{};
    const from_vllm = try parse(arena, .{
        .config_json = vllm,
        .generation_json = generation,
        .specials = &specials,
    }, &diagnostics);
    const from_native = try parse(arena, .{
        .config_json = native,
        .generation_json = generation,
        .tokenizer_config_json = token_config,
        .specials = &specials,
    }, &diagnostics);

    // The two checkpoints describe one model, so their runtime configurations
    // must be byte-identical: a difference here would mean the two variants
    // load different weights.
    try std.testing.expectEqualSlices(
        u8,
        std.mem.asBytes(&from_vllm.config),
        std.mem.asBytes(&from_native.config),
    );
    try std.testing.expect(from_vllm.output_is_tied);
    try std.testing.expect(from_native.output_is_tied);
    try std.testing.expectEqualStrings("bfloat16", from_vllm.source_dtype);
    try std.testing.expectEqualStrings("bfloat16", from_native.source_dtype);

    const config = from_vllm.config;
    try std.testing.expectEqual(@as(u32, 896), config.audio_d_model);
    try std.testing.expectEqual(@as(u32, 18), config.audio_layers);
    try std.testing.expectEqual(@as(u32, 28), config.text_layers);
    try std.testing.expectEqual(@as(u32, 13), config.audio_max_position_steps);
    try std.testing.expectEqual(@as(f32, 1e-5), config.audio_layer_norm_eps);
    try std.testing.expectEqual(@as(u32, 16384), config.max_positions);
    try std.testing.expectEqual(max_decode_tokens_default, config.max_decode_tokens);
    try std.testing.expectEqual(@as(u32, 65536), from_vllm.config_positions_advertised);
    // Ids the configuration supplies directly.
    try std.testing.expectEqual(@as(u32, 151669), config.token_audio_start);
    try std.testing.expectEqual(@as(u32, 151670), config.token_audio_end);
    try std.testing.expectEqual(@as(u32, 151676), config.token_audio_pad);
    // Ids that only the tokenizer files know.
    try std.testing.expectEqual(@as(u32, 151644), config.token_im_start);
    try std.testing.expectEqual(@as(u32, 151645), config.token_im_end);
    try std.testing.expectEqual(@as(u32, 151643), config.token_endoftext);
    try std.testing.expectEqual(@as(u32, 151704), config.token_asr_text);
    try std.testing.expectEqual(@as(u32, 151643), config.token_eos_primary);
    try std.testing.expectEqual(@as(u32, 151645), config.token_eos_secondary);
    try std.testing.expectEqual(@as(u32, 151643), config.token_pad);
    try std.testing.expectEqualStrings("", diagnostics.field);
}

test "fields the configuration must supply are reported by name" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostics: Diagnostics = .{};

    const without_text =
        \\{"model_type":"qwen3_asr",
        \\ "audio_config":{"d_model":896,"encoder_layers":18,"encoder_attention_heads":14,
        \\   "encoder_ffn_dim":3584,"downsample_hidden_size":480,"n_window":50,
        \\   "n_window_infer":800,"output_dim":1024,"num_mel_bins":128}}
    ;
    try std.testing.expectError(Error.MissingTextConfig, parse(arena, .{
        .config_json = without_text,
    }, &diagnostics));
    try std.testing.expectEqualStrings("text_config", diagnostics.field);

    const without_audio =
        \\{"model_type":"qwen3_asr","text_config":{"hidden_size":1024,"num_hidden_layers":28,
        \\ "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,
        \\ "intermediate_size":3072,"vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000}}
    ;
    diagnostics = .{};
    try std.testing.expectError(Error.MissingAudioConfig, parse(arena, .{
        .config_json = without_audio,
    }, &diagnostics));
    try std.testing.expectEqualStrings("audio_config", diagnostics.field);

    // A mistyped field is named and flagged, not silently defaulted.
    const mistyped =
        \\{"model_type":"qwen3_asr",
        \\ "audio_config":{"d_model":"896","encoder_layers":18,"encoder_attention_heads":14,
        \\   "encoder_ffn_dim":3584,"downsample_hidden_size":480,"n_window":50,
        \\   "n_window_infer":800,"output_dim":1024,"num_mel_bins":128},
        \\ "text_config":{"hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":16,
        \\   "num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,
        \\   "vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000}}
    ;
    diagnostics = .{};
    try std.testing.expectError(Error.MissingField, parse(arena, .{
        .config_json = mistyped,
    }, &diagnostics));
    try std.testing.expectEqualStrings("audio_config.d_model", diagnostics.field);
    try std.testing.expect(diagnostics.mistyped);

    // A missing layer count is not a zero-layer model.
    const without_layers =
        \\{"model_type":"qwen3_asr",
        \\ "audio_config":{"d_model":896,"encoder_attention_heads":14,"encoder_ffn_dim":3584,
        \\   "downsample_hidden_size":480,"n_window":50,"n_window_infer":800,"output_dim":1024,
        \\   "num_mel_bins":128},
        \\ "text_config":{"hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":16,
        \\   "num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,
        \\   "vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000}}
    ;
    diagnostics = .{};
    try std.testing.expectError(Error.MissingField, parse(arena, .{
        .config_json = without_layers,
    }, &diagnostics));
    try std.testing.expectEqualStrings("audio_config.encoder_layers", diagnostics.field);
}

test "geometry the runtime cannot represent is rejected before a single tensor is read" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostics: Diagnostics = .{};

    const text = "\"text_config\":{\"hidden_size\":1024,\"num_hidden_layers\":28," ++
        "\"num_attention_heads\":16,\"num_key_value_heads\":8,\"head_dim\":128," ++
        "\"intermediate_size\":3072,\"vocab_size\":151936,\"rms_norm_eps\":1e-06," ++
        "\"rope_theta\":1000000}";

    // 896 is not a whole number of 13-head groups.
    const mismatch = try std.fmt.allocPrint(arena, "{{\"model_type\":\"qwen3_asr\"," ++
        "\"audio_config\":{{\"d_model\":896,\"encoder_layers\":18," ++
        "\"encoder_attention_heads\":13,\"encoder_ffn_dim\":3584," ++
        "\"downsample_hidden_size\":480,\"n_window\":50,\"n_window_infer\":800," ++
        "\"output_dim\":1024,\"num_mel_bins\":128}},{s}}}", .{text});
    try std.testing.expectError(Error.HeadGeometryMismatch, parse(arena, .{
        .config_json = mismatch,
    }, &diagnostics));

    // Zero heads is not a model with no attention.
    const zero_heads = try std.fmt.allocPrint(arena, "{{\"model_type\":\"qwen3_asr\"," ++
        "\"audio_config\":{{\"d_model\":896,\"encoder_layers\":18," ++
        "\"encoder_attention_heads\":0,\"encoder_ffn_dim\":3584," ++
        "\"downsample_hidden_size\":480,\"n_window\":50,\"n_window_infer\":800," ++
        "\"output_dim\":1024,\"num_mel_bins\":128}},{s}}}", .{text});
    try std.testing.expectError(Error.ZeroHeadCount, parse(arena, .{
        .config_json = zero_heads,
    }, &diagnostics));

    // The audio tower is not a grouped-query model.
    const grouped = try std.fmt.allocPrint(arena, "{{\"model_type\":\"qwen3_asr\"," ++
        "\"audio_config\":{{\"d_model\":896,\"encoder_layers\":18," ++
        "\"encoder_attention_heads\":14,\"num_key_value_heads\":7," ++
        "\"encoder_ffn_dim\":3584,\"downsample_hidden_size\":480,\"n_window\":50," ++
        "\"n_window_infer\":800,\"output_dim\":1024,\"num_mel_bins\":128}},{s}}}", .{text});
    try std.testing.expectError(Error.HeadGeometryMismatch, parse(arena, .{
        .config_json = grouped,
    }, &diagnostics));

    // A sinusoidal table too short for one chunk.
    const short_table = try std.fmt.allocPrint(arena, "{{\"model_type\":\"qwen3_asr\"," ++
        "\"audio_config\":{{\"d_model\":896,\"encoder_layers\":18," ++
        "\"encoder_attention_heads\":14,\"encoder_ffn_dim\":3584," ++
        "\"downsample_hidden_size\":480,\"n_window\":50,\"n_window_infer\":800," ++
        "\"output_dim\":1024,\"num_mel_bins\":128,\"max_position_embeddings\":5}},{s}}}", .{text});
    try std.testing.expectError(Error.PositionTableTooShort, parse(arena, .{
        .config_json = short_table,
    }, &diagnostics));

    // An odd head width cannot be handled by the runtime's vector kernels.
    const odd_head_dim = try std.fmt.allocPrint(arena, "{{\"model_type\":\"qwen3_asr\"," ++
        "\"audio_config\":{{\"d_model\":896,\"encoder_layers\":18," ++
        "\"encoder_attention_heads\":14,\"encoder_ffn_dim\":3584," ++
        "\"downsample_hidden_size\":480,\"n_window\":50,\"n_window_infer\":800," ++
        "\"output_dim\":1024,\"num_mel_bins\":128}}," ++
        "\"text_config\":{{\"hidden_size\":1024,\"num_hidden_layers\":28," ++
        "\"num_attention_heads\":16,\"num_key_value_heads\":8,\"head_dim\":127," ++
        "\"intermediate_size\":3072,\"vocab_size\":151936,\"rms_norm_eps\":1e-06," ++
        "\"rope_theta\":1000000}}}}", .{});
    try std.testing.expectError(
        model_config.Error.InvalidHeadGeometry,
        parse(arena, .{ .config_json = odd_head_dim }, &diagnostics),
    );
}

test "the model type and dtype gate the conversion" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostics: Diagnostics = .{};

    try std.testing.expectError(Error.MissingModelType, parse(arena, .{
        .config_json = "{\"audio_config\":{},\"text_config\":{}}",
    }, &diagnostics));

    diagnostics = .{};
    try std.testing.expectError(Error.UnknownModelType, parse(arena, .{
        .config_json = "{\"model_type\":\"qwen2_audio\"}",
    }, &diagnostics));

    // A dtype that is not a float means the checkpoint is already quantized.
    const quantized = "{\"model_type\":\"qwen3_asr\",\"dtype\":\"int8\",\"audio_config\":" ++
        "{\"d_model\":896,\"encoder_layers\":18,\"encoder_attention_heads\":14," ++
        "\"encoder_ffn_dim\":3584,\"downsample_hidden_size\":480,\"n_window\":50," ++
        "\"n_window_infer\":800,\"output_dim\":1024,\"num_mel_bins\":128}," ++
        "\"text_config\":{\"hidden_size\":1024,\"num_hidden_layers\":28," ++
        "\"num_attention_heads\":16,\"num_key_value_heads\":8,\"head_dim\":128," ++
        "\"intermediate_size\":3072,\"vocab_size\":151936,\"rms_norm_eps\":1e-06," ++
        "\"rope_theta\":1000000}}";
    diagnostics = .{};
    try std.testing.expectError(Error.UnsupportedDtype, parse(arena, .{
        .config_json = quantized,
    }, &diagnostics));
    try std.testing.expectEqualStrings("dtype", diagnostics.field);

    // A configuration that is not JSON at all.
    try std.testing.expectError(Error.MalformedJson, parse(arena, .{
        .config_json = "{",
    }, &diagnostics));
}

test "special token ids are resolved from the files that carry them" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostics: Diagnostics = .{};

    // The native export has no audio marker ids: they come from
    // `tokenizer_config.json` naming the tokens, and from the tokenizer's added
    // tokens carrying the ids.
    const native =
        \\{"model_type":"qwen3_asr","dtype":"bfloat16","audio_token_id":151676,
        \\ "eos_token_id":[151643,151645],"pad_token_id":151645,
        \\ "audio_config":{"d_model":896,"encoder_layers":18,"encoder_attention_heads":14,
        \\   "encoder_ffn_dim":3584,"downsample_hidden_size":480,"n_window":50,
        \\   "n_window_infer":800,"output_dim":1024,"num_mel_bins":128,
        \\   "max_position_embeddings":13},
        \\ "text_config":{"hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":16,
        \\   "num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,
        \\   "vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000}}
    ;
    const token_config =
        \\{"audio_bos_token":"<|audio_start|>","audio_eos_token":"<|audio_end|>"}
    ;
    const specials = [_]tokenizer_file.TokenSpecials{
        .{ .name = "<|endoftext|>", .id = 151643 },
        .{ .name = "<|im_start|>", .id = 151644 },
        .{ .name = "<|im_end|>", .id = 151645 },
        .{ .name = "<|audio_start|>", .id = 151669 },
        .{ .name = "<|audio_end|>", .id = 151670 },
        .{ .name = "<|audio_pad|>", .id = 151676 },
        .{ .name = "<asr_text>", .id = 151704 },
    };
    const parsed = try parse(arena, .{
        .config_json = native,
        .tokenizer_config_json = token_config,
        .specials = &specials,
    }, &diagnostics);
    try std.testing.expectEqual(@as(u32, 151669), parsed.config.token_audio_start);
    try std.testing.expectEqual(@as(u32, 151670), parsed.config.token_audio_end);
    try std.testing.expectEqual(@as(u32, 151676), parsed.config.token_audio_pad);

    // Without the tokenizer's added tokens nothing can supply the audio markers.
    diagnostics = .{};
    try std.testing.expectError(Error.MissingAudioStartToken, parse(arena, .{
        .config_json = native,
        .tokenizer_config_json = token_config,
    }, &diagnostics));

    // With the tokenizer's names but no tokens behind them, the audio start
    // marker is the first id that cannot be resolved.
    diagnostics = .{};
    try std.testing.expectError(Error.MissingAudioStartToken, parse(arena, .{
        .config_json = native,
        .tokenizer_config_json = token_config,
    }, &diagnostics));

    // With the audio ids in the configuration but no tokenizer tokens at all,
    // the prompt marker is the first one that cannot be resolved: it lives only
    // in the tokenizer.
    const with_audio_ids =
        \\{"model_type":"qwen3_asr","dtype":"bfloat16","audio_token_id":151676,
        \\ "audio_start_token_id":151669,"audio_end_token_id":151670,
        \\ "eos_token_id":[151643,151645],"pad_token_id":151643,
        \\ "audio_config":{"d_model":896,"encoder_layers":18,"encoder_attention_heads":14,
        \\   "encoder_ffn_dim":3584,"downsample_hidden_size":480,"n_window":50,
        \\   "n_window_infer":800,"output_dim":1024,"num_mel_bins":128,
        \\   "max_position_embeddings":13},
        \\ "text_config":{"hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":16,
        \\   "num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,
        \\   "vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000}}
    ;
    diagnostics = .{};
    try std.testing.expectError(Error.MissingImStartToken, parse(arena, .{
        .config_json = with_audio_ids,
    }, &diagnostics));

    // An empty end-of-sequence list is a configuration error, not an empty set.
    const empty_eos =
        \\{"model_type":"qwen3_asr","dtype":"bfloat16","audio_token_id":151676,
        \\ "eos_token_id":[],
        \\ "audio_config":{"d_model":896,"encoder_layers":18,"encoder_attention_heads":14,
        \\   "encoder_ffn_dim":3584,"downsample_hidden_size":480,"n_window":50,
        \\   "n_window_infer":800,"output_dim":1024,"num_mel_bins":128,
        \\   "max_position_embeddings":13},
        \\ "text_config":{"hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":16,
        \\   "num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,
        \\   "vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000}}
    ;
    diagnostics = .{};
    try std.testing.expectError(Error.MissingEosToken, parse(arena, .{
        .config_json = empty_eos,
        .specials = &specials,
    }, &diagnostics));
}

test "the released checkpoints' configurations parse when they are present" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    // The named checkpoints are downloads, so this test checks whichever are
    // present and skips the rest rather than failing the suite. When they are
    // present it also proves the two variants describe one model, since they
    // are the same weights in two spellings.
    const paths = [_][]const u8{
        "models/Qwen3-ASR-0.6B",
        "models/Qwen3-ASR-0.6B-hf",
    };
    var previous: ?model_config.Config = null;
    for (paths) |path| {
        var dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch continue;
        defer dir.close(io);

        var token_diagnostics: tokenizer_file.Diagnostics = .{};
        const tokens = tokenizer_file.read(arena, io, dir, &token_diagnostics) catch |err| {
            // The checkpoint may still be downloading; that is not a failure of
            // this module.
            if (err == error.FileNotFound) continue;
            return err;
        };
        const config_json = try dir.readFileAlloc(io, "config.json", arena, .limited(1 << 20));
        const generation = readTextFile(arena, io, dir, "generation_config.json");
        const token_config = readTextFile(arena, io, dir, "tokenizer_config.json");

        var diagnostics: Diagnostics = .{};
        const parsed = try parse(arena, .{
            .config_json = config_json,
            .generation_json = generation,
            .tokenizer_config_json = token_config,
            .specials = tokens.specials,
        }, &diagnostics);

        try std.testing.expectEqual(@as(u32, 896), parsed.config.audio_d_model);
        try std.testing.expectEqual(@as(u32, 18), parsed.config.audio_layers);
        try std.testing.expectEqual(@as(u32, 14), parsed.config.audio_attention_heads);
        try std.testing.expectEqual(@as(u32, 3584), parsed.config.audio_ffn_dim);
        try std.testing.expectEqual(@as(u32, 1024), parsed.config.text_hidden_size);
        try std.testing.expectEqual(@as(u32, 28), parsed.config.text_layers);
        try std.testing.expectEqual(@as(u32, 128), parsed.config.text_head_dim);
        try std.testing.expectEqual(@as(u32, 151936), parsed.config.vocab_size);
        try std.testing.expectEqual(@as(f32, 1e-6), parsed.config.text_rms_norm_eps);
        try std.testing.expectEqual(@as(f32, 1e6), parsed.config.rope_theta);
        try std.testing.expectEqual(@as(u32, 13), parsed.config.audio_max_position_steps);
        try std.testing.expectEqualStrings("bfloat16", parsed.source_dtype);
        // Both released variants are weight-tied.
        try std.testing.expect(parsed.output_is_tied);
        // The audio markers resolve even though the native configuration does
        // not carry their ids.
        try std.testing.expectEqual(@as(u32, 151669), parsed.config.token_audio_start);
        try std.testing.expectEqual(@as(u32, 151670), parsed.config.token_audio_end);
        try std.testing.expectEqual(@as(u32, 151676), parsed.config.token_audio_pad);
        try std.testing.expectEqual(@as(u32, 151644), parsed.config.token_im_start);
        try std.testing.expectEqual(@as(u32, 151645), parsed.config.token_im_end);
        try std.testing.expectEqual(@as(u32, 151704), parsed.config.token_asr_text);

        if (previous) |earlier| {
            try expectSameGeometry(earlier, parsed.config);
        } else {
            previous = parsed.config;
        }
    }
}

/// The two released checkpoints differ in their `pad_token_id` (151643 against
/// 151645, both defensible), so the comparison is over the geometry, which must
/// agree exactly: a difference there would mean the two variants would load
/// different weights.
fn expectSameGeometry(expected: model_config.Config, actual: model_config.Config) !void {
    try std.testing.expectEqual(expected.audio_d_model, actual.audio_d_model);
    try std.testing.expectEqual(expected.audio_layers, actual.audio_layers);
    try std.testing.expectEqual(expected.audio_attention_heads, actual.audio_attention_heads);
    try std.testing.expectEqual(expected.audio_ffn_dim, actual.audio_ffn_dim);
    try std.testing.expectEqual(expected.audio_output_dim, actual.audio_output_dim);
    try std.testing.expectEqual(expected.audio_max_position_steps, actual.audio_max_position_steps);
    try std.testing.expectEqual(expected.text_hidden_size, actual.text_hidden_size);
    try std.testing.expectEqual(expected.text_layers, actual.text_layers);
    try std.testing.expectEqual(expected.text_attention_heads, actual.text_attention_heads);
    try std.testing.expectEqual(expected.text_key_value_heads, actual.text_key_value_heads);
    try std.testing.expectEqual(expected.text_head_dim, actual.text_head_dim);
    try std.testing.expectEqual(expected.text_ffn_dim, actual.text_ffn_dim);
    try std.testing.expectEqual(expected.vocab_size, actual.vocab_size);
    try std.testing.expectEqual(expected.audio_layer_norm_eps, actual.audio_layer_norm_eps);
    try std.testing.expectEqual(expected.text_rms_norm_eps, actual.text_rms_norm_eps);
    try std.testing.expectEqual(expected.rope_theta, actual.rope_theta);
    try std.testing.expectEqual(expected.max_positions, actual.max_positions);
    try std.testing.expectEqual(expected.max_decode_tokens, actual.max_decode_tokens);
    try std.testing.expectEqual(expected.token_audio_start, actual.token_audio_start);
    try std.testing.expectEqual(expected.token_audio_end, actual.token_audio_end);
    try std.testing.expectEqual(expected.token_audio_pad, actual.token_audio_pad);
}

/// Reads a file that may legitimately be absent, such as a generation
/// configuration a checkpoint does not ship.
fn readTextFile(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    name: []const u8,
) ?[]u8 {
    return dir.readFileAlloc(io, name, arena, .limited(1 << 20)) catch return null;
}
