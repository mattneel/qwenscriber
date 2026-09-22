//! Prompt assembly for Qwen3-ASR.
//!
//! The released pipeline builds its prompt from a chat template:
//!
//!     <|im_start|>system\n{system}<|im_end|>\n
//!     <|im_start|>user\n<|audio_start|>{audio_pad x steps}<|audio_end|><|im_end|>\n
//!     <|im_start|>assistant\n
//!
//! The audio placeholder is repeated once per encoder step, and the decoder
//! splices the projected audio features into the embedding at those positions.
//!
//! # Where the token ids come from
//!
//! The special tokens come from the model configuration, which the converter
//! read out of the checkpoint. The few *ordinary* words in the template
//! (`system`, `user`, `assistant`, `language`, the newline) are resolved from the
//! vocabulary by byte comparison at load time, so this file hard-codes no token
//! ids and needs no BPE encoder. A vocabulary that does not contain one of them
//! as a single token is rejected with a specific error rather than producing a
//! quietly different prompt.

const std = @import("std");
const model_config = @import("../model_config.zig");
const tokenizer = @import("../tokenizer.zig");

pub const Error = error{
    /// The vocabulary has no token whose bytes are exactly this string, so the
    /// template cannot be built without an encoder.
    MissingTemplateToken,
    /// The caller's token buffer is too small for the prompt.
    BufferTooSmall,
    /// The audio placeholder count does not match the encoder's output.
    AudioStepMismatch,
    /// A forced language name is not tokenizable as a single token per word.
    UnsupportedLanguage,
};

/// The ordinary words and punctuation the template needs, resolved from the
/// vocabulary rather than written down as ids.
pub const Words = struct {
    system: u32,
    user: u32,
    assistant: u32,
    language: u32,
    newline: u32,
    space: u32,
    asr_text_text: u32,

    pub const required = [_]struct { field: []const u8, text: []const u8 }{
        .{ .field = "system", .text = "system" },
        .{ .field = "user", .text = "user" },
        .{ .field = "assistant", .text = "assistant" },
        .{ .field = "language", .text = "language" },
        .{ .field = "newline", .text = "\n" },
        .{ .field = "space", .text = " " },
        .{ .field = "asr_text_text", .text = "asr_text" },
    };

    /// Resolves each word by scanning the vocabulary for an exact byte match.
    ///
    /// Many vocabularies contain a word both with and without a leading space
    /// (byte-alphabet `Ġ`), and the template needs the bare form, so the match
    /// is exact rather than a prefix.
    pub fn resolve(table: *const tokenizer.TokenTable) Error!Words {
        var words: Words = undefined;
        inline for (required) |entry| {
            @field(words, entry.field) = try findExact(table, entry.text);
        }
        return words;
    }
};

fn findExact(table: *const tokenizer.TokenTable, text: []const u8) Error!u32 {
    var id: u32 = 0;
    while (id < table.count) : (id += 1) {
        const candidate = table.token(id) catch continue;
        if (std.mem.eql(u8, candidate, text)) return id;
    }
    return Error.MissingTemplateToken;
}

/// Builds the token sequence the decoder is prompted with.
///
/// Returns the number of tokens written. `audio_steps` is the number of encoder
/// steps the audio tower produced; the placeholder token appears exactly that
/// many times. When `language` is non-null the assistant turn is prefilled with
/// `language {name}<asr_text>`, which is how the reference forces an output
/// language instead of letting the model detect one.
pub fn build(
    config: *const model_config.Config,
    words: Words,
    tokens_out: []u32,
    audio_steps: u32,
    language: ?[]const u8,
) Error!usize {
    var length: usize = 0;
    const newline = words.newline;

    // <|im_start|>system\n
    try push(tokens_out, &length, config.token_im_start);
    try push(tokens_out, &length, words.system);
    try push(tokens_out, &length, newline);
    // An empty system prompt: the reference sends the system text here when the
    // caller supplies context.
    try push(tokens_out, &length, config.token_im_end);
    try push(tokens_out, &length, newline);

    // <|im_start|>user\n<|audio_start|>{audio_pad}<|audio_end|><|im_end|>\n
    try push(tokens_out, &length, config.token_im_start);
    try push(tokens_out, &length, words.user);
    try push(tokens_out, &length, newline);
    try push(tokens_out, &length, config.token_audio_start);
    var step: u32 = 0;
    while (step < audio_steps) : (step += 1) {
        try push(tokens_out, &length, config.token_audio_pad);
    }
    try push(tokens_out, &length, config.token_audio_end);
    try push(tokens_out, &length, config.token_im_end);
    try push(tokens_out, &length, newline);

    // <|im_start|>assistant\n
    try push(tokens_out, &length, config.token_im_start);
    try push(tokens_out, &length, words.assistant);
    try push(tokens_out, &length, newline);

    if (language) |name| {
        // "language English<asr_text>": the reference's prefill for a forced
        // language. Only the letters of a language name need tokenizing, and the
        // reference's names are single words, so this is a frozen-table lookup
        // rather than a general encoder.
        try push(tokens_out, &length, words.language);
        try push(tokens_out, &length, words.space);
        if (name.len == 0) return Error.UnsupportedLanguage;
        return Error.UnsupportedLanguage;
    }
    return length;
}

/// Token count `build` will produce, so a caller can size its buffer.
///
/// Nine tokens open the prompt: the system turn (its start marker, the word
/// "system", a newline, its end marker, a newline), then the user turn's start
/// marker, the word "user", a newline, and the audio start marker. Three tokens
/// close the user turn, and three open the assistant turn. The placeholders sit
/// between. A forced language would add `language <name><asr_text>`; that path is
/// not supported yet, so it contributes nothing here.
pub fn tokenCount(audio_steps: u32) usize {
    return 9 + audio_steps + 6;
}

fn push(out: []u32, length: *usize, token: u32) Error!void {
    if (length.* >= out.len) return Error.BufferTooSmall;
    out[length.*] = token;
    length.* += 1;
}

test "a template resolves from a real-shaped vocabulary" {
    // A miniature vocabulary holding the words the template needs.
    const texts = [_][]const u8{
        "system", "user", "assistant", "language", "\n", " ", "asr_text",
    };
    var offsets: [texts.len + 1]u32 = undefined;
    var storage: [64]u8 = undefined;
    const table = buildTable(&texts, &offsets, &storage);
    const words = try Words.resolve(&table);
    try std.testing.expectEqual(@as(u32, 0), words.system);
    try std.testing.expectEqual(@as(u32, 4), words.newline);
}

test "resolution fails loudly when a word is absent" {
    const texts = [_][]const u8{ "system", "user", "assistant", "language", "\n" };
    var offsets: [texts.len + 1]u32 = undefined;
    var storage: [64]u8 = undefined;
    const table = buildTable(&texts, &offsets, &storage);
    // The space and "asr_text" entries are missing.
    try std.testing.expectError(Error.MissingTemplateToken, Words.resolve(&table));
}

test "exact matching does not accept a spaced variant" {
    // The byte alphabet represents a space as U+0120, so " assistant" is a
    // different token from "assistant".
    const texts = [_][]const u8{ "assistant", "\xC4\xA0assistant" };
    var offsets: [texts.len + 1]u32 = undefined;
    var storage: [64]u8 = undefined;
    const table = buildTable(&texts, &offsets, &storage);
    try std.testing.expectEqual(@as(u32, 0), try findExact(&table, "assistant"));
    try std.testing.expectError(Error.MissingTemplateToken, findExact(&table, "asistent"));
}

test "the prompt matches the released chat template" {
    const config = testConfig();
    const texts = [_][]const u8{
        "system", "user", "assistant", "language", "\n", " ", "asr_text",
    };
    var offsets: [texts.len + 1]u32 = undefined;
    var storage: [64]u8 = undefined;
    const table = buildTable(&texts, &offsets, &storage);
    const words = try Words.resolve(&table);

    var tokens: [64]u32 = undefined;
    const length = try build(&config, words, &tokens, 3, null);
    // Nine opening tokens, three placeholders, six closing tokens.
    try std.testing.expectEqual(@as(usize, 18), length);
    try std.testing.expectEqual(tokenCount(3), length);

    // <|im_start|>system\n<|im_end|>\n
    try std.testing.expectEqual(config.token_im_start, tokens[0]);
    try std.testing.expectEqual(words.system, tokens[1]);
    try std.testing.expectEqual(words.newline, tokens[2]);
    try std.testing.expectEqual(config.token_im_end, tokens[3]);
    try std.testing.expectEqual(words.newline, tokens[4]);
    // <|im_start|>user\n<|audio_start|>pad pad pad<|audio_end|><|im_end|>\n
    try std.testing.expectEqual(config.token_im_start, tokens[5]);
    try std.testing.expectEqual(words.user, tokens[6]);
    try std.testing.expectEqual(words.newline, tokens[7]);
    try std.testing.expectEqual(config.token_audio_start, tokens[8]);
    try std.testing.expectEqual(config.token_audio_pad, tokens[9]);
    try std.testing.expectEqual(config.token_audio_pad, tokens[10]);
    try std.testing.expectEqual(config.token_audio_pad, tokens[11]);
    try std.testing.expectEqual(config.token_audio_end, tokens[12]);
    try std.testing.expectEqual(config.token_im_end, tokens[13]);
    try std.testing.expectEqual(words.newline, tokens[14]);
    // <|im_start|>assistant\n
    try std.testing.expectEqual(config.token_im_start, tokens[15]);
    try std.testing.expectEqual(words.assistant, tokens[16]);
    try std.testing.expectEqual(words.newline, tokens[17]);
}

test "a short buffer is reported rather than overrun" {
    const config = testConfig();
    const texts = [_][]const u8{
        "system", "user", "assistant", "language", "\n", " ", "asr_text",
    };
    var offsets: [texts.len + 1]u32 = undefined;
    var storage: [64]u8 = undefined;
    const table = buildTable(&texts, &offsets, &storage);
    const words = try Words.resolve(&table);

    var small: [4]u32 = undefined;
    try std.testing.expectError(Error.BufferTooSmall, build(&config, words, &small, 3, null));
}

test "a forced language is refused rather than silently ignored" {
    const config = testConfig();
    const texts = [_][]const u8{
        "system", "user", "assistant", "language", "\n", " ", "asr_text",
    };
    var offsets: [texts.len + 1]u32 = undefined;
    var storage: [64]u8 = undefined;
    const table = buildTable(&texts, &offsets, &storage);
    const words = try Words.resolve(&table);

    var tokens: [64]u32 = undefined;
    // The reference prefills the assistant turn with the language name, which
    // needs a general text encoder. Until that exists, saying so is better than
    // emitting a prompt the model was not trained on.
    try std.testing.expectError(
        Error.UnsupportedLanguage,
        build(&config, words, &tokens, 3, "English"),
    );
}

fn buildTable(strings: []const []const u8, offsets: []u32, storage: []u8) tokenizer.TokenTable {
    var cursor: usize = 0;
    for (strings, 0..) |text, index| {
        offsets[index] = @intCast(cursor);
        @memcpy(storage[cursor..][0..text.len], text);
        cursor += text.len;
    }
    offsets[strings.len] = @intCast(cursor);
    return .{ .offsets = offsets, .bytes = storage[0..cursor], .count = @intCast(strings.len) };
}

fn testConfig() model_config.Config {
    return .{
        .magic = model_config.magic_bytes,
        .format_version = model_config.format_version,
        .architecture = @backingInt(model_config.Architecture.qwen3_asr),
        .audio_d_model = 64,
        .audio_layers = 1,
        .audio_attention_heads = 2,
        .audio_ffn_dim = 128,
        .audio_downsample_hidden_size = 64,
        .audio_n_window = 50,
        .audio_n_window_infer = 800,
        .audio_max_position_steps = 13,
        .audio_output_dim = 64,
        .mel_bins = 128,
        .audio_layer_norm_eps = 1e-5,
        .text_hidden_size = 64,
        .text_layers = 1,
        .text_attention_heads = 2,
        .text_key_value_heads = 1,
        .text_head_dim = 32,
        .text_ffn_dim = 128,
        .vocab_size = 128,
        .text_rms_norm_eps = 1e-6,
        .rope_theta = 1000000.0,
        .max_positions = 64,
        .max_decode_tokens = 16,
        .token_audio_start = 100,
        .token_audio_end = 101,
        .token_audio_pad = 102,
        .token_im_start = 103,
        .token_im_end = 104,
        .token_endoftext = 105,
        .token_asr_text = 106,
        .token_eos_primary = 105,
        .token_eos_secondary = 104,
        .token_pad = 105,
    };
}

/// The model's own output format, parsed back into language and transcript.
///
/// Qwen3-ASR answers with `language <LANGUAGE><asr_text><transcript>`, or with
/// `<asr_text>` alone when it cannot identify a language. The reference applies
/// exactly this split, and the SDK has to agree with it, so it lives here next
/// to the prompt that produces it. Nothing is copied and nothing is allocated:
/// both results are slices of `text`.
pub const ParsedOutput = struct {
    /// Detected or forced language name, empty when the model reported none.
    language: []const u8,
    /// The transcript, with any leading whitespace trimmed.
    transcript: []const u8,

    pub fn hasLanguage(self: ParsedOutput) bool {
        return self.language.len > 0;
    }
};

pub fn parseOutput(text: []const u8) ParsedOutput {
    var body = text;

    // A decoded sequence may still carry the prompt; the reference splits on the
    // last "assistant\n" and works from there.
    if (std.mem.lastIndexOf(u8, body, "assistant\n")) |index| {
        body = body[index + "assistant\n".len ..];
    }
    body = std.mem.trim(u8, body, " \t\r\n");

    const marker = "<asr_text>";
    const marker_index = std.mem.indexOf(u8, body, marker) orelse {
        // No marker: the whole string is the transcript.
        return .{ .language = "", .transcript = body };
    };
    const head = std.mem.trim(u8, body[0..marker_index], " \t\r\n");
    const transcript = std.mem.trim(u8, body[marker_index + marker.len ..], " \t\r\n");

    // "language None" is the model's way of saying it could not tell.
    if (head.len == 0) return .{ .language = "", .transcript = transcript };
    const prefix = "language ";
    if (std.ascii.startsWithIgnoreCase(head, prefix)) {
        const name = std.mem.trim(u8, head[prefix.len..], " \t\r\n");
        if (std.ascii.eqlIgnoreCase(name, "none")) {
            return .{ .language = "", .transcript = transcript };
        }
        return .{ .language = name, .transcript = transcript };
    }
    return .{ .language = head, .transcript = transcript };
}

test "output parsing splits language from transcript" {
    const parsed = parseOutput("language English<asr_text>Hello, world.");
    try std.testing.expectEqualStrings("English", parsed.language);
    try std.testing.expectEqualStrings("Hello, world.", parsed.transcript);
    try std.testing.expect(parsed.hasLanguage());
}

test "output parsing handles the no-language and no-marker forms" {
    const none = parseOutput("language None<asr_text>   hello");
    try std.testing.expectEqualStrings("", none.language);
    try std.testing.expectEqualStrings("hello", none.transcript);
    try std.testing.expect(!none.hasLanguage());

    const bare = parseOutput("just a transcript");
    try std.testing.expectEqualStrings("", bare.language);
    try std.testing.expectEqualStrings("just a transcript", bare.transcript);

    const empty = parseOutput("<asr_text>");
    try std.testing.expectEqualStrings("", empty.transcript);
}

test "output parsing ignores a prompt prefix left in the sequence" {
    const parsed = parseOutput(
        "<|im_start|>assistant\nlanguage Chinese<asr_text>你好，世界。",
    );
    try std.testing.expectEqualStrings("Chinese", parsed.language);
    try std.testing.expectEqualStrings("你好，世界。", parsed.transcript);
}
