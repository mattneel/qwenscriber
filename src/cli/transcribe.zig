//! `qwenscriber-transcribe`: the native end-to-end runner.
//!
//! This is the bring-up instrument. It loads a converted model directory, reads
//! a WAV file, and runs exactly the pipeline the browser will run: the same log-mel
//! frontend, the same audio tower, the same decoder, the same detokenizer. Its
//! output is what gets compared against the reference `transformers` pipeline.
//!
//! It is a tool, not a library: it prints a report and exits nonzero on failure,
//! so it can be used as a regression gate.
//!
//! Usage:
//!
//!     qwenscriber-transcribe --model <model dir> --audio <wav> [--max-tokens N]
//!                            [--dump <dir>] [--dump-logits] [--metrics <path>]

const std = @import("std");
const Io = std.Io;
const qw = @import("qwenscriber");
const wav = @import("wav.zig");
const fixture = @import("fixture.zig");
const host = @import("host");

const AudioTooLong = error{AudioTooLong};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.arena.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    const args = try init.minimal.args.toSlice(gpa);
    var options = Options{};
    try options.parse(args, out, io, gpa);
    if (options.help) {
        try printUsage(out);
        try out.flush();
        return;
    }

    const started = nowMs(io);
    const model_dir = try Io.Dir.cwd().openDir(io, options.model_path, .{ .iterate = true });
    defer model_dir.close(io);

    // Configuration first: it sizes everything else.
    const config_bytes = try readFile(io, gpa, model_dir, "config.bin", Io.Limit.limited(4096));
    const config = try qw.model_config.parse(config_bytes);
    try out.print(
        "model: audio {d}x{d} layers, text {d}x{d} layers, vocab {d}\n",
        .{
            config.audio_d_model,
            config.audio_layers,
            config.text_hidden_size,
            config.text_layers,
            config.vocab_size,
        },
    );

    // Shards are self-describing, so the runner does not need the manifest.
    var shard_bytes = std.ArrayList([]align(16) const u8).empty;
    var iterator = model_dir.iterate();
    while (try iterator.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".qw")) continue;
        const bytes = try readFileAligned(io, gpa, model_dir, entry.name, shard_limit);
        try shard_bytes.append(gpa, bytes);
        try out.print("shard: {s} ({d} MiB)\n", .{
            entry.name,
            bytes.len / (1024 * 1024),
        });
    }
    if (shard_bytes.items.len == 0) {
        try out.print("no .qw shards found in {s}\n", .{options.model_path});
        try out.flush();
        std.process.exit(2);
    }

    var model = try qw.qwen3_asr.model.Model.loadFromShardBytes(
        gpa,
        config.*,
        shard_bytes.items,
    );
    const load_ms = nowMs(io) - started;
    try out.print("loaded in {d} ms, key/value cache {d} MiB\n", .{
        load_ms,
        model.cacheBytes() / (1024 * 1024),
    });

    // Vocabulary, for both the template words and detokenization.
    // The vocabulary's offset array is read as u32, so it needs the same
    // alignment guarantee the shards get.
    const token_bytes = try readFileAligned(io, gpa, model_dir, "tokens.bin", token_limit);
    const vocabulary = try parseVocabulary(token_bytes);
    const words = try qw.qwen3_asr.prompt.Words.resolve(&vocabulary);

    // Audio.
    const wav_bytes = try readFile(io, gpa, Io.Dir.cwd(), options.audio_path, audio_limit);
    const frames_in_file = try wav.frameCount(wav_bytes);
    const samples_buffer = try gpa.alloc(f32, frames_in_file);
    const audio = try wav.decode(wav_bytes, samples_buffer);
    try out.print("audio: {d} samples ({d:.2} s)\n", .{
        audio.samples.len,
        @as(f64, @floatFromInt(audio.samples.len)) / 16000.0,
    });

    // The reference zero-pads clips shorter than 8000 samples before
    // transforming them, which changes how many frames it sees.
    const padded = try padToMinimum(io, gpa, audio.samples, qw.mel.min_samples);

    const preprocessing_started = nowMs(io);
    const frames = qw.mel.framesForSamples(padded.len);
    const log_mel = try gpa.alloc(f32, @as(usize, qw.mel.mel_bins) * frames);
    const global_max_log = try qw.mel.compute(padded, log_mel, frames);
    const projected_steps = config.packedStepCount(@intCast(frames));
    const mel_ms = nowMs(io) - preprocessing_started;
    try out.print(
        "mel: {d} frames in {d} ms (global max log {d:.6}), encoder steps {d}\n",
        .{ frames, mel_ms, global_max_log, projected_steps },
    );

    var dump = try Dump.open(io, gpa, options.dump_path);
    defer dump.close(io);

    const projected = try gpa.alloc(f32, @as(usize, projected_steps) * config.text_hidden_size);
    const encoder_started = nowMs(io);
    // The reference dumps the convolution stack's output on its own, so the dump
    // is written at the same boundary: comparing it separates a convolution
    // defect from an audio tower block defect.
    const steps = try model.encodeConvStage(log_mel, @intCast(frames));
    try dump.write(
        io,
        "audio_conv_out.f32",
        &.{ config.audioChunkSteps(), config.audio_d_model },
        model.encodedSteps(steps)[0 .. config.audioChunkSteps() * config.audio_d_model],
    );
    try model.encodeTowerStage(steps);
    const encoder_ms = nowMs(io) - encoder_started;
    try out.print("encoder: {d} steps in {d} ms\n", .{
        steps,
        encoder_ms,
    });
    try dump.write(io, "audio_encoded.f32", &.{ steps, config.audio_d_model }, model.encodedSteps(steps));
    try dump.write(io, "input_features.f32", &.{ config.mel_bins, @intCast(frames) }, log_mel);
    try model.projectSteps(steps, projected);
    try dump.write(
        io,
        "audio_projected.f32",
        &.{ steps, config.text_hidden_size },
        projected[0 .. @as(usize, steps) * config.text_hidden_size],
    );

    // Prompt, with one placeholder per encoder step.
    const prompt_tokens = try gpa.alloc(u32, qw.qwen3_asr.prompt.tokenCount(steps));
    const prompt_length = try qw.qwen3_asr.prompt.build(
        config,
        words,
        prompt_tokens,
        steps,
        null,
    );
    try out.print("prompt: {d} tokens\n", .{prompt_length});
    const prompt_as_f32 = try gpa.alloc(f32, prompt_length);
    for (prompt_tokens[0..prompt_length], prompt_as_f32) |token, *value| {
        value.* = @floatFromInt(token);
    }
    try dump.write(io, "decoder_input_ids.f32", &.{@intCast(prompt_length)}, prompt_as_f32);

    var decoder = qw.qwen3_asr.decoder.Decoder.init(&model);
    const decode_started = nowMs(io);
    var next = try decoder.prefill(prompt_tokens[0..prompt_length], projected, steps);
    if (options.dump_logits) {
        // The row the first generated token comes from, which is the only
        // decoding stage that can be compared against the reference's
        // `logits_step0.f32` fixture.
        try dump.write(
            io,
            "logits_step0.f32",
            &.{config.vocab_size},
            decoder.model.scratch.logits,
        );
    }
    const first_token_ms = nowMs(io) - decode_started;
    try out.print("first token in {d} ms\n", .{first_token_ms});

    // Generation is timed apart from the prompt's prefill: a user waits through
    // both, but only this phase scales with how long the answer is.
    const generation_started = nowMs(io);
    const generated = try gpa.alloc(u32, options.max_tokens);
    var generated_count: usize = 0;
    var token_count: u32 = 0;
    while (generated_count < generated.len) {
        if (next == config.token_eos_primary or next == config.token_eos_secondary) break;
        if (next == config.token_endoftext) break;
        generated[generated_count] = next;
        generated_count += 1;
        token_count += 1;
        if (generated_count == generated.len) break;
        next = try decoder.step(next);
    }

    const decode_ms = nowMs(io) - generation_started;
    const tokens_per_second = if (decode_ms > 0)
        @as(f64, @floatFromInt(token_count)) * 1000.0 / @as(f64, @floatFromInt(decode_ms))
    else
        0.0;
    try out.print("decode: {d} tokens in {d} ms ({d:.1} tokens/s)\n", .{
        token_count,
        decode_ms,
        tokens_per_second,
    });

    const generated_as_f32 = try gpa.alloc(f32, generated_count);
    for (generated[0..generated_count], generated_as_f32) |token, *value| {
        value.* = @floatFromInt(token);
    }
    try dump.write(io, "generated_ids.f32", &.{@intCast(generated_count)}, generated_as_f32);
    const text_buffer = try gpa.alloc(u8, generated_count * 8 + 64);
    const text_length = try qw.tokenizer.detokenize(
        &vocabulary,
        generated[0..generated_count],
        text_buffer,
    );
    const parsed = qw.qwen3_asr.prompt.parseOutput(text_buffer[0..text_length]);
    try out.print("language: {s}\n", .{if (parsed.hasLanguage()) parsed.language else "(none)"});
    try out.print("transcript: {s}\n", .{parsed.transcript});
    try out.print("generated ids: {any}\n", .{generated[0..generated_count]});
    const total_ms = nowMs(io) - started;
    const audio_ms = @divTrunc(@as(i64, @intCast(audio.samples.len)) * 1000, 16000);
    try out.print(
        "total: {d} ms for {d:.2} s of audio\n",
        .{
            total_ms,
            @as(f64, @floatFromInt(audio.samples.len)) / 16000.0,
        },
    );
    if (options.metrics_path.len != 0) {
        const identity = host.metrics.readIdentity(io, gpa, model_dir);
        var shard_bytes_total: u64 = 0;
        for (shard_bytes.items) |shard| shard_bytes_total += shard.len;
        var cpu_buffer: [128]u8 = undefined;
        try host.metrics.writeFile(io, options.metrics_path, .{
            .model_id = identity.model_id,
            .quantization = identity.quantization,
            .bits_per_weight = identity.bits_per_weight,
            .tensors = identity.tensors,
            .parameters = identity.parameters,
            .payload_bytes = identity.payload_bytes,
            .shard_count = @intCast(shard_bytes.items.len),
            .shard_bytes = shard_bytes_total,
            .audio_d_model = config.audio_d_model,
            .audio_layers = config.audio_layers,
            .text_hidden_size = config.text_hidden_size,
            .text_layers = config.text_layers,
            .vocab_size = config.vocab_size,
            .kv_cache_bytes = model.cacheBytes(),
            .frames = @intCast(frames),
            .encoder_steps = @intCast(steps),
            .prompt_tokens = @intCast(prompt_length),
            .generated_tokens = token_count,
            .load_ms = load_ms,
            .mel_ms = mel_ms,
            .encoder_ms = encoder_ms,
            .first_token_ms = first_token_ms,
            .decode_ms = decode_ms,
            .total_ms = total_ms,
            .audio_ms = audio_ms,
        }, host.metrics.detectEnvironment(io, &cpu_buffer));
        try out.print("metrics: {s}\n", .{options.metrics_path});
    }
    try out.flush();
}

/// Milliseconds on a clock that cannot jump backwards, for the report's
/// timings only.
fn nowMs(io: Io) i64 {
    return Io.Timestamp.now(io, .awake).toMilliseconds();
}

const shard_limit = Io.Limit.limited(2 * 1024 * 1024 * 1024);
const token_limit = Io.Limit.limited(64 * 1024 * 1024);
const audio_limit = Io.Limit.limited(512 * 1024 * 1024);

const Options = struct {
    model_path: []const u8 = "models/qwen3-asr-0.6b-q4",
    audio_path: []const u8 = "",
    max_tokens: usize = 256,
    dump_path: []const u8 = "",
    dump_logits: bool = false,
    /// Where to write the run's metrics record. Empty writes none.
    metrics_path: []const u8 = "",
    help: bool = false,

    fn parse(
        self: *Options,
        args: []const [:0]const u8,
        out: *Io.Writer,
        io: Io,
        gpa: std.mem.Allocator,
    ) !void {
        _ = io;
        _ = gpa;
        var index: usize = 1;
        while (index < args.len) : (index += 1) {
            const arg = args[index];
            if (std.mem.eql(u8, arg, "--help")) {
                self.help = true;
            } else if (std.mem.eql(u8, arg, "--model")) {
                index += 1;
                if (index >= args.len) return fail(out, "missing value for --model");
                self.model_path = args[index];
            } else if (std.mem.eql(u8, arg, "--audio")) {
                index += 1;
                if (index >= args.len) return fail(out, "missing value for --audio");
                self.audio_path = args[index];
            } else if (std.mem.eql(u8, arg, "--dump")) {
                index += 1;
                if (index >= args.len) return fail(out, "missing value for --dump");
                self.dump_path = args[index];
            } else if (std.mem.eql(u8, arg, "--dump-logits")) {
                self.dump_logits = true;
            } else if (std.mem.eql(u8, arg, "--metrics")) {
                index += 1;
                if (index >= args.len) return fail(out, "missing value for --metrics");
                self.metrics_path = args[index];
            } else if (std.mem.eql(u8, arg, "--max-tokens")) {
                index += 1;
                if (index >= args.len) return fail(out, "missing value for --max-tokens");
                self.max_tokens = std.fmt.parseInt(usize, args[index], 10) catch
                    return fail(out, "invalid --max-tokens");
            } else {
                return fail(out, "unknown argument");
            }
        }
        if (!self.help and self.audio_path.len == 0) {
            return fail(out, "--audio is required");
        }
    }
};

fn fail(out: *Io.Writer, message: []const u8) error{InvalidArguments} {
    out.print("error: {s}\n", .{message}) catch {};
    out.flush() catch {};
    return error.InvalidArguments;
}

fn printUsage(out: *Io.Writer) !void {
    try out.print(
        \\usage: qwenscriber-transcribe --model <dir> --audio <wav> [options]
        \\
        \\  --model <dir>       converted model directory (config.bin, tokens.bin, *.qw)
        \\  --audio <wav>       16 kHz mono WAV file
        \\  --max-tokens <n>    generation limit (default 256)
        \\  --dump <dir>        write intermediate tensors for reference comparison
        \\  --dump-logits       also write the first step's logits
        \\  --metrics <path>    write this run's measurements as a JSON record
        \\  --help              show this message
        \\
    , .{});
}

/// Writes intermediates in the `QWFIX001` container so a Python driver can
/// compare them against the reference pipeline stage by stage. A no-op when no
/// dump directory was requested.
const Dump = struct {
    dir: ?Io.Dir = null,

    fn open(io: Io, gpa: std.mem.Allocator, path: []const u8) !Dump {
        _ = gpa;
        if (path.len == 0) return .{};
        const cwd = Io.Dir.cwd();
        try cwd.createDirPath(io, path);
        const dir = try cwd.openDir(io, path, .{});
        return .{ .dir = dir };
    }

    fn write(
        self: *Dump,
        io: Io,
        name: []const u8,
        dims: []const u32,
        values: []const f32,
    ) !void {
        const dir = self.dir orelse return;
        try fixture.write(io, dir, name, dims, values);
    }

    fn close(self: *Dump, io: Io) void {
        if (self.dir) |dir| dir.close(io);
    }
};

/// Reads a whole file into 16-byte-aligned storage, which is what the container
/// parser requires before it will hand out tensor views.
fn readFileAligned(
    io: Io,
    gpa: std.mem.Allocator,
    dir: Io.Dir,
    path: []const u8,
    limit: Io.Limit,
) ![]align(16) const u8 {
    const bytes = try dir.readFileAllocOptions(io, path, gpa, limit, .@"16", null);
    return @alignCast(bytes);
}

fn readFile(io: Io, gpa: std.mem.Allocator, dir: Io.Dir, path: []const u8, limit: Io.Limit) ![]u8 {
    return dir.readFileAlloc(io, path, gpa, limit);
}

/// Zero-pads a clip up to the reference's minimum length, in a fresh buffer.
fn padToMinimum(
    io: Io,
    gpa: std.mem.Allocator,
    samples: []const f32,
    minimum: usize,
) ![]const f32 {
    _ = io;
    if (samples.len >= minimum) return samples;
    const padded = try gpa.alloc(f32, minimum);
    @memset(padded, 0.0);
    @memcpy(padded[0..samples.len], samples);
    return padded;
}

/// Parses the tokenizer table the converter emits: a u32 offset per token,
/// followed by the concatenated token bytes.
fn parseVocabulary(bytes: []align(16) const u8) !qw.tokenizer.TokenTable {
    if (bytes.len < 4) return error.MalformedVocabulary;
    // The table is a standalone file, so it carries its own count.
    const count = std.mem.readInt(u32, bytes[0..4], .little);
    const offsets_length = @as(usize, count + 1) * 4;
    if (bytes.len < 4 + offsets_length) return error.MalformedVocabulary;
    const offsets_bytes = bytes[4..][0..offsets_length];
    const aligned: []align(4) const u8 = @alignCast(offsets_bytes);
    const offsets = std.mem.bytesAsSlice(u32, aligned);
    // Offsets are relative to the start of the byte pool.
    const pool = bytes[4 + offsets_length ..];
    const table = qw.tokenizer.TokenTable{
        .offsets = offsets,
        .bytes = pool,
        .count = count,
    };
    table.validate() catch return error.MalformedVocabulary;
    return table;
}

test "the vocabulary table parser rejects malformed input" {
    // The parser requires the 16-byte alignment the converter's files carry.
    // Count says two tokens but the offsets are not there.
    var bytes: [8]u8 align(16) = undefined;
    std.mem.writeInt(u32, bytes[0..4], 2, .little);
    std.mem.writeInt(u32, bytes[4..8], 0, .little);
    try std.testing.expectError(error.MalformedVocabulary, parseVocabulary(&bytes));

    // An empty buffer is not a vocabulary either.
    var empty: [4]u8 align(16) = undefined;
    std.mem.writeInt(u32, empty[0..4], 0, .little);
    try std.testing.expectError(error.MalformedVocabulary, parseVocabulary(&empty));
}
