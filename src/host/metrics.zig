//! What one `qwenscriber-transcribe` run measured, written as a JSON record.
//!
//! The runner prints timings for a human to read. This writes the same numbers
//! for a machine to compare, because the questions this repository has to answer
//! -- is q5 fast enough to ship, did a kernel change cost us the encoder, how
//! much slower is 1.7B than 0.6B -- are comparisons between runs, not readings of
//! one.
//!
//! A duration on its own is not comparable to anything, so the record carries
//! the model's identity (from the model directory's manifest), the shape the run
//! actually executed, and the machine and build mode it ran on. Every field is a
//! measurement or a value read from the model directory; nothing is estimated.
//!
//! The two rates in `derived` are computed here from the durations in the same
//! record, so a reader cannot find them disagreeing with the numbers they came
//! from.

const std = @import("std");
const Io = std.Io;
const builtin = @import("builtin");

const json = @import("json_writer.zig");
const manifest = @import("manifest.zig");
const build_options = @import("build_options");

const assert = std.debug.assert;

/// One run's measurements. Durations are milliseconds; `audio_ms` is the clip's
/// own length, and `generated_tokens` counts the tokens the decoder emitted
/// after the prompt.
pub const Metrics = struct {
    /// From the model directory's manifest; empty when the directory has none.
    model_id: []const u8 = "",
    quantization: []const u8 = "",
    bits_per_weight: f64 = 0,
    tensors: u32 = 0,
    parameters: u64 = 0,
    payload_bytes: u64 = 0,
    shard_count: u32 = 0,
    shard_bytes: u64 = 0,
    audio_d_model: u32 = 0,
    audio_layers: u32 = 0,
    text_hidden_size: u32 = 0,
    text_layers: u32 = 0,
    vocab_size: u32 = 0,
    kv_cache_bytes: u64 = 0,
    frames: u32 = 0,
    encoder_steps: u32 = 0,
    prompt_tokens: u32 = 0,
    generated_tokens: u32 = 0,
    load_ms: i64 = 0,
    mel_ms: i64 = 0,
    encoder_ms: i64 = 0,
    first_token_ms: i64 = 0,
    decode_ms: i64 = 0,
    total_ms: i64 = 0,
    audio_ms: i64 = 0,
};

/// The machine and the build mode. A latency measured by a debug build on an
/// unnamed processor is not comparable to anything, so both travel with the
/// record.
pub const Environment = struct {
    build_mode: []const u8,
    os: []const u8,
    arch: []const u8,
    /// Processor model name, or `unknown` where it cannot be read.
    cpu: []const u8,
    cpu_count: u32,
    /// The revision the binary was built from, or `unknown`.
    revision: []const u8,
    zig: []const u8,
};

/// Longest record written. The field list is fixed and every number fits inside
/// 32 characters, so the bound is generous rather than tight.
pub const record_bytes_max: u32 = 4096;

/// Largest prefix of `/proc/cpuinfo` scanned for the processor model.
const cpuinfo_bytes_max: u32 = 64 * 1024;

/// Who the model was, read from the model directory's manifest.
///
/// A directory without a manifest still produces a record: the run's own
/// measurements are what the record is for, and the manifest only says which
/// model they were taken on.
pub const Identity = struct {
    model_id: []const u8 = "",
    quantization: []const u8 = "",
    bits_per_weight: f64 = 0,
    tensors: u32 = 0,
    parameters: u64 = 0,
    payload_bytes: u64 = 0,
};

/// A manifest lists shards and their digests, so it is small; the bound keeps a
/// wrong path from reading a model's weights as if they were one.
pub fn readIdentity(io: Io, gpa: std.mem.Allocator, dir: Io.Dir) Identity {
    const bytes = dir.readFileAlloc(
        io,
        "manifest.json",
        gpa,
        Io.Limit.limited(1 << 20),
    ) catch return .{};
    const value = manifest.read(gpa, bytes) catch return .{};
    return .{
        .model_id = value.model_id,
        .quantization = value.quantization,
        .bits_per_weight = value.bits_per_weight,
        .tensors = value.tensors,
        .parameters = value.parameters_approximate,
        .payload_bytes = value.payload_bytes,
    };
}

pub fn writeFile(io: Io, path: []const u8, metrics: Metrics, environment: Environment) !void {
    var buffer: [record_bytes_max]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try write(&writer, metrics, environment);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buffer[0..writer.end] });
}

/// Writes the record. Takes the writer rather than a path so a test can read the
/// text it produces without touching the filesystem.
pub fn write(writer: *std.Io.Writer, metrics: Metrics, environment: Environment) !void {
    assert(environment.cpu_count >= 1);
    assert(metrics.load_ms >= 0);
    assert(metrics.mel_ms >= 0);
    assert(metrics.encoder_ms >= 0);
    assert(metrics.first_token_ms >= 0);
    assert(metrics.decode_ms >= 0);
    // The phases run one after another, so the total cannot be shorter than any
    // of them. A record that says otherwise was assembled from the wrong clock.
    assert(metrics.total_ms >= metrics.load_ms);
    assert(metrics.total_ms >= metrics.mel_ms);
    assert(metrics.total_ms >= metrics.encoder_ms);
    assert(metrics.total_ms >= metrics.decode_ms);

    try writer.writeAll("{\n");
    try json.textField(writer, json.top_indent, "model_id", metrics.model_id, false);
    try json.textField(writer, json.top_indent, "quantization", metrics.quantization, false);
    try json.floatField(writer, json.top_indent, "bits_per_weight", metrics.bits_per_weight, false);
    try json.numberField(writer, json.top_indent, "tensors", metrics.tensors, false);
    try json.numberField(writer, json.top_indent, "parameters", metrics.parameters, false);
    try json.numberField(writer, json.top_indent, "payload_bytes", metrics.payload_bytes, false);
    try json.numberField(writer, json.top_indent, "shard_count", metrics.shard_count, false);
    try json.numberField(writer, json.top_indent, "shard_bytes", metrics.shard_bytes, false);

    try writer.writeAll("  \"model\": {\n");
    try json.numberField(writer, json.nested_indent, "audio_d_model", metrics.audio_d_model, false);
    try json.numberField(writer, json.nested_indent, "audio_layers", metrics.audio_layers, false);
    try json.numberField(
        writer,
        json.nested_indent,
        "text_hidden_size",
        metrics.text_hidden_size,
        false,
    );
    try json.numberField(writer, json.nested_indent, "text_layers", metrics.text_layers, false);
    try json.numberField(writer, json.nested_indent, "vocab_size", metrics.vocab_size, false);
    try json.numberField(
        writer,
        json.nested_indent,
        "kv_cache_bytes",
        metrics.kv_cache_bytes,
        true,
    );
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"run\": {\n");
    try json.numberField(writer, json.nested_indent, "frames", metrics.frames, false);
    try json.numberField(writer, json.nested_indent, "encoder_steps", metrics.encoder_steps, false);
    try json.numberField(writer, json.nested_indent, "prompt_tokens", metrics.prompt_tokens, false);
    try json.numberField(
        writer,
        json.nested_indent,
        "generated_tokens",
        metrics.generated_tokens,
        false,
    );
    try json.numberField(writer, json.nested_indent, "audio_ms", metrics.audio_ms, true);
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"timings_ms\": {\n");
    try json.numberField(writer, json.nested_indent, "load", metrics.load_ms, false);
    try json.numberField(writer, json.nested_indent, "mel", metrics.mel_ms, false);
    try json.numberField(writer, json.nested_indent, "encoder", metrics.encoder_ms, false);
    // Includes the prompt's prefill, which is the part of the first token a user
    // waits through after the audio is already in the runtime.
    try json.numberField(
        writer,
        json.nested_indent,
        "first_token",
        metrics.first_token_ms,
        false,
    );
    try json.numberField(writer, json.nested_indent, "decode", metrics.decode_ms, false);
    try json.numberField(writer, json.nested_indent, "total", metrics.total_ms, true);
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"derived\": {\n");
    try json.floatField(
        writer,
        json.nested_indent,
        "tokens_per_second",
        tokensPerSecond(metrics),
        false,
    );
    try json.floatField(
        writer,
        json.nested_indent,
        "realtime_factor",
        realtimeFactor(metrics),
        true,
    );
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"environment\": {\n");
    try json.textField(writer, json.nested_indent, "build_mode", environment.build_mode, false);
    try json.textField(writer, json.nested_indent, "os", environment.os, false);
    try json.textField(writer, json.nested_indent, "arch", environment.arch, false);
    try json.textField(writer, json.nested_indent, "cpu", environment.cpu, false);
    try json.numberField(writer, json.nested_indent, "cpu_count", environment.cpu_count, false);
    try json.textField(writer, json.nested_indent, "revision", environment.revision, false);
    try json.textField(writer, json.nested_indent, "zig", environment.zig, true);
    try writer.writeAll("  }\n");

    try writer.writeAll("}\n");
}

/// Generated tokens per second across the decode phase, which is the rate a user
/// waits on. Zero rather than an infinity when the phase did not run.
fn tokensPerSecond(metrics: Metrics) f64 {
    if (metrics.decode_ms <= 0) return 0;
    const tokens: f64 = @floatFromInt(metrics.generated_tokens);
    const duration: f64 = @floatFromInt(metrics.decode_ms);
    return tokens * 1000.0 / duration;
}

/// Wall time over audio time: below one is faster than the clip plays. Zero
/// rather than an infinity when the clip length is unknown.
fn realtimeFactor(metrics: Metrics) f64 {
    if (metrics.audio_ms <= 0) return 0;
    const elapsed: f64 = @floatFromInt(metrics.total_ms);
    const audio: f64 = @floatFromInt(metrics.audio_ms);
    return elapsed / audio;
}

/// Reads the machine's identity once per run. `cpu_buffer` receives the
/// processor's model name; the returned slice points into it.
pub fn detectEnvironment(io: Io, cpu_buffer: []u8) Environment {
    assert(cpu_buffer.len > 0);
    return .{
        .build_mode = @tagName(builtin.mode),
        .os = @tagName(builtin.os.tag),
        .arch = @tagName(builtin.cpu.arch),
        .cpu = cpuModelName(io, cpu_buffer),
        .cpu_count = cpuCount(),
        .revision = build_options.revision,
        .zig = builtin.zig_version_string,
    };
}

/// The processor's model name, or `unknown` when it cannot be read. Only Linux
/// publishes it where this can find it, and an unnamed machine is recorded as
/// unnamed rather than guessed at.
fn cpuModelName(io: Io, buffer: []u8) []const u8 {
    if (builtin.os.tag != .linux) return "unknown";
    // `/proc/cpuinfo` reports a length of zero, and the ordinary reader
    // memoizes that length and returns nothing. The streaming mode never asks
    // for a size and reads until end of file, which is what a shell's `cat`
    // does.
    const file = Io.Dir.cwd().openFile(io, "/proc/cpuinfo", .{}) catch return "unknown";
    defer file.close(io);
    var read_buffer: [1024]u8 = undefined;
    var reader = file.readerStreaming(io, &read_buffer);
    const bytes = reader.interface.allocRemaining(
        std.heap.page_allocator,
        Io.Limit.limited(cpuinfo_bytes_max),
    ) catch return "unknown";
    defer std.heap.page_allocator.free(bytes);

    // The first `model name` line, which is the processor itself rather than a
    // per-core repeat of it.
    const marker = "model name";
    const marker_at = std.mem.indexOf(u8, bytes, marker) orelse return "unknown";
    const colon_at = std.mem.indexOfScalarPos(u8, bytes, marker_at, ':') orelse return "unknown";
    const line_end = std.mem.indexOfScalarPos(u8, bytes, colon_at, '\n') orelse bytes.len;
    const name = std.mem.trim(u8, bytes[colon_at + 1 .. line_end], " \t\r");
    if (name.len == 0 or name.len > buffer.len) return "unknown";
    @memcpy(buffer[0..name.len], name);
    return buffer[0..name.len];
}

fn cpuCount() u32 {
    const count = std.Thread.getCpuCount() catch return 1;
    return @intCast(@min(count, 1 << 20));
}

test "the record parses and its rates come from its own durations" {
    var buffer: [record_bytes_max]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    var metrics = Metrics{};
    metrics.generated_tokens = 12;
    metrics.decode_ms = 6000;
    metrics.audio_ms = 4000;
    metrics.total_ms = 8000;
    try write(&writer, metrics, .{
        .build_mode = "ReleaseFast",
        .os = "linux",
        .arch = "x86_64",
        .cpu = "test",
        .cpu_count = 1,
        .revision = "deadbee",
        .zig = "0.0.0",
    });
    const record = buffer[0..writer.end];

    // A record that does not parse is not a record. Parsing is the assertion
    // that catches a missing comma, which is the one mistake this writer can
    // make.
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, record, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    for ([_][]const u8{ "model", "run", "timings_ms", "derived", "environment" }) |group| {
        try std.testing.expect(root.get(group) != null);
    }
    // Twelve tokens in six seconds is two per second, and eight seconds of work
    // for four seconds of audio is twice real time.
    const derived = root.get("derived").?.object;
    try std.testing.expectEqual(@as(f64, 2.0), derived.get("tokens_per_second").?.float);
    try std.testing.expectEqual(@as(f64, 2.0), derived.get("realtime_factor").?.float);
    try std.testing.expectEqual(@as(i64, 8000), root.get("timings_ms").?.object.get("total").?.integer);
    try std.testing.expectEqualStrings(
        "ReleaseFast",
        root.get("environment").?.object.get("build_mode").?.string,
    );
    try std.testing.expectEqualStrings(
        "deadbee",
        root.get("environment").?.object.get("revision").?.string,
    );
}

test "a run that generated nothing reports zero rather than dividing by zero" {
    var buffer: [record_bytes_max]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try write(&writer, .{}, .{
        .build_mode = "Debug",
        .os = "linux",
        .arch = "x86_64",
        .cpu = "test",
        .cpu_count = 1,
        .revision = "unknown",
        .zig = "0.0.0",
    });
    var parsed = try std.json.parseFromSlice(
        std.json.Value,
        std.testing.allocator,
        buffer[0..writer.end],
        .{},
    );
    defer parsed.deinit();
    const derived = parsed.value.object.get("derived").?.object;
    try std.testing.expectEqual(@as(f64, 0.0), derived.get("tokens_per_second").?.float);
    try std.testing.expectEqual(@as(f64, 0.0), derived.get("realtime_factor").?.float);
}
