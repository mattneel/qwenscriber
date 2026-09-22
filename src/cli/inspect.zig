//! `qwenscriber-inspect`: prints what a model directory contains.
//!
//!     qwenscriber-inspect <model dir>
//!
//! This is the tool to reach for when a model misbehaves: it re-reads every file
//! the runtime reads, validates each one the way the runtime validates it, and
//! prints the tensor table shard by shard. It also checks the one thing the
//! runtime can only discover at load time -- that every tensor the architecture
//! requires is present -- and says which ones are missing.

const std = @import("std");
const host = @import("host");
const qw = @import("qwenscriber");

const usage =
    \\usage: qwenscriber-inspect <model dir>
    \\
    \\Prints the manifest, the parsed configuration, and every shard's tensor
    \\table. Exits non-zero when any file fails to validate or a tensor the
    \\runtime requires is missing.
    \\
;

const arguments_max = 8;

const assert = std.debug.assert;

/// Prints a failure and exits non-zero. A write that fails -- a consumer that
/// closed the pipe, as `head` does -- is a failure of the same kind, so it exits
/// the same way instead of surfacing a stack trace for an ordinary end of
/// output.
fn failWith(out: *std.Io.Writer, comptime format: []const u8, args: anytype) noreturn {
    out.print(format, args) catch {};
    out.flush() catch {};
    std.process.exit(1);
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_file_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    const args = try std.process.Args.toSlice(init.minimal.args, arena);
    if (args.len > arguments_max) {
        failWith(out, "error: too many arguments\n\n{s}", .{usage});
    }
    if (args.len <= 1 or std.mem.eql(u8, args[1], "--help")) {
        out.print("{s}", .{usage}) catch return;
        out.flush() catch return;
        return;
    }
    const path = args[1];

    var dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch |err| {
        failWith(out, "error: cannot open {s}: {s}\n", .{ path, @errorName(err) });
    };
    defer dir.close(io);

    const status = inspect(arena, io, dir, path, out) catch |err| {
        // A report that could not be written in full is reported the same way as
        // a model that did not validate: quietly, and with a non-zero exit.
        failWith(out, "error: {s}: {s}\n", .{ path, @errorName(err) });
    };
    out.flush() catch std.process.exit(1);
    if (!status) {
        // A model that does not validate must fail a script, not just print.
        std.process.exit(1);
    }
}

/// Returns false when the directory parses but does not hold what the runtime
/// needs.
fn inspect(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    path: []const u8,
    out: *std.Io.Writer,
) !bool {
    const manifest_bytes = try readFile(arena, io, dir, "manifest.json");
    const value = try host.manifest.read(arena, manifest_bytes);
    try printManifest(out, &value, path);

    const config_bytes = try readAligned(arena, io, dir, "config.bin", .@"4");
    const config = try qw.model_config.parse(config_bytes);
    try printConfig(out, config);

    // The runtime decides from `config.flags` whether the output projection is
    // the token embedding, so that bit is what the coverage check must follow.
    // A manifest that disagrees with it would describe a directory that does not
    // load, which is worth saying rather than discovering at load time.
    const tied = config.outputProjectionTied();
    var consistent = true;
    if (tied != value.output_is_tied) {
        try out.print(
            "\ninconsistent: config.bin says tied output projection {s}, " ++
                "manifest.json says {s}\n",
            .{ yesNo(tied), yesNo(value.output_is_tied) },
        );
        consistent = false;
    }

    var tensors: u32 = 0;
    const positions = qw.qwen3_asr.layout.Iterator.count(config);
    const seen = try arena.alloc(bool, positions);
    @memset(seen, false);
    for (value.shards) |shard| {
        tensors += try printShard(arena, io, dir, out, config, shard, seen);
    }

    try out.print("\ncoverage\n", .{});
    try out.print("  tensors in shards {d}\n", .{tensors});
    var missing: u32 = 0;
    var position: u32 = 0;
    while (position < positions) : (position += 1) {
        if (seen[position]) continue;
        const required = qw.qwen3_asr.layout.at(config, position).?;
        if (tied and required.kind == .decoder_output_weight) continue;
        var buffer: [128]u8 = undefined;
        const name = host.tensor_names.officialName(
            required.kind,
            required.layer,
            &buffer,
        ) catch "unknown";
        try out.print("  missing {s} (layer {d})\n", .{ name, required.layer });
        missing += 1;
    }
    try out.print("  expected          {d}\n", .{positions});
    if (tied) {
        try out.writeAll("  output projection tied to the token embedding\n");
    }
    const covered = positions - @as(u32, if (tied) 1 else 0);
    if (missing == 0 and tensors == covered and consistent) {
        try out.writeAll("  complete\n");
        return true;
    }
    return false;
}

fn printManifest(out: *std.Io.Writer, value: *const host.manifest.Value, path: []const u8) !void {
    try out.print("manifest {s}\n", .{path});
    try out.print("  format version    {d}\n", .{value.format_version});
    try out.print("  architecture      {s}\n", .{value.architecture});
    try out.print("  model id          {s}\n", .{value.model_id});
    try out.print("  quantization      {s}\n", .{value.quantization});
    try out.print("  tool              {s}\n", .{value.tool_version});
    try out.print("  config            {s} ({d} bytes)\n", .{
        value.config_file,
        value.config_bytes,
    });
    try out.print("  tokenizer         {s} ({d} bytes, {d} tokens)\n", .{
        value.tokenizer_file,
        value.tokenizer_bytes,
        value.token_count,
    });
    if (value.merges_file.len > 0) {
        try out.print("  merges            {s} ({d} bytes)\n", .{
            value.merges_file,
            value.merges_bytes,
        });
    }
    try out.print("  output tied       {s}\n", .{if (value.output_is_tied) "yes" else "no"});
    try out.print("  totals            {d} tensors, {d} parameters, {d} payload bytes, " ++
        "{d:.4} bits per weight\n", .{
        value.tensors,
        value.parameters_approximate,
        value.payload_bytes,
        value.bits_per_weight,
    });
    try out.print("  shards\n", .{});
    for (value.shards) |shard| {
        try out.print("    {s: <16} {d: >10} bytes  {d: >4} tensors  layers {d}..{d}\n", .{
            shard.name,
            shard.bytes,
            shard.tensor_count,
            shard.first_layer,
            shard.last_layer,
        });
        try out.print("      sha256 {s}\n", .{shard.sha256_hex});
    }
}

fn printConfig(out: *std.Io.Writer, config: *const qw.model_config.Config) !void {
    try out.print("\nconfig\n", .{});
    try out.print("  architecture          {d} (magic {s}, version {d})\n", .{
        config.architecture,
        config.magic,
        config.format_version,
    });
    try out.print("  audio tower           d_model {d}, layers {d}, heads {d} (head dim {d}), " ++
        "ffn {d}\n", .{
        config.audio_d_model,
        config.audio_layers,
        config.audio_attention_heads,
        config.audioHeadDim(),
        config.audio_ffn_dim,
    });
    try out.print("  audio geometry        downsample {d}, n_window {d} " ++
        "({d} frames, {d} steps), n_window_infer {d}\n", .{
        config.audio_downsample_hidden_size,
        config.audio_n_window,
        config.audioChunkFrames(),
        config.audioChunkSteps(),
        config.audio_n_window_infer,
    });
    try out.print("  audio tower out       {d} classes from {d} features, position steps {d}\n", .{
        config.audio_output_dim,
        config.audioConvOutInputFeatures(),
        config.audio_max_position_steps,
    });
    try out.print("  mel                   {d} bins (layer norm eps {e})\n", .{
        config.mel_bins,
        config.audio_layer_norm_eps,
    });
    try out.print("  decoder               hidden {d}, layers {d}, " ++
        "heads {d}/{d} (head dim {d}), ffn {d}\n", .{
        config.text_hidden_size,
        config.text_layers,
        config.text_attention_heads,
        config.text_key_value_heads,
        config.text_head_dim,
        config.text_ffn_dim,
    });
    try out.print("  decoder projections   query {d}, key/value {d}, vocab {d} (rms eps {e})\n", .{
        config.textQueryElements(),
        config.textKeyValueElements(),
        config.vocab_size,
        config.text_rms_norm_eps,
    });
    try out.print(
        "  positions             rope theta {d}, max positions {d}, decode tokens {d}\n",
        .{
            config.rope_theta,
            config.max_positions,
            config.max_decode_tokens,
        },
    );
    try out.print("  tokens                audio {d}..{d} (pad {d}), " ++
        "im {d}..{d}, endoftext {d}, asr_text {d}\n", .{
        config.token_audio_start,
        config.token_audio_end,
        config.token_audio_pad,
        config.token_im_start,
        config.token_im_end,
        config.token_endoftext,
        config.token_asr_text,
    });
    try out.print("  flags                 0x{x} ({s})\n", .{
        config.flags,
        if (config.outputProjectionTied()) "output projection tied" else "no flags",
    });
    try out.print("  tokens (decode)       eos {d}/{d}, pad {d}\n", .{
        config.token_eos_primary,
        config.token_eos_secondary,
        config.token_pad,
    });
}

/// Prints one shard's tensor table and marks the inventory positions it covers.
/// Returns the number of tensors it held.
fn printShard(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    out: *std.Io.Writer,
    config: *const qw.model_config.Config,
    shard: host.manifest.Shard,
    seen: []bool,
) !u32 {
    const bytes = try readAligned(arena, io, dir, shard.name, .@"16");
    const file = try qw.container.File.parse(bytes);
    try file.verifyChecksum();
    const digest = host.manifest.sha256Hex(bytes);
    if (!std.mem.eql(u8, &digest, shard.sha256_hex)) return error.DigestMismatch;
    if (bytes.len != shard.bytes) return error.ByteLengthMismatch;

    try out.print("\nshard {s} ({d} bytes, {d} tensors, checksum 0x{x:0>8})\n", .{
        shard.name,
        bytes.len,
        file.header.tensor_count,
        file.header.payload_checksum,
    });
    try out.print("  {s: <40} {s: >5} {s: >6} {s: >18} {s: >10} {s: >12}\n", .{
        "kind",
        "layer",
        "format",
        "shape",
        "bytes",
        "offset",
    });
    for (file.index) |*entry| {
        const required = qw.qwen3_asr.layout.find(
            config,
            @fromBackingInt(entry.kind),
            entry.layer,
        );
        var shape_buffer: [48]u8 = undefined;
        const shape_text = try formatShape(&shape_buffer, &(try entry.shape()));
        const kind_name = @as(qw.container.TensorKind, @fromBackingInt(entry.kind)).name();
        try out.print("  {s: <40} {d: >5} {s: >6} {s: >18} {d: >10} {d: >12}\n", .{
            kind_name,
            entry.layer,
            (try entry.storageFormat()).name(),
            shape_text,
            entry.len_bytes,
            entry.offset_bytes,
        });
        if (required) |described| {
            const position = host.tensor_names.positionOf(
                config,
                described.kind,
                described.layer,
            ).?;
            assert(position < seen.len);
            seen[position] = true;
        } else {
            try out.print("    ^ not described by this configuration\n", .{});
        }
    }
    return file.header.tensor_count;
}

fn yesNo(value: bool) []const u8 {
    return if (value) "yes" else "no";
}

fn formatShape(buffer: []u8, shape: *const qw.tensor.Shape) ![]const u8 {
    assert(shape.rank <= qw.tensor.rank_max);
    var stream = std.Io.Writer.fixed(buffer);
    for (shape.dims[0..shape.rank], 0..) |dim, index| {
        if (index != 0) try stream.writeByte('x');
        try stream.print("{d}", .{dim});
    }
    return stream.buffered();
}

fn readFile(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    name: []const u8,
) ![]u8 {
    return dir.readFileAlloc(io, name, arena, .limited(1 << 30));
}

/// Reads into a buffer with the alignment the file's parser requires.
fn readAligned(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    name: []const u8,
    comptime alignment: std.mem.Alignment,
) ![]align(alignment.toByteUnits()) u8 {
    return dir.readFileAllocOptions(io, name, arena, .limited(1 << 30), alignment, null);
}
