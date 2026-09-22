//! `qwenscriber-convert`: converts an official Qwen3-ASR checkpoint into a
//! model directory.
//!
//!     qwenscriber-convert --input <checkpoint dir> --output <model dir> \
//!         [--quant q4|q5|q8|none] [--shard-bytes N] [--verify] [--model-id NAME]
//!
//! The tool is a shell over `host.convert.run`: it parses arguments, opens the
//! two directories, and prints what the conversion reports. Everything that can
//! be tested lives in the library, which is where the synthetic end-to-end test
//! checks it.

const std = @import("std");
const host = @import("host");
const qw = @import("qwenscriber");

const usage =
    \\usage: qwenscriber-convert --input <checkpoint dir> --output <model dir> [options]
    \\
    \\  --input <dir>        checkpoint directory holding config.json and *.safetensors
    \\  --output <dir>       model directory to create or overwrite
    \\  --quant <format>     q4 (default), q5, q8, or none for f16 weights
    \\  --shard-bytes <n>    shard budget in bytes (default 201326592, 192 MiB)
    \\  --model-id <name>    identifier recorded in the manifest (default: input name)
    \\  --verify             re-read every shard and compare it with the checkpoint
    \\  --help               print this text
    \\
;

const arguments_max = 64;

const Arguments = struct {
    input: []const u8 = "",
    output: []const u8 = "",
    model_id: []const u8 = "",
    quantization_text: []const u8 = "q4",
    shard_bytes: u64 = host.convert.shard_bytes_default,
    verify: bool = false,
    help: bool = false,

    fn parse(args: []const [:0]const u8) !Arguments {
        var parsed = Arguments{};
        var index: usize = 1;
        while (index < args.len) : (index += 1) {
            const flag = args[index];
            if (std.mem.eql(u8, flag, "--help")) {
                parsed.help = true;
            } else if (std.mem.eql(u8, flag, "--verify")) {
                parsed.verify = true;
            } else if (std.mem.eql(u8, flag, "--input")) {
                parsed.input = try value(args, &index);
            } else if (std.mem.eql(u8, flag, "--output")) {
                parsed.output = try value(args, &index);
            } else if (std.mem.eql(u8, flag, "--model-id")) {
                parsed.model_id = try value(args, &index);
            } else if (std.mem.eql(u8, flag, "--quant")) {
                parsed.quantization_text = try value(args, &index);
            } else if (std.mem.eql(u8, flag, "--shard-bytes")) {
                parsed.shard_bytes = try std.fmt.parseInt(u64, try value(args, &index), 10);
            } else {
                return error.UnknownArgument;
            }
        }
        return parsed;
    }
};

fn value(args: []const [:0]const u8, index: *usize) ![]const u8 {
    index.* += 1;
    if (index.* >= args.len) return error.MissingArgumentValue;
    return args[index.*];
}

/// `none` is the flag spelling of "do not quantize": the storage policy then
/// keeps matrix weights in f16.
fn quantizationFromText(text: []const u8) ?qw.dtype.Format {
    if (std.mem.eql(u8, text, "none")) return .f16;
    const format = qw.dtype.Format.parse(text) orelse return null;
    if (!format.isQuantized() and format != .f16) return null;
    return format;
}

/// The last path component, which is the default model id.
fn baseName(path: []const u8) []const u8 {
    if (path.len == 0) return path;
    const trimmed = std.mem.trimEnd(u8, path, "/");
    if (trimmed.len == 0) return path;
    const slash = std.mem.lastIndexOfScalar(u8, trimmed, '/') orelse return trimmed;
    return trimmed[slash + 1 ..];
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_file_writer: std.io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    const args = try std.process.Args.toSlice(init.minimal.args, arena);
    if (args.len > arguments_max) {
        try out.print("error: too many arguments\n", .{});
        try out.flush();
        std.process.exit(1);
    }
    const parsed = Arguments.parse(args) catch |err| {
        try out.print("error: {s}\n\n{s}", .{ @errorName(err), usage });
        try out.flush();
        std.process.exit(1);
    };
    if (parsed.help or args.len <= 1) {
        // `--help` is a successful run: it was asked for exactly this.
        try out.print("{s}", .{usage});
        try out.flush();
        return;
    }
    if (parsed.input.len == 0 or parsed.output.len == 0) {
        try out.print("error: --input and --output are required\n\n{s}", .{usage});
        try out.flush();
        std.process.exit(1);
    }
    const quantization = quantizationFromText(parsed.quantization_text) orelse {
        try out.print("error: --quant {s} is not q4, q5, q8, or none\n", .{
            parsed.quantization_text,
        });
        try out.flush();
        std.process.exit(1);
    };

    // The checkpoint directory must be iterable, because which `*.safetensors`
    // files a checkpoint has is the directory's answer.
    var input_dir = std.Io.Dir.cwd().openDir(io, parsed.input, .{ .iterate = true }) catch |err| {
        try out.print("error: cannot open checkpoint {s}: {s}\n", .{ parsed.input, @errorName(err) });
        try out.flush();
        std.process.exit(1);
    };
    defer input_dir.close(io);
    var output_dir = std.Io.Dir.cwd().createDirPathOpen(io, parsed.output, .{}) catch |err| {
        try out.print("error: cannot create {s}: {s}\n", .{ parsed.output, @errorName(err) });
        try out.flush();
        std.process.exit(1);
    };
    defer output_dir.close(io);

    const model_id = if (parsed.model_id.len > 0) parsed.model_id else baseName(parsed.input);
    const start = std.Io.Clock.awake.now(io);
    var diagnostics: host.convert.Diagnostics = .{};
    const report = host.convert.run(arena, io, .{
        .input_dir = input_dir,
        .output_dir = output_dir,
        .input_name = parsed.input,
        .output_name = parsed.output,
        .model_id = model_id,
        .quantization = quantization,
        .shard_bytes_max = parsed.shard_bytes,
        .verify = parsed.verify,
    }, &diagnostics) catch |err| {
        try printFailure(out, err, &diagnostics, parsed.input);
        try out.flush();
        std.process.exit(1);
    };
    const elapsed_ns = std.Io.Clock.awake.now(io).nanoseconds - start.nanoseconds;

    try printReport(out, &report, elapsed_ns);
    try out.flush();
}

fn printFailure(
    out: *std.Io.Writer,
    err: anyerror,
    diagnostics: *const host.convert.Diagnostics,
    input_name: []const u8,
) !void {
    try out.print("error: {s}", .{@errorName(err)});
    if (diagnostics.tensor_name.len > 0) {
        try out.print(": {s}", .{diagnostics.tensor_name});
    }
    if (diagnostics.detail.len > 0) {
        try out.print(" ({s})", .{diagnostics.detail});
    }
    try out.writeAll("\n");
    // A missing tensor is the failure an operator has to act on, so say where to
    // look rather than only what is wrong.
    if (err == host.convert.Error.MissingTensor) {
        try out.print(
            "hint: {s} does not contain every tensor this runtime needs\n",
            .{input_name},
        );
    }
}

fn printReport(out: *std.Io.Writer, report: *const host.convert.Report, elapsed_ns: i96) !void {
    try out.print("model         {s}\n", .{report.model_id});
    try out.print("quantization  {s}\n", .{report.quantization.name()});
    try out.print("output        tied output projection: {s}\n", .{
        if (report.output_is_tied) "yes" else "no",
    });
    try out.print("tensors       {d}", .{report.tensors});
    if (report.unused_tensors != 0) {
        try out.print(" (+{d} checkpoint tensors unused)", .{report.unused_tensors});
    }
    try out.writeAll("\n");
    try out.print("tokens        {d} in {d} bytes\n", .{ report.token_count, report.tokenizer_bytes });
    try out.print("\nshard            tensors  layers      bytes      checksum\n", .{});
    for (report.shards) |shard| {
        try out.print("{s: <16} {d: >7}  {d: >4}..{d: <4} {d: >10}  0x{x:0>8}\n", .{
            shard.name,
            shard.tensor_count,
            shard.first_layer,
            shard.last_layer,
            shard.bytes,
            shard.payload_checksum,
        });
    }
    try out.print("\ntotals\n", .{});
    try out.print("  source bytes      {d}\n", .{report.source_bytes});
    try out.print("  payload bytes     {d}\n", .{report.payload_bytes});
    try out.print("  output bytes      {d}\n", .{report.output_bytes});
    try out.print("  parameters        {d}\n", .{report.parameters});
    try out.print("  bits per weight   {d:.4}\n", .{report.bits_per_weight});
    try out.print("  merges            {d} bytes\n", .{report.merges_bytes});
    try out.print("  seconds           {d:.2}\n", .{
        @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s,
    });
    if (report.verify.checked) {
        try out.print("verify\n", .{});
        try out.print("  shards parsed     {d}\n", .{report.verify.shards_parsed});
        try out.print("  tensors compared  {d}\n", .{report.verify.tensors_compared});
        try out.print("  rows compared     {d}\n", .{report.verify.rows_compared});
        try out.print("  elements compared {d}\n", .{report.verify.elements_compared});
        try out.print("  largest error     {e}\n", .{report.verify.error_max});
    }
}
