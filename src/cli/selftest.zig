//! Prints the numerical core's self-test report.
//!
//! `tools/wasm_selftest.mjs` compares the freestanding build's report against
//! the values this tool prints, so the host build is the reference for the
//! browser build. Run `zig build selftest` after changing a numeric routine and
//! update the constants in that harness in the same commit if they moved.

const std = @import("std");
const qw = @import("qwenscriber");

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const writer = &stdout_file_writer.interface;

    const report = qw.selftest.run();
    try writer.print("core_version   {d}\n", .{qw.core_version});
    try writer.print("failures       0x{x:0>8}\n", .{report.failures});
    try writer.print("quant_hash     0x{x:0>16}\n", .{report.quant_hash});
    try writer.print("mel_hash       0x{x:0>16}\n", .{report.mel_hash});
    try writer.print("loudest_band   {d}\n", .{report.loudest_band});
    try writer.print("silence_value  {d}\n", .{report.silence_value});
    try writer.flush();

    if (report.failures != 0) {
        // A non-zero exit makes a broken numeric core fail scripts, not just
        // print a warning nobody reads.
        std.process.exit(1);
    }
}
