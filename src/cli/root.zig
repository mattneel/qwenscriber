//! Root of the command line tools, so their tests can run under `zig build
//! test` alongside the core's.
//!
//! `std.testing.refAllDecls` is one level deep, so every module that owns tests
//! is listed here explicitly.

const std = @import("std");

pub const wav = @import("wav.zig");
pub const transcribe = @import("transcribe.zig");
pub const fixture = @import("fixture.zig");

test {
    std.testing.refAllDecls(@This());
}
