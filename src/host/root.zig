//! Host-side checkpoint tooling.
//!
//! The portable core (`qwenscriber`) never reads a file, never parses JSON, and
//! never touches a checkpoint. This module is the other half of that boundary:
//! it reads the official Qwen3-ASR checkpoints, checks them against the core's
//! tensor inventory, quantizes them into the core's storage formats, and writes
//! a model directory the browser can load.
//!
//! Nothing here is compiled for `wasm32-freestanding`, so it may use `std.json`,
//! an allocator, and the operating system freely.

const std = @import("std");

pub const safetensors = @import("safetensors.zig");
pub const tensor_names = @import("tensor_names.zig");
pub const checkpoint_config = @import("checkpoint_config.zig");
pub const tokenizer_file = @import("tokenizer_file.zig");
pub const manifest = @import("manifest.zig");
pub const convert = @import("convert.zig");

/// Version of the conversion tooling, recorded in every manifest. Bumped when a
/// change alters the bytes a model directory contains.
pub const tool_version = "qwenscriber-convert 1";

test {
    std.testing.refAllDecls(@This());
}
