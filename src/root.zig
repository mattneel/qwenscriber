//! Qwenscriber core: portable, browser-agnostic speech-to-text for Qwen3-ASR.
//!
//! This module is compiled twice, unchanged: once for the host (tests, the
//! conversion tooling, and the native bring-up runner) and once for
//! `wasm32-freestanding` (the browser runtime). It must therefore never import
//! anything that requires an operating system, libc, WASI, or JavaScript.
//!
//! Responsibilities, in the order data flows through the runtime:
//!
//!   * `audio`      — PCM conversion and resampling to 16 kHz mono f32.
//!   * `mel`        — Whisper-style log-mel frontend (128 bins, 10 ms hop).
//!   * `tokenizer`  — byte-level BPE for prompt assembly and detokenization.
//!   * `config`     — validated Qwen3-ASR architecture configuration.
//!   * `manifest`   — the `.qw` container index that describes a model.
//!   * `qwen3_asr`  — the audio encoder, projector, and text decoder.
//!
//! Everything here is deterministic: identical inputs produce identical bits on
//! every target. That property is what lets the WASM backend serve as the
//! correctness oracle for the WebGPU kernels.

const std = @import("std");

pub const math = @import("core/math.zig");
pub const half_float = @import("core/half_float.zig");
pub const dtype = @import("core/dtype.zig");
pub const quant = @import("core/quant.zig");
pub const tensor = @import("core/tensor.zig");
pub const mel = @import("core/mel.zig");
pub const tokenizer = @import("core/tokenizer.zig");
pub const container = @import("core/container.zig");
pub const model_config = @import("core/model_config.zig");
pub const qwen3_asr = struct {
    pub const layout = @import("core/qwen3_asr/layout.zig");
    pub const kernels = @import("core/qwen3_asr/kernels.zig");
    pub const model = @import("core/qwen3_asr/model.zig");
    pub const decoder = @import("core/qwen3_asr/decoder.zig");
    pub const prompt = @import("core/qwen3_asr/prompt.zig");
};
pub const selftest = @import("core/selftest.zig");

/// Version of the deterministic core, reported through the WASM ABI. Bumped
/// whenever a numeric layout or algorithm changes in a way that alters output.
pub const core_version: u32 = 1;

test {
    std.testing.refAllDecls(@This());
    // `refAllDecls` is deliberately one level deep, so the architecture modules
    // nested under `qwen3_asr` have to be listed explicitly. Without this line
    // their tests -- the model's tensor inventory and every kernel -- would
    // silently not run.
    std.testing.refAllDecls(qwen3_asr);
}
