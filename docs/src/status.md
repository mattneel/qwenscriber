# Project status

Qwenscriber is **pre-alpha**. Nothing here is a supported release, and no artifact has been
published. This page records what the code on the current revision actually does, with the command
that demonstrates it.

## Status vocabulary

| Label | Meaning |
| --- | --- |
| Implemented | Present, tested, and available on the documented revision |
| Partial | Present with documented gaps or unsupported cases |
| Experimental | Usable for evaluation; compatibility and behavior may change |
| Planned | Directional design only; do not depend on it yet |

## Capability matrix

| Capability | Status | Evidence on the current revision |
| --- | --- | --- |
| Zig core: math, dtypes, tensors, containers | Implemented | 107 `src/core` unit tests pass under `zig build test` |
| Log-mel frontend | Implemented | Fixture comparison against `tools/reference/gen_fixtures.py` output in `src/core/mel.zig` tests |
| Tokenizer byte alphabet and detokenization | Implemented | `src/core/tokenizer.zig` tests, including malformed tables |
| Q4/Q5/Q8 packing and dequantization | Implemented | `src/core/quant.zig` tests plus `tests/gpu/layout_drift.mjs` |
| `wasm32-freestanding` module | Implemented | `zig build wasm` emits a 963 KiB module that resolves zero imports and exports its own memory |
| Threaded WASM build (`wasm32-emscripten`) | Planned | Sanctioned by ADR-0005 for thread support. Nothing is built yet: there is no emscripten target, no thread-enabled artifact, and no measurement showing threads pay for cross-origin isolation. `capabilities()` already reports `wasmThreads`/`sharedArrayBuffer`/`crossOriginIsolated` so the decision can be made from data |
| Versioned C-like ABI (v1) | Partial | Feature bits report 0x1f: version, allocator, log-mel, tokenizer, self-test, model, decode. The model family (begin, add_shard, finish, requirements) and decode (begin, step, tokens, end) are exported and covered by `tools/wasm_selftest.mjs` — invalid-state matrix, shard validation, requirements arithmetic, and a real decode. A native C header and the audio family are still missing |
| ABI conformance harness | Implemented | `zig build test-wasm` runs `tools/wasm_selftest.mjs` against the real module in Node |
| TypeScript SDK | Partial | `Qwenscriber.capabilities()`, typed errors, resumable download with digest verification, IndexedDB cache, worker channel, preprocessing, WebGPU adapter/limit probing with an upload planner, and the model/decode ABI bindings. `transcribe()` still stops with a typed error when no converted model is loaded; `stream()` has no surface |
| Browser example page | Experimental | `examples/browser` loads the module in a worker, reports capabilities, self-tests the core, preprocesses audio with measured throughput, plans GPU uploads from the adapter's real limits, and runs `matmul_q4` against the CPU reference. It is not a transcription demo, because the repository ships no model artifact |
| Qwen3-ASR architecture parsing | Implemented | `src/core/model_config.zig` plus `src/core/qwen3_asr/layout.zig` derive every shape from `config.bin`; validated against the 0.6B and 1.7B upstream configs |
| CPU reference encoder and decoder | Implemented | `src/core/qwen3_asr/{kernels,model,decoder}.zig`; 4 `tests/reference_check.zig` tests compare against `tools/reference` fixtures |
| Safetensors reading and model conversion | Partial | `src/host/{safetensors,manifest,convert}.zig` and `qwenscriber-convert`; conversion of the official 0.6B checkpoint is in progress |
| Qwen3-ASR 0.6B transcription | Partial | The native runner loads the converted checkpoint and runs the encoder's convolution stack; a defect in the tower's attention path is being fixed. The reference transcript for the fixture clip exists (`tools/reference/transcribe_reference.py`), so the comparison target is in place |
| WASM SIMD execution path | Partial | The core is built for `simd128` and host/WASM self-tests agree; full-model inference has not been run through WASM |
| WASM linear-memory budget | Measured | A wasm32 instance tops out at 2 GiB of linear memory (Node: 1.9 GiB accepted, 2.0 GiB refused; `--wasm-max-mem-pages` does not move it). A 2048-position model fits at 0.855 GiB (weights 422.8 MB + cache 469.8 MB + scratch 25.5 MB) with 2.066 GiB resident after load; 8192 positions does not — `qw_model_finish` returns `out_of_memory` before the cache can be allocated. `--max-positions` selects the runnable budget, and the next change here is a smaller or narrower (f16/q8) cache |
| WGSL kernel suite | Experimental | 8 kernels in `gpu/shaders`; `node tests/gpu/harness.mjs` compares 11 cases against Zig CPU references and passes under SwiftShader on this host (worst case: RoPE `max_abs` 5.0e-4), including a `quant_hash` oracle that matches the Zig self-test bit for bit |
| WebGPU backend integration | Planned | Kernels exist, but the SDK does not yet dispatch them |
| Qwen3-ASR 1.7B | Planned | The configuration, layout, and quantized formats are model-agnostic; no 1.7B run has been attempted |
| Streaming transcription | Planned | No surface implemented |
| Native C FFI packages | Planned | No header or shared library is emitted |
| Elixir/Zigler integration | Planned | Nothing implemented |

## How to reproduce the claims

```sh
zig build test          # core, reference-math, and CLI unit tests
zig build wasm          # wasm32-freestanding module
zig build test-wasm     # ABI conformance against the built module
node tests/gpu/layout_drift.mjs   # shader/constant drift gate
```

The GPU conformance harness (`node tests/gpu/harness.mjs`) needs a Chromium with WebGPU enabled and a
GPU adapter; it is not part of the default command set.

## Not yet true

- No release, package, or artifact has been published.
- CI has run once on `main`. The Zig gate, shader drift gate, and TypeScript SDK jobs passed; the
  Documentation job failed on an mdBook config field that mdBook 0.5 removed, which is fixed. The book
  is now published to GitHub Pages from `main`, which requires the repository's Pages source to be set
  to "GitHub Actions" once.
- No performance number on this page has been measured under the benchmark hooks described in the
  architecture pages.

Update this page from evidence: merged code, passing tests, and released artifacts. Do not advance a
status because an interface has merely been designed.
