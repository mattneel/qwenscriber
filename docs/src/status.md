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
| `wasm32-freestanding` module | Implemented | `zig build wasm` emits a 963 KiB module with no imports |
| Versioned C-like ABI (v1) | Partial | Exported: version, feature bits, allocator, log-mel, tokenizer, self-test. Model/decode/audio families are not exported yet, and no native C header exists |
| ABI conformance harness | Implemented | `zig build test-wasm` runs `tools/wasm_selftest.mjs` against the real module in Node |
| TypeScript SDK | Partial | `Qwenscriber.capabilities()`, typed errors, resumable download with digest verification, IndexedDB cache, worker channel, and preprocessing. `transcribe`/`stream` exist as declared surfaces only |
| Browser example page | Experimental | `examples/browser` loads the module in a worker and reports capabilities; not yet a transcription demo |
| Qwen3-ASR architecture parsing | Implemented | `src/core/model_config.zig` plus `src/core/qwen3_asr/layout.zig` derive every shape from `config.bin`; validated against the 0.6B and 1.7B upstream configs |
| CPU reference encoder and decoder | Implemented | `src/core/qwen3_asr/{kernels,model,decoder}.zig`; 4 `tests/reference_check.zig` tests compare against `tools/reference` fixtures |
| Safetensors reading and model conversion | Partial | `src/host/{safetensors,manifest,convert}.zig` and `qwenscriber-convert`; conversion of the official 0.6B checkpoint is in progress |
| Qwen3-ASR 0.6B transcription | Partial | `qwenscriber-transcribe` drives the real checkpoint end to end; transcript validation in progress |
| WASM SIMD execution path | Partial | The core is built for `simd128` and host/WASM self-tests agree; full-model inference has not been run through WASM |
| WGSL kernel suite | Experimental | 8 kernels in `gpu/shaders`; compared against Zig CPU references by `tests/gpu` on this host, not in CI |
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

The GPU conformance harness (`node tests/gpu/driver.mjs`) needs a Chromium with WebGPU enabled and a
GPU adapter; it is not part of the default command set.

## Not yet true

- No release, package, or artifact has been published.
- No CI workflow exists yet, so ADR-0004 has no enforcement behind it.
- No performance number on this page has been measured under the benchmark hooks described in the
  architecture pages.

Update this page from evidence: merged code, passing tests, and released artifacts. Do not advance a
status because an interface has merely been designed.
