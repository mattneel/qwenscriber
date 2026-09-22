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
| Versioned C-like ABI (v1) | Partial | Feature bits report 0x1f: version, allocator, log-mel, tokenizer, self-test, model, decode. The model family (begin, add_shard, finish, set_cache_format, requirements, audio_config) and decode (begin, step, tokens, end) are exported and covered by `tools/wasm_selftest.mjs` — invalid-state matrix, shard validation, requirements arithmetic, and a real decode. A native C header and the audio family are still missing |
| ABI conformance harness | Implemented | `zig build test-wasm` runs `tools/wasm_selftest.mjs` against the real module in Node |
| TypeScript SDK | Partial | `Qwenscriber.capabilities()`, typed errors, resumable download with digest verification, IndexedDB cache, worker channel, preprocessing, WebGPU adapter/limit probing with an upload planner, and the model/decode ABI bindings. `transcribe()` still stops with a typed error when no converted model is loaded; `stream()` has no surface |
| Browser example page | Experimental | `examples/browser` loads the module in a worker, reports capabilities, self-tests the core, preprocesses audio with measured throughput, plans GPU uploads from the adapter's real limits, and runs `matmul_q4` against the CPU reference. It is not a transcription demo, because the repository ships no model artifact |
| Qwen3-ASR architecture parsing | Implemented | `src/core/model_config.zig` plus `src/core/qwen3_asr/layout.zig` derive every shape from `config.bin`; validated against the 0.6B and 1.7B upstream configs |
| CPU reference encoder and decoder | Implemented | `src/core/qwen3_asr/{kernels,model,decoder}.zig`; 4 `tests/reference_check.zig` tests compare against `tools/reference` fixtures |
| Safetensors reading and model conversion | Implemented | `src/host/{safetensors,manifest,convert}.zig` and `qwenscriber-convert`. The 0.6B checkpoint converts and verifies at q4, q5, q8, and f16 (`--verify` re-reads every shard: f16 reproduces the checkpoint exactly, each quantized level within its own step). The reader takes every `*.safetensors` in a checkpoint directory, so upstream's multi-file 1.7B checkpoint needs no special case; the 1.7B configuration parses through the same validator |
| Qwen3-ASR 0.6B transcription | Implemented | `qwenscriber-transcribe` decodes `tests/fixtures/audio/asr_zh.wav` to the reference's own transcript 「甚至出现交易几乎停滞的情况。」 at q5, q8, and f16; `tools/reference/compare_fixtures.py` agrees on every compared stage — the prompt and the generated tokens are exact, `logits_step0` differs by `max|d|` 1.8e-4, and the convolution, encoder, and projector outputs differ by less than 1e-5. At q4 the audio tower output is too coarse for the first decision, which is a 0.095-point near-tie, and the run produces no tokens — see the quantization row |
| Quantization floor (0.6B) | Measured | The same clip at four conversion levels: q4 426 MB (4.32 bits/weight) produces no tokens, q5 520 MB (5.32), q8 812 MB (8.30), and f16 1.57 GB (16.0) all produce the reference transcript. The failure is a decision margin, not a broken stage: q4's mean first-step logit error (~3.6) exceeds the 0.095 gap between the two candidate tokens on this clip |
| WASM SIMD execution path | Partial | The core is built for `simd128` and host/WASM self-tests agree; full-model inference has not been run through WASM |
| WASM linear-memory budget | Measured | A wasm32 instance tops out at 2 GiB of linear memory (Node: 1.9 GiB accepted, 2.0 GiB refused; `--wasm-max-mem-pages` does not move it). A 2048-position model fits at 0.855 GiB (weights 422.8 MB + cache 469.8 MB + scratch 25.5 MB) with 2.066 GiB resident after load; 8192 positions does not — `qw_model_finish` returns `out_of_memory` before the cache can be allocated. `--max-positions` selects the runnable budget, and the next change here is a smaller or narrower (f16/q8) cache |
| WGSL kernel suite | Experimental | 11 kernels in `gpu/shaders`: the decoder's matmuls, RMSNorm, RoPE, causal attention, and gated SiLU, plus the audio tower's LayerNorm, GELU, and 3x3 stride-2 convolution. `node tests/gpu/harness.mjs` compares 14 cases against Zig CPU references and passes under SwiftShader on this host (worst case: RoPE `max_abs` 5.0e-4), including a `quant_hash` oracle that matches the Zig self-test bit for bit |
| WebGPU backend integration | Partial | The SDK acquires a device, reads the adapter's real limits, requests the raised `max` limits a caller declares — a device starts at the specification's 16384-byte workgroup-storage default, which the convolution kernel's 18432 staged bytes exceed — plans sharded uploads, and runs `matmul_q4` end to end (`packages/qwenscriber/src/gpu/{matmul_q4,runtime,upload_plan}.ts`), exercised by `examples/browser`; no encoder or decoder kernel is dispatched yet, so `backend: "webgpu"` does not accelerate inference |
| Run metrics recording | Implemented | `qwenscriber-transcribe --metrics <path>` writes a JSON record of model identity, the shape a run executed, per-phase timings, derived rates, and the build mode, processor, and revision (`src/host/metrics.zig`; documented in [Performance](development/performance.md)) |
| Qwen3-ASR 1.7B | Implemented | The 4.7 GB two-shard checkpoint converts at f16 (4.08 GB, verified element for element) and at q5 (1.34 GB), and `qwenscriber-transcribe` decodes the same clip to the reference transcript. `tools/reference/compare_fixtures.py` agrees on every compared stage against the `Qwen3-ASR-1.7B-hf` checkpoint upstream publishes — the prompt and the generated tokens are exact, `logits_step0` differs by `max\|d\|` 9.5e-05, and the convolution, encoder, and projector outputs by less than 1e-5 — which also settles the interleaved MRoPE the checkpoint declares: the runtime does not model it and matches anyway. Read the `-hf` repository and not the raw checkpoint, which is the converter's input; pointing `transformers` at the raw one produces errors that look like missing library support. No browser execution yet, and the remaining obstacle is dispatch rather than memory: the key/value cache is quantized through the same plane format as the weights and selectable through the ABI (462 MiB at 8192 positions against 1792 MiB in f32, at the same decode speed) |
| Quantized key/value cache | Implemented | `--cache q8`, and `qw_model_set_cache_format` through the ABI, store the cache in the model's own quantized plane layout: 462 MiB at 8192 positions against 1792 MiB in f32, at the same decode speed on the same clip (1.55 against 1.49 tokens per second) and with identical token ids. `tools/wasm_selftest.mjs` covers the refusals and checks the reported bytes against the f32 ones (115.5 MiB against 469.8 MB for a 2048-position 0.6B). The SDK does not call the setter yet, so a browser still gets the f32 cache |
| Streaming transcription | Partial | Microphone capture is implemented and measured in a browser: `MicrophoneCapture` runs an AudioWorklet that sends a block only on credit, a queue bounded by the core's own 30-second capacity, and counters for what each bound dropped. `tests/browser/audio_capture.html` drives both cases against a synthetic `MediaStream` — a 440 Hz tone arrives intact and converts to 16 kHz, and a reader that never reads loses the oldest audio and can see it — with no microphone and no permission. `stream()`, meaning decoded segments as they are produced, has no surface; the runtime still transcribes whole clips |
| Native C FFI packages | Planned | No header or shared library is emitted |
| Elixir/Zigler integration | Planned | Nothing implemented |

## How to reproduce the claims

```sh
zig build test          # core, reference-math, and CLI unit tests
zig build wasm          # wasm32-freestanding module
zig build test-wasm     # ABI conformance against the built module
node tests/gpu/layout_drift.mjs   # shader/constant drift gate
```

Conversion, transcription, and the independent measurements behind the rows above:

```sh
zig build                                     # native tools, ReleaseFast with -Doptimize=ReleaseFast
zig-out/bin/qwenscriber-convert \
  --input models/Qwen3-ASR-0.6B --output models/qwen3-asr-0.6b-q5 --quant q5 --verify
zig-out/bin/qwenscriber-transcribe \
  --model models/qwen3-asr-0.6b-q5 --audio tests/fixtures/audio/asr_zh.wav \
  --metrics run.json                          # per-phase timings and the machine they were taken on
```

The stage-by-stage comparison against the released implementation needs the Python reference in
`tools/reference`, which is bring-up tooling rather than a runtime dependency. It reads the `-hf`
repository upstream publishes for each checkpoint, not the conversion source:

```sh
# once per checkpoint, from https://huggingface.co/Qwen/Qwen3-ASR-<size>-hf
tools/reference/transcribe_reference.py --model models/Qwen3-ASR-0.6B-hf --audio <wav> --output reference/
zig-out/bin/qwenscriber-transcribe --model <model dir> --audio <wav> --dump ours/
tools/reference/compare_fixtures.py --ours ours --reference reference
```

Passing a raw checkpoint to `transformers` produces errors that look like library gaps and are not: its
`thinker_config` is not read, and its tensor names are pre-conversion. `tools/reference/prepare_hf_layout.py`
explains both and refuses to guess unless told to.

Microphone capture needs a browser and nothing else — the harness feeds the SDK a synthetic
`MediaStream` from the browser's own audio graph, so there is no microphone to grant:

```sh
npm --prefix packages/qwenscriber run build
python3 -m http.server 8791        # from the repository root
# open http://127.0.0.1:8791/tests/browser/audio_capture.html, click "start", read window.__result
```

That comparison runs for either size, against the `-hf` repository each checkpoint publishes. It is
not available for a *raw* checkpoint: `transformers` reads the conversion output, and pointing it at
the conversion source produces errors that look like missing library support. See the 1.7B row and the
notes in `tools/reference/prepare_hf_layout.py`.

The GPU conformance harness (`node tests/gpu/harness.mjs`) needs a Chromium with WebGPU enabled and a
GPU adapter; it is not part of the default command set.

## Not yet true

- No release, package, or artifact has been published.
- The Zig gate failed on `main` before this work, and the reason was the repository's state rather
  than the compiler: `tools/wasm_selftest.mjs` was committed while the `qw_features`, `qw_model_*`, and
  `qw_decode_*` exports it checks still existed only in the working tree, so the gate built the module
  from the commit and found them missing. Committing that work turned the gate green — verified by the
  run for `34b2cad` — and the TypeScript SDK job needed the mel-frame expectation and the default
  model corrected in the same way.
- No performance number on this page has been measured under the benchmark hooks described in the
  architecture pages.

Update this page from evidence: merged code, passing tests, and released artifacts. Do not advance a
status because an interface has merely been designed.
