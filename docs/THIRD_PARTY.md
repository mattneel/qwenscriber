# Third-party material

Qwenscriber's runtime is written from scratch. This file records what was read
while doing that, under what license, and what was taken from it, so the
provenance of every non-obvious decision is traceable.

Nothing in this repository copies implementation code from another ASR runtime.
Where a project is listed as "reference", it was read to establish exact
semantics (tensor names, arithmetic order, token ids, file layouts), and the code
in `src/` was written against those observations.

## Model artifacts and their configuration

| Source | License | Used for |
|---|---|---|
| `Qwen/Qwen3-ASR-0.6B` (Hugging Face) | Apache-2.0 | Checkpoint the converter reads; architecture configuration; chat template; tokenizer vocabulary and merges |
| `Qwen/Qwen3-ASR-0.6B-hf` (Hugging Face) | Apache-2.0 | The transformers-native checkpoint layout: canonical tensor names and a weight-tied output projection |
| `Qwen/Qwen3-ASR-0.6B` example audio (`asr_zh.wav`, `asr_en.wav`) | Apache-2.0 | End-to-end fixture audio, currently `tests/fixtures/audio/asr_zh.wav` |

No checkpoint weights are committed to this repository. `models/` and
`tests/fixtures/reference/` are gitignored, and `tests/fixtures/audio/` holds one
134 KiB clip so the end-to-end comparison has a deterministic input.

## Reference implementations read for semantics

| Source | License | What was learned from it, not copied |
|---|---|---|
| `transformers` `Qwen3ASRForConditionalGeneration`, `Qwen3ASREncoder`, `Qwen3AsrFeatureExtractor`, `Qwen3ASRProcessor` | Apache-2.0 | The exact convolution geometry and per-chunk packing rule, the windowed attention partition, the sinusoidal position embedding, the prompt template, the output format (`language X<asr_text>...`), and the log-mel tail (`clamp`, global-max dynamic range, normalization) |
| `transformers` `Qwen3RMSNorm`, `Qwen3Attention`, `Qwen3RotaryEmbedding` | Apache-2.0 | That the text decoder applies per-head query/key normalization before rotation, and that rotary embeddings are the non-interleaved `rotate_half` pairing over a plain (non-MRoPE) frequency base despite the `mrope_section` key present in the released configuration |
| `transformers` `audio_utils.mel_filter_bank` | Apache-2.0 | The Slaney mel scale and Slaney area normalization; reimplemented in Zig and pinned by a bit-exact fixture |
| `transformers` `Qwen2Tokenizer` | Apache-2.0 | The GPT-2 byte-to-unicode alphabet, which the detokenizer reimplements |

These packages are **not** runtime dependencies and appear nowhere in the
shipped code. The Python scripts under `tools/reference/` use them so that
`zig build test` has a trustworthy oracle; they are bring-up tooling, and the
Zig converter and runtime do not call into Python.

## Algorithms with published origins

| Where | Origin | Notes |
|---|---|---|
| `src/core/math.zig` `erf` | Numerical Recipes, Chebyshev fit of `erfc` | Accuracy better than 1.2e-7, verified against a table of high-precision values in the unit tests |
| `src/core/mel.zig` mel frontend | Whisper's log-mel convention, via the Qwen3-ASR feature extractor | Slaney mel scale, periodic Hann window, `center=True` reflect padding, final frame dropped |
| `src/core/mel.zig` FFT | Cooley-Tukey, 400 = 25 x 16 | Twiddle tables are computed at compile time; the sub-transforms are evaluated directly |
| `src/core/qwen3_asr/kernels.zig` attention | Scaled dot-product attention with a numerically stable softmax | Windowed for the audio tower, causal with grouped query heads for the decoder |

## Tooling

| Tool | License | Role |
|---|---|---|
| Zig | MIT | Compiler, build system, standard library |
| Node.js | MIT | Runs the WASM ABI harness (`tools/wasm_selftest.mjs`) |
| TypeScript | Apache-2.0 | The SDK's only development dependency |
| Playwright | Apache-2.0 | Drives a real browser for GPU verification (`tools/browser/run-page.mjs`) |
| PyTorch, NumPy, SoundFile | BSD-3-Clause / BSD-3-Clause / BSD-3-Clause | Used by the reference scripts under `tools/reference/` only |
| Emscripten | MIT / University of Illinois | Planned build toolchain for the thread-enabled `wasm32-emscripten` module only ([ADR-0005](src/project/decisions/0005-thread-enabled-wasm-build.md)); no thread-enabled artifact exists yet |

None of these are runtime dependencies of the browser build. The default
`wasm32-freestanding` browser build has no dependencies at all: one TypeScript
package and one WASM module that resolves zero imports. The planned threaded build
would be Emscripten-compiled and would therefore carry Emscripten's in-module
runtime and host glue, which is the cost ADR-0005 records.
