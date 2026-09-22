# Roadmap

The roadmap is milestone-based, not date theater. A milestone completes when its exit criteria are
demonstrated by tests and artifacts.

## M0 — Foundation

- Replace the `zig init` example with a clean Zig-centric repository.
- Pin current Zig master (`0.17.0-dev` lineage).
- Establish native/test/WASM build steps and repository hygiene.
- Define ABI v1 and instantiate the freestanding WASM module from TypeScript.
- Establish the TypeScript package, Worker boundary, and minimal browser example.
- Add WebGPU capability/device initialization.

**Exit:** CI builds/tests a tiny native core and `wasm32-freestanding` artifact; TypeScript calls the
versioned ABI in a browser test.

## M1 — Correct 0.6B vertical slice

- Implement bounded audio conversion, resampling, log-mel features, and normalization.
- Implement tokenizer/detokenizer and model/manifest parsing.
- Implement deterministic CPU tensor references and decode state.
- Inspect and convert the official Qwen3-ASR 0.6B topology.
- Produce the smallest real end-to-end transcript.

**Exit:** real Qwen3-ASR audio produces correct decoded text through Qwenscriber's own pipeline.

## M2 — WebGPU execution

- Add WGSL kernels one family at a time with CPU conformance tests.
- Add adapter-aware shard/buffer planning and GPU-resident weights.
- Implement Q4 first, then evaluate Q5 from measured behavior.
- Measure load, latency, real-time factor, memory, and kernel time.

**Exit:** 0.6B WebGPU output conforms to the reference path and has reproducible measurements.

## M3 — 1.7B production architecture

- Validate every layout and memory assumption against Qwen3-ASR 1.7B.
- Tune sharding, fused quantized kernels, activation reuse, and cache behavior.
- Define the initial supported browser/device envelope from evidence.

**Exit:** quantized 1.7B completes end-to-end browser transcription on documented hardware within
published resource/performance bounds.

## M4 — Product surfaces

- Stabilize and publish the TypeScript SDK.
- Package native Zig and C ABI artifacts.
- Add Elixir/Zigler and selected native-language integrations.
- Harden Node/Bun adapters and evaluate Worker environments honestly.
- Add streaming with bounded queues, backpressure, cancellation, and partial/final segments.

**Exit:** CI publishes versioned, mutually compatible artifacts and docs with a declared support
policy.

## Later work

Additional model variants, quantizations, sampling modes, timestamps, diarization, or platform
integrations require separate evidence and design. They are not implied by the milestones above.

