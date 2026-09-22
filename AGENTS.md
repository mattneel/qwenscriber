# AGENTS.md

This file is the operating contract for humans and coding agents working in this repository.
Read it before changing code. The complete style rationale lives in
[`docs/TIGER_STYLE.md`](docs/TIGER_STYLE.md).

## Project truth

Qwenscriber is a small, high-performance, local-first runtime for Qwen3-ASR.

- Qwen3-ASR 0.6B is the bring-up target.
- Qwen3-ASR 1.7B is the production architecture target.
- Browser inference is client-side. No server or cloud transcription API is required.
- The intended stack is Zig + WASM SIMD + WebGPU/WGSL + TypeScript.
- The runtime owns its inference implementation. Heavy general-purpose ML runtimes are not
  runtime dependencies.

Never describe a planned capability as implemented. Documentation, tests, examples, and release
notes must distinguish **implemented**, **partial**, **experimental**, and **planned** behavior.

## Priorities

Resolve tradeoffs in this order:

1. Safety and correctness.
2. Predictable performance.
3. Developer experience.

Simplicity serves all three. It does not excuse an incomplete model, an accidental ABI, an
unbounded resource path, or an unmeasured performance claim.

## Ownership boundaries

Keep each responsibility in its intended layer.

| Layer | Owns |
| --- | --- |
| TypeScript | Public API, browser lifecycle, feature detection, downloads, caching, workers,
  WebGPU objects, GPU buffers, pipeline creation, and orchestration |
| Zig | Portable systems logic, audio preprocessing, tokenizer, metadata parsing, decode state,
  CPU reference kernels, WASM SIMD kernels, allocation, and stable exported ABI |
| WGSL | Large parallel tensor operations and fused GPU kernels |

Do not build a large Zig-to-JavaScript binding layer for WebGPU objects. Do not route every tensor
operation through WASM. Long-lived weights should be parsed, uploaded, and kept GPU-resident where
possible.

## Zig and build policy

- Target the repository-pinned Zig master toolchain (`0.17.0-dev` lineage), not an older stable
  release by habit.
- Verify APIs against the installed compiler. Zig master moves; stale examples are not evidence.
- The browser core targets `wasm32-freestanding`.
- The WASM core must not require WASI, Emscripten, libc, Node, or an embedded JavaScript runtime.
- The root `build.zig` is authoritative for Zig artifacts.
- Keep the expected commands working as their steps land:
  `zig build`, `zig build test`, `zig build wasm`, and `zig build test-wasm`.
- Format and test after meaningful changes. Inspect compiler errors immediately rather than
  accumulating speculative code.

## Dependencies

Prefer dependencies in this order:

1. Pure Zig.
2. A Zig-native wrapper over a C or C++ library.
3. A C or C++ library consumed through `@cImport`/translate-c where practical.
4. A thin wrapper we own that exposes Zig-native idioms internally.
5. An external build mechanism only when the previous options are genuinely inadequate.

Prefer implementing small, bounded functionality over importing a large general-purpose package.
The browser runtime should have zero production dependencies unless a dependency clears a high
bar. Do not introduce ONNX Runtime, TensorFlow, PyTorch, llama.cpp, ggml, Emscripten, or a giant
JavaScript ML framework as a runtime dependency.

Record each new dependency's purpose, license, update mechanism, and why a smaller option was not
sufficient.

## Control flow and bounds

- Use explicit, simple control flow. Avoid recursion in runtime code.
- Put an explicit upper bound on loops, queues, buffers, shards, tensor ranks, audio duration, and
  decode work. Assert a deliberate infinite event loop where one is truly intended.
- Use explicitly sized integer types at serialized and ABI boundaries. Avoid accidental
  architecture-sized layouts.
- Keep variables in the smallest useful scope.
- Keep functions at or below 70 lines. Centralize branching; push leaf computation into focused
  helpers.
- Keep lines at or below 100 columns unless a machine-generated format makes that unreasonable.
- Split compound assertions and complicated conditions so the valid and invalid spaces are clear.
- Handle all expected errors. Traps and panics are for violated invariants, not malformed input or
  unsupported models.

## Assertions and invariants

Assertions document and enforce programmer assumptions.

- Assert inputs, outputs, preconditions, postconditions, and internal relationships.
- Pair important assertions across different paths, such as before serialization and after parse.
- Assert compile-time layout, size, alignment, and constant relationships.
- Test the positive space and the negative space.
- Never use assertions as a replacement for understanding the model topology or buffer lifetime.

## Memory discipline

- Make ownership and lifetime explicit.
- Separate long-lived weights from ephemeral activations.
- Prefer bounded arenas and reuse for hot paths.
- Avoid invisible copies across network buffers, JavaScript, WASM memory, and GPU buffers.
- Treat browser memory and WebGPU binding limits as architectural constraints.
- Inspect `maxBufferSize` and `maxStorageBufferBindingSize`; never assume one enormous GPU buffer.
- Do not dynamically allocate in steady-state hot loops unless the design explicitly justifies it
  and measurements support it.

## ABI policy

The exported ABI is a product surface, not a compiler accident.

- Use C semantics: fixed-width integers, explicit byte buffers, integer or opaque handles, and
  documented ownership.
- Version the ABI from the beginning.
- Do not export Zig slices, error unions, optionals, or structs with accidental layout.
- Return machine-readable error codes and expose contextual error information deliberately.
- Validate every pointer, length, alignment, handle, enum value, and state transition.
- Add layout and compatibility tests before changing the ABI.
- The CLI and language bindings depend on the core SDK; the core SDK never depends on them.

## Model and tensor policy

- Implement the actual Qwen3-ASR audio encoder, projection, and decoder topology. Do not assume it
  is a generic text model with an audio tensor attached.
- Do not hard-code 0.6B dimensions in ways that block 1.7B.
- Keep tensor names, shapes, layouts, and conversions validated at every boundary.
- The distribution format is Qwenscriber-owned and optimized for streaming, caching, sharding,
  and direct WebGPU use. GGUF may be an import source, not the architecture.
- Prefer fused unpack/dequantize/multiply kernels over materializing full floating-point weights.
- Share format and tensor definitions between converter and runtime wherever feasible.
- Never commit downloaded checkpoints or generated weight shards.

## WebGPU and WGSL

- Establish correctness against deterministic Zig CPU references before optimizing a kernel.
- Test small tensors, edge shapes, invalid metadata, and numerical tolerances.
- Minimize CPU/GPU synchronization and read back only compact results when possible.
- Measure adapter limits and design for multiple layer- or shard-oriented buffers.
- Keep shader interfaces explicit and validate bindings, offsets, alignment, and dispatch bounds.
- Do not claim speedups without benchmark data and a reproducible baseline.

## Testing expectations

Changes should add or update the smallest test that proves the behavior.

Minimum test layers include:

- Zig unit and deterministic math tests.
- ABI version, size, layout, and invalid-input tests.
- Manifest and tensor-shape validation tests.
- Audio preprocessing, tokenizer, quantization, and dequantization tests.
- CPU-versus-WGSL conformance tests.
- Browser integration tests for WebGPU and WASM SIMD.
- Small end-to-end fixtures that are legal and practical to keep in git.

The first meaningful milestone is a correct real transcript through our preprocessing, runtime,
backend, and decoder—not a large abstraction graph.

## Performance work

- Sketch network, disk, memory, and CPU costs before implementation.
- Record model load, cache-hit load, first-token latency, real-time factor, preprocessing
  throughput, GPU upload and kernel time, peak WASM/GPU memory, model size, and decoder tokens/sec.
- Benchmark release builds on identified hardware and browser versions.
- Keep correctness baselines when adding fast paths.
- Optimize measured bottlenecks without erasing clear ownership or bounded behavior.

## Documentation

- Keep `README.md` short and honest; detailed material belongs in the mdBook under `docs/`.
- Update documentation in the same change as a public API, ABI, command, configuration, format, or
  support-status change.
- Mark illustrative interfaces as provisional.
- Explain why a design exists, not only what it does.
- Add or update an ADR when a change alters a durable cross-layer decision.
- The published site is `https://mattneel.github.io/qwenscriber` and is built from `docs/` by CI.

## Git and generated files

Check repository status before and after work. Do not add Zig caches, `node_modules`, browser
caches, model checkpoints, generated shards, large benchmark output, or temporary fixtures.

Do not modify generated artifacts by hand. Change their source or generator and reproduce them.
Do not commit or publish unless explicitly asked.

## Licensing and provenance

Prefer official model files/specifications, Zig sources/docs, WebGPU/WGSL specifications, and
browser vendor documentation. Track the license and provenance of adapted code, algorithms,
fixtures, tokenizer data, and model artifacts. Study compatible implementations, but independently
implement where licensing or architecture requires it.

## Definition of done

A change is done when:

- the implementation is bounded and its ownership is clear;
- expected errors are handled and invariants are asserted;
- relevant unit, conformance, and integration tests pass;
- formatting and repository checks pass;
- performance claims have measurements;
- public behavior and status documentation are current; and
- no generated junk or model weights have entered the repository.

