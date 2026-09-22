# Runtime boundaries

The boundary rule is simple: TypeScript owns host objects, Zig owns portable state machines and
byte-level semantics, and WGSL owns parallel tensor compute.

## TypeScript boundary

TypeScript can directly use browser APIs without forcing them through a WASM binding layer. It owns:

- `GPUAdapter`, `GPUDevice`, `GPUBuffer`, pipelines, command encoders, and error scopes;
- `fetch`, range requests, response streams, IndexedDB, and cache policy;
- Worker and AudioWorklet lifecycle;
- feature detection and backend choice;
- package-facing types, promises, cancellation, and progress; and
- conversion of stable ABI failures into typed JavaScript errors.

TypeScript must not duplicate manifest validation, tokenizer semantics, decode rules, or tensor
layout definitions in ways that can silently drift from the core.

## Zig boundary

Zig owns behavior that must match across browser and native builds:

- fixed-width serialized formats and validation;
- PCM normalization, resampling, and feature extraction;
- tokenizer and detokenizer behavior;
- architecture and tensor metadata;
- decode state, sampling/greedy decisions, and KV-cache bookkeeping;
- reference kernels and CPU/WASM SIMD fast paths; and
- stable handles, ownership, and error semantics.

Zig exports operations over numbers and bytes. It does not export compiler-defined slices,
optionals, error unions, or accidental struct layouts.

## WGSL boundary

WGSL receives validated metadata, bindings, dimensions, and dispatch sizes. Kernels own math, not
policy. A kernel should not infer model identity from magic tensor shapes or reach into unrelated
buffers.

Every kernel needs a deterministic CPU reference and conformance tests before it becomes a fast
path.

## Crossing a boundary

Before adding a crossing, answer:

1. Which side owns the allocation?
2. How is length/alignment validated?
3. Is the transfer a copy, a view, an upload, or a readback?
4. What is the maximum size and work bound?
5. How are cancellation and cleanup expressed?
6. What error crosses back?
7. Can the operation be batched or kept resident instead?

If these answers are fuzzy, the interface is not ready.

