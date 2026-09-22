# ADR-0001: Runtime boundaries

- **Status:** Accepted
- **Date:** 2026-09-22

## Context

Qwenscriber must use browser-native WebGPU effectively while sharing preprocessing, format, and
decode semantics across browser and native targets. Making Zig own JavaScript WebGPU objects would
require a large binding layer. Making TypeScript own model semantics would duplicate logic and make
native/browser behavior drift. Running tensor math through WASM for every operation would add copies
and synchronization.

## Decision

- TypeScript owns public API, browser lifecycle, downloads/cache, Workers, and WebGPU objects.
- Zig owns portable byte-level and state-machine semantics, audio preprocessing, tokenizer, model
  validation, decode state, reference/SIMD CPU kernels, and the exported ABI.
- WGSL owns large parallel tensor operations.
- Weights are uploaded in validated shards and kept GPU-resident where possible.

## Consequences

Interfaces between layers must be explicit about ownership, bytes, bounds, and synchronization.
Some metadata representation must be shared/generated to prevent drift. The design avoids a giant
WebGPU binding layer and keeps CPU/GPU transfers out of the per-layer steady state.

## Rejected alternatives

- Zig directly wrapping the complete JavaScript WebGPU API.
- TypeScript reimplementing tokenizer, manifest, and decode semantics.
- A WASM dispatcher mediating every GPU tensor operation.
- A heavyweight third-party ML runtime owning all layers.

