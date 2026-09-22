# Zig core

The Zig core is the portable semantic center of Qwenscriber. The same source should support native
integration, `wasm32-freestanding`, tooling, and deterministic reference tests without dragging
host-specific APIs into the core.

## Modules

The exact source tree may evolve, but responsibilities should remain separable:

| Area | Responsibility |
| --- | --- |
| ABI | Version negotiation, handles, byte ranges, error codes, and exported entry points |
| Audio | PCM conversion, channel handling, resampling, log-mel features, normalization |
| Format | Manifest/container parsing, bounds/alignment/checksum metadata, tensor descriptors |
| Tokenizer | Vocabulary loading, tokenization, detokenization, text assembly |
| Decode | Decode lifecycle, token selection, stop rules, KV-cache bookkeeping |
| Tensor reference | Straightforward deterministic kernels used as the correctness oracle |
| Tensor SIMD | Measured vectorized CPU/WASM implementations |
| Runtime | Bounded state, arenas, scratch planning, model/session ownership |

## Freestanding WASM

The browser module targets `wasm32-freestanding`. It supplies its own explicit allocation surface
and imports only the minimal host functions the ABI documents. It must not assume filesystem,
environment, clock, threads, sockets, libc, WASI, Node, or Emscripten.

Use idiomatic Zig vectors first for SIMD. Inspect emitted code and benchmarks before adding raw WASM
intrinsics or handwritten modules.

## Native foundation

Native builds expose an idiomatic Zig API and a stable C ABI. Shared protocol, model, and decode
semantics live once in Zig. Bindings adapt the core; they do not reimplement it.

The Elixir integration uses Zigler so BEAM-facing scheduling and resource semantics remain explicit.
Other languages can bind the C ABI with their ordinary native FFI.

## Allocation

Initialization calculates and bounds persistent and scratch requirements. Long-lived model/session
state is distinct from reusable per-run scratch. Hot loops avoid allocation. Development builds make
allocation counts and peaks observable.

