# Node, Bun, and Workers

The TypeScript package uses the WASM ABI outside browsers unless a runtime-specific native path is
explicitly selected. Host capabilities vary; package compatibility does not imply that a full 1.7B
model fits every environment.

## Node.js

Node uses the freestanding WASM core through the package's host adapter. WebGPU availability and
behavior depend on the selected Node/runtime environment; otherwise the WASM backend is the portable
path. Filesystem model loading is a host adapter concern, not a core import.

## Bun

Bun uses the same WASM path by default. A future `bun:ffi` adapter may bind the native C ABI for
applications that deliberately choose native artifacts. The two paths must share observable model,
decode, and error semantics.

## Cloudflare Workers and edge runtimes

The freestanding WASM module avoids WASI, Node, and libc assumptions, making it structurally suitable
for Worker-like hosts — and it is the default precisely because most edge runtimes cannot grant
cross-origin isolation or a `SharedArrayBuffer`, which the thread-enabled build needs. Actual full-model
viability depends on runtime bytecode, memory, CPU-duration, artifact-fetch, and GPU limits. Do not
market compatibility until a target is tested end to end.

A Worker adapter must provide explicit bytes, cache/fetch operations, and timing/cancellation from
the host. The core must not grow ambient network or filesystem assumptions to support it.

## Capability errors

When a host cannot meet a model's requirements, fail before large downloads or allocations when
possible. Return which requirement failed—feature, memory, artifact size, backend, or execution
limit—rather than a generic initialization error.

