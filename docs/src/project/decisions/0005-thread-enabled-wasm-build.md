# ADR-0005: Thread-enabled WASM build via Emscripten

- **Status:** Accepted
- **Date:** 2026-09-22
- **Supersedes:** the freestanding-only constraint in AGENTS.md and
  [ADR-0001](0001-runtime-boundaries.md) (partially — see Consequences)

## Context

The browser core was specified as `wasm32-freestanding`, with no WASI, Emscripten, libc, or embedded
JavaScript runtime. That build works today: the module has **zero imports**, exports its own memory,
and passes its ABI self-test in Node and in a browser worker.

The cost of the constraint is thread *creation*. The WebAssembly threads proposal gives a module
atomics, `memory.atomic.wait`/`notify`, and shared memory, but no way to start a thread: the module
can only become multi-threaded if something outside it instantiates it in several workers and hands it
work. `wasm32-freestanding` has no thread-spawn primitive at all, so a hand-rolled pool means owning
that handshake in TypeScript. Emscripten ships exactly that runtime (`-pthread`), and that is the only
reason it is being considered: not for performance libraries, not for a general-purpose ML runtime.

Threads matter for the CPU-side path — preprocessing, and the WASM fallback decoder that has to be a
credible alternative when no WebGPU adapter exists. They do nothing for the WebGPU path, which is the
primary performance target and is already asynchronous.

## Decision

Two WASM targets are sanctioned, and threads are the reason for the second:

1. **`wasm32-freestanding`** — the default build. Single-threaded, zero imports, no libc, no JS glue.
   It stays the portability build, the correctness oracle for the WebGPU kernels, and the artifact
   whose cross-target parity (native ↔ WASM) is asserted by tests.
2. **`wasm32-emscripten`** — the thread-enabled build. It may link libc and pthread support inside the
   module and may depend on Emscripten's JavaScript glue.

Both builds expose the same ABI, and the ABI contract in [ADR-0002](0002-stable-abi.md) is unchanged:
fixed-width integers, linear-memory offsets, integer handles, machine-readable status codes, no Zig
types on the boundary. Core sources stay free of operating-system and libc assumptions; only the build,
its host glue, and the thread runtime differ.

Emscripten is a **toolchain**, never an application runtime dependency: an application embedding
Qwenscriber does not add Emscripten to its own dependency list, and no general-purpose ML runtime
becomes acceptable by association.

## Consequences

- **Cross-origin isolation becomes a real requirement for threads.** Any thread-enabled build needs a
  `SharedArrayBuffer`, and browsers gate *instantiating* shared memory behind cross-origin isolation,
  so the embedding page must serve COOP/COEP headers. `Qwenscriber.capabilities()` already reports
  `wasmThreads`, `sharedArrayBuffer`, and `crossOriginIsolated` separately, because
  `wasmThreads: true, crossOriginIsolated: false` is a real combination. The threaded path is selected
  only when every condition holds; otherwise the single-threaded path is used, so a page that cannot
  set headers still works.
- **The loader must provide imports for the threaded build.** The freestanding build requires zero
  resolvable imports; the Emscripten build imports its `env`. The SDK's instantiation error message
  must stop implying that every build is import-free.
- **Artifact identity must record the build.** A release ships one or both, and the manifest or package
  metadata has to say which, because they are not interchangeable at the byte level even though the ABI
  matches.
- **The ABI self-test runs against both builds.** Parity between them is a test, not an assumption.
- **CI gains an emsdk step** when the threaded target lands; today's `check` gate is unaffected.
- **Threads must earn their place by measurement.** Real-time factor, preprocessing throughput, and
  WASM fallback tokens/second have to improve enough to justify the isolation requirement, the larger
  artifact, and the extra toolchain. Until that is measured, the threaded build is *planned*, not
  default: the freestanding build remains the one CI builds and the one the SDK prefers when the host
  cannot supply a shared memory.
