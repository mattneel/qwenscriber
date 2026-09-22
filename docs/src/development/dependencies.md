# Dependency policy

Qwenscriber is intentionally dependency-austere. Every dependency increases supply-chain surface,
build complexity, binary size, compatibility work, and the number of semantics outside our control.

## Preference order

1. Pure Zig.
2. A Zig-ified wrapper over a C/C++ library.
3. A C/C++ library consumed with `@cImport`/translate-c where practical.
4. Our own thin Zig wrapper that exposes Zig-native internal idioms.
5. A separate external build mechanism only when the previous paths are inadequate.

Implement small bounded functionality locally rather than importing a general-purpose system.

## Browser rule

Strongly prefer zero production dependencies in the browser runtime. Qwenscriber owns its inference
runtime. Do not add ONNX Runtime, TensorFlow, PyTorch, llama.cpp, ggml, or a giant JavaScript ML
framework as a runtime dependency.

Emscripten is the one sanctioned exception, and only as a **build toolchain** for the thread-enabled
`wasm32-emscripten` module ([ADR-0005](../project/decisions/0005-thread-enabled-wasm-build.md)): it
supplies the pthread runtime that a freestanding WASM module cannot create for itself. It is not
required to embed or use Qwenscriber, it does not become part of an application's dependency list, and
the default `wasm32-freestanding` build stays free of it. Its cost is recorded like any other
dependency: larger artifact, host glue to load, an emsdk step in CI, and a `SharedArrayBuffer`
requirement that follows from threads rather than from Emscripten.

Projects may be studied for algorithms, tensor layouts, quantization schemes, formats, and
correctness comparisons subject to their licenses. Study is not permission to copy incompatible
code or turn the studied runtime into a hidden dependency.

## Proposal checklist

A dependency change records:

- exact purpose and call surface;
- why standard library/local implementation is insufficient;
- license and provenance;
- native, WASM, and supported-platform implications;
- transitive dependencies and external build tools;
- binary/package-size cost;
- security/update ownership;
- failure behavior and deterministic/reproducible pin; and
- removal or replacement path.

## Development dependencies

Test, documentation, and release tools are still dependencies. Pin them where reproducibility
matters, minimize privileged third-party CI actions, and do not allow tooling to redefine runtime
formats or semantics independently of Zig source.

