# FAQ

## Is Qwenscriber usable today?

Not as a supported release. It is in pre-alpha design and bring-up. The
[status page](../status.md) is intentionally pessimistic until code and CI prove otherwise.

## Why 0.6B first if 1.7B is the goal?

The smaller model shortens the correctness/debug loop. The architecture treats 1.7B as a constraint
from the start so bring-up cannot bake 0.6B dimensions or memory assumptions into every layer.

## Does audio leave the device?

Qwenscriber's intended inference path does not require audio egress. The embedding application still
controls its own network behavior and can upload anything it chooses.

## Why not use ONNX Runtime, TensorFlow, PyTorch, llama.cpp, or ggml?

The project goal is a small purpose-built runtime with direct control over browser memory, packed
weights, WebGPU kernels, WASM builds, and distribution. Those projects can inform correctness
work subject to licensing; they are not runtime dependencies.

## Why not use GGUF directly?

GGUF may be a conversion input. Qwenscriber's distribution format is designed around browser
sharding, IndexedDB/cacheability, strict bounded parsing, adapter limits, and fused WebGPU execution.

## Why split TypeScript, Zig, and WGSL?

TypeScript naturally owns browser objects, Zig provides portable explicit systems logic, and WGSL
executes parallel tensor math where the data lives. Forcing one language to impersonate all three
creates larger bindings, more copies, and murkier ownership.

## What happens without WebGPU?

`backend: "auto"` is intended to use WASM SIMD when compatible. The WASM path is also the correctness
oracle. Actual model/device support will be published from measurements; 1.7B CPU inference is not
promised to be pleasant everywhere.

## Does it require SharedArrayBuffer?

Not for the default build. The non-streaming API avoids requiring it, and the default
`wasm32-freestanding` module runs in a worker without it. Two things do use it, both capability-gated:
streaming, where shared memory materially improves latency, and the thread-enabled
`wasm32-emscripten` build ([ADR-0005](../project/decisions/0005-thread-enabled-wasm-build.md)), which
needs a `SharedArrayBuffer` and therefore cross-origin isolation (COOP/COEP) on the embedding page. A
page that cannot set those headers still transcribes; it just runs single-threaded.

## Why does Emscripten appear at all if dependencies are so restricted?

Because the WASM threads proposal cannot create a thread: a module can use atomics and shared memory,
but only the host can instantiate it in several workers. Emscripten supplies that runtime, so it is
sanctioned as a build toolchain for one optional artifact —
[ADR-0005](../project/decisions/0005-thread-enabled-wasm-build.md) — and never as a runtime dependency
of an application or of the default build.

## Can I use it from Elixir or another native language?

The planned native foundation provides an idiomatic Zig API and stable C ABI. Elixir uses Zigler;
other native-FFI languages bind the C ABI. These integrations remain planned until released.

## Where are model weights stored?

Not in the source repository. Browser artifacts are downloaded and cached under explicit model and
format identities. Native applications choose their storage/resolver. Every artifact must retain
provenance, license, format, quantization, and integrity metadata.

## What browsers are supported?

No support matrix is declared before end-to-end conformance and performance tests. WebGPU presence
alone is insufficient; relevant limits and required shader behavior matter.

