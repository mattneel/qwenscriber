# TypeScript SDK

`@qwenscriber/qwenscriber` is the intended public package for JavaScript and TypeScript consumers.
It wraps the raw WASM ABI, owns host/runtime objects, and presents typed asynchronous operations.

> The signatures on this page are provisional until the first published package declares a
> compatibility policy.

## Create options

```ts
type Backend = "auto" | "webgpu" | "wasm";
type Quantization = "q4" | "q5";

interface QwenscriberOptions {
  model: string;
  quantization: Quantization;
  backend?: Backend;
}
```

Unknown options should not be silently ignored once configuration is stable. Defaults must be
visible in API documentation and passed explicitly at important internal call sites.

## Lifecycle

```ts
const asr = await Qwenscriber.create({
  model: "qwen3-asr-0.6b",
  quantization: "q4",
  backend: "auto",
});

try {
  const result = await asr.transcribe(audio);
  console.log(result.text);
} finally {
  asr.dispose();
}
```

Creation can perform network and cache I/O. Transcription is asynchronous and should execute in a
Worker-backed runtime. `dispose()` is idempotent at the TypeScript surface or produces a documented
state error; it must never double-free the core handle.

## Capabilities

`Qwenscriber.capabilities()` should report enough structured information to explain whether a model
can run and why a backend was selected. Expected categories include:

- WebGPU presence and relevant adapter limits;
- WASM SIMD support;
- WASM thread support, `SharedArrayBuffer` availability, and cross-origin isolation as three separate
  facts, because threads need all three and a page can easily have one or two;
- Worker and AudioWorklet support;
- compatible quantization/layout features; and
- selected backend after creation.

Raw adapter/vendor detail should be opt-in diagnostics, not ordinary application API surface. Thread
support is a *request*, not an assumption: the default artifact is single-threaded, and the SDK never
requires the embedder to set COOP/COEP ([ADR-0005](../project/decisions/0005-thread-enabled-wasm-build.md)).

## Errors

The SDK maps numeric core errors and host failures to typed JavaScript errors with a stable code,
human-readable message, operation context, and a preserved cause when one exists. Applications must
not parse error-message prose to make decisions.

## Streaming

`stream()` is planned after non-streaming correctness. Its eventual contract must define
backpressure, cancellation, partial/final segments, timestamps, audio queue bounds, and what happens
when input outruns inference. No example in this book overrides that future contract.

