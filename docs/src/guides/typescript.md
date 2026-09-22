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

## Microphone capture

Realtime input is implemented, and it is deliberately narrower than `stream()`. `MicrophoneCapture`
turns a microphone, or any `MediaStream` a caller already owns, into mono f32 samples:

```ts
const capture = await MicrophoneCapture.start(); // acquires a microphone
// or: await MicrophoneCapture.attach(stream);   // borrows one the caller owns

let chunk = capture.read(16000); // up to one second of mono f32
if (!capture.isNativeRate) {
  chunk = resample(chunk, capture.state.input_rate_hz, 16000);
}

const { dropped_samples, dropped_blocks } = capture.state; // what was thrown away, and how much
await capture.stop(); // detaches; stops the microphone only if this session acquired it
```

Every stage between the audio thread and the reader is bounded, and each bound reports what it cost:

| Stage | Bound | When it is reached |
| --- | --- | --- |
| Worklet send | `CAPTURE_BLOCKS_IN_FLIGHT_MAX` blocks, held by credit | the worklet drops the block and counts it in `dropped_blocks` |
| Sample queue | `CAPTURE_SAMPLES_MAX`, the core's own 30-second capacity | the oldest samples go, counted in `dropped_samples` |

The reader grants one block of credit per block it consumes, so a stalled consumer cannot make the
audio thread queue without limit. A caller that never reads loses the oldest audio, never memory, and
can see that it happened.

**The capture rate is the track's, not the context's.** Chromium grants the audio device's rate to an
`AudioContext` even when 16 kHz is requested, and hands the worklet a `MediaStream` track at the
track's own rate. The two disagree, and the samples follow the track, so `state.input_rate_hz` reports
what they actually are while `state.context_rate_hz` reports what the context claimed. Convert with
`resample` whenever `isNativeRate` is false. Believing the context's rate is what made this path
produce audio three times too fast the first time it was measured in a browser.

Acquiring a microphone needs a user gesture in most browsers, so `start()` belongs in a click or
keypress handler; a rejection from the user, or from a page without permission, reaches the caller
unchanged.

## Streaming

`stream()` — decoded segments as they are produced — is still planned. Its contract must define partial
and final segments, timestamps, cancellation, and what happens when input outruns inference. The
capture path above settles the audio-bounds half of that question and none of the rest.

