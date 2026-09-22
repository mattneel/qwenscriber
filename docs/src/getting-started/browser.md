# Browser quickstart

> **Planned API:** this page fixes the intended developer experience while the first browser slice
> is implemented. Check [project status](../status.md) before depending on it.

## Create a runtime

```ts
import { Qwenscriber } from "@qwenscriber/qwenscriber";

const asr = await Qwenscriber.create({
  model: "qwen3-asr-1.7b",
  quantization: "q4",
  backend: "auto",
});
```

`create()` is responsible for capability checks, manifest validation, cache lookup/download, backend
selection, and runtime initialization. Applications should surface download progress and cancellation
because model artifacts are large.

## Inspect capabilities

```ts
const capabilities = await Qwenscriber.capabilities();

console.log(capabilities.webgpu);
console.log(capabilities.wasmSimd);
console.log(capabilities.worker);
```

Capability reporting should include the facts needed to explain backend selection without exposing
ordinary callers to raw GPU implementation detail.

## Transcribe

```ts
const result = await asr.transcribe(audio);
console.log(result.text);
```

Accepted audio containers and JavaScript input types will be documented only after the audio input
contract is implemented. Internally, the runtime converges on validated PCM and performs explicit
channel conversion, resampling, and feature extraction.

## Dispose

```ts
asr.dispose();
```

Disposal releases GPU buffers, WASM state, workers, and other runtime-owned resources. Cached model
artifacts are governed separately by cache policy.

## Execution model

Inference belongs in a Worker so it cannot monopolize the UI thread. WebGPU owns large tensor work;
Zig/WASM owns preprocessing, decode control, and fallback/reference computation. See
[browser execution](../architecture/browser-execution.md).

