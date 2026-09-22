# Qwenscriber

Qwenscriber is a tiny, high-performance, local-first speech-to-text runtime for Qwen3-ASR.
Qwen3-ASR 0.6B is the first bring-up target; Qwen3-ASR 1.7B is the primary production target and an
architectural constraint from the beginning.

The defining browser stack is:

| Layer | Role |
| --- | --- |
| TypeScript | Public API, browser integration, workers, model download/cache, and orchestration |
| WebGPU/WGSL | Large parallel tensor operations and fused quantized kernels |
| Zig/WASM SIMD | Portable systems logic, preprocessing, tokenization, decode control, and CPU
  fallback/reference math |

The model is intended to execute entirely on the user's machine. Browser inference should need no
server, cloud transcription API, or audio egress.

## Target developer experience

```ts
import { Qwenscriber } from "@qwenscriber/qwenscriber";

const asr = await Qwenscriber.create({
  model: "qwen3-asr-1.7b",
  quantization: "q4",
  backend: "auto",
});

const result = await asr.transcribe(audio);
asr.dispose();
```

Streaming is a later surface:

```ts
for await (const segment of asr.stream(microphone)) {
  console.log(segment.text);
}
```

These examples define the intended shape, not a promise that an unreleased build already implements
every method.

## Design goals

1. Correct execution of the actual Qwen3-ASR topology.
2. Local privacy by default.
3. Predictable memory use in browser-constrained environments.
4. Direct WebGPU execution without a heavyweight general-purpose ML runtime.
5. A small, explicit, versioned ABI shared by WASM and native integrations.
6. Reproducible artifacts built and released by CI.

## Non-goals

- A hosted transcription service.
- A wrapper around ONNX Runtime, TensorFlow, PyTorch, llama.cpp, or ggml.
- Treating GGUF as the internal architecture.
- Hiding incomplete features behind aspirational documentation.

Start with the [project status](status.md), then read the
[architecture overview](architecture/overview.md).

