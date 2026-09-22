# Qwenscriber

Local-first Qwen3-ASR inference with a small stack: Zig, WebAssembly SIMD, WebGPU/WGSL,
and TypeScript.

Qwenscriber is intended to run speech-to-text locally, including entirely in modern
browsers. Audio should not need to leave the user's machine, and applications should not
need a hosted transcription service.

> **Status:** pre-alpha design and bring-up. Qwen3-ASR 0.6B is the first correctness target;
> Qwen3-ASR 1.7B is the production architecture target. Nothing here should be read as a
> claim that a public release is already usable.

## Documentation

- [Read the documentation](https://mattneel.github.io/qwenscriber)
- [Browse the mdBook source](docs/src/SUMMARY.md)
- [Architecture](docs/src/architecture/overview.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## Target API

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

The API above is directional until the implementation and compatibility policy are released.

