# Performance

Performance is an architectural property, but every claimed improvement needs a reproducible
measurement.

## Required metrics

| Area | Metrics |
| --- | --- |
| Loading | Cold model load, cache-hit load, manifest parse, shard fetch, GPU upload |
| Latency | First-token latency, final-result latency, streaming chunk latency when implemented |
| Throughput | Audio preprocessing, encoder time, decoder tokens/sec, real-time factor |
| Memory | Peak WASM memory, JavaScript staging bytes, GPU weights/activations, cache footprint |
| Distribution | Quantized model/shard sizes and compression/transfer behavior |
| Kernels | Dispatch time, synchronization/readback time, effective bandwidth, relevant shape |

## Benchmark record

Every published number identifies:

- revision and build mode;
- model, quantization, format version, and shard layout;
- browser/runtime and version;
- OS, CPU, GPU, memory, and relevant adapter limits;
- audio duration/content class without exposing private audio;
- warm-up and sample counts;
- cold versus warm/cache state; and
- median plus a useful tail/spread measure.

Do not compare a debug reference with an optimized release path and call the result an architecture
win.

## Recording a run

`qwenscriber-transcribe --metrics <path>` writes one run as a JSON record:

```sh
zig build -Doptimize=ReleaseFast
zig-out/bin/qwenscriber-transcribe \
  --model models/qwen3-asr-0.6b-q5 \
  --audio tests/fixtures/audio/asr_zh.wav \
  --metrics q5.json
```

The record carries what a comparison needs, so that none of it has to be remembered from the shell
that produced it:

| Group | Fields |
| --- | --- |
| identity | `model_id`, `quantization`, `bits_per_weight`, `tensors`, `parameters`, `payload_bytes`, `shard_count`, `shard_bytes` |
| `model` | the audio and text dimensions the run actually executed, and `kv_cache_bytes` |
| `run` | mel `frames`, `encoder_steps`, `prompt_tokens`, `generated_tokens`, `audio_ms` |
| `timings_ms` | `load`, `mel`, `encoder`, `first_token` (the prompt's prefill), `decode` (generation only), `total` |
| `derived` | `tokens_per_second` and `realtime_factor`, computed from the durations in the same record |
| `environment` | `build_mode`, `os`, `arch`, processor `cpu`, `cpu_count`, `revision`, `zig` |

Two records are comparable when their identity and `environment` groups agree. "A debug build was
faster" is a claim this format makes hard to state by accident.

`revision` is git's short HEAD at build time, `unknown` in a build made outside a checkout, and
whatever `-Drevision=<value>` says, which is what a release build passes.

Not yet recorded: warm-up and sample counts, and a spread across repeats. Until those exist, publish
a single run only next to the exact command that produced it, as the numbers in
[Project status](../status.md) do.

## Design sketches

Before implementing a hot path, sketch network, disk/cache, memory, and CPU/GPU bandwidth/latency.
Account for how many times each byte moves. A nominally fast resource used repeatedly can dominate.

## Browser-specific traps

- Unnecessary network/JS/WASM/GPU copies.
- Pipeline construction during steady-state inference.
- Full-tensor readbacks or CPU/GPU synchronization per layer.
- One giant buffer assumption instead of adapter-aware shards.
- Materializing full FP16 weights from packed Q4/Q5.
- Allocation and garbage-collection pressure in the decode loop.
- Benchmarks that exclude first-use compilation or silently reuse caches.

Keep correctness/reference implementations even after fast paths land.

