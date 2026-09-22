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

