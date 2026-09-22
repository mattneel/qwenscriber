# Browser execution

Inference must not lock the browser UI.

## Threads

| Context | Responsibilities |
| --- | --- |
| Main/UI thread | Application UI, user gestures, progress presentation, and message coordination |
| Worker | WASM instance, WebGPU orchestration where supported, model state, and inference loop |
| AudioWorklet | Time-sensitive capture/conditioning for streaming, when streaming is implemented |

The non-streaming API should not require `SharedArrayBuffer` when ordinary transferable buffers can
provide a reasonable path. Streaming may use shared memory where it materially improves latency,
with capability-gated alternatives.

Inside a worker, the core may itself use threads. Threads require a `SharedArrayBuffer`, and browsers
only let a page *instantiate* shared memory when it is cross-origin isolated, so a thread-enabled build
means the embedding page sends COOP/COEP headers. Because that is a host-page policy the SDK cannot
impose, thread use is capability-gated end to end: `capabilities()` reports `wasmThreads`,
`sharedArrayBuffer`, and `crossOriginIsolated` separately, the threaded build is selected only when all
three hold, and the single-threaded `wasm32-freestanding` build remains the default artifact. See
[ADR-0005](../project/decisions/0005-thread-enabled-wasm-build.md).

## Model loading

Model loading is an observable, cancellable process:

1. Fetch a small manifest.
2. Validate its schema, format version, model identity, and declared bounds.
3. Compare shard identity/checksums with the cache.
4. Fetch missing ranges or shards with bounded concurrency.
5. Upload validated ranges into GPU buffers sized for the adapter.
6. Release network and staging buffers promptly.

Avoid the copy chain `network → JavaScript → WASM → JavaScript → GPU` when TypeScript can validate
metadata through a narrow core call and upload the original range directly.

## Caching

IndexedDB is the intended browser cache. Cache identity must include model, format version,
quantization, shard checksum, and any layout feature that changes interpretation. An incomplete or
failed download must never masquerade as a valid cached model.

## Device loss and cancellation

WebGPU device loss, page lifecycle changes, aborted fetches, worker termination, and application
cancellation are expected operating conditions. They require deterministic cleanup and actionable
errors, not traps.

