# Architecture overview

Qwenscriber separates orchestration, portable systems logic, and parallel tensor compute so each
layer uses the environment it understands best.

| Component | Primary responsibilities | Explicitly does not own |
| --- | --- | --- |
| TypeScript SDK | Public API, downloads, cache, Workers, WebGPU device/buffer/pipeline lifecycle,
  shard orchestration | Tensor math implementation, tokenizer internals, stable binary ABI |
| Zig core | PCM conversion, resampling, log-mel features, tokenization, metadata validation, decode
  state, KV-cache bookkeeping, CPU reference/SIMD kernels | Browser objects and JavaScript WebGPU
  bindings |
| WGSL kernels | Quantized matmul, fused dequantization, attention, normalization, RoPE,
  activations, convolution, encoder/decoder tensor operations | Downloads, caching, public API, model
  policy |
| Conversion tools | Read official metadata/checkpoints, map/validate/repack/quantize/shard tensors,
  generate manifests and checksums | Runtime inference and browser lifecycle |

## Data path

1. TypeScript resolves and validates a model manifest.
2. Shards are fetched or read from cache in bounded pieces.
3. Metadata is validated by the shared format logic.
4. Weight ranges are uploaded to multiple GPU buffers that respect adapter limits.
5. Zig/WASM converts audio and produces model features.
6. WebGPU executes large encoder and decoder tensor operations.
7. Compact logits or token decisions cross the synchronization boundary when required.
8. Zig owns decode state and token-to-text conversion.
9. TypeScript returns stable public result objects and releases temporary resources.

The WASM backend follows the same semantic model with CPU reference/SIMD kernels. That makes it a
fallback and an oracle for GPU conformance, not an excuse to compromise the WebGPU data path.

## Architectural constraints

- Qwen3-ASR 0.6B must not create hard-coded assumptions that prevent 1.7B.
- Browser builds are `wasm32-freestanding`, with no WASI, Emscripten, or libc requirement.
- The ABI is small, explicit, versioned, and C-like.
- WebGPU limits require shard- or layer-oriented buffers, not one multi-gigabyte allocation.
- Quantized weights should be unpacked/dequantized inside fused GPU kernels.
- No heavyweight general-purpose ML runtime sits underneath Qwenscriber.

Read [runtime boundaries](runtime-boundaries.md) before adding a cross-layer feature.

