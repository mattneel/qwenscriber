# WebGPU and WGSL

WebGPU performs essentially all large tensor operations in the browser backend. Zig/WASM remains in
control of portable semantics without becoming a per-operation transport layer.

## Kernel families

The expected kernel set includes:

- packed Q4/Q5 matrix multiplication with fused unpack/dequantization;
- attention and KV-cache access;
- RMSNorm;
- RoPE;
- SiLU and other required activations;
- convolution required by the audio encoder;
- elementwise and reduction operations; and
- encoder/decoder block execution.

The actual Qwen3-ASR configuration is authoritative. Kernel names do not imply that a generic Qwen
text topology is sufficient.

## Correctness before fusion

For each kernel:

1. Generate small deterministic tensors.
2. Compute a straightforward Zig result.
3. Execute WGSL on the same logical inputs.
4. Read back the minimum output needed for comparison.
5. Compare with datatype-appropriate absolute/relative tolerances.
6. Include edge dimensions and invalid binding metadata.
7. Only then introduce packing, tiling, fusion, or specialization.

Keep the reference path after optimization; it is the regression oracle.

## Limits and residency

Never assume a single multi-gigabyte buffer. Inspect at least:

- `maxBufferSize`;
- `maxStorageBufferBindingSize`;
- relevant binding counts;
- workgroup dimensions and storage; and
- alignment requirements.

Partition weights by layer or shard from the beginning. Keep them GPU-resident after upload and
reuse activation buffers where lifetimes permit. Avoid full tensor readbacks and repeated pipeline
construction.

## Quantized execution

The desired path is packed weights → fused unpack/dequantize → multiply → accumulation. Expanding an
entire Q4 tensor to FP16 before matmul forfeits the intended memory-bandwidth advantage and should be
a reference/debug path, not the production design.

