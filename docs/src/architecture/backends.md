# Backends

The public backend selector has three values:

| Value | Semantics |
| --- | --- |
| `"auto"` | Select WebGPU when the device and model requirements are compatible; otherwise select
  WASM SIMD when supported |
| `"webgpu"` | Require the WebGPU backend; return an actionable unsupported/capability error instead
  of silently changing semantics |
| `"wasm"` | Require the CPU/WASM backend |

## Shared semantics

Backend choice must not change tokenizer behavior, model-format interpretation, stop conditions, or
result-object meaning. Floating-point execution can produce bounded numerical differences; tests and
release notes must document accepted tolerances and any known transcript divergence.

## WebGPU backend

WebGPU is the primary browser performance architecture, especially for the 1.7B target. It keeps
large weights and activations on the GPU and minimizes synchronization with the host.

## WASM backend

The WASM backend is:

- the correctness/reference implementation;
- a portability fallback;
- a conformance oracle for WGSL kernels; and
- a serious optimization target for the 0.6B model where practical.

It is not a reason to route the WebGPU backend through CPU memory for every tensor operation.

## Capability selection

`auto` should evaluate the whole requirement set: WebGPU availability, adapter limits, required
shader features, model/quantization compatibility, WASM SIMD, Workers, and memory budget. Selection
must be explainable through capability diagnostics.

