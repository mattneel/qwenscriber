//! The kernel catalogue: one literal entry per WGSL kernel in `gpu/shaders/`.
//!
//! Every number here is copied from the kernel's own header comment, because the header is what a
//! reviewer reads and this file is what the driver executes. The two are checked against each other
//! by `tests/gpu/harness.mjs` (which runs the same shaders through the same shapes) and by the
//! example page, which prints the measured error of one kernel end to end.
//!
//! Explicit, not generated: eight entries, eight sources, no reflection and no build step. A new
//! kernel is a new entry plus a new `.wgsl` file, and forgetting one is a readable diff rather than
//! a missing table row somewhere.
//!
//! `bindings` are the *WebGPU* binding numbers from each shader's `@group(0) @binding(n)` lines, in
//! the order the shader declares them, so a bind group can be built by walking the list.

import { SHADER_STAGE } from "./webgpu_flags.ts";

/** Every kernel this SDK knows how to build a pipeline for, named as its file is named. */
export type WebGpuKernelName =
  | "matmul_f32"
  | "matmul_q4"
  | "matmul_q5"
  | "matmul_f16"
  | "transpose_f32"
  | "add_f32"
  | "add_bias_f32"
  | "rmsnorm"
  | "rope"
  | "attention"
  | "silu_mul"
  | "gelu"
  | "layernorm"
  | "conv3x3_stride2_gelu"
  | "dequant_reference";

/** What a shader declares at `@group(0) @binding(n)`. */
export interface WebGpuKernelBinding {
  readonly binding: number;
  readonly kind: "uniform" | "storage-read" | "storage-read-write";
  /** Bytes the layout requires of this binding, or `0` when any size is acceptable. */
  readonly minBindingSizeBytes: number;
}

export interface WebGpuKernelDescriptor {
  readonly name: WebGpuKernelName;
  /** File name inside the shader directory. */
  readonly file: string;
  readonly entryPoint: string;
  readonly workgroupSize: readonly [number, number, number];
  /** `var<workgroup>` bytes the kernel declares, checked against `maxComputeWorkgroupStorageSize`. */
  readonly workgroupStorageBytes: number;
  readonly bindings: readonly WebGpuKernelBinding[];
}

/** Uniform block sizes, as `abi`-style structs: four 4-byte fields, or two of them for attention. */
// 16x16 tile plus the padding column that keeps the read side off one shared-memory bank.
const TRANSPOSE_TILE_STORAGE_BYTES = 16 * 17 * 4;
const PARAMS_BYTES = 16;
const ATTENTION_PARAMS_BYTES = 32;

const MATMUL_TILE_WORKGROUP = [16, 16, 1] as const;
const MATMUL_TILE_STORAGE_BYTES = 2 * 16 * 16 * 4;

/**
 * The catalogue.
 *
 * `matmul_f32`, `matmul_q4` and `matmul_q5` share a workgroup shape and a dispatch rule so their
 * outputs can be compared against each other; `dequant_reference` exists only as a test instrument
 * for the packing.
 */
export const WEBGPU_KERNELS: Readonly<Record<WebGpuKernelName, WebGpuKernelDescriptor>> = {
  matmul_f32: {
    name: "matmul_f32",
    file: "matmul_f32.wgsl",
    entryPoint: "matmul_f32_main",
    workgroupSize: MATMUL_TILE_WORKGROUP,
    workgroupStorageBytes: MATMUL_TILE_STORAGE_BYTES,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 3, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  matmul_q4: {
    name: "matmul_q4",
    file: "matmul_q4.wgsl",
    entryPoint: "matmul_q4_main",
    workgroupSize: MATMUL_TILE_WORKGROUP,
    workgroupStorageBytes: MATMUL_TILE_STORAGE_BYTES,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 3, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  matmul_q5: {
    name: "matmul_q5",
    file: "matmul_q5.wgsl",
    entryPoint: "matmul_q5_main",
    workgroupSize: MATMUL_TILE_WORKGROUP,
    workgroupStorageBytes: MATMUL_TILE_STORAGE_BYTES,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 3, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  matmul_f16: {
    name: "matmul_f16",
    file: "matmul_f16.wgsl",
    entryPoint: "matmul_f16_main",
    workgroupSize: [16, 16, 1],
    workgroupStorageBytes: 2 * 16 * 16 * 4,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 3, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  transpose_f32: {
    name: "transpose_f32",
    file: "transpose_f32.wgsl",
    entryPoint: "transpose_f32_main",
    workgroupSize: [16, 16, 1],
    workgroupStorageBytes: TRANSPOSE_TILE_STORAGE_BYTES,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  add_f32: {
    name: "add_f32",
    file: "add_f32.wgsl",
    entryPoint: "add_f32_main",
    workgroupSize: [256, 1, 1],
    workgroupStorageBytes: 0,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 3, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  add_bias_f32: {
    name: "add_bias_f32",
    file: "add_bias_f32.wgsl",
    entryPoint: "add_bias_f32_main",
    workgroupSize: [256, 1, 1],
    workgroupStorageBytes: 0,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  rmsnorm: {
    name: "rmsnorm",
    file: "rmsnorm.wgsl",
    entryPoint: "rmsnorm_main",
    workgroupSize: [256, 1, 1],
    workgroupStorageBytes: 256 * 4,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 3, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  rope: {
    name: "rope",
    file: "rope.wgsl",
    entryPoint: "rope_main",
    workgroupSize: [64, 1, 1],
    workgroupStorageBytes: 0,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  attention: {
    name: "attention",
    file: "attention.wgsl",
    entryPoint: "attention_main",
    workgroupSize: [128, 1, 1],
    workgroupStorageBytes: (128 + 256 + 128) * 4,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: ATTENTION_PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 3, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 4, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  silu_mul: {
    name: "silu_mul",
    file: "silu_mul.wgsl",
    entryPoint: "silu_mul_main",
    workgroupSize: [256, 1, 1],
    workgroupStorageBytes: 0,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 3, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  gelu: {
    name: "gelu",
    file: "gelu.wgsl",
    entryPoint: "gelu_main",
    workgroupSize: [256, 1, 1],
    workgroupStorageBytes: 0,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  layernorm: {
    name: "layernorm",
    file: "layernorm.wgsl",
    entryPoint: "layernorm_main",
    workgroupSize: [256, 1, 1],
    workgroupStorageBytes: 256 * 4,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 3, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 4, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  conv3x3_stride2_gelu: {
    name: "conv3x3_stride2_gelu",
    file: "conv3x3_stride2_gelu.wgsl",
    entryPoint: "conv3x3_stride2_gelu_main",
    workgroupSize: [256, 1, 1],
    // The decoded 3x3 weights for one output channel: 512 input channels * 9 taps * 4 bytes. This is
    // the number that makes the kernel need a device limit above the specification's 16384-byte
    // default, so a caller must declare `maxComputeWorkgroupStorageSize` in `requiredLimits` --
    // which `WebGpuRuntime.create` then asks the adapter for. An under-declared device does not
    // necessarily fail: the host this was first run on created the pipeline, dispatched, and wrote
    // zeros, because the staging writes had nowhere to go. A caller that sees a zero output from
    // this kernel should check its declared limits before its input.
    workgroupStorageBytes: 512 * 9 * 4,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 3, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 4, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
  dequant_reference: {
    name: "dequant_reference",
    file: "dequant_reference.wgsl",
    entryPoint: "dequant_reference_main",
    workgroupSize: [64, 1, 1],
    workgroupStorageBytes: 0,
    bindings: [
      { binding: 0, kind: "uniform", minBindingSizeBytes: PARAMS_BYTES },
      { binding: 1, kind: "storage-read", minBindingSizeBytes: 0 },
      { binding: 2, kind: "storage-read-write", minBindingSizeBytes: 0 },
    ],
  },
};

const BUFFER_TYPES = {
  uniform: "uniform",
  "storage-read": "read-only-storage",
  "storage-read-write": "storage",
} as const;

/**
 * The bind group layout the kernel's shader declares, for `createBindGroupLayout`.
 *
 * Built from the catalogue so that the layout cannot disagree with the bindings the shader uses:
 * WebGPU validates the two against each other at pipeline creation, and a mismatch there is a
 * `GpuShaderError` at the first dispatch rather than a silently unbound buffer.
 */
export function bindGroupLayoutEntries(
  kernel: WebGpuKernelDescriptor,
): GPUBindGroupLayoutEntry[] {
  return kernel.bindings.map((entry) => ({
    binding: entry.binding,
    visibility: SHADER_STAGE.COMPUTE,
    buffer: {
      type: BUFFER_TYPES[entry.kind],
      minBindingSize: entry.minBindingSizeBytes,
    },
  }));
}
