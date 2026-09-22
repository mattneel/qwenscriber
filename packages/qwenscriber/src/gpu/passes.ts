//! The dispatch patterns every kernel in this directory is driven through.
//!
//! A pass is one kernel, its uniform, and the bindings it declares, with the temporary uniform
//! released as soon as the dispatch is submitted. They live here rather than in each driver because
//! the audio tower and the text decoder run the same matmuls and normalizations, and two copies of
//! "which binding is the weight" is two places for that to be wrong.

import { QwenscriberError } from "../errors.ts";
import { STATUS } from "../wasm/abi.ts";
import type { WebGpuDispatchGeometry, WebGpuRuntime } from "./runtime.ts";
import type { QuantFormat } from "./quant_layout.ts";

/** A matrix on the GPU: the packed weight, its shape, and the plane offset when it is quantized. */
export interface MatrixWeights {
  readonly weight: GPUBuffer;
  readonly rows: number;
  readonly cols: number;
  readonly quant?: { readonly format: QuantFormat; readonly dataOffsetBytes: number } | undefined;
}

/** A uniform block: u32 fields, with `floats` naming the indices that carry an f32 instead. */
export function packUniform(fields: readonly number[], floats: readonly number[] = []): Uint8Array {
  const bytes = new Uint8Array(fields.length * 4);
  const view = new DataView(bytes.buffer);
  fields.forEach((value, index) => {
    if (floats.includes(index)) view.setFloat32(index * 4, value, true);
    else view.setUint32(index * 4, value, true);
  });
  return bytes;
}

/**
 * `x * weight`, with the optional bias added afterwards.
 *
 * The kernel follows the weight's format. The bias is a separate pass on purpose: the matmul kernels
 * compute the product alone so that their accumulation is the only thing a comparison has to
 * explain, and the core adds the bias separately for the same reason.
 */
export async function matmulPass(
  runtime: WebGpuRuntime,
  matrix: MatrixWeights,
  input: GPUBuffer,
  tokens: number,
  label: string,
  bias?: GPUBuffer | undefined,
): Promise<{ readonly output: GPUBuffer; readonly geometry: readonly WebGpuDispatchGeometry[] }> {
  const { rows, cols, quant } = matrix;
  if (tokens < 1 || rows < 1 || cols < 1) {
    throw new QwenscriberError(STATUS.shape_mismatch, "gpu.passes", {
      message: `${label}: a matmul needs positive tokens, rows, and cols, got ${tokens}, ${rows}, ${cols}`,
      context: { tokens, rows, cols },
    });
  }
  const output = runtime.createOutputBuffer(
    tokens * rows * Float32Array.BYTES_PER_ELEMENT,
    `${label}.out`,
  );
  const params = runtime.createUniformBuffer(
    packUniform([tokens, rows, cols, quant?.dataOffsetBytes ?? 0]),
    `${label}.params`,
  );
  const geometry = [
    await runtime.dispatch(
      quant === undefined ? "matmul_f16" : "matmul_q5",
      [params, input, matrix.weight, output],
      [Math.ceil(rows / 16), Math.ceil(tokens / 16), 1],
    ),
  ];
  params.destroy();

  if (bias === undefined) return { output, geometry };
  const bias_params = runtime.createUniformBuffer(packUniform([tokens, rows, 0, 0]), `${label}.bias`);
  geometry.push(
    await runtime.dispatch(
      "add_bias_f32",
      [bias_params, bias, output],
      [Math.ceil((tokens * rows) / 256), 1, 1],
    ),
  );
  bias_params.destroy();
  return { output, geometry };
}

/** Per-row RMSNorm with a weight vector, one workgroup per row. */
export async function rmsNormPass(
  runtime: WebGpuRuntime,
  weight: GPUBuffer,
  input: GPUBuffer,
  rows: number,
  cols: number,
  eps: number,
  label: string,
): Promise<{ readonly output: GPUBuffer; readonly geometry: WebGpuDispatchGeometry }> {
  const output = runtime.createOutputBuffer(
    rows * cols * Float32Array.BYTES_PER_ELEMENT,
    `${label}.out`,
  );
  const params = runtime.createUniformBuffer(packUniform([rows, cols, eps, 0], [2]), `${label}.params`);
  const geometry = await runtime.dispatch(
    "rmsnorm",
    [params, input, weight, output],
    [rows, 1, 1],
  );
  params.destroy();
  return { output, geometry };
}

/** Per-row LayerNorm with an affine weight and bias, one workgroup per row. */
export async function layerNormPass(
  runtime: WebGpuRuntime,
  norm: { readonly weight: GPUBuffer; readonly bias: GPUBuffer },
  input: GPUBuffer,
  rows: number,
  cols: number,
  eps: number,
  label: string,
): Promise<{ readonly output: GPUBuffer; readonly geometry: WebGpuDispatchGeometry }> {
  const output = runtime.createOutputBuffer(
    rows * cols * Float32Array.BYTES_PER_ELEMENT,
    `${label}.out`,
  );
  const params = runtime.createUniformBuffer(packUniform([rows, cols, eps, 0], [2]), `${label}.params`);
  const geometry = await runtime.dispatch(
    "layernorm",
    [params, input, norm.weight, norm.bias, output],
    [rows, 1, 1],
  );
  params.destroy();
  return { output, geometry };
}

/** `left[i] + right[i]`, into a buffer of its own. */
export async function addPass(
  runtime: WebGpuRuntime,
  left: GPUBuffer,
  right: GPUBuffer,
  count: number,
  label: string,
): Promise<{ readonly output: GPUBuffer; readonly geometry: WebGpuDispatchGeometry }> {
  const output = runtime.createOutputBuffer(count * Float32Array.BYTES_PER_ELEMENT, `${label}.out`);
  const params = runtime.createUniformBuffer(packUniform([count, 0, 0, 0]), `${label}.params`);
  const geometry = await runtime.dispatch(
    "add_f32",
    [params, left, right, output],
    [Math.ceil(count / 256), 1, 1],
  );
  params.destroy();
  return { output, geometry };
}

/**
 * Replaces `target` with the sum of it and `addend`, releasing the buffer it replaces.
 *
 * A bind group cannot use one buffer as both a read-only and a writable binding, so a residual moves
 * to a fresh buffer rather than accumulating in place. The core's `addInPlace` mutates only because
 * linear memory has no such rule.
 */
export async function residualPass(
  runtime: WebGpuRuntime,
  target: GPUBuffer,
  addend: GPUBuffer,
  count: number,
  label: string,
): Promise<{ readonly output: GPUBuffer; readonly geometry: WebGpuDispatchGeometry }> {
  const sum = await addPass(runtime, target, addend, count, label);
  target.destroy();
  return sum;
}
