//! `matmul_q4` end to end: packed Q4 weights in, f32 activations in, f32 products out.
//!
//! This is the kernel the SDK can run today, and it is the one worth running: it reads the
//! two-plane quantized layout straight out of the container (`src/core/quant.zig` writes it, the
//! shader decodes it) and never materializes a dense copy of the weights. The SDK is transport
//! here, not layout authority: the caller supplies the packed bytes and the data-plane offset that
//! `quant.zig`'s `planeLayout` computed, so there is exactly one implementation of the layout --
//! the Zig one -- and no second copy of those constants to drift in JavaScript.
//!
//! Geometry, from the shader's own header:
//!
//!   out[token][row] = dot(activation[token], weight[row])
//!   16x16 workgroup, dispatch (ceil(rows / 16), ceil(tokens / 16), 1)
//!   group size 64 weights, 32 code bytes per q4 group after an f16 scale plane

import { QwenscriberError } from "../errors.ts";
import { STATUS } from "../wasm/abi.ts";
import { QUANT_CODE_BYTES_PER_GROUP, QUANT_GROUP_SIZE } from "./quant_layout.ts";
import type { WebGpuRuntime } from "./runtime.ts";

/** Weights per quantization group. Re-exported from `quant_layout.ts`, which the drift gate checks. */
export const Q4_GROUP_SIZE = QUANT_GROUP_SIZE;
/** Packed code bytes per q4 group: 64 four-bit codes, two to a byte. */
export const Q4_CODE_BYTES_PER_GROUP = QUANT_CODE_BYTES_PER_GROUP.q4;
/** Weights decoded per tile edge: the workgroup is 16x16, so the tile is 16x16. */
const MATMUL_TILE = 16;

export interface MatmulQ4Request {
  /** The two-plane payload: f16 scales first, then the code plane at `dataOffsetBytes`. */
  readonly packed: Uint8Array;
  /** Offset of the code plane inside `packed`, as `planeLayout` aligned it (16-byte multiple). */
  readonly dataOffsetBytes: number;
  /** Weight matrix rows: the output width. */
  readonly rows: number;
  /** Weight matrix columns; must be a multiple of `Q4_GROUP_SIZE`. */
  readonly cols: number;
  /** Activation matrix, `tokens * cols` f32, row major. */
  readonly activations: Float32Array;
  /** Activation rows: the output height. */
  readonly tokens: number;
}

export interface MatmulQ4Result {
  /** `tokens * rows` f32, row major: `out[token][row]`. */
  readonly values: Float32Array;
  readonly workgroupSize: readonly [number, number, number];
  readonly workgroupCounts: readonly [number, number, number];
  readonly dataOffsetBytes: number;
  readonly packedBytes: number;
  readonly groupCount: number;
}

/** Rejects a shape the kernel cannot decode, naming the numbers, before any buffer exists. */
function assertShape(request: MatmulQ4Request): void {
  const groups_per_row = request.cols / Q4_GROUP_SIZE;
  const group_count = request.rows * groups_per_row;
  const code_bytes = group_count * Q4_CODE_BYTES_PER_GROUP;
  const scale_bytes = group_count * 2;
  const needed = request.dataOffsetBytes + code_bytes;
  const problems: string[] = [];
  if (!Number.isInteger(request.rows) || request.rows < 1) problems.push("rows must be positive");
  if (!Number.isInteger(request.cols) || request.cols < 1) problems.push("cols must be positive");
  if (!Number.isInteger(request.tokens) || request.tokens < 1) problems.push("tokens must be positive");
  if (Number.isInteger(request.cols) && request.cols % Q4_GROUP_SIZE !== 0) {
    problems.push(`cols must be a multiple of ${Q4_GROUP_SIZE}`);
  }
  if (request.activations.length !== request.tokens * request.cols) {
    problems.push(`activations must be tokens * cols = ${request.tokens * request.cols}`);
  }
  if (request.dataOffsetBytes < scale_bytes) {
    problems.push(`dataOffsetBytes must clear the ${scale_bytes}-byte scale plane`);
  }
  if (request.packed.length < needed) {
    problems.push(`packed must hold ${needed} bytes for ${group_count} groups`);
  }
  if (problems.length > 0) {
    throw new QwenscriberError(STATUS.shape_mismatch, "gpu.matmulQ4", {
      message: `gpu.matmulQ4: ${problems.join("; ")}`,
      context: {
        rows: request.rows,
        cols: request.cols,
        tokens: request.tokens,
        activations: request.activations.length,
        packedBytes: request.packed.length,
        dataOffsetBytes: request.dataOffsetBytes,
        groupCount: group_count,
      },
    });
  }
}

/** Views the packed bytes as u32 words: the shader binds the payload as `array<u32>`. */
function asWords(bytes: Uint8Array): Uint8Array {
  if (bytes.byteOffset % 4 === 0 && bytes.byteLength % 4 === 0) return bytes;
  const padded = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
  padded.set(bytes);
  return padded;
}

/**
 * Runs the fused dequantize+matmul and returns the f32 products.
 *
 * The uniform block is the four u32 fields the shader declares, packed here rather than in the
 * caller: `tokens, rows, cols, data_offset_bytes`, which is the order `matmul_q4.wgsl` documents.
 */
export async function matmulQ4(
  runtime: WebGpuRuntime,
  request: MatmulQ4Request,
): Promise<MatmulQ4Result> {
  assertShape(request);
  const { tokens, rows, cols } = request;
  const group_count = rows * (cols / Q4_GROUP_SIZE);
  const packed_bytes = asWords(request.packed);
  runtime.checkRequirements(
    { bufferBytes: packed_bytes.byteLength, storageBindingBytes: packed_bytes.byteLength },
    "gpu.matmulQ4",
  );

  const params = new ArrayBuffer(16);
  const params_view = new DataView(params);
  params_view.setUint32(0, tokens, true);
  params_view.setUint32(4, rows, true);
  params_view.setUint32(8, cols, true);
  params_view.setUint32(12, request.dataOffsetBytes, true);

  const activations = runtime.uploadBytes(request.activations, "matmul_q4.activations");
  const weights = runtime.uploadBytes(packed_bytes, "matmul_q4.packed");
  const output = runtime.createOutputBuffer(tokens * rows * 4, "matmul_q4.out");
  const uniform = runtime.createUniformBuffer(new Uint8Array(params), "matmul_q4.params");

  const workgroup_counts: readonly [number, number, number] = [
    Math.ceil(rows / MATMUL_TILE),
    Math.ceil(tokens / MATMUL_TILE),
    1,
  ];
  const geometry = await runtime.dispatch(
    "matmul_q4",
    [uniform, activations, weights, output],
    workgroup_counts,
  );
  const values = await runtime.readFloats(output, tokens * rows);
  for (const buffer of [activations, weights, output, uniform]) buffer.destroy();

  return {
    values,
    workgroupSize: geometry.workgroupSize,
    workgroupCounts: geometry.workgroupCounts,
    dataOffsetBytes: request.dataOffsetBytes,
    packedBytes: packed_bytes.byteLength,
    groupCount: group_count,
  };
}
