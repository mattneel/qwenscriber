//! The audio tower's blocks, on the GPU: eighteen pre-norm layers and the final normalization.
//!
//! The structure is `src/core/qwen3_asr/model.zig`'s `runAudioTower`, block for block and pass for
//! pass, because the point of this module is to produce the same numbers by different means:
//!
//!     for each layer:
//!       normed = LayerNorm(encoder, attention norm)
//!       q, k, v = normed * w + b            (one projection each, same input)
//!       block  = attention over the window
//!       block  = block * out_w + out_b        (per window)
//!       encoder += block                      (residual)
//!       normed = LayerNorm(encoder, ffn norm)
//!       wide   = gelu(normed * in_w + in_b)
//!       block  = wide * out_w + out_b
//!       encoder += block                      (residual)
//!     encoder = LayerNorm(encoder, ln_post)
//!
//! Attention is not causal: an audio encoder's every step attends to its whole window. The window is
//! `n_window_infer` mel frames' worth of steps, and a sequence that fits in one window -- which is
//! every clip short enough to be transcribed in one pass through a single `n_window_infer` group --
//! is one call. A longer sequence needs each window's queries and keys bound as a buffer *range*,
//! which `WebGpuRuntime.dispatch` does not take yet, so this refuses such a sequence rather than
//! attending across a window boundary and returning a plausible wrong answer.
//!
//! Every weight is read from the model by kind, and the kernel for each projection follows the
//! tensor's own format: the converter quantizes the attention and feed-forward matrices, while the
//! biases and both LayerNorms stay f32.

import { QwenscriberError, SDK_STATUS } from "../errors.ts";
import { STATUS, type AudioConfig } from "../wasm/abi.ts";
import type { WasmModel } from "../decode.ts";
import { dataPlaneOffsetBytes, quantFormatOf, type QuantFormat } from "./quant_layout.ts";
import { AUDIO_LAYER_BASE, TENSOR_KIND } from "./tensor_kind.ts";
import type { WebGpuRuntime, WebGpuDispatchGeometry } from "./runtime.ts";

/** A projection weight and its bias: `x * weight + bias`, with the shape the descriptor reports. */
export interface ProjectionWeights {
  readonly weight: GPUBuffer;
  readonly bias: GPUBuffer;
  readonly rows: number;
  readonly cols: number;
  readonly quant?: { readonly format: QuantFormat; readonly dataOffsetBytes: number } | undefined;
}

/** One layer's weights, all fourteen of them. */
export interface AudioLayerWeights {
  readonly attention_norm: { readonly weight: GPUBuffer; readonly bias: GPUBuffer };
  readonly q: ProjectionWeights;
  readonly k: ProjectionWeights;
  readonly v: ProjectionWeights;
  readonly out: ProjectionWeights;
  readonly ffn_norm: { readonly weight: GPUBuffer; readonly bias: GPUBuffer };
  readonly ffn_in: ProjectionWeights;
  readonly ffn_out: ProjectionWeights;
}

export interface AudioTowerWeights {
  readonly layers: readonly AudioLayerWeights[];
  readonly final_norm: { readonly weight: GPUBuffer; readonly bias: GPUBuffer };
}

export interface AudioTowerResult {
  /** The tower's output on the GPU: `[steps][d_model]` f32, row major. */
  readonly output: GPUBuffer;
  readonly steps: number;
  readonly dispatches: readonly WebGpuDispatchGeometry[];
}

function tensorOrThrow(model: WasmModel, name: keyof typeof TENSOR_KIND, layer: number): {
  readonly bytes: Uint8Array;
  readonly dims: readonly [number, number, number, number];
  readonly rank: number;
  readonly format: number;
} {
  const kind = TENSOR_KIND[name];
  const tensor = model.tensor(kind, layer);
  if (tensor === undefined) {
    throw new QwenscriberError(SDK_STATUS.protocol, "audio_tower", {
      message: `${model.modelId} holds no ${name} tensor at layer ${layer} (kind ${kind})`,
      context: { tensor: name, kind, layer },
    });
  }
  return {
    bytes: tensor.bytes,
    dims: tensor.descriptor.dims,
    rank: tensor.descriptor.rank,
    format: tensor.descriptor.format,
  };
}

function projection(runtime: WebGpuRuntime, model: WasmModel,
  name: keyof typeof TENSOR_KIND, layer: number): ProjectionWeights {
  const weight = tensorOrThrow(model, name, layer);
  // The kinds pair a projection with its bias by replacing the suffix, not by appending to it:
  // `audio_layer_attention_q_weight` and `audio_layer_attention_q_bias`.
  const bias_name = `${String(name).replace(/_weight$/, "")}_bias` as keyof typeof TENSOR_KIND;
  if (TENSOR_KIND[bias_name] === undefined) {
    throw new QwenscriberError(SDK_STATUS.protocol, "audio_tower", {
      message: `${String(name)} has no bias kind beside it (${bias_name})`,
      context: { tensor: String(name), bias: bias_name },
    });
  }
  const bias = tensorOrThrow(model, bias_name, layer);
  if (weight.rank !== 2 || bias.rank !== 1) {
    throw new QwenscriberError(SDK_STATUS.protocol, "audio_tower", {
      message:
        `${String(name)} must be [rows][cols] with a [rows] bias at layer ${layer}, got rank ` +
        `${weight.rank} and ${bias.rank}`,
      context: { tensor: String(name), layer, weight_rank: weight.rank, bias_rank: bias.rank },
    });
  }
  const quantized = quantFormatOf(weight.format);
  return {
    weight: runtime.uploadBytes(weight.bytes, `tower.${String(name)}.${layer}`),
    bias: runtime.uploadBytes(bias.bytes, `tower.${String(name)}.bias.${layer}`),
    rows: weight.dims[0],
    cols: weight.dims[1],
    ...(quantized === undefined
      ? {}
      : {
          quant: {
            format: quantized,
            dataOffsetBytes: dataPlaneOffsetBytes(weight.dims[0], weight.dims[1]),
          },
        }),
  };
}

function normPair(runtime: WebGpuRuntime, model: WasmModel,
  weight_name: keyof typeof TENSOR_KIND, bias_name: keyof typeof TENSOR_KIND, layer: number): {
    readonly weight: GPUBuffer;
    readonly bias: GPUBuffer;
  } {
  return {
    weight: runtime.uploadBytes(tensorOrThrow(model, weight_name, layer).bytes,
      `tower.${String(weight_name)}.${layer}`),
    bias: runtime.uploadBytes(tensorOrThrow(model, bias_name, layer).bytes,
      `tower.${String(bias_name)}.${layer}`),
  };
}

/** Every weight the tower's blocks read, uploaded once. The caller destroys them. */
export function uploadAudioTowerWeights(
  runtime: WebGpuRuntime,
  model: WasmModel,
  config: AudioConfig,
): AudioTowerWeights {
  const layers: AudioLayerWeights[] = [];
  for (let layer = 0; layer < config.layers; layer += 1) {
    // The container tags the tower's blocks above the decoder's range, so the descriptor's layer is
    // the block index plus that base -- the same arithmetic the core's `buildLayers` does.
    const block = AUDIO_LAYER_BASE + layer;
    layers.push({
      attention_norm: normPair(runtime, model, "audio_layer_attention_norm_weight",
        "audio_layer_attention_norm_bias", block),
      q: projection(runtime, model, "audio_layer_attention_q_weight", block),
      k: projection(runtime, model, "audio_layer_attention_k_weight", block),
      v: projection(runtime, model, "audio_layer_attention_v_weight", block),
      out: projection(runtime, model, "audio_layer_attention_out_weight", block),
      ffn_norm: normPair(runtime, model, "audio_layer_final_norm_weight",
        "audio_layer_final_norm_bias", block),
      ffn_in: projection(runtime, model, "audio_layer_ffn_in_weight", block),
      ffn_out: projection(runtime, model, "audio_layer_ffn_out_weight", block),
    });
  }
  return {
    layers,
    final_norm: normPair(runtime, model, "audio_final_norm_weight", "audio_final_norm_bias", 0),
  };
}

/** A uniform block: u32 fields, with `floats` naming the indices that carry an f32 instead. */
function packUniform(fields: readonly number[], floats: readonly number[] = []): Uint8Array {
  const bytes = new Uint8Array(fields.length * 4);
  const view = new DataView(bytes.buffer);
  fields.forEach((value, index) => {
    if (floats.includes(index)) view.setFloat32(index * 4, value, true);
    else view.setUint32(index * 4, value, true);
  });
  return bytes;
}

// ---------------------------------------------------------------------------
// The passes
// ---------------------------------------------------------------------------

async function layerNormPass(
  runtime: WebGpuRuntime,
  norm: { readonly weight: GPUBuffer; readonly bias: GPUBuffer },
  input: GPUBuffer,
  rows: number,
  cols: number,
  eps: number,
  label: string,
): Promise<{ readonly output: GPUBuffer; readonly geometry: WebGpuDispatchGeometry }> {
  const output = runtime.createOutputBuffer(rows * cols * Float32Array.BYTES_PER_ELEMENT, `${label}.out`);
  const params = runtime.createUniformBuffer(packUniform([rows, cols, eps, 0], [2]), `${label}.params`);
  const geometry = await runtime.dispatch(
    "layernorm",
    [params, input, norm.weight, norm.bias, output],
    [rows, 1, 1],
  );
  params.destroy();
  return { output, geometry };
}

/**
 * `x * weight + bias` for one projection.
 *
 * The kernel follows the weight's format, and the bias is a separate pass: the matmul kernels
 * compute the product alone, so that their accumulation is the only thing a comparison has to
 * explain.
 */
async function projectionPass(
  runtime: WebGpuRuntime,
  projection_weights: ProjectionWeights,
  input: GPUBuffer,
  tokens: number,
  label: string,
): Promise<{ readonly output: GPUBuffer; readonly geometry: readonly WebGpuDispatchGeometry[] }> {
  const { rows, cols, quant } = projection_weights;
  if (tokens < 1 || rows < 1 || cols < 1) {
    throw new QwenscriberError(STATUS.shape_mismatch, "audio_tower", {
      message: `${label}: a projection needs positive tokens, rows, and cols, got ${tokens}, ${rows}, ${cols}`,
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
  const product = await runtime.dispatch(
    quant === undefined ? "matmul_f16" : "matmul_q5",
    [params, input, projection_weights.weight, output],
    [Math.ceil(rows / 16), Math.ceil(tokens / 16), 1],
  );
  params.destroy();

  const bias_params = runtime.createUniformBuffer(packUniform([tokens, rows, 0, 0]), `${label}.bias`);
  const biased = await runtime.dispatch(
    "add_bias_f32",
    [bias_params, projection_weights.bias, output],
    [Math.ceil((tokens * rows) / 256), 1, 1],
  );
  bias_params.destroy();
  return { output, geometry: [product, biased] };
}

/** `out[i] = gelu(x[i])`, into a buffer of its own: a bind group cannot read and write one buffer. */
async function geluPass(
  runtime: WebGpuRuntime,
  input: GPUBuffer,
  count: number,
  label: string,
): Promise<{ readonly output: GPUBuffer; readonly geometry: WebGpuDispatchGeometry }> {
  const output = runtime.createOutputBuffer(count * Float32Array.BYTES_PER_ELEMENT, `${label}.out`);
  const params = runtime.createUniformBuffer(packUniform([count, 0, 0, 0]), `${label}.params`);
  const geometry = await runtime.dispatch(
    "gelu",
    [params, input, output],
    [Math.ceil(count / 256), 1, 1],
  );
  params.destroy();
  return { output, geometry };
}

/**
 * `left[i] + right[i]`, into a buffer of its own.
 *
 * The sum cannot be written back over an operand: a bind group cannot use one buffer as both a
 * read-only and a writable binding, which is why the residual swaps to a fresh buffer and releases
 * the one it replaces. The core's `addInPlace` mutates because linear memory has no such rule.
 */
async function addPass(
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

/** Replaces `target` with the sum, releasing the buffer it replaces. */
async function residual(
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

/**
 * Attention over one window, non-causal.
 *
 * The kernel's non-causal mode is "the last `window` keys", so passing the whole sequence length as
 * the window is full attention inside it. `window` is both the query count and the key count here
 * because one window's queries and keys are the same steps.
 */
async function attentionPass(
  runtime: WebGpuRuntime,
  config: AudioConfig,
  q: GPUBuffer,
  k: GPUBuffer,
  v: GPUBuffer,
  steps: number,
  label: string,
): Promise<{ readonly output: GPUBuffer; readonly geometry: WebGpuDispatchGeometry }> {
  const output = runtime.createOutputBuffer(
    steps * config.d_model * Float32Array.BYTES_PER_ELEMENT,
    `${label}.out`,
  );
  const scale = 1 / Math.sqrt(config.head_dim);
  const params = runtime.createUniformBuffer(
    packUniform([steps, config.attention_heads, steps, config.head_dim, scale, steps, 0, 0], [4]),
    `${label}.params`,
  );
  const geometry = await runtime.dispatch(
    "attention",
    [params, q, k, v, output],
    [config.attention_heads, steps, 1],
  );
  params.destroy();
  return { output, geometry };
}

/** Steps one attention window spans, as `maxWindow` in the core computes it. */
export function windowSteps(config: AudioConfig): number {
  return Math.floor(config.n_window_infer / config.chunk_frames) * config.chunk_steps;
}

/**
 * Runs the tower's blocks over a packed sequence and returns `ln_post`'s output.
 *
 * `input` is `[steps][d_model]`, the convolution stage's output. The first residual adds into it, so
 * this takes ownership of that buffer: it comes back as part of the tower's scratch and the caller
 * must not use it again.
 */
export async function runAudioTower(
  runtime: WebGpuRuntime,
  weights: AudioTowerWeights,
  config: AudioConfig,
  input: GPUBuffer,
  steps: number,
): Promise<AudioTowerResult> {
  if (steps < 1) {
    throw new QwenscriberError(STATUS.invalid_argument, "audio_tower", {
      message: `the tower needs at least one step, got ${steps}`,
      context: { steps },
    });
  }
  const window_steps = windowSteps(config);
  if (steps > window_steps || window_steps > 256) {
    throw new QwenscriberError(SDK_STATUS.not_implemented, "audio_tower", {
      message:
        `a ${steps}-step sequence does not fit one ${window_steps}-step attention window; ` +
        "multi-window attention needs each window's queries and keys bound as a buffer range, " +
        "which the dispatch path does not take yet",
      context: { steps, window_steps },
    });
  }
  const dispatches: WebGpuDispatchGeometry[] = [];
  let encoder = input;
  for (const layer of weights.layers) {
    const normed = await layerNormPass(
      runtime, layer.attention_norm, encoder, steps, config.d_model, config.layer_norm_eps,
      "tower.attention.norm",
    );
    dispatches.push(normed.geometry);
    const q = await projectionPass(runtime, layer.q, normed.output, steps, "tower.q");
    const k = await projectionPass(runtime, layer.k, normed.output, steps, "tower.k");
    const v = await projectionPass(runtime, layer.v, normed.output, steps, "tower.v");
    dispatches.push(...q.geometry, ...k.geometry, ...v.geometry);
    normed.output.destroy();

    const attended = await attentionPass(runtime, config, q.output, k.output, v.output, steps,
      "tower.attention");
    dispatches.push(attended.geometry);
    for (const buffer of [q.output, k.output, v.output]) buffer.destroy();

    const projected = await projectionPass(runtime, layer.out, attended.output, steps,
      "tower.attention.out");
    dispatches.push(...projected.geometry);
    attended.output.destroy();
    const attended_residual = await residual(
      runtime, encoder, projected.output, steps * config.d_model, "tower.residual.attention",
    );
    dispatches.push(attended_residual.geometry);
    encoder = attended_residual.output;
    projected.output.destroy();

    const ffn_normed = await layerNormPass(
      runtime, layer.ffn_norm, encoder, steps, config.d_model, config.layer_norm_eps,
      "tower.ffn.norm",
    );
    dispatches.push(ffn_normed.geometry);
    const wide = await projectionPass(runtime, layer.ffn_in, ffn_normed.output, steps, "tower.ffn.in");
    dispatches.push(...wide.geometry);
    ffn_normed.output.destroy();
    const activated = await geluPass(runtime, wide.output, steps * config.ffn_dim, "tower.ffn.gelu");
    dispatches.push(activated.geometry);
    wide.output.destroy();
    const block = await projectionPass(runtime, layer.ffn_out, activated.output, steps,
      "tower.ffn.out");
    dispatches.push(...block.geometry);
    activated.output.destroy();
    const ffn_residual = await residual(
      runtime, encoder, block.output, steps * config.d_model, "tower.residual.ffn",
    );
    dispatches.push(ffn_residual.geometry);
    encoder = ffn_residual.output;
    block.output.destroy();
  }

  const final = await layerNormPass(
    runtime, weights.final_norm, encoder, steps, config.d_model, config.layer_norm_eps,
    "tower.ln_post",
  );
  dispatches.push(final.geometry);
  encoder.destroy();
  return { output: final.output, steps, dispatches };
}
