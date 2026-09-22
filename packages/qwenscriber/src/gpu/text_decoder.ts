//! The decoder, one token at a time, on the GPU, over a q8 key/value cache.
//!
//! The structure is `src/core/qwen3_asr/decoder.zig`'s `forwardToken`, layer for layer and pass for
//! pass, so that the numbers are the same ones arrived at by different means:
//!
//!     per layer:
//!       normed = RMSNorm(hidden, attention norm)
//!       q = normed * q_weight            k = ... * k_weight        v = ... * v_weight
//!       q = RMSNorm per head (heads, head_dim)                     k likewise, over kv_heads
//!       q = RoPE(q, heads)               k = RoPE(k, kv_heads)
//!       append k and v to the cache, quantized
//!       attention over the cached positions
//!       hidden += attention * out_weight
//!       normed = RMSNorm(hidden, ffn norm)
//!       gate = normed * gate_weight, up = normed * up_weight, gate = silu(gate) * up
//!       hidden += gate * down_weight
//!     logits = RMSNorm(hidden, ln_f) * output_weight;  the next token is its argmax
//!
//! Prompt tokens take the same path as generated ones -- the core's comment says why, and it is what
//! makes the prompt and the decode loop comparable to the reference by construction.
//!
//! The cache is layer-major and holds codes, not f32: each layer's planes are an f16 scale per 64
//! values followed by one code per value, at the offsets `cachePlane` computes in the core. The
//! whole cache is two buffers, and the per-layer, per-position offsets arrive as uniforms, which is
//! why the appends and the attention read and write a shared buffer without a binding per layer.

import { QwenscriberError, SDK_STATUS } from "../errors.ts";
import { STATUS } from "../wasm/abi.ts";
import type { WasmModel } from "../decode.ts";
import { dataPlaneOffsetBytes, quantFormatOf } from "./quant_layout.ts";
import { DECODER_LAYER_BASE, TENSOR_KIND } from "./tensor_kind.ts";
import {
  matmulPass,
  packUniform,
  residualPass,
  rmsNormPass,
  ropePass,
  siluMulPass,
  type MatrixWeights,
} from "./passes.ts";
import type { WebGpuDispatchGeometry, WebGpuRuntime } from "./runtime.ts";

/** One decoder layer's weights. The decoder has no biases on any projection. */
export interface DecoderLayerWeights {
  readonly attention_norm: GPUBuffer;
  readonly q: MatrixWeights;
  readonly q_norm: GPUBuffer;
  readonly k: MatrixWeights;
  readonly k_norm: GPUBuffer;
  readonly v: MatrixWeights;
  readonly out: MatrixWeights;
  readonly ffn_norm: GPUBuffer;
  readonly gate: MatrixWeights;
  readonly up: MatrixWeights;
  readonly down: MatrixWeights;
}

/**
 * The token embedding, kept packed on the GPU: a lookup is a dequantization, not a copy.
 *
 * `buffer` is the output projection's own buffer when the checkpoint ties them, which is the case
 * the released models ship.
 */
export interface EmbeddingWeights {
  readonly buffer: GPUBuffer;
  readonly vocab: number;
  readonly hidden: number;
  readonly data_offset_bytes: number;
  readonly format: number;
}

export interface TextDecoderWeights {
  readonly embed: EmbeddingWeights;
  readonly layers: readonly DecoderLayerWeights[];
  readonly final_norm: GPUBuffer;
  readonly output: MatrixWeights;
}

/** What the weights say about the model's shape, so a caller does not have to. */
export interface DecoderGeometry {
  readonly hidden: number;
  readonly heads: number;
  readonly kv_heads: number;
  readonly head_dim: number;
  readonly ffn_dim: number;
  readonly vocab: number;
  readonly layers: number;
}

/**
 * The key/value cache: two buffers, layer-major, with the q8 planes inside them.
 *
 * `plane_scale_bytes` is where a layer's code plane begins, and `plane_stride` is the distance
 * between layers, both as `cachePlane` in the core computes them.
 */
export interface DecoderCache {
  readonly keys_scales: GPUBuffer;
  readonly keys_codes: GPUBuffer;
  readonly values_scales: GPUBuffer;
  readonly values_codes: GPUBuffer;
  readonly max_positions: number;
  readonly row_width: number;
  readonly groups_per_row: number;
  /** Where a layer's code plane begins inside each buffer, as `cachePlane` aligns it. */
  readonly plane_code_offset: number;
  /** Distance between layers within a plane buffer. */
  readonly plane_stride: number;
  /** Positions written so far: the attention reads `position + 1` rows and the append writes one. */
  position: number;
}

export interface DecoderRun {
  readonly token: number;
  readonly dispatches: readonly WebGpuDispatchGeometry[];
  /** Present when the caller asked for it: `[1][vocab]` f32, the logits the argmax came from. */
  readonly logits?: GPUBuffer | undefined;
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
    throw new QwenscriberError(SDK_STATUS.protocol, "text_decoder", {
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

function matrixWeights(
  runtime: WebGpuRuntime,
  model: WasmModel,
  name: keyof typeof TENSOR_KIND,
  layer: number,
): MatrixWeights {
  const tensor = tensorOrThrow(model, name, layer);
  if (tensor.rank !== 2) {
    throw new QwenscriberError(SDK_STATUS.protocol, "text_decoder", {
      message: `${String(name)} must be [rows][cols] at layer ${layer}, got rank ${tensor.rank}`,
      context: { tensor: String(name), layer, rank: tensor.rank },
    });
  }
  const quantized = quantFormatOf(tensor.format);
  return {
    weight: runtime.uploadBytes(tensor.bytes, `decoder.${String(name)}.${layer}`),
    rows: tensor.dims[0],
    cols: tensor.dims[1],
    ...(quantized === undefined
      ? {}
      : {
          quant: {
            format: quantized,
            dataOffsetBytes: dataPlaneOffsetBytes(tensor.dims[0], tensor.dims[1]),
          },
        }),
  };
}

function normWeight(
  runtime: WebGpuRuntime,
  model: WasmModel,
  name: keyof typeof TENSOR_KIND,
  layer: number,
): GPUBuffer {
  return runtime.uploadBytes(
    tensorOrThrow(model, name, layer).bytes,
    `decoder.${String(name)}.${layer}`,
  );
}

/**
 * Uploads the decoder's weights and reports the shape they describe.
 *
 * Every dimension comes from a tensor's own descriptor: the head width from `q_norm` and `k_norm`,
 * the head counts from the query and key/value widths divided by it, the feed-forward width from the
 * gate, the vocabulary from the output projection. Only `text_rms_norm_eps` and `rope_theta` have no
 * tensor to read them from, so they are the caller's.
 */
export function uploadTextDecoderWeights(
  runtime: WebGpuRuntime,
  model: WasmModel,
): { readonly weights: TextDecoderWeights; readonly geometry: DecoderGeometry } {
  const layers: DecoderLayerWeights[] = [];
  // The layer count comes from the container rather than a configuration value: the index entries
  // are the authority on which layers exist, and walking them is how a caller that has no parsed
  // configuration finds the last one.
  let layer_count = 0;
  while (model.tensor(TENSOR_KIND.decoder_layer_attention_q_weight,
    DECODER_LAYER_BASE + layer_count) !== undefined) {
    layer_count += 1;
  }
  for (let index = 0; index < layer_count; index += 1) {
    const base = DECODER_LAYER_BASE + index;
    layers.push({
      attention_norm: normWeight(runtime, model, "decoder_layer_attention_norm_weight", base),
      q: matrixWeights(runtime, model, "decoder_layer_attention_q_weight", base),
      q_norm: normWeight(runtime, model, "decoder_layer_attention_q_norm_weight", base),
      k: matrixWeights(runtime, model, "decoder_layer_attention_k_weight", base),
      k_norm: normWeight(runtime, model, "decoder_layer_attention_k_norm_weight", base),
      v: matrixWeights(runtime, model, "decoder_layer_attention_v_weight", base),
      out: matrixWeights(runtime, model, "decoder_layer_attention_out_weight", base),
      ffn_norm: normWeight(runtime, model, "decoder_layer_ffn_norm_weight", base),
      gate: matrixWeights(runtime, model, "decoder_layer_ffn_gate_weight", base),
      up: matrixWeights(runtime, model, "decoder_layer_ffn_up_weight", base),
      down: matrixWeights(runtime, model, "decoder_layer_ffn_down_weight", base),
    });
  }
  // The released checkpoints tie the output projection to the embedding, in which case the tensor is
  // the embedding's and `resolveRequired` in the core has already substituted it.
  const first = layers[0];
  if (first === undefined) {
    throw new QwenscriberError(STATUS.shape_mismatch, "text_decoder", {
      message: "a decoder needs at least one layer",
      context: {},
    });
  }
  const head_dim = tensorOrThrow(model, "decoder_layer_attention_q_norm_weight",
    DECODER_LAYER_BASE).dims[0];
  const query_width = first.q.rows;
  const key_value_width = first.k.rows;
  if (query_width % head_dim !== 0 || key_value_width % head_dim !== 0) {
    throw new QwenscriberError(STATUS.shape_mismatch, "text_decoder", {
      message: `head width ${head_dim} does not divide the query width ${query_width} or the ` +
        `key/value width ${key_value_width}`,
      context: { head_dim, query_width, key_value_width },
    });
  }

  const embed = tensorOrThrow(model, "decoder_embed_tokens_weight", 0);
  const quantized = quantFormatOf(embed.format);
  const embed_matrix: MatrixWeights = {
    weight: runtime.uploadBytes(embed.bytes, "decoder.embed_tokens.weight"),
    rows: embed.dims[0],
    cols: embed.dims[1],
    ...(quantized === undefined
      ? {}
      : {
          quant: {
            format: quantized,
            dataOffsetBytes: dataPlaneOffsetBytes(embed.dims[0], embed.dims[1]),
          },
        }),
  };
  // A tied checkpoint ships no output projection: the embedding matrix is the unembedding, so the
  // same buffer serves both and the model is not uploaded twice.
  const output = model.tensor(TENSOR_KIND.decoder_output_weight, 0) !== undefined
    ? matrixWeights(runtime, model, "decoder_output_weight", 0)
    : embed_matrix;

  return {
    weights: {
      embed: {
        buffer: embed_matrix.weight,
        vocab: embed.dims[0],
        hidden: embed.dims[1],
        data_offset_bytes: dataPlaneOffsetBytes(embed.dims[0], embed.dims[1]),
        format: embed.format,
      },
      layers,
      final_norm: normWeight(runtime, model, "decoder_final_norm_weight", 0),
      output,
    },
    geometry: {
      hidden: first.q.cols,
      heads: query_width / head_dim,
      kv_heads: key_value_width / head_dim,
      head_dim,
      ffn_dim: first.gate.rows,
      vocab: output.rows,
      layers: layers.length,
    },
  };
}

/**
 * Allocates the key/value cache for `max_positions` positions.
 *
 * The layout is the core's: a layer's f16 scales for every position come first, then its codes, and
 * the next layer starts at the next alignment boundary. Both planes are zero, which decodes to zero
 * -- a zero scale with midpoint codes -- so a position that has not been written reads as absent
 * rather than as garbage.
 */
export function allocateDecoderCache(
  runtime: WebGpuRuntime,
  geometry: DecoderGeometry,
  max_positions: number,
): DecoderCache {
  const row_width = geometry.kv_heads * geometry.head_dim;
  const groups_per_row = row_width / 64;
  // One layer's planes: the f16 scales for every position, then the codes at the next alignment
  // boundary, which is what `dataPlaneOffsetBytes` computes. Four buffers rather than two, because
  // the append binds a scale plane and a code plane as writable in the same bind group.
  const plane_code_offset = dataPlaneOffsetBytes(max_positions, row_width);
  const plane_stride = plane_code_offset + max_positions * row_width;
  const total = plane_stride * geometry.layers;
  return {
    keys_scales: runtime.createOutputBuffer(total, "decoder.cache.keys.scales"),
    keys_codes: runtime.createOutputBuffer(total, "decoder.cache.keys.codes"),
    values_scales: runtime.createOutputBuffer(total, "decoder.cache.values.scales"),
    values_codes: runtime.createOutputBuffer(total, "decoder.cache.values.codes"),
    max_positions,
    row_width,
    groups_per_row,
    plane_code_offset,
    plane_stride,
    position: 0,
  };
}

/** Drops the cache back to an empty context. The planes are read only up to `position`, so this is
 * a counter reset rather than a clear: whatever a previous utterance left beyond it is never read. */
export function resetDecoderCache(cache: DecoderCache): void {
  cache.position = 0;
}

/** Appends one position's key or value row, quantizing it into the layer's planes as it goes. */
async function quantizeCacheRow(
  runtime: WebGpuRuntime,
  cache: DecoderCache,
  scales: GPUBuffer,
  codes: GPUBuffer,
  layer: number,
  row: GPUBuffer,
  cols: number,
  label: string,
): Promise<WebGpuDispatchGeometry> {
  const position = cache.position;
  const scale_base = layer * cache.plane_stride + position * cache.groups_per_row * 2;
  const code_base = layer * cache.plane_stride + cache.plane_code_offset +
    position * cache.row_width;
  const params = runtime.createUniformBuffer(
    packUniform([1, cols, scale_base, code_base]),
    `${label}.params`,
  );
  const geometry = await runtime.dispatch(
    "quantize_q8_group",
    [params, row, scales, codes],
    [cache.groups_per_row, 1, 1],
  );
  params.destroy();
  return geometry;
}

/** One query's attention over the positions written so far, reading the q8 planes. */
async function decodeAttentionPass(
  runtime: WebGpuRuntime,
  geometry: DecoderGeometry,
  cache: DecoderCache,
  layer: number,
  query: GPUBuffer,
  label: string,
): Promise<{ readonly output: GPUBuffer; readonly geometry: WebGpuDispatchGeometry }> {
  const output = runtime.createOutputBuffer(
    geometry.heads * geometry.head_dim * Float32Array.BYTES_PER_ELEMENT,
    `${label}.out`,
  );
  const scale_base = layer * cache.plane_stride;
  const code_base = layer * cache.plane_stride + cache.plane_code_offset;
  const params = runtime.createUniformBuffer(
    packUniform([
      geometry.heads,
      geometry.kv_heads,
      geometry.head_dim,
      cache.position + 1,
      cache.groups_per_row,
      scale_base,
      code_base,
      0,
    ]),
    `${label}.params`,
  );
  const dispatched = await runtime.dispatch(
    "decode_attention_q8",
    [params, query, cache.keys_scales, cache.keys_codes, cache.values_scales, cache.values_codes,
      output],
    [geometry.heads, 1, 1],
  );
  params.destroy();
  return { output, geometry: dispatched };
}

export interface DecoderSettings {
  /** `text_rms_norm_eps`; the released checkpoints all use 1e-6, and there is no tensor to read it
   * from, so the caller supplies it rather than having it baked in here. */
  readonly eps: number;
  /** `rope_theta`, likewise a configuration value with no tensor behind it. */
  readonly rope_theta: number;
  /** Ask for the logits the argmax came from, for a caller comparing them against a reference. */
  readonly want_logits?: boolean | undefined;
}

/**
 * Runs one token through the decoder and returns the next one.
 *
 * `input` is either a token id, whose embedding is gathered from the quantized matrix, or a hidden
 * state the caller already has -- which is how the audio tower's rows enter the prompt, exactly as
 * the core's `forwardToken` takes an optional audio row.
 *
 * Every intermediate is released as the next pass replaces it. Buffer pooling would cut the traffic;
 * a check that runs one token at a time does not need it yet, and a pool that is never measured is
 * an optimization nobody can justify.
 */
export async function decodeToken(
  runtime: WebGpuRuntime,
  weights: TextDecoderWeights,
  geometry: DecoderGeometry,
  cache: DecoderCache,
  input: { readonly token: number } | { readonly hidden: GPUBuffer },
  settings: DecoderSettings,
): Promise<DecoderRun> {
  const dispatches: WebGpuDispatchGeometry[] = [];
  if (cache.position >= cache.max_positions) {
    throw new QwenscriberError(STATUS.limit_exceeded, "text_decoder", {
      message: `the cache holds ${cache.max_positions} positions and they are all written`,
      context: { position: cache.position, max_positions: cache.max_positions },
    });
  }
  const { eps, rope_theta } = settings;
  const hidden_size = geometry.hidden;

  let hidden: GPUBuffer;
  if ("hidden" in input) {
    hidden = input.hidden;
  } else {
    const row = runtime.createOutputBuffer(hidden_size * Float32Array.BYTES_PER_ELEMENT,
      "decoder.embed.row");
    const params = runtime.createUniformBuffer(
      packUniform([input.token, weights.embed.hidden, weights.embed.data_offset_bytes,
        weights.embed.format]),
      "decoder.embed.params",
    );
    dispatches.push(await runtime.dispatch(
      "gather_row",
      [params, weights.embed.buffer, row],
      [Math.ceil(weights.embed.hidden / 64), 1, 1],
    ));
    params.destroy();
    hidden = row;
  }

  for (let index = 0; index < weights.layers.length; index += 1) {
    const layer = weights.layers[index] as DecoderLayerWeights;
    const normed = await rmsNormPass(runtime, layer.attention_norm, hidden, 1, hidden_size, eps,
      "decoder.attention.norm");
    dispatches.push(normed.geometry);
    const query = await matmulPass(runtime, layer.q, normed.output, 1, "decoder.q");
    const key = await matmulPass(runtime, layer.k, normed.output, 1, "decoder.k");
    const value = await matmulPass(runtime, layer.v, normed.output, 1, "decoder.v");
    dispatches.push(...query.geometry, ...key.geometry, ...value.geometry);
    normed.output.destroy();

    const query_normed = await rmsNormPass(runtime, layer.q_norm, query.output, geometry.heads,
      geometry.head_dim, eps, "decoder.q.norm");
    const key_normed = await rmsNormPass(runtime, layer.k_norm, key.output, geometry.kv_heads,
      geometry.head_dim, eps, "decoder.k.norm");
    dispatches.push(query_normed.geometry, key_normed.geometry);
    query.output.destroy();
    key.output.destroy();

    const query_rotated = await ropePass(runtime, query_normed.output, geometry.heads,
      geometry.head_dim, rope_theta, cache.position, "decoder.q.rope");
    const key_rotated = await ropePass(runtime, key_normed.output, geometry.kv_heads,
      geometry.head_dim, rope_theta, cache.position, "decoder.k.rope");
    dispatches.push(query_rotated.geometry, key_rotated.geometry);
    query_normed.output.destroy();
    key_normed.output.destroy();

    dispatches.push(await quantizeCacheRow(runtime, cache, cache.keys_scales, cache.keys_codes,
      index, key_rotated.output, cache.row_width, "decoder.cache.key"));
    dispatches.push(await quantizeCacheRow(runtime, cache, cache.values_scales, cache.values_codes,
      index, value.output, cache.row_width, "decoder.cache.value"));
    key_rotated.output.destroy();

    const attended = await decodeAttentionPass(runtime, geometry, cache, index, query_rotated.output,
      "decoder.attention");
    dispatches.push(attended.geometry);
    query_rotated.output.destroy();
    value.output.destroy();

    const projected = await matmulPass(runtime, layer.out, attended.output, 1, "decoder.attention.out");
    dispatches.push(...projected.geometry);
    attended.output.destroy();
    const attention_residual = await residualPass(runtime, hidden, projected.output, hidden_size,
      "decoder.residual.attention");
    dispatches.push(attention_residual.geometry);
    projected.output.destroy();
    hidden = attention_residual.output;

    const ffn_normed = await rmsNormPass(runtime, layer.ffn_norm, hidden, 1, hidden_size, eps,
      "decoder.ffn.norm");
    dispatches.push(ffn_normed.geometry);
    const gate = await matmulPass(runtime, layer.gate, ffn_normed.output, 1, "decoder.ffn.gate");
    const up = await matmulPass(runtime, layer.up, ffn_normed.output, 1, "decoder.ffn.up");
    dispatches.push(...gate.geometry, ...up.geometry);
    ffn_normed.output.destroy();
    const gated = await siluMulPass(runtime, gate.output, up.output, geometry.ffn_dim,
      "decoder.ffn.silu");
    dispatches.push(gated.geometry);
    gate.output.destroy();
    up.output.destroy();
    const down = await matmulPass(runtime, layer.down, gated.output, 1, "decoder.ffn.down");
    dispatches.push(...down.geometry);
    gated.output.destroy();
    const ffn_residual = await residualPass(runtime, hidden, down.output, hidden_size,
      "decoder.residual.ffn");
    dispatches.push(ffn_residual.geometry);
    down.output.destroy();
    hidden = ffn_residual.output;
  }
  cache.position += 1;

  const final_normed = await rmsNormPass(runtime, weights.final_norm, hidden, 1, hidden_size, eps,
    "decoder.final.norm");
  dispatches.push(final_normed.geometry);
  hidden.destroy();
  const logits = await matmulPass(runtime, weights.output, final_normed.output, 1, "decoder.logits");
  dispatches.push(...logits.geometry);
  final_normed.output.destroy();

  const picked = runtime.createOutputBuffer(Uint32Array.BYTES_PER_ELEMENT, "decoder.argmax");
  const argmax_params = runtime.createUniformBuffer(
    packUniform([geometry.vocab, 0, 0, 0]),
    "decoder.argmax.params",
  );
  dispatches.push(await runtime.dispatch(
    "argmax_f32",
    [argmax_params, logits.output, picked],
    [1, 1, 1],
  ));
  argmax_params.destroy();
  const token = (await runtime.readWords(picked, 1))[0] as number;
  picked.destroy();

  return {
    token,
    dispatches,
    ...(settings.want_logits === true ? { logits: logits.output } : { logits: undefined }),
  };
}
