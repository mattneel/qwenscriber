//! One chunk of the audio tower's convolution stack and downsample projection, on the GPU.
//!
//! The core's own forward pass does exactly this per chunk of `audio_n_window * 2` mel frames:
//! three 3x3 stride-2 convolutions with GELU fused into each, then a linear projection from
//! `downsample_hidden_size * frequency_bins` to `audio_d_model`. The reference for this file is
//! `src/core/qwen3_asr/model.zig`'s convolution stage, which the released implementation's
//! `audio_conv_out` fixture was generated from -- so a caller can compare the result against
//! `tests/fixtures/reference/audio_conv_out.f32` and see the same numbers the CPU path produces.
//!
//! The layouts are the container's, unchanged: a convolution reads `[in_channels][in_height]
//! [in_width]` and writes `[out_channels][out_height][out_width]`, all channel planes contiguous,
//! so stage `n`'s output is stage `n + 1`'s input with no unpacking. The stack's output is then
//! transposed to `[steps][channels * bins]` -- the row-major flatten of the first two axes makes
//! that a plain 2-D transpose -- because the projection reads activations token-major.

import { QwenscriberError, SDK_STATUS } from "../errors.ts";
import { STATUS, type AudioConfig } from "../wasm/abi.ts";
import type { WebGpuRuntime, WebGpuDispatchGeometry } from "./runtime.ts";
import type { ConvStackWeights, TowerWeights } from "./tower_weights.ts";

/** What one chunk's tower stage produced, and what it cost. */
export interface ConvStageResult {
  /** The stage's output on the GPU: `[chunk_steps][d_model]` f32, row major. */
  readonly output: GPUBuffer;
  readonly steps: number;
  readonly d_model: number;
  /** The dispatches in order, with the geometry each one ran at. */
  readonly dispatches: readonly WebGpuDispatchGeometry[];
  /**
   * The two buffers between the stack and the projection, for a caller that wants to see where a
   * disagreement starts: `stack` is `[channels][frequency][steps]` and `transposed` is
   * `[steps][channels * frequency]`. The caller destroys them; nothing else refers to them.
   */
  readonly intermediates: {
    readonly conv1: GPUBuffer;
    readonly conv2: GPUBuffer;
    readonly stack: GPUBuffer;
    readonly transposed: GPUBuffer;
  };
}

/**
 * The sinusoidal position embedding for `steps` time steps, `[steps][channels]`.
 *
 * Mirrors `addSinusoidalPosition` in `src/core/qwen3_asr/model.zig`, f64 arithmetic included:
 * `log_timescale_increment = ln(10000) / (channels / 2 - 1)`, the row is `concat(sin, cos)` over
 * `channels / 2` timescales, and the embedding is a term *added* to the projected output. The core's
 * comment records why that last part matters: an implementation that wrote the embedding over the
 * row instead discarded the audio entirely, and a tone and a speech clip decoded the same tokens.
 */
export function positionEmbedding(steps: number, channels: number): Float32Array {
  const half = channels / 2;
  const log_timescale_increment = Math.log(10000) / (half - 1);
  const table = new Float32Array(steps * channels);
  for (let step = 0; step < steps; step += 1) {
    for (let index = 0; index < half; index += 1) {
      const inverse_timescale = Math.exp(-log_timescale_increment * index);
      const angle = step * inverse_timescale;
      table[step * channels + index] = Math.sin(angle);
      table[step * channels + half + index] = Math.cos(angle);
    }
  }
  return table;
}

/** Output-height/width of a 3x3 stride-2 pad-1 convolution, as the shader and the core compute it. */
function convolved(size: number): number {
  if (size < 1) {
    throw new QwenscriberError(SDK_STATUS.protocol, "conv_stack", {
      message: `a convolution input axis must be at least one element, got ${size}`,
      context: { size },
    });
  }
  return Math.floor((size - 1) / 2) + 1;
}

function uniform(values: readonly number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setUint32(index * 4, value, true));
  return bytes;
}

/** Dispatches one convolution over a `[in_channels][height][width]` input. */
async function convolve(
  runtime: WebGpuRuntime,
  stage: ConvStackWeights,
  input: GPUBuffer,
  channels: number,
  height: number,
  width: number,
  label: string,
): Promise<{
  readonly output: GPUBuffer;
  readonly height: number;
  readonly width: number;
  readonly geometry: WebGpuDispatchGeometry;
}> {
  const out_height = convolved(height);
  const out_width = convolved(width);
  const plane = out_height * out_width;
  const output = runtime.createOutputBuffer(
    stage.out_channels * plane * Float32Array.BYTES_PER_ELEMENT,
    `${label}.out`,
  );
  // The shader derives its own output plane from the input's, so passing the input's dimensions is
  // the whole contract; `in_channels` is what bounds the staged weight block it decodes.
  const params = runtime.createUniformBuffer(
    uniform([stage.out_channels, channels, height, width]),
    `${label}.params`,
  );
  const geometry = await runtime.dispatch(
    "conv3x3_stride2_gelu",
    [params, input, stage.weight, stage.bias, output],
    [Math.ceil(plane / 256), stage.out_channels, 1],
  );
  params.destroy();
  return { output, height: out_height, width: out_width, geometry };
}

/**
 * Runs one chunk through the convolution stack and the downsample projection.
 *
 * `features` is `[mel_bins][chunk_frames]` row major -- one chunk of the log-mel block the frontend
 * produces, which is `[mel_bins][frames]` with at least `chunk_frames` columns. The buffers this
 * allocates are intermediate activations: the returned output is the caller's to destroy, and
 * everything else is released here.
 */
export async function runConvStage(
  runtime: WebGpuRuntime,
  weights: TowerWeights,
  config: AudioConfig,
  features: Float32Array,
): Promise<ConvStageResult> {
  if (features.length !== config.mel_bins * config.chunk_frames) {
    throw new QwenscriberError(STATUS.invalid_argument, "conv_stack", {
      message:
        `one chunk is ${config.mel_bins} x ${config.chunk_frames} features, got ${features.length}`,
      context: {
        expected: config.mel_bins * config.chunk_frames,
        got: features.length,
        mel_bins: config.mel_bins,
        chunk_frames: config.chunk_frames,
      },
    });
  }
  const dispatches: WebGpuDispatchGeometry[] = [];
  const stage1 = runtime.uploadBytes(new Uint8Array(features.buffer, features.byteOffset,
    features.byteLength), "tower.chunk.input");
  const first = await convolve(runtime, weights.conv1, stage1, 1, config.mel_bins,
    config.chunk_frames, "tower.conv1");
  const second = await convolve(runtime, weights.conv2, first.output, weights.conv1.out_channels,
    first.height, first.width, "tower.conv2");
  const third = await convolve(runtime, weights.conv3, second.output, weights.conv2.out_channels,
    second.height, second.width, "tower.conv3");
  dispatches.push(first.geometry, second.geometry, third.geometry);
  stage1.destroy();

  // The stack's output is `[channels][bins][steps]`; `channel * bins + bin` is the row-major flatten
  // of the first two axes, so the projection's `[steps][channels * bins]` is a 2-D transpose.
  const stack_rows = weights.conv3.out_channels * third.height;
  const steps = third.width;
  const transposed = runtime.createOutputBuffer(
    stack_rows * steps * Float32Array.BYTES_PER_ELEMENT,
    "tower.transposed",
  );
  const transpose_params = runtime.createUniformBuffer(
    uniform([stack_rows, steps, 0, 0]),
    "tower.transpose.params",
  );
  dispatches.push(await runtime.dispatch(
    "transpose_f32",
    [transpose_params, third.output, transposed],
    [Math.ceil(steps / 16), Math.ceil(stack_rows / 16), 1],
  ));
  transpose_params.destroy();

  if (weights.downsample.cols !== stack_rows) {
    throw new QwenscriberError(SDK_STATUS.protocol, "conv_stack", {
      message:
        `the stack emits ${stack_rows} features per step but audio.conv_out.weight takes ` +
        `${weights.downsample.cols}`,
      context: { stack_rows, downsample_cols: weights.downsample.cols },
    });
  }
  const output = runtime.createOutputBuffer(
    steps * weights.downsample.rows * Float32Array.BYTES_PER_ELEMENT,
    "tower.conv_out",
  );
  // The projection's kernel follows its format: an f16 conversion keeps element weights, every
  // other conversion quantizes the matrix, and the two kernels read different planes.
  const quant = weights.downsample.quant;
  const matmul_params = runtime.createUniformBuffer(
    uniform([steps, weights.downsample.rows, weights.downsample.cols, quant?.dataOffsetBytes ?? 0]),
    "tower.conv_out.params",
  );
  dispatches.push(await runtime.dispatch(
    quant === undefined ? "matmul_f16" : "matmul_q5",
    [matmul_params, transposed, weights.downsample.weight, output],
    [Math.ceil(weights.downsample.rows / 16), Math.ceil(steps / 16), 1],
  ));
  matmul_params.destroy();

  // `conv_out += positional_embedding[:time_steps]`, the last thing the core's convolution stage
  // does. A separate buffer because a bind group cannot read and write one storage buffer at once.
  const embedding = runtime.uploadBytes(
    new Uint8Array(positionEmbedding(steps, weights.downsample.rows).buffer),
    "tower.position",
  );
  const projected = runtime.createOutputBuffer(
    steps * weights.downsample.rows * Float32Array.BYTES_PER_ELEMENT,
    "tower.projected",
  );
  const add_params = runtime.createUniformBuffer(
    uniform([steps * weights.downsample.rows, 0, 0, 0]),
    "tower.position.params",
  );
  dispatches.push(await runtime.dispatch(
    "add_f32",
    [add_params, output, embedding, projected],
    [Math.ceil((steps * weights.downsample.rows) / 256), 1, 1],
  ));
  add_params.destroy();
  embedding.destroy();
  output.destroy();

  return {
    output: projected,
    steps,
    d_model: weights.downsample.rows,
    dispatches,
    intermediates: {
      conv1: first.output,
      conv2: second.output,
      stack: third.output,
      transposed,
    },
  };
}
