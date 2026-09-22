//! The audio tower's weights, resident on the GPU.
//!
//! The tower's convolutions and its downsample projection are the only tensors it opens with, and
//! all of them keep the checkpoint's element formats: f16 weights and f32 biases. They are uploaded
//! once and read as the packed bytes the container stores, which is what the shaders decode -- no
//! f32 expansion anywhere on this side.
//!
//! Dimensions come from each tensor's own descriptor rather than from a table kept here: the
//! container already records `[out_channels][in_channels][3][3]` for a convolution and
//! `[rows][cols]` for the projection, and a second copy of those numbers is a second thing to keep
//! in step.

import { QwenscriberError, SDK_STATUS } from "../errors.ts";
import { dataPlaneOffsetBytes, quantFormatOf, type QuantFormat } from "./quant_layout.ts";
import type { WasmModel } from "../decode.ts";
import { TENSOR_KIND } from "./tensor_kind.ts";
import type { WebGpuRuntime } from "./runtime.ts";

/** One convolution of the tower: its weights and bias, with the shape they were written at. */
export interface ConvStackWeights {
  readonly weight: GPUBuffer;
  readonly bias: GPUBuffer;
  readonly in_channels: number;
  readonly out_channels: number;
}

/**
 * The downsample projection that follows the stack: `[rows][cols]`.
 *
 * `quant` is present when the converter quantized the projection, which it does for every
 * conversion except the f16 one: the projection is a weight matrix like any other, so the q4/q5/q8
 * builds carry it in a two-plane format and the dispatch has to pick the matching kernel and pass
 * the plane offset. Its `format` is the descriptor's own value rather than a copy this file keeps.
 */
export interface DownsampleWeights {
  readonly weight: GPUBuffer;
  readonly rows: number;
  readonly cols: number;
  readonly quant?: { readonly format: QuantFormat; readonly dataOffsetBytes: number } | undefined;
}

/** Everything the tower's first stage reads. */
export interface TowerWeights {
  readonly conv1: ConvStackWeights;
  readonly conv2: ConvStackWeights;
  readonly conv3: ConvStackWeights;
  readonly downsample: DownsampleWeights;
}

function tensorOrThrow(model: WasmModel, name: keyof typeof TENSOR_KIND): {
  readonly bytes: Uint8Array;
  readonly dims: readonly [number, number, number, number];
  readonly rank: number;
  readonly format: number;
} {
  const kind = TENSOR_KIND[name];
  const tensor = model.tensor(kind, 0);
  if (tensor === undefined) {
    // A loaded model resolved every tensor its architecture names, so this is a model whose shards
    // do not match the architecture the core parsed rather than a caller's mistake.
    throw new QwenscriberError(SDK_STATUS.protocol, "tower_weights", {
      message: `${model.modelId} holds no ${name} tensor (kind ${kind})`,
      context: { tensor: name, kind },
    });
  }
  return {
    bytes: tensor.bytes,
    dims: tensor.descriptor.dims,
    rank: tensor.descriptor.rank,
    format: tensor.descriptor.format,
  };
}

function convStackWeights(
  runtime: WebGpuRuntime,
  model: WasmModel,
  index: 1 | 2 | 3,
): ConvStackWeights {
  const weight = tensorOrThrow(model, `audio_conv${index}_weight`);
  const bias = tensorOrThrow(model, `audio_conv${index}_bias`);
  if (weight.rank !== 4 || bias.rank !== 1) {
    throw new QwenscriberError(SDK_STATUS.protocol, "tower_weights", {
      message:
        `audio.conv${index}.weight must be [out][in][kernel][kernel] and its bias [out], got rank ` +
        `${weight.rank} and ${bias.rank}`,
      context: { index, weight_rank: weight.rank, bias_rank: bias.rank },
    });
  }
  return {
    weight: runtime.uploadBytes(weight.bytes, `tower.conv${index}.weight`),
    bias: runtime.uploadBytes(bias.bytes, `tower.conv${index}.bias`),
    out_channels: weight.dims[0],
    in_channels: weight.dims[1],
  };
}

/**
 * Uploads every weight the tower opens with.
 *
 * The buffers are the caller's to destroy: they are weights, so they outlive any single clip and
 * should be released with the model rather than per utterance.
 */
export function uploadTowerWeights(runtime: WebGpuRuntime, model: WasmModel): TowerWeights {
  const downsample = tensorOrThrow(model, "audio_conv_out_weight");
  if (downsample.rank !== 2) {
    throw new QwenscriberError(SDK_STATUS.protocol, "tower_weights", {
      message: `audio.conv_out.weight must be [rows][cols], got rank ${downsample.rank}`,
      context: { rank: downsample.rank },
    });
  }
  const quantized = quantFormatOf(downsample.format);
  return {
    conv1: convStackWeights(runtime, model, 1),
    conv2: convStackWeights(runtime, model, 2),
    conv3: convStackWeights(runtime, model, 3),
    downsample: {
      weight: runtime.uploadBytes(downsample.bytes, "tower.conv_out.weight"),
      rows: downsample.dims[0],
      cols: downsample.dims[1],
      ...(quantized === undefined
        ? {}
        : {
            quant: {
              format: quantized,
              dataOffsetBytes: dataPlaneOffsetBytes(downsample.dims[0], downsample.dims[1]),
            },
          }),
    },
  };
}
