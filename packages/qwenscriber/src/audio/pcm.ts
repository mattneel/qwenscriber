//! Turning whatever a caller has into mono f32 samples.
//!
//! Accepted, in the order they are recognised:
//!
//!   * `Blob`/`File` -- read as bytes, then as a WAVE container or raw PCM. Bytes are sniffed
//!     rather than trusted, so a WAVE inside a `Blob` reports `source_kind: "wave"`: the kind
//!     records what the bytes turned out to be, which is the part a report needs.
//!   * `Float32Array` -- mono f32 in `[-1, 1]`, taken as 16 kHz unless told otherwise.
//!   * `Int16Array` / `Int32Array` -- integer PCM, scaled by 32768 / 2147483648.
//!   * `ArrayBuffer` / `Uint8Array` -- 16-bit or 32-bit PCM, or a RIFF/WAVE container (sniffed).
//!   * An `AudioBuffer`-like object -- anything with `getChannelData`, `sampleRate`, and
//!     `numberOfChannels`: a real `AudioBuffer` from `decodeAudioData` fits without a dependency.
//!
//! Every path returns a fresh `Float32Array` that this SDK owns. Nothing here ever hands back a view
//! of the caller's buffer, because the worker boundary transfers what it sends, and transfer detaches
//! the source -- a caller who passed their own array would find it emptied afterwards.

import { MEL_SAMPLE_RATE_HZ, STATUS } from "../wasm/abi.ts";
import { QwenscriberError } from "../errors.ts";
import { looksLikeWave, readWave } from "./wav.ts";
import type { AudioSourceKind } from "../types.ts";

export type PcmFormat = "f32" | "i16" | "i32";

/** The part of `AudioBuffer` this SDK uses. Structural, so no DOM type is required. */
export interface AudioBufferLike {
  readonly sampleRate: number;
  readonly numberOfChannels?: number | undefined;
  readonly length?: number | undefined;
  readonly duration?: number | undefined;
  getChannelData(channel: number): Float32Array;
}

export type AudioInput =
  | Blob
  | Float32Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | ArrayBuffer
  | AudioBufferLike;

export interface DecodeOptions {
  /** How to read bytes that are not a WAVE container. Defaults to `"i16"`. */
  readonly format?: PcmFormat | undefined;
  /** Sample rate for PCM that carries none. Defaults to 16000, the rate the core wants. */
  readonly sample_rate_hz?: number | undefined;
}

export interface DecodedAudio {
  /** Mono, f32, in `[-1, 1]`. Owned by the SDK, never a view of the caller's buffer. */
  readonly samples: Float32Array;
  readonly sample_rate_hz: number;
  readonly channel_count: number;
  readonly frames: number;
  readonly duration_ms: number;
  readonly source_kind: AudioSourceKind;
}

function finish(
  samples: Float32Array,
  sample_rate_hz: number,
  channel_count: number,
  source_kind: AudioSourceKind,
): DecodedAudio {
  if (!Number.isFinite(sample_rate_hz) || sample_rate_hz <= 0) {
    throw new QwenscriberError(STATUS.invalid_argument, "decodeAudio", {
      message: `sample rate must be positive, got ${sample_rate_hz}`,
      context: { sample_rate_hz, source_kind },
    });
  }
  return {
    samples,
    sample_rate_hz,
    channel_count,
    frames: samples.length,
    duration_ms: (samples.length / sample_rate_hz) * 1000,
    source_kind,
  };
}

/** Rejects NaN and infinities early; every later stage assumes finite input. */
function validateFinite(samples: Float32Array, source_kind: AudioSourceKind): void {
  for (let index = 0; index < samples.length; index += 1) {
    const value = samples[index] ?? 0;
    if (!Number.isFinite(value)) {
      throw new QwenscriberError(STATUS.invalid_argument, "decodeAudio", {
        message: `sample ${index} is not a finite number`,
        context: { sample_index: index, source_kind },
      });
    }
  }
}

function int16ToFloat32(values: Int16Array): Float32Array {
  const samples = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    // Divided by 32768, not 32767: the full int16 range then maps into [-1, 1), so the conversion
    // cannot produce a sample outside the range the core documents.
    samples[index] = (values[index] ?? 0) / 32768;
  }
  return samples;
}

function int32ToFloat32(values: Int32Array): Float32Array {
  const samples = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    samples[index] = (values[index] ?? 0) / 2147483648;
  }
  return samples;
}

/** Reads raw PCM out of a byte range. `DataView` reads, so an odd byte offset is not a problem. */
function pcmBytesToFloat32(
  bytes: Uint8Array,
  format: PcmFormat,
  sample_rate_hz: number,
): Float32Array {
  const bytes_per_sample = format === "i16" ? 2 : 4;
  if (bytes.byteLength % bytes_per_sample !== 0) {
    throw new QwenscriberError(STATUS.truncated, "decodeAudio", {
      message: `${bytes.byteLength} bytes is not a whole number of ${format} samples`,
      context: { byte_length: bytes.byteLength, format },
    });
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const frames = bytes.byteLength / bytes_per_sample;
  const samples = new Float32Array(frames);
  for (let index = 0; index < frames; index += 1) {
    const position = index * bytes_per_sample;
    if (format === "i16") {
      samples[index] = view.getInt16(position, true) / 32768;
    } else if (format === "i32") {
      samples[index] = view.getInt32(position, true) / 2147483648;
    } else {
      const value = view.getFloat32(position, true);
      if (!Number.isFinite(value)) {
        throw new QwenscriberError(STATUS.invalid_argument, "decodeAudio", {
          message: `sample ${index} is not a finite number`,
          context: { sample_index: index, sample_rate_hz },
        });
      }
      samples[index] = value;
    }
  }
  return samples;
}

/** Averages an `AudioBuffer`-like object's channels down to mono. */
function audioBufferToMono(buffer: AudioBufferLike): DecodedAudio {
  const channel_count = buffer.numberOfChannels ?? 1;
  if (!Number.isInteger(channel_count) || channel_count < 1) {
    throw new QwenscriberError(STATUS.invalid_argument, "decodeAudio", {
      message: `numberOfChannels must be a positive integer, got ${channel_count}`,
      context: { channel_count },
    });
  }
  const first = buffer.getChannelData(0);
  const frames = first.length;
  const samples = new Float32Array(frames);
  if (channel_count === 1) {
    samples.set(first);
  } else {
    // Averaging is the conventional mixdown for a stereo pair captured by two nearby microphones.
    // A caller with genuinely decorrelated channels should mix them before handing audio over.
    for (let channel = 0; channel < channel_count; channel += 1) {
      const data = buffer.getChannelData(channel);
      if (data.length !== frames) {
        throw new QwenscriberError(STATUS.shape_mismatch, "decodeAudio", {
          message: `channel ${channel} holds ${data.length} frames; channel 0 holds ${frames}`,
          context: { channel, channel_frames: data.length, frames },
        });
      }
      for (let index = 0; index < frames; index += 1) {
        samples[index] = (samples[index] ?? 0) + (data[index] ?? 0);
      }
    }
    for (let index = 0; index < frames; index += 1) {
      samples[index] = (samples[index] ?? 0) / channel_count;
    }
  }
  validateFinite(samples, "audio-buffer-like");
  return finish(samples, buffer.sampleRate, channel_count, "audio-buffer-like");
}

function isAudioBufferLike(value: unknown): value is AudioBufferLike {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as { getChannelData?: unknown; sampleRate?: unknown };
  if (typeof candidate.getChannelData !== "function") return false;
  return typeof candidate.sampleRate === "number";
}

/** Bytes are either a WAVE container or headerless PCM; the magic decides which. */
function fromBytes(
  bytes: Uint8Array,
  source_kind: AudioSourceKind,
  options: DecodeOptions,
): DecodedAudio {
  if (looksLikeWave(bytes)) {
    const wave = readWave(bytes);
    return finish(wave.samples, wave.sample_rate_hz, wave.channel_count, "wave");
  }
  const format = options.format ?? "i16";
  const sample_rate_hz = options.sample_rate_hz ?? MEL_SAMPLE_RATE_HZ;
  const samples = pcmBytesToFloat32(bytes, format, sample_rate_hz);
  return finish(samples, sample_rate_hz, 1, source_kind);
}

/**
 * Decodes one clip to mono f32 at whatever rate it actually is.
 *
 * Resampling to the core's 16 kHz is a separate step (`resample`) so the caller can see both rates.
 */
export async function decodeAudio(
  input: AudioInput,
  options: DecodeOptions = {},
): Promise<DecodedAudio> {
  const received: unknown = input;
  if (typeof Blob === "function" && input instanceof Blob) {
    return fromBytes(new Uint8Array(await input.arrayBuffer()), "blob", options);
  }
  if (input instanceof Float32Array) {
    const samples = input.slice();
    validateFinite(samples, "float32");
    return finish(samples, options.sample_rate_hz ?? MEL_SAMPLE_RATE_HZ, 1, "float32");
  }
  if (input instanceof Int16Array) {
    const samples = int16ToFloat32(input);
    return finish(samples, options.sample_rate_hz ?? MEL_SAMPLE_RATE_HZ, 1, "int16");
  }
  if (input instanceof Int32Array) {
    const samples = int32ToFloat32(input);
    return finish(samples, options.sample_rate_hz ?? MEL_SAMPLE_RATE_HZ, 1, "int32");
  }
  if (input instanceof Uint8Array) {
    return fromBytes(input, "pcm-bytes", options);
  }
  if (input instanceof ArrayBuffer) {
    return fromBytes(new Uint8Array(input), "pcm-bytes", options);
  }
  if (isAudioBufferLike(input)) return audioBufferToMono(input);
  throw new QwenscriberError(STATUS.invalid_argument, "decodeAudio", {
    message:
      "unrecognised audio input: pass a Float32Array, Int16Array, Int32Array, ArrayBuffer, " +
      "WAVE bytes, a Blob, or an AudioBuffer",
    context: { input_type: received === null ? "null" : Object.prototype.toString.call(received) },
  });
}
