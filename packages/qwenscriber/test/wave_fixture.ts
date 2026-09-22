//! A tiny mono WAVE writer, so the audio tests can build a container byte by byte.
//!
//! Tests need bytes whose every field they chose: a header with a wrong `block_align`, a stereo
//! declaration, a compressed format tag. Producing those from a fixture file would mean guessing
//! what a generator wrote; producing them here means the test states the bytes it is asserting about.

import { WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_PCM } from "../src/audio/wav.ts";

export type WaveSampleFormat = "i16" | "f32";

function writeAscii(target: Uint8Array, offset: number, text: string): number {
  for (let index = 0; index < text.length; index += 1) {
    target[offset + index] = text.charCodeAt(index);
  }
  return offset + 4;
}

export function writeWave(
  samples: Float32Array,
  sample_rate_hz: number,
  format: WaveSampleFormat = "i16",
  channel_count = 1,
): Uint8Array<ArrayBuffer> {
  const bytes_per_sample = format === "i16" ? 2 : 4;
  const data_bytes = samples.length * bytes_per_sample;
  const bytes = new Uint8Array(44 + data_bytes);
  const view = new DataView(bytes.buffer);
  let cursor = writeAscii(bytes, 0, "RIFF");
  view.setUint32(cursor, 36 + data_bytes, true);
  cursor += 4;
  cursor = writeAscii(bytes, cursor, "WAVE");
  cursor = writeAscii(bytes, cursor, "fmt ");
  view.setUint32(cursor, 16, true);
  cursor += 4;
  view.setUint16(cursor, format === "i16" ? WAVE_FORMAT_PCM : WAVE_FORMAT_IEEE_FLOAT, true);
  cursor += 2;
  view.setUint16(cursor, channel_count, true);
  cursor += 2;
  view.setUint32(cursor, sample_rate_hz, true);
  cursor += 4;
  view.setUint32(cursor, sample_rate_hz * channel_count * bytes_per_sample, true);
  cursor += 4;
  view.setUint16(cursor, channel_count * bytes_per_sample, true);
  cursor += 2;
  view.setUint16(cursor, bytes_per_sample * 8, true);
  cursor += 2;
  cursor = writeAscii(bytes, cursor, "data");
  view.setUint32(cursor, data_bytes, true);
  cursor += 4;
  const data = new DataView(bytes.buffer, cursor, data_bytes);
  for (let index = 0; index < samples.length; index += 1) {
    const value = samples[index] ?? 0;
    if (format === "i16") {
      const clamped = value < -1 ? -1 : value > 1 ? 1 : value;
      data.setInt16(index * 2, Math.round(clamped * 32767), true);
    } else {
      data.setFloat32(index * 4, value, true);
    }
  }
  return bytes;
}

/** Overwrites the 16-bit `audio format` field, for the rejection tests. */
export function withAudioFormat(bytes: Uint8Array, audio_format: number): Uint8Array {
  const mutated = bytes.slice();
  new DataView(mutated.buffer).setUint16(20, audio_format, true);
  return mutated;
}

/** Overwrites the `block align` field, so a header can contradict itself. */
export function withBlockAlign(bytes: Uint8Array, block_align: number): Uint8Array {
  const mutated = bytes.slice();
  new DataView(mutated.buffer).setUint16(32, block_align, true);
  return mutated;
}
