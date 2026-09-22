//! A small RIFF/WAVE reader.
//!
//! Scope: the containers a browser recording or a test fixture actually produces -- uncompressed PCM
//! (8/16/24/32-bit, including `WAVE_FORMAT_EXTENSIBLE`) and 32-bit IEEE float, mono. Everything else
//! is refused with a typed error naming the reason, because silently mis-reading a compressed or
//! multi-channel container would poison the mel features in a way no downstream check can detect.
//!
//! Multi-channel audio is rejected rather than mixed: this reader's job is to reproduce a fixture
//! exactly, and a caller with a stereo file has `AudioContext.decodeAudioData` (which the
//! `AudioBuffer` path of `decodeAudio` accepts) to mix it with the platform's own rules.

import { STATUS } from "../wasm/abi.ts";
import { QwenscriberError } from "../errors.ts";

export const WAVE_FORMAT_PCM = 1;
export const WAVE_FORMAT_IEEE_FLOAT = 3;
export const WAVE_FORMAT_EXTENSIBLE = 0xfffe;
/** Last 12 bytes of KSDATAFORMAT_SUBTYPE_PCM / _IEEE_FLOAT, which share this tail. */
const SUBFORMAT_TAIL = [0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71];
/** A WAVE file with more chunks than this is not a WAVE file this reader wants to meet. */
const WAVE_CHUNKS_MAX = 64;

export interface WaveAudio {
  /** Mono samples in `[-1, 1]`. */
  readonly samples: Float32Array;
  readonly sample_rate_hz: number;
  readonly channel_count: number;
  readonly bits_per_sample: number;
  readonly frame_count: number;
  /** True for 32-bit IEEE float data, false for integer PCM. */
  readonly is_float: boolean;
}

interface WaveFormat {
  readonly format: number;
  readonly channel_count: number;
  readonly sample_rate_hz: number;
  readonly byte_rate: number;
  readonly block_align: number;
  readonly bits_per_sample: number;
}

function ascii(bytes: Uint8Array, offset: number, length: number): string {
  let text = "";
  for (let index = 0; index < length; index += 1) {
    text += String.fromCharCode(bytes[offset + index] ?? 0);
  }
  return text;
}

/** True when the bytes start with `RIFF....WAVE`. */
export function looksLikeWave(bytes: Uint8Array): boolean {
  if (bytes.length < 12) return false;
  if (ascii(bytes, 0, 4) !== "RIFF") return false;
  return ascii(bytes, 8, 4) === "WAVE";
}

function unsupportedWave(reason: string, context: Record<string, unknown>): QwenscriberError {
  return new QwenscriberError(STATUS.unsupported, "readWave", {
    message: `the WAVE container is not one this reader accepts: ${reason}`,
    context,
  });
}

function parseFormat(view: DataView, offset: number, size: number): WaveFormat {
  if (size < 16) {
    throw new QwenscriberError(STATUS.truncated, "readWave", {
      message: `the fmt chunk is ${size} bytes; the smallest valid one is 16`,
      context: { chunk_bytes: size },
    });
  }
  let format = view.getUint16(offset, true);
  if (format === WAVE_FORMAT_EXTENSIBLE) {
    // `fmt` for extensible: 16 bytes of the classic layout, then cbSize, valid bits, channel mask,
    // and a 16-byte subformat GUID. The GUID's first two bytes repeat the real format tag.
    if (size < 40) {
      throw new QwenscriberError(STATUS.truncated, "readWave", {
        message: `an extensible fmt chunk is 40 bytes; this one is ${size}`,
        context: { chunk_bytes: size },
      });
    }
    const extension_bytes = view.getUint16(offset + 16, true);
    if (extension_bytes < 22) {
      throw unsupportedWave(`extensible fmt declares ${extension_bytes} extension bytes`, {
        extension_bytes,
      });
    }
    for (let index = 0; index < SUBFORMAT_TAIL.length; index += 1) {
      if (view.getUint8(offset + 26 + index) !== SUBFORMAT_TAIL[index]) {
        throw unsupportedWave("the extensible subformat GUID is not PCM or IEEE float", {
          subformat_byte: index,
        });
      }
    }
    format = view.getUint16(offset + 24, true);
  }
  if (format !== WAVE_FORMAT_PCM) {
    if (format !== WAVE_FORMAT_IEEE_FLOAT) {
      throw unsupportedWave(`audio format 0x${format.toString(16)} is not PCM or IEEE float`, {
        audio_format: format,
      });
    }
  }
  const channel_count = view.getUint16(offset + 2, true);
  const sample_rate_hz = view.getUint32(offset + 4, true);
  const byte_rate = view.getUint32(offset + 8, true);
  const block_align = view.getUint16(offset + 12, true);
  const bits_per_sample = view.getUint16(offset + 14, true);
  if (channel_count === 0) {
    throw unsupportedWave("the container declares zero channels", { channel_count });
  }
  if (sample_rate_hz === 0) {
    throw unsupportedWave("the container declares a zero sample rate", { sample_rate_hz });
  }
  const expected_align = (channel_count * bits_per_sample) / 8;
  if (block_align !== expected_align) {
    throw new QwenscriberError(STATUS.shape_mismatch, "readWave", {
      message: `block alignment ${block_align} contradicts ${channel_count} channels at ` +
        `${bits_per_sample} bits`,
      context: { block_align, expected_block_align: expected_align },
    });
  }
  if (byte_rate !== sample_rate_hz * block_align) {
    throw new QwenscriberError(STATUS.shape_mismatch, "readWave", {
      message: `byte rate ${byte_rate} contradicts sample rate ${sample_rate_hz} at ` +
        `${block_align} bytes per frame`,
      context: { byte_rate, expected_byte_rate: sample_rate_hz * block_align },
    });
  }
  return {
    format,
    channel_count,
    sample_rate_hz,
    byte_rate,
    block_align,
    bits_per_sample,
  };
}

function decodeSamples(
  bytes: Uint8Array,
  offset: number,
  size: number,
  format: WaveFormat,
): Float32Array {
  const { bits_per_sample } = format;
  const is_float = format.format === WAVE_FORMAT_IEEE_FLOAT;
  if (is_float) {
    if (bits_per_sample !== 32) {
      throw unsupportedWave(`IEEE float samples must be 32-bit, got ${bits_per_sample}`, {
        bits_per_sample,
      });
    }
  } else if (
    bits_per_sample !== 8 &&
    bits_per_sample !== 16 &&
    bits_per_sample !== 24 &&
    bits_per_sample !== 32
  ) {
    throw unsupportedWave(`PCM samples of ${bits_per_sample} bits are not supported`, {
      bits_per_sample,
    });
  }
  const bytes_per_sample = bits_per_sample / 8;
  if (size % bytes_per_sample !== 0) {
    throw new QwenscriberError(STATUS.truncated, "readWave", {
      message: `${size} data bytes is not a whole number of ${bits_per_sample}-bit samples`,
      context: { data_bytes: size, bits_per_sample },
    });
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const frame_count = size / bytes_per_sample;
  const samples = new Float32Array(frame_count);
  for (let index = 0; index < frame_count; index += 1) {
    const position = offset + index * bytes_per_sample;
    let value: number;
    if (is_float) {
      value = view.getFloat32(position, true);
    } else if (bits_per_sample === 8) {
      // 8-bit PCM is unsigned in WAVE; 128 is silence.
      value = ((bytes[position] ?? 128) - 128) / 128;
    } else if (bits_per_sample === 16) {
      value = view.getInt16(position, true) / 32768;
    } else if (bits_per_sample === 24) {
      const raw =
        (bytes[position] ?? 0) | ((bytes[position + 1] ?? 0) << 8) | ((bytes[position + 2] ?? 0) << 16);
      const signed = (raw & 0x800000) === 0 ? raw : raw - 0x1000000;
      value = signed / 8388608;
    } else {
      value = view.getInt32(position, true) / 2147483648;
    }
    if (!Number.isFinite(value)) {
      throw new QwenscriberError(STATUS.invalid_argument, "readWave", {
        message: `sample ${index} is not a finite number`,
        context: { sample_index: index, sample_rate_hz: format.sample_rate_hz },
      });
    }
    samples[index] = value;
  }
  return samples;
}

/**
 * Reads a mono PCM/float WAVE container.
 *
 * The declared RIFF size is checked against the bytes actually present, because a truncated download
 * is the common failure and a container that lies about its length would otherwise decode as a clip
 * that silently ends early.
 */
export function readWave(bytes: Uint8Array): WaveAudio {
  if (!looksLikeWave(bytes)) {
    throw new QwenscriberError(STATUS.bad_magic, "readWave", {
      message: "the bytes do not start with RIFF....WAVE",
      context: { byte_length: bytes.length },
    });
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const declared_bytes = view.getUint32(4, true) + 8;
  if (declared_bytes > bytes.length) {
    throw new QwenscriberError(STATUS.truncated, "readWave", {
      message: `the header declares ${declared_bytes} bytes but only ${bytes.length} are present`,
      context: { declared_bytes, byte_length: bytes.length },
    });
  }
  let format: WaveFormat | undefined;
  let data_offset = -1;
  let data_bytes = 0;
  let cursor = 12;
  let chunks_seen = 0;
  while (cursor + 8 <= bytes.length) {
    if (chunks_seen >= WAVE_CHUNKS_MAX) {
      throw new QwenscriberError(STATUS.limit_exceeded, "readWave", {
        message: `a WAVE file has at most ${WAVE_CHUNKS_MAX} chunks; this one claims more`,
        context: { chunks_seen },
      });
    }
    chunks_seen += 1;
    const identifier = ascii(bytes, cursor, 4);
    const size = view.getUint32(cursor + 4, true);
    const body = cursor + 8;
    if (body + size > bytes.length) {
      throw new QwenscriberError(STATUS.truncated, "readWave", {
        message: `the ${identifier} chunk declares ${size} bytes but the file ends first`,
        context: { chunk: identifier, chunk_bytes: size, byte_length: bytes.length },
      });
    }
    if (identifier === "fmt ") {
      format = parseFormat(view, body, size);
    } else if (identifier === "data") {
      data_offset = body;
      data_bytes = size;
    }
    // Chunks are word-aligned: an odd-sized chunk is followed by one pad byte.
    cursor = body + size + (size % 2);
  }
  if (format === undefined) {
    throw new QwenscriberError(STATUS.not_found, "readWave", { message: "the fmt chunk is missing" });
  }
  if (data_offset < 0) {
    throw new QwenscriberError(STATUS.not_found, "readWave", {
      message: "the data chunk is missing",
    });
  }
  if (format.channel_count !== 1) {
    throw unsupportedWave(
      `the container has ${format.channel_count} channels and this reader is mono-only`,
      { channel_count: format.channel_count },
    );
  }
  const samples = decodeSamples(bytes, data_offset, data_bytes, format);
  return {
    samples,
    sample_rate_hz: format.sample_rate_hz,
    channel_count: format.channel_count,
    bits_per_sample: format.bits_per_sample,
    frame_count: samples.length,
    is_float: format.format === WAVE_FORMAT_IEEE_FLOAT,
  };
}
