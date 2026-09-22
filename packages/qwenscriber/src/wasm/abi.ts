//! The WASM ABI v1 wire contract, transcribed from `src/wasm/abi.zig`.
//!
//! This file is the TypeScript half of a two-sided contract, and nothing here may drift from the
//! Zig side: every offset, status code, and geometry constant below is what `src/wasm/exports.zig`
//! reads and writes. Field names keep the ABI's spelling (`global_max_log`, `ids_ptr`) because they
//! *are* the ABI's names; every other module in the SDK uses camelCase.
//!
//! Linear memory is little-endian (WebAssembly requires it), so every `DataView` access here passes
//! `littleEndian = true` explicitly rather than relying on the default of the host platform.

/** ABI major version this SDK speaks. `qw_abi_version` returns `major << 16 | minor`. */
export const ABI_MAJOR = 1;
/** ABI minor version this SDK was built against. A newer minor is additive and still accepted. */
export const ABI_MINOR = 0;
/** `0x00010000`: the exact version `src/wasm/abi.zig` packs for `major = 1, minor = 0`. */
export const ABI_VERSION_EXPECTED = (ABI_MAJOR << 16) | ABI_MINOR;

/** `0x00010001` -> `"v1.1"`, for error messages and the demo page. */
export function formatAbiVersion(version: number): string {
  return `v${version >>> 16}.${version & 0xffff}`;
}

/** ABI v1 exposes exactly one runtime instance; `qw_create` returns 1 or 0. */
export const HANDLE_MAX = 1;

/** Largest alignment `qw_alloc` honours, and the closed set it accepts. */
export const ALIGNMENT_MAX = 64;
export const SUPPORTED_ALIGNMENTS = [1, 2, 4, 8, 16, 32, 64] as const;
export type Alignment = (typeof SUPPORTED_ALIGNMENTS)[number];

/**
 * Status codes, mirroring `abi.Status`.
 *
 * Zero is success; every failure is a distinct negative value so callers branch on numbers instead
 * of parsing text. `-14 + 1` is the count of failure codes the ABI defines.
 */
export const STATUS = {
  ok: 0,
  /** A pointer, length, or alignment argument was not usable. */
  invalid_argument: -1,
  /** The allocator could not satisfy a request. */
  out_of_memory: -2,
  /** The requested capability does not exist in this build. */
  unsupported: -3,
  /** A container's magic bytes did not match. */
  bad_magic: -4,
  /** A container or manifest version this build cannot read. */
  bad_version: -5,
  /** An integrity check failed. */
  checksum_mismatch: -6,
  /** A tensor or buffer shape violated the model's contract. */
  shape_mismatch: -7,
  /** A call arrived in the wrong state (for example decode before begin). */
  invalid_state: -8,
  /** A fixed capacity would have been exceeded. */
  limit_exceeded: -9,
  /** A referenced object does not exist in the container. */
  not_found: -10,
  /** Input ended before the declared structure did. */
  truncated: -11,
  /** Text was not valid UTF-8, or a token was outside the byte alphabet. */
  invalid_encoding: -12,
  /** The model configuration is not one this build supports. */
  unsupported_model: -13,
  /** More audio was supplied than the runtime accepts in one call. */
  audio_too_long: -14,
} as const;
export type StatusName = keyof typeof STATUS;

/** Failure codes in `-1 .. -14` order, so `STATUS_NAMES[-code - 1]` is the name of a code. */
export const STATUS_NAMES: readonly StatusName[] = [
  "invalid_argument",
  "out_of_memory",
  "unsupported",
  "bad_magic",
  "bad_version",
  "checksum_mismatch",
  "shape_mismatch",
  "invalid_state",
  "limit_exceeded",
  "not_found",
  "truncated",
  "invalid_encoding",
  "unsupported_model",
  "audio_too_long",
];

export function statusName(code: number): StatusName | "unknown" {
  if (code === STATUS.ok) return "ok";
  if (code > STATUS.ok) return "unknown";
  if (code < STATUS.audio_too_long) return "unknown";
  return STATUS_NAMES[-code - 1] ?? "unknown";
}

/**
 * Mirrors `abi.statusMessage`, which is the text `qw_error_message_ptr` points at.
 *
 * The module is the authority while it is loaded; this table exists for the codes the SDK can
 * report before a module exists (a failed instantiation, a rejected argument).
 */
export const STATUS_FALLBACK_MESSAGES: Readonly<Record<StatusName, string>> = {
  ok: "ok",
  invalid_argument: "invalid argument",
  out_of_memory: "out of memory",
  unsupported: "unsupported",
  bad_magic: "bad magic",
  bad_version: "bad version",
  checksum_mismatch: "checksum mismatch",
  shape_mismatch: "shape mismatch",
  invalid_state: "invalid state",
  limit_exceeded: "limit exceeded",
  not_found: "not found",
  truncated: "truncated input",
  invalid_encoding: "invalid encoding",
  unsupported_model: "unsupported model",
  audio_too_long: "audio too long",
};

export function statusFallbackMessage(code: number): string {
  const name = statusName(code);
  return name === "unknown" ? `status ${code}` : STATUS_FALLBACK_MESSAGES[name];
}

// ---------------------------------------------------------------------------
// Log-mel geometry (src/core/mel.zig)
// ---------------------------------------------------------------------------

export const MEL_BINS = 128;
export const MEL_N_FFT = 400;
export const MEL_HOP_LENGTH = 160;
export const MEL_SAMPLE_RATE_HZ = 16000;
/** Audio capacity of the reference buffer: 30 seconds at 16 kHz. */
export const MEL_CAPACITY_SAMPLES = 480000;
export const MEL_CAPACITY_FRAMES = MEL_CAPACITY_SAMPLES / MEL_HOP_LENGTH;
/** Clips below this length are zero-padded before the transform. */
export const MEL_MIN_SAMPLES = 8000;
/** `log10(1e-10)`: the floor a frame of digital silence clamps to before the range clamp. */
export const MEL_SILENCE_LOG = -10;
export const MEL_DYNAMIC_RANGE = 8;
export const MEL_NORMALIZATION_OFFSET = 4;
export const MEL_NORMALIZATION_SCALE = 4;

/** Mirrors `mel.paddingFrameValue`: the value a zero-padded frame carries in the mel domain. */
export function melPaddingValueFromGlobalMax(global_max_log: number): number {
  const clamped = Math.max(global_max_log - MEL_DYNAMIC_RANGE, MEL_SILENCE_LOG);
  return (clamped + MEL_NORMALIZATION_OFFSET) / MEL_NORMALIZATION_SCALE;
}

// ---------------------------------------------------------------------------
// Struct layouts, as `exports.zig` writes them
// ---------------------------------------------------------------------------

/** `abi.Status` is a 4-byte integer. */
export const STATUS_BYTES = 4;

/**
 * `qw_mel_result`, written by `qw_mel_compute`. 16 bytes, 4-byte aligned.
 *
 *     offset  0  u32  frames_written
 *     offset  4  u32  bytes_written
 *     offset  8  f32  global_max_log
 *     offset 12  u32  reserved
 */
export const MEL_RESULT_BYTES = 16;
export const MEL_RESULT_OFFSET = {
  frames_written: 0,
  bytes_written: 4,
  global_max_log: 8,
  reserved: 12,
} as const;

export interface MelResult {
  readonly frames_written: number;
  readonly bytes_written: number;
  readonly global_max_log: number;
  readonly reserved: number;
}

export function readMelResult(view: DataView, offset = 0): MelResult {
  return {
    frames_written: view.getUint32(offset + MEL_RESULT_OFFSET.frames_written, true),
    bytes_written: view.getUint32(offset + MEL_RESULT_OFFSET.bytes_written, true),
    global_max_log: view.getFloat32(offset + MEL_RESULT_OFFSET.global_max_log, true),
    reserved: view.getUint32(offset + MEL_RESULT_OFFSET.reserved, true),
  };
}

/**
 * `qw_selftest_result`, written by `qw_selftest`. 32 bytes, 8-byte aligned.
 *
 *     offset  0  u32  failures (bit mask of failed checks)
 *     offset  4  u32  loudest_band
 *     offset  8  f32  silence_value
 *     offset 12  u32  reserved
 *     offset 16  u64  quant_hash
 *     offset 24  u64  mel_hash
 */
export const SELF_TEST_RESULT_BYTES = 32;
export const SELF_TEST_RESULT_OFFSET = {
  failures: 0,
  loudest_band: 4,
  silence_value: 8,
  reserved: 12,
  quant_hash: 16,
  mel_hash: 24,
} as const;

export interface SelfTestResult {
  readonly failures: number;
  readonly loudest_band: number;
  readonly silence_value: number;
  readonly reserved: number;
  readonly quant_hash: bigint;
  readonly mel_hash: bigint;
}

export function readSelfTestResult(view: DataView, offset = 0): SelfTestResult {
  return {
    failures: view.getUint32(offset + SELF_TEST_RESULT_OFFSET.failures, true),
    loudest_band: view.getUint32(offset + SELF_TEST_RESULT_OFFSET.loudest_band, true),
    silence_value: view.getFloat32(offset + SELF_TEST_RESULT_OFFSET.silence_value, true),
    reserved: view.getUint32(offset + SELF_TEST_RESULT_OFFSET.reserved, true),
    quant_hash: view.getBigUint64(offset + SELF_TEST_RESULT_OFFSET.quant_hash, true),
    mel_hash: view.getBigUint64(offset + SELF_TEST_RESULT_OFFSET.mel_hash, true),
  };
}

/** The `failures` bit mask, mirroring `selftest.Report.check_*`. */
export const SELF_TEST_CHECKS: readonly { readonly mask: number; readonly name: string }[] = [
  { mask: 1 << 0, name: "quantization" },
  { mask: 1 << 1, name: "log-mel" },
  { mask: 1 << 2, name: "tokenizer" },
  { mask: 1 << 3, name: "simd" },
  { mask: 1 << 4, name: "math" },
];

/** Names of the checks a nonzero `failures` mask reports, in bit order. */
export function failedSelfTestChecks(failures: number): string[] {
  const failed: string[] = [];
  for (const check of SELF_TEST_CHECKS) {
    if ((failures & check.mask) !== 0) failed.push(check.name);
  }
  return failed;
}

/**
 * `qw_tokenizer`, the vocabulary the runtime decodes with. 24 bytes, 4-byte aligned.
 *
 *     offset  0  u32  offsets_ptr   (u32 per token, count + 1 entries)
 *     offset  4  u32  offsets_len
 *     offset  8  u32  bytes_ptr
 *     offset 12  u32  bytes_len
 *     offset 16  u32  token_count
 *     offset 20  u32  reserved
 *
 * The runtime does not copy the vocabulary: the caller keeps both buffers alive until
 * `qw_tokenizer_set` runs again or the instance is destroyed.
 */
export const TOKENIZER_DESCRIPTOR_BYTES = 24;
export const TOKENIZER_DESCRIPTOR_OFFSET = {
  offsets_ptr: 0,
  offsets_len: 4,
  bytes_ptr: 8,
  bytes_len: 12,
  token_count: 16,
  reserved: 20,
} as const;

export interface TokenizerDescriptor {
  readonly offsets_ptr: number;
  readonly offsets_len: number;
  readonly bytes_ptr: number;
  readonly bytes_len: number;
  readonly token_count: number;
  readonly reserved: number;
}

export function writeTokenizerDescriptor(
  view: DataView,
  offset: number,
  descriptor: TokenizerDescriptor,
): void {
  view.setUint32(offset + TOKENIZER_DESCRIPTOR_OFFSET.offsets_ptr, descriptor.offsets_ptr, true);
  view.setUint32(offset + TOKENIZER_DESCRIPTOR_OFFSET.offsets_len, descriptor.offsets_len, true);
  view.setUint32(offset + TOKENIZER_DESCRIPTOR_OFFSET.bytes_ptr, descriptor.bytes_ptr, true);
  view.setUint32(offset + TOKENIZER_DESCRIPTOR_OFFSET.bytes_len, descriptor.bytes_len, true);
  view.setUint32(offset + TOKENIZER_DESCRIPTOR_OFFSET.token_count, descriptor.token_count, true);
  view.setUint32(offset + TOKENIZER_DESCRIPTOR_OFFSET.reserved, descriptor.reserved, true);
}

// ---------------------------------------------------------------------------
// Capability families (`qw_features`, abi.Feature)
// ---------------------------------------------------------------------------

/**
 * The families `qw_features` reports, one bit each.
 *
 * A module built from an older source of the same ABI version reports fewer bits, which is how the
 * SDK tells "this build cannot load a model" from "the call is wrong" without probing for exports
 * one at a time.
 */
export const FEATURE = {
  /** `qw_mel_*`: the log-mel frontend. */
  mel: 1 << 0,
  /** `qw_tokenizer_set` and `qw_detokenize`. */
  tokenizer: 1 << 1,
  /** `qw_selftest`. */
  selftest: 1 << 2,
  /** `qw_model_*`: container parsing and model loading. */
  model: 1 << 3,
  /** `qw_decode_*`: audio tower, projector, and greedy decoding. */
  decode: 1 << 4,
} as const;
export type FeatureName = keyof typeof FEATURE;

/** Family names in bit order, so `featureNames(mask)` reports them the way `abi.Feature` declares them. */
export const FEATURE_NAMES: readonly FeatureName[] = [
  "mel",
  "tokenizer",
  "selftest",
  "model",
  "decode",
];

/** Every family `abi.features` publishes. */
export const FEATURE_ALL: number = FEATURE_NAMES.reduce((mask, name) => mask | FEATURE[name], 0);

// ---------------------------------------------------------------------------
// Model requirements (`qw_model_requirements`, abi.ModelRequirements)
// ---------------------------------------------------------------------------

/**
 * `abi.ModelRequirements`: what a loaded model keeps resident, and the limits it decodes within.
 * 48 bytes, 8-byte aligned.
 *
 *     offset  0  u64  weight_bytes        shard bytes the caller supplied
 *     offset  8  u64  cache_bytes         key and value cache, all layers
 *     offset 16  u64  scratch_bytes       activations, rotary tables, logits
 *     offset 24  u64  total_bytes         weight + cache + scratch
 *     offset 32  u32  max_positions       decoder positions the cache addresses
 *     offset 36  u32  max_audio_frames    mel frames one clip may hold
 *     offset 40  u32  max_decode_tokens   token budget of one utterance
 *     offset 44  u32  reserved
 *
 * `weight_bytes` counts the shard buffers as the caller supplied them, container padding included,
 * because the model borrows them instead of copying: they are resident for as long as the model is.
 * The runtime's own bookkeeping (tensor bindings, layer tables, prompt and token buffers) is not in
 * `total_bytes`; for a 0.6B model it stays under 64 KiB.
 *
 * The byte counts are read as `u64` and returned as numbers. Every count here is a resident byte
 * count of a single model, so it is far below `Number.MAX_SAFE_INTEGER`; a model that large could
 * not be resident in a 32-bit address space at all.
 */
export const MODEL_REQUIREMENTS_BYTES = 48;
export const MODEL_REQUIREMENTS_OFFSET = {
  weight_bytes: 0,
  cache_bytes: 8,
  scratch_bytes: 16,
  total_bytes: 24,
  max_positions: 32,
  max_audio_frames: 36,
  max_decode_tokens: 40,
  reserved: 44,
} as const;

export interface ModelRequirements {
  readonly weight_bytes: number;
  readonly cache_bytes: number;
  readonly scratch_bytes: number;
  readonly total_bytes: number;
  readonly max_positions: number;
  readonly max_audio_frames: number;
  readonly max_decode_tokens: number;
  readonly reserved: number;
}

export function readModelRequirements(view: DataView, offset = 0): ModelRequirements {
  return {
    weight_bytes: Number(view.getBigUint64(offset + MODEL_REQUIREMENTS_OFFSET.weight_bytes, true)),
    cache_bytes: Number(view.getBigUint64(offset + MODEL_REQUIREMENTS_OFFSET.cache_bytes, true)),
    scratch_bytes: Number(
      view.getBigUint64(offset + MODEL_REQUIREMENTS_OFFSET.scratch_bytes, true),
    ),
    total_bytes: Number(view.getBigUint64(offset + MODEL_REQUIREMENTS_OFFSET.total_bytes, true)),
    max_positions: view.getUint32(offset + MODEL_REQUIREMENTS_OFFSET.max_positions, true),
    max_audio_frames: view.getUint32(offset + MODEL_REQUIREMENTS_OFFSET.max_audio_frames, true),
    max_decode_tokens: view.getUint32(offset + MODEL_REQUIREMENTS_OFFSET.max_decode_tokens, true),
    reserved: view.getUint32(offset + MODEL_REQUIREMENTS_OFFSET.reserved, true),
  };
}

// ---------------------------------------------------------------------------
// Model directory format
// ---------------------------------------------------------------------------

/** Exact size of a `config.bin` (`model_config.size_bytes`); the runtime refuses anything else. */
export const MODEL_CONFIG_BYTES = 160;

/**
 * Alignment every shard buffer must have.
 *
 * `container.File.parse` hands out tensor views that alias the caller's buffer, so the buffer has to
 * start on a 16-byte boundary: `qw_alloc(size, SHARD_ALIGNMENT)`.
 */
export const SHARD_ALIGNMENT = 16;

/** Every shard file ends in this, which is how a directory's shards are recognized. */
export const SHARD_SUFFIX = ".qw";
