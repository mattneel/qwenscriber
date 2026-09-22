//! `WasmCore`: the typed JavaScript face of the WASM ABI.
//!
//! One class owns one instantiated module and one `qw_create` handle. It does four things:
//!
//!   1. Refuses a module whose ABI major version it does not speak, before any other call.
//!   2. Rebuilds every `TypedArray`/`DataView` from `exports.memory.buffer`, because growing linear
//!      memory detaches existing views and `qw_alloc` grows memory. No view is ever cached.
//!   3. Owns the alloc/free pairing: an `Allocation` remembers the size and alignment it was made
//!      with, which the ABI requires it to repeat on release.
//!   4. Turns every nonzero status into a `QwenscriberError` carrying the module's own error text.

import {
  ALIGNMENT_MAX,
  ABI_MAJOR,
  ABI_VERSION_EXPECTED,
  FEATURE,
  MEL_BINS,
  MEL_CAPACITY_FRAMES,
  MEL_RESULT_BYTES,
  MODEL_CONFIG_BYTES,
  MODEL_REQUIREMENTS_BYTES,
  SELF_TEST_RESULT_BYTES,
  SHARD_ALIGNMENT,
  STATUS,
  SUPPORTED_ALIGNMENTS,
  TOKENIZER_DESCRIPTOR_BYTES,
  failedSelfTestChecks,
  readMelResult,
  readModelRequirements,
  readSelfTestResult,
  writeTokenizerDescriptor,
  type Alignment,
  type ModelRequirements,
  type TokenizerDescriptor,
} from "./abi.ts";
import { AbiMismatchError, NotImplementedError, QwenscriberError, SDK_STATUS, throwForStatus } from "../errors.ts";

/**
 * Everything `WasmCore.load` accepts.
 *
 * A URL is fetched (relative URLs resolve against the caller's document or worker), a byte source is
 * compiled directly, and an already-compiled `WebAssembly.Module` is instantiated as-is.
 */
export type WasmSource = WebAssembly.Module | ArrayBuffer | ArrayBufferView | string | URL;

/** The core exports every ABI v1 module publishes. A module missing one of these is refused. */
interface CoreExports {
  readonly memory: WebAssembly.Memory;
  qw_abi_version(): number;
  qw_core_version(): number;
  qw_create(): number;
  qw_destroy(handle: number): number;
  qw_alloc(size: number, alignment: number): number;
  qw_free(pointer: number, size: number, alignment: number): void;
  qw_memory_bytes(): number;
  qw_error_message_ptr(code: number): number;
  qw_error_message_len(code: number): number;
  qw_mel_frames_for_samples(sample_count: number): number;
  qw_mel_min_samples(): number;
  qw_mel_compute(
    handle: number,
    samples_ptr: number,
    sample_count: number,
    out_ptr: number,
    out_bytes: number,
    result_ptr: number,
  ): number;
  qw_mel_padding_value(global_max_log: number): number;
  qw_tokenizer_set(handle: number, descriptor_ptr: number): number;
  qw_detokenize(
    handle: number,
    ids_ptr: number,
    id_count: number,
    out_ptr: number,
    out_bytes: number,
    written_ptr: number,
  ): number;
  qw_selftest(result_ptr: number): number;
  /** Capability bits, absent from a module built before the model family existed. */
  qw_features?(): number;
  qw_model_begin?(handle: number): number;
  qw_model_add_shard?(handle: number, shard_ptr: number, shard_len: number): number;
  qw_model_finish?(handle: number, config_ptr: number, config_len: number): number;
  qw_model_requirements?(handle: number, out_ptr: number): number;
  qw_decode_begin?(
    handle: number,
    features_ptr: number,
    features_len: number,
    max_tokens: number,
  ): number;
  qw_decode_step?(handle: number, token_out_ptr: number): number;
  qw_decode_tokens?(handle: number, out_ptr: number, out_capacity_tokens: number): number;
  qw_decode_end?(handle: number): number;
}

/** The model family, bound only when the module reports its feature bits. */
interface ModelExports {
  qw_model_begin(handle: number): number;
  qw_model_add_shard(handle: number, shard_ptr: number, shard_len: number): number;
  qw_model_finish(handle: number, config_ptr: number, config_len: number): number;
  qw_model_requirements(handle: number, out_ptr: number): number;
  qw_decode_begin(
    handle: number,
    features_ptr: number,
    features_len: number,
    max_tokens: number,
  ): number;
  qw_decode_step(handle: number, token_out_ptr: number): number;
  qw_decode_tokens(handle: number, out_ptr: number, out_capacity_tokens: number): number;
  qw_decode_end(handle: number): number;
}

/** What `validateExports` produced: the always-present core, and the negotiated model family. */
interface BoundExports {
  readonly core: CoreExports;
  readonly model: ModelExports | undefined;
}

const TEXT_ENCODER = new TextEncoder();
const TEXT_DECODER = new TextDecoder("utf-8", { fatal: false, ignoreBOM: true });

/** Status codes are a closed table; caching their text bounded keeps lookups off the ABI. */
const ERROR_MESSAGE_CACHE_MAX = 32;

/** Ids per `detokenize` call. Bounded so a bad length cannot become a huge allocation. */
const DETOKENIZE_IDS_MAX = 1 << 20;
/** Tokens per vocabulary. Qwen3-ASR ships ~150k; this leaves room and stays bounded. */
const VOCABULARY_TOKENS_MAX = 1 << 21;

export interface MelComputation {
  /** Frames written, `ceil(sample_count / 160)`. */
  readonly frames: number;
  readonly bytesWritten: number;
  /** The largest log10 mel magnitude before clamping, as the reference measured it. */
  readonly globalMaxLog: number;
  /** Mel value a zero-padded frame carries: `(max(globalMaxLog - 8, -10) + 4) / 4`. */
  readonly paddingValue: number;
  /** `[mel_bin][frame]` row-major, `128 * frames` values. A copy, owned by the caller. */
  readonly features: Float32Array;
}

export interface VocabularyDescriptor {
  /** Token byte strings in id order: what each id maps to, before UTF-8 assembly. */
  readonly tokens: readonly (string | Uint8Array)[];
}

export interface SelfTestReport {
  /** Bit mask of failed checks; zero means the core verified itself against its golden constants. */
  readonly failures: number;
  /** Mel band that responds most strongly to a 1 kHz tone (42 for the reference geometry). */
  readonly loudestBand: number;
  /** Log-mel value the frontend produces for digital silence (-1.5). */
  readonly silenceValue: number;
  readonly quantHash: bigint;
  readonly melHash: bigint;
  /** Names of the failed checks, decoded from `failures` for display. */
  readonly failedChecks: readonly string[];
}

function unsupportedAlignment(alignment: number): QwenscriberError {
  return new QwenscriberError(STATUS.invalid_argument, "qw_alloc", {
    message: `qw_alloc accepts alignments {${SUPPORTED_ALIGNMENTS.join(", ")}}, got ${alignment}`,
    context: { alignment, alignment_max: ALIGNMENT_MAX },
  });
}

/**
 * A table whose first offset is not zero is not a vocabulary: offsets are relative to the start of
 * the token bytes, so the first one has to be zero. Refusing here is what keeps a table written with
 * an unexpected prefix from decoding into the middle of every token.
 */
function unsupportedTable(headerBytes: number, tokenCount: number, firstOffset: number): never {
  throw new QwenscriberError(STATUS.invalid_encoding, "qw_tokenizer_set", {
    message: `the vocabulary table's first offset is ${firstOffset}, not zero`,
    context: { token_count: tokenCount, header_bytes: headerBytes, first_offset: firstOffset },
  });
}

function requireExport(raw: WebAssembly.Exports, name: string): (...args: number[]) => number {
  const value = raw[name];
  if (typeof value !== "function") {
    throw new QwenscriberError(SDK_STATUS.protocol, "instantiate", {
      message: `the module does not export ${name}; it was not built by \`zig build wasm\``,
      context: { missing_export: name },
    });
  }
  return value as (...args: number[]) => number;
}

function requireVoidExport(raw: WebAssembly.Exports, name: string): (...args: number[]) => void {
  const value = raw[name];
  if (typeof value !== "function") {
    throw new QwenscriberError(SDK_STATUS.protocol, "instantiate", {
      message: `the module does not export ${name}; it was not built by \`zig build wasm\``,
      context: { missing_export: name },
    });
  }
  return value as (...args: number[]) => void;
}

/**
 * Binds the model and decode family when the module publishes it.
 *
 * The family landed after the first ABI v1 modules shipped, so it is negotiated rather than
 * required: `qw_features` says whether this build has it, and every export is checked for presence
 * in the same step. A module that reports the bits without the exports is refused, because that
 * combination means the module is not the one it claims to be.
 */
function bindModelFamily(raw: WebAssembly.Exports): ModelExports | undefined {
  const features = (raw["qw_features"] as (() => number) | undefined)?.() ?? 0;
  const wanted = (features & (FEATURE.model | FEATURE.decode)) === 
    (FEATURE.model | FEATURE.decode);
  if (!wanted) return undefined;
  return {
    qw_model_begin: requireExport(raw, "qw_model_begin"),
    qw_model_add_shard: requireExport(raw, "qw_model_add_shard"),
    qw_model_finish: requireExport(raw, "qw_model_finish"),
    qw_model_requirements: requireExport(raw, "qw_model_requirements"),
    qw_decode_begin: requireExport(raw, "qw_decode_begin"),
    qw_decode_step: requireExport(raw, "qw_decode_step"),
    qw_decode_tokens: requireExport(raw, "qw_decode_tokens"),
    qw_decode_end: requireExport(raw, "qw_decode_end"),
  };
}

/** Maps the raw instance exports onto `CoreExports`, refusing a module that is not ABI v1. */
function validateExports(raw: WebAssembly.Exports): BoundExports {
  const memory = raw["memory"];
  if (!(memory instanceof WebAssembly.Memory)) {
    throw new QwenscriberError(SDK_STATUS.protocol, "instantiate", {
      message: "the module does not export its linear memory as `memory`",
    });
  }
  const core: CoreExports = {
    memory,
    qw_abi_version: requireExport(raw, "qw_abi_version"),
    qw_core_version: requireExport(raw, "qw_core_version"),
    qw_create: requireExport(raw, "qw_create"),
    qw_destroy: requireExport(raw, "qw_destroy"),
    qw_alloc: requireExport(raw, "qw_alloc"),
    qw_free: requireVoidExport(raw, "qw_free"),
    qw_memory_bytes: requireExport(raw, "qw_memory_bytes"),
    qw_error_message_ptr: requireExport(raw, "qw_error_message_ptr"),
    qw_error_message_len: requireExport(raw, "qw_error_message_len"),
    qw_mel_frames_for_samples: requireExport(raw, "qw_mel_frames_for_samples"),
    qw_mel_min_samples: requireExport(raw, "qw_mel_min_samples"),
    qw_mel_compute: requireExport(raw, "qw_mel_compute"),
    qw_mel_padding_value: requireExport(raw, "qw_mel_padding_value"),
    qw_tokenizer_set: requireExport(raw, "qw_tokenizer_set"),
    qw_detokenize: requireExport(raw, "qw_detokenize"),
    qw_selftest: requireExport(raw, "qw_selftest"),
  };
  const qw_features = raw["qw_features"];
  if (typeof qw_features === "function") core.qw_features = qw_features as () => number;
  return { core, model: bindModelFamily(raw) };
}

async function fetchModuleBytes(url: string | URL): Promise<Uint8Array<ArrayBuffer>> {
  const href = typeof url === "string" ? url : url.href;
  if (href.startsWith("file:")) {
    // Browsers refuse file: URLs at fetch() and Node's fetch has no file: support either. The SDK
    // targets the browser; a Node caller reads the file itself and passes bytes.
    throw new QwenscriberError(SDK_STATUS.protocol, "load", {
      message:
        `cannot fetch ${href}: file: URLs are not fetchable. Read the bytes with node:fs and pass ` +
        `a Uint8Array (or load the module over http) instead.`,
      context: { url: href },
    });
  }
  let response: Response;
  try {
    response = await fetch(href);
  } catch (error) {
    throw new QwenscriberError(SDK_STATUS.protocol, "load", {
      message: `fetching ${href} failed`,
      context: { url: href },
      cause: error,
    });
  }
  if (!response.ok) {
    throw new QwenscriberError(SDK_STATUS.protocol, "load", {
      message: `fetching ${href} returned ${response.status} ${response.statusText}`,
      context: { url: href, http_status: response.status },
    });
  }
  return new Uint8Array(await response.arrayBuffer());
}

/** Compiles `source` into a module, or returns the one it already is. */
async function toModule(source: WasmSource): Promise<{ module: WebAssembly.Module; bytes: number }> {
  if (source instanceof WebAssembly.Module) return { module: source, bytes: 0 };
  let bytes: Uint8Array<ArrayBuffer>;
  if (typeof source === "string" || source instanceof URL) {
    bytes = await fetchModuleBytes(source);
  } else if (source instanceof ArrayBuffer) {
    bytes = new Uint8Array(source);
  } else {
    // Copied into a standalone buffer: `WebAssembly.compile` wants an ArrayBuffer-backed source,
    // and a caller's view of a larger buffer may be mutated while the module compiles.
    bytes = new Uint8Array(
      new Uint8Array(source.buffer, source.byteOffset, source.byteLength),
    );
  }
  if (bytes.byteLength === 0) {
    throw new QwenscriberError(SDK_STATUS.protocol, "load", { message: "the wasm source is empty" });
  }
  try {
    return { module: await WebAssembly.compile(bytes), bytes: bytes.byteLength };
  } catch (error) {
    throw new QwenscriberError(SDK_STATUS.protocol, "load", {
      message: "the bytes are not a valid WebAssembly module",
      context: { bytes: bytes.byteLength },
      cause: error,
    });
  }
}

/**
 * A linear-memory buffer this side of the ABI owns.
 *
 * `qw_free` must repeat the size and alignment of the matching `qw_alloc`, so the allocation
 * remembers both. Views are built on demand: an allocation is exactly the operation that can detach
 * a previously created view, so caching one would hand out a zero-length array.
 */
export class Allocation {
  readonly pointer: number;
  readonly size: number;
  readonly alignment: Alignment;
  private readonly core: WasmCore;
  private released = false;

  constructor(core: WasmCore, pointer: number, size: number, alignment: Alignment) {
    this.core = core;
    this.pointer = pointer;
    this.size = size;
    this.alignment = alignment;
  }

  /** True once `dispose()` has run. A released buffer must not be handed to the runtime. */
  get disposed(): boolean {
    return this.released;
  }

  /** A fresh view over the whole buffer. */
  get bytes(): Uint8Array {
    this.assertLive();
    return new Uint8Array(this.core.memoryBuffer(), this.pointer, this.size);
  }

  f32(count: number, byteOffset = 0): Float32Array {
    this.assertLive();
    this.assertFits(byteOffset + count * 4);
    return new Float32Array(this.core.memoryBuffer(), this.pointer + byteOffset, count);
  }

  u32(count: number, byteOffset = 0): Uint32Array {
    this.assertLive();
    this.assertFits(byteOffset + count * 4);
    return new Uint32Array(this.core.memoryBuffer(), this.pointer + byteOffset, count);
  }

  dataView(byteLength = this.size, byteOffset = 0): DataView {
    this.assertLive();
    this.assertFits(byteOffset + byteLength);
    return new DataView(this.core.memoryBuffer(), this.pointer + byteOffset, byteLength);
  }

  /** Copies `source` into this buffer, writing at offset zero. */
  copyFrom(source: ArrayBufferView | ArrayBuffer): void {
    const source_bytes =
      source instanceof ArrayBuffer
        ? new Uint8Array(source)
        : new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
    this.assertLive();
    this.assertFits(source_bytes.byteLength);
    this.bytes.set(source_bytes);
  }

  /** Releases the buffer. Idempotent, so a `finally` block and `using` cannot double-free. */
  dispose(): void {
    if (this.released) return;
    this.released = true;
    this.core.release(this);
  }

  /**
   * Explicit resource management (`using buffer = core.alloc(...)`).
   *
   * Where the engine has no `Symbol.dispose`, this compiles to a method keyed by the string
   * `"undefined"` and is simply never called; `dispose()` is the portable path.
   */
  [Symbol.dispose](): void {
    this.dispose();
  }

  private assertLive(): void {
    if (this.released) {
      throw new QwenscriberError(SDK_STATUS.protocol, "Allocation", {
        message: `the buffer at ${this.pointer} was already released`,
        context: { pointer: this.pointer, size: this.size },
      });
    }
  }

  private assertFits(bytes: number): void {
    if (bytes > this.size) {
      throw new QwenscriberError(STATUS.invalid_argument, "Allocation", {
        message: `the view needs ${bytes} bytes but the buffer holds ${this.size}`,
        context: { pointer: this.pointer, size: this.size, requested: bytes },
      });
    }
  }
}

/** One instantiated ABI v1 module with its `qw_create` handle. */
export class WasmCore {
  private readonly exports: CoreExports;
  /** The negotiated model family, or `undefined` when this build has none. */
  private readonly modelFamilyValue: ModelExports | undefined;
  private readonly errorMessages = new Map<number, string>();
  private readonly moduleBytesValue: number;
  private handleValue: number;
  private vocabulary:
    | { readonly allocations: readonly Allocation[]; readonly idBytes: Uint32Array }
    | undefined;

  private constructor(
    bound: BoundExports,
    handle: number,
    moduleBytes: number,
  ) {
    this.exports = bound.core;
    this.modelFamilyValue = bound.model;
    this.handleValue = handle;
    this.moduleBytesValue = moduleBytes;
  }

  /**
   * Compiles (if needed), instantiates with no imports, checks the ABI major version, and creates
   * the runtime handle.
   *
   * Instantiating twice from the same bytes yields two independent modules: `qw_create` returning
   * zero means the caller passed a module that already holds an instance, which v1 forbids.
   */
  static async load(source: WasmSource): Promise<WasmCore> {
    const prepared =
      source instanceof WebAssembly.Module
        ? { module: source, bytes: 0 }
        : await toModule(source);
    let instance: WebAssembly.Instance;
    try {
      instance = await WebAssembly.instantiate(prepared.module, {});
    } catch (error) {
      throw new QwenscriberError(SDK_STATUS.protocol, "instantiate", {
        message:
          "instantiation failed; this build resolves no imports, so a module that needs any was " +
          "compiled for a different target",
        cause: error,
      });
    }
    const bound = validateExports(instance.exports);
    WasmCore.assertAbiVersion(bound.core.qw_abi_version());
    const handle = bound.core.qw_create();
    if (handle === 0) {
      throw new QwenscriberError(SDK_STATUS.protocol, "qw_create", {
        message: "the module already holds a runtime instance; ABI v1 allows exactly one",
      });
    }
    return new WasmCore(bound, handle, prepared.bytes);
  }

  /**
   * Refuses a module whose ABI major version differs.
   *
   * Minor versions are additive by contract, so a newer minor is accepted; a different major means
   * struct layouts and semantics may differ, which no amount of defensive coding can paper over.
   */
  static assertAbiVersion(version: number): void {
    if ((version >>> 16) !== ABI_MAJOR) {
      throw new AbiMismatchError(version, ABI_VERSION_EXPECTED);
    }
  }

  get handle(): number {
    return this.handleValue;
  }

  get disposed(): boolean {
    return this.handleValue === 0;
  }

  abiVersion(): number {
    return this.exports.qw_abi_version();
  }

  coreVersion(): number {
    return this.exports.qw_core_version();
  }

  /** Current linear memory size in bytes. */
  memoryBytes(): number {
    // The export returns `u32`; a wasm result arrives as a signed `i32`, so a
    // memory past 2 GiB would read back negative. `>>> 0` restores the value.
    return this.exports.qw_memory_bytes() >>> 0;
  }

  /** Compiled module size in bytes, or 0 when the caller passed an already-compiled module. */
  moduleBytes(): number {
    return this.moduleBytesValue;
  }

  /** `exports.memory.buffer`, re-read on every call: allocation detaches earlier buffers. */
  memoryBuffer(): ArrayBuffer {
    return this.exports.memory.buffer as ArrayBuffer;
  }

  /** The module's own text for a status code, read through `qw_error_message_ptr`. */
  errorMessage(code: number): string {
    const cached = this.errorMessages.get(code);
    if (cached !== undefined) return cached;
    const pointer = this.exports.qw_error_message_ptr(code) >>> 0;
    const length = this.exports.qw_error_message_len(code) >>> 0;
    const text =
      length === 0
        ? ""
        : TEXT_DECODER.decode(new Uint8Array(this.memoryBuffer(), pointer, length));
    if (this.errorMessages.size < ERROR_MESSAGE_CACHE_MAX) this.errorMessages.set(code, text);
    return text;
  }

  /** Allocates inside linear memory. The caller must `dispose()` the result. */
  alloc(size: number, alignment: number = 8): Allocation {
    if (!Number.isInteger(size) || size <= 0) {
      throw new QwenscriberError(STATUS.invalid_argument, "qw_alloc", {
        message: `qw_alloc needs a positive byte size, got ${size}`,
        context: { size, alignment },
      });
    }
    if (!isSupportedAlignment(alignment)) throw unsupportedAlignment(alignment);
    // `qw_alloc` returns a `u32` offset, and a wasm result arrives in JavaScript as a signed i32:
    // an allocation above 2 GiB -- which is exactly where a model's weights and cache land --
    // would otherwise read back negative and be rejected as a failed call.
    const pointer = this.exports.qw_alloc(size, alignment) >>> 0;
    if (pointer === 0) {
      throw new QwenscriberError(STATUS.out_of_memory, "qw_alloc", {
        message: `qw_alloc could not satisfy ${size} bytes aligned to ${alignment}`,
        context: { size, alignment },
      });
    }
    return new Allocation(this, pointer, size, alignment);
  }

  /** Called by `Allocation.dispose`; the ABI requires the original size and alignment. */
  release(allocation: Allocation): void {
    this.exports.qw_free(allocation.pointer, allocation.size, allocation.alignment);
  }

  melFramesForSamples(sampleCount: number): number {
    if (!Number.isInteger(sampleCount) || sampleCount <= 0) {
      throw new QwenscriberError(STATUS.invalid_argument, "qw_mel_frames_for_samples", {
        message: `sample_count must be a positive integer, got ${sampleCount}`,
        context: { sample_count: sampleCount },
      });
    }
    return this.exports.qw_mel_frames_for_samples(sampleCount);
  }

  /** Mono samples below which the reference pipeline zero-pads a clip (8000 at 16 kHz). */
  melMinSamples(): number {
    return this.exports.qw_mel_min_samples();
  }

  /** Mel value for frames inside a partially filled encoder chunk. */
  melPaddingValue(globalMaxLog: number): number {
    return this.exports.qw_mel_padding_value(globalMaxLog);
  }

  /**
   * Whisper log-mel features for one clip, computed by the freestanding core.
   *
   * `samples` are mono f32 in `[-1, 1]` at 16 kHz. Input, output, and result buffers are allocated
   * here and released in a `finally`, so a failed call leaks nothing; the returned features are a
   * copy, because the buffer they came from is gone by the time the caller sees them.
   */
  melCompute(samples: Float32Array, handle: number = this.handleValue): MelComputation {
    if (samples.length === 0) {
      // The ABI reports zero samples as invalid_argument, but the output allocation below would fail
      // first, so this has to be checked here to keep the reported status the honest one.
      throw new QwenscriberError(STATUS.invalid_argument, "qw_mel_compute", {
        message: "mel_compute needs at least one sample",
        context: { sample_count: 0 },
      });
    }
    const frames = this.exports.qw_mel_frames_for_samples(samples.length);
    if (frames > MEL_CAPACITY_FRAMES) {
      throw new QwenscriberError(STATUS.audio_too_long, "qw_mel_compute", {
        message:
          `mel_compute accepts at most ${MEL_CAPACITY_FRAMES} frames (30 s at 16 kHz); ` +
          `${samples.length} samples is ${frames} frames`,
        context: { sample_count: samples.length, frames, frames_max: MEL_CAPACITY_FRAMES },
      });
    }
    const input = this.alloc(samples.length * Float32Array.BYTES_PER_ELEMENT, 4);
    const output = this.alloc(frames * MEL_BINS * Float32Array.BYTES_PER_ELEMENT, 4);
    const result = this.alloc(MEL_RESULT_BYTES, 4);
    try {
      input.copyFrom(samples);
      const status = this.exports.qw_mel_compute(
        handle,
        input.pointer,
        samples.length,
        output.pointer,
        output.size,
        result.pointer,
      );
      this.check(status, "qw_mel_compute", { sample_count: samples.length, frames });
      const written = readMelResult(result.dataView(), 0);
      return {
        frames: written.frames_written,
        bytesWritten: written.bytes_written,
        globalMaxLog: written.global_max_log,
        paddingValue: this.melPaddingValue(written.global_max_log),
        features: output.f32(written.frames_written * MEL_BINS).slice(),
      };
    } finally {
      result.dispose();
      output.dispose();
      input.dispose();
    }
  }

  /**
   * Points the runtime at a vocabulary.
   *
   * The ABI keeps the caller's pointer instead of copying, so the three buffers written here are
   * retained until this method runs again or the core is disposed.
   */
  setVocabulary(descriptor: VocabularyDescriptor): void {
    const tokens = descriptor.tokens.map((token) =>
      typeof token === "string" ? TEXT_ENCODER.encode(token) : token,
    );
    if (tokens.length === 0) {
      throw new QwenscriberError(STATUS.invalid_argument, "qw_tokenizer_set", {
        message: "a vocabulary needs at least one token",
        context: { token_count: 0 },
      });
    }
    if (tokens.length > VOCABULARY_TOKENS_MAX) {
      throw new QwenscriberError(STATUS.limit_exceeded, "qw_tokenizer_set", {
        message: `a vocabulary holds at most ${VOCABULARY_TOKENS_MAX} tokens`,
        context: { token_count: tokens.length, token_count_max: VOCABULARY_TOKENS_MAX },
      });
    }
    let bytesLength = 0;
    for (const token of tokens) bytesLength += token.length;
    if (bytesLength === 0) {
      throw new QwenscriberError(STATUS.invalid_argument, "qw_tokenizer_set", {
        message: "a vocabulary needs at least one byte",
        context: { token_count: tokens.length, bytes_len: 0 },
      });
    }
    const bytes = this.alloc(bytesLength, 1);
    const offsets = this.alloc((tokens.length + 1) * Uint32Array.BYTES_PER_ELEMENT, 4);
    const descriptor_buffer = this.alloc(TOKENIZER_DESCRIPTOR_BYTES, 4);
    const fresh: readonly Allocation[] = [bytes, offsets, descriptor_buffer];
    try {
      const byte_view = bytes.bytes;
      const offset_view = offsets.u32(tokens.length + 1);
      let cursor = 0;
      for (const [index, token] of tokens.entries()) {
        offset_view[index] = cursor;
        byte_view.set(token, cursor);
        cursor += token.length;
      }
      offset_view[tokens.length] = cursor;
      const layout: TokenizerDescriptor = {
        offsets_ptr: offsets.pointer,
        offsets_len: offsets.size,
        bytes_ptr: bytes.pointer,
        bytes_len: bytes.size,
        token_count: tokens.length,
        reserved: 0,
      };
      writeTokenizerDescriptor(descriptor_buffer.dataView(), 0, layout);
      const status = this.exports.qw_tokenizer_set(this.handleValue, descriptor_buffer.pointer);
      this.check(status, "qw_tokenizer_set", { token_count: tokens.length, bytes_len: bytesLength });
    } catch (error) {
      for (const allocation of fresh) allocation.dispose();
      throw error;
    }
    // Accepted: the runtime now reads this memory, so only now may the previous vocabulary go.
    this.releaseVocabulary();
    this.vocabulary = {
      allocations: fresh,
      idBytes: Uint32Array.from(tokens, (token) => token.length),
    };
  }

  /** Decodes token ids into UTF-8, sized from the vocabulary's own token lengths. */
  detokenize(ids: Uint32Array): string {
    if (ids.length === 0) return "";
    if (ids.length > DETOKENIZE_IDS_MAX) {
      throw new QwenscriberError(STATUS.limit_exceeded, "qw_detokenize", {
        message: `detokenize accepts at most ${DETOKENIZE_IDS_MAX} ids`,
        context: { id_count: ids.length, id_count_max: DETOKENIZE_IDS_MAX },
      });
    }
    const input = this.alloc(ids.length * Uint32Array.BYTES_PER_ELEMENT, 4);
    const written = this.alloc(Uint32Array.BYTES_PER_ELEMENT, 4);
    const needed = this.detokenizedByteLength(ids);
    const output = this.alloc(needed, 1);
    try {
      input.copyFrom(ids);
      const status = this.exports.qw_detokenize(
        this.handleValue,
        input.pointer,
        ids.length,
        output.pointer,
        output.size,
        written.pointer,
      );
      this.check(status, "qw_detokenize", { id_count: ids.length, out_bytes: needed });
      const written_bytes = written.dataView().getUint32(0, true);
      return TEXT_DECODER.decode(output.bytes.subarray(0, written_bytes));
    } finally {
      output.dispose();
      written.dispose();
      input.dispose();
    }
  }

  /** Runs the core's self-test. Failures are reported in the result, not as a thrown error. */
  selfTest(): SelfTestReport {
    const result = this.alloc(SELF_TEST_RESULT_BYTES, 8);
    try {
      const status = this.exports.qw_selftest(result.pointer);
      this.check(status, "qw_selftest");
      const raw = readSelfTestResult(result.dataView(), 0);
      return {
        failures: raw.failures,
        loudestBand: raw.loudest_band,
        silenceValue: raw.silence_value,
        quantHash: raw.quant_hash,
        melHash: raw.mel_hash,
        failedChecks: failedSelfTestChecks(raw.failures),
      };
    } finally {
      result.dispose();
    }
  }

  /**
   * The capability bits this module reports (`qw_features`).
   *
   * Zero means the module was built before that export existed, which is how an older ABI v1 build
   * is told apart from a broken one. Use `FEATURE`/`FEATURE_NAMES` from `abi.ts` to read it.
   */
  features(): number {
    return this.exports.qw_features?.() ?? 0;
  }

  /**
   * Allocates the model's own arena.
   *
   * Legal when no model is being built and when a previous attempt is still pending: shards can be
   * added but never removed, so starting over is the only way to retry a load that failed.
   */
  modelBegin(): void {
    const model = this.modelExports("qw_model_begin");
    this.check(model.qw_model_begin(this.handleValue), "qw_model_begin");
  }

  /**
   * Hands the runtime one `.qw` shard.
   *
   * The runtime parses the container in place and keeps the view: the weight tensors point into
   * this buffer, so it stays resident and unmodified until the model is released. Allocate it with
   * `alloc(size, SHARD_ALIGNMENT)`, which is what `decode.ts` does.
   */
  modelAddShard(shard: Allocation): void {
    const model = this.modelExports("qw_model_add_shard");
    if (shard.disposed) {
      throw new QwenscriberError(STATUS.invalid_argument, "qw_model_add_shard", {
        message: "the shard buffer was released; the runtime keeps the bytes, not a copy",
        context: { pointer: shard.pointer, size: shard.size },
      });
    }
    if (shard.pointer % SHARD_ALIGNMENT !== 0) {
      throw new QwenscriberError(STATUS.invalid_argument, "qw_model_add_shard", {
        message: `a shard buffer must be ${SHARD_ALIGNMENT}-byte aligned`,
        context: { pointer: shard.pointer, alignment: SHARD_ALIGNMENT },
      });
    }
    this.check(
      model.qw_model_add_shard(this.handleValue, shard.pointer, shard.size),
      "qw_model_add_shard",
      { shard_bytes: shard.size },
    );
  }

  /**
   * Parses `config.bin`, resolves every tensor, and prepares the decoder.
   *
   * The configuration is copied into the runtime, so the buffer may be released as soon as this
   * returns. A failure releases whatever the attempt had allocated and leaves the handle ready to
   * start over with `modelBegin`.
   */
  modelFinish(config: Uint8Array): void {
    const model = this.modelExports("qw_model_finish");
    if (config.byteLength !== MODEL_CONFIG_BYTES) {
      throw new QwenscriberError(STATUS.truncated, "qw_model_finish", {
        message: `config.bin is exactly ${MODEL_CONFIG_BYTES} bytes, got ${config.byteLength}`,
        context: { config_bytes: config.byteLength },
      });
    }
    const buffer = this.alloc(MODEL_CONFIG_BYTES, 4);
    try {
      buffer.copyFrom(config);
      this.check(
        model.qw_model_finish(this.handleValue, buffer.pointer, buffer.size),
        "qw_model_finish",
        { config_bytes: buffer.size },
      );
    } finally {
      buffer.dispose();
    }
  }

  /**
   * What the loaded model keeps resident, and the limits it decodes within.
   *
   * This is the measurement a caller plans against: `total_bytes` is the weight bytes it must keep
   * resident plus the runtime's own cache and scratch.
   */
  modelRequirements(): ModelRequirements {
    const model = this.modelExports("qw_model_requirements");
    const result = this.alloc(MODEL_REQUIREMENTS_BYTES, 8);
    try {
      this.check(
        model.qw_model_requirements(this.handleValue, result.pointer),
        "qw_model_requirements",
      );
      return readModelRequirements(result.dataView(), 0);
    } finally {
      result.dispose();
    }
  }

  /**
   * Decodes one utterance: encode the features, prefill the prompt, and generate up to `maxTokens`
   * ids with `qw_decode_begin`, `qw_decode_step`, and `qw_decode_tokens`.
   *
   * `features` is the `[mel_bin][frame]` block `melCompute` returns; it is read during this call and
   * never retained. The key/value cache is per utterance, so the utterance is ended here -- that is
   * what keeps the next clip independent of this one -- and the model stays resident for it.
   *
   * Returns the produced token ids, ready for `detokenize`.
   */
  decodeUtterance(features: Float32Array, maxTokens: number): Uint32Array {
    const model = this.modelExports("qw_decode_begin");
    if (features.length === 0) {
      throw new QwenscriberError(STATUS.invalid_argument, "qw_decode_begin", {
        message: "decoding needs at least one log-mel frame",
        context: { frames: 0 },
      });
    }
    if (!Number.isInteger(maxTokens) || maxTokens <= 0) {
      throw new QwenscriberError(STATUS.invalid_argument, "qw_decode_begin", {
        message: `max_tokens must be a positive integer, got ${maxTokens}`,
        context: { max_tokens: maxTokens },
      });
    }
    const input = this.alloc(features.length * Float32Array.BYTES_PER_ELEMENT, 4);
    const token = this.alloc(Uint32Array.BYTES_PER_ELEMENT, 4);
    const ids = this.alloc(maxTokens * Uint32Array.BYTES_PER_ELEMENT, 4);
    try {
      input.copyFrom(features);
      this.check(
        model.qw_decode_begin(this.handleValue, input.pointer, input.size, maxTokens),
        "qw_decode_begin",
        { frames: features.length / MEL_BINS, max_tokens: maxTokens },
      );
      try {
        return this.generateTokens(model, token, ids, maxTokens);
      } finally {
        this.check(model.qw_decode_end(this.handleValue), "qw_decode_end");
      }
    } finally {
      ids.dispose();
      token.dispose();
      input.dispose();
    }
  }

  /**
   * Points the runtime at a vocabulary in the ABI's own table format: `count + 1` little-endian
   * `u32` offsets followed by the concatenated token bytes, which is what the converter writes to
   * `tokens.bin`. `count` comes from the directory's manifest.
   *
   * The table is copied into linear memory once and kept for the runtime's lifetime. No token is
   * ever materialized as a JavaScript string, so a 150k-entry vocabulary costs one copy rather than
   * a string per token.
   *
   * A table may also carry the count itself as a leading word. That is indistinguishable from an
   * offset by size, so the layout is read from the first word: it is either zero, which is what a
   * table without a prefix starts with, or exactly `count`, which is the prefix. Any other value is
   * refused, because a table shifted by one word would decode every token into the middle of its
   * neighbour rather than fail.
   */
  setVocabularyFromTable(table: Uint8Array, tokenCount: number): void {
    if (!Number.isInteger(tokenCount) || tokenCount <= 0 || tokenCount > VOCABULARY_TOKENS_MAX) {
      throw new QwenscriberError(STATUS.invalid_argument, "qw_tokenizer_set", {
        message: `a vocabulary holds 1 to ${VOCABULARY_TOKENS_MAX} tokens, got ${tokenCount}`,
        context: { token_count: tokenCount },
      });
    }
    const offsetsBytes = (tokenCount + 1) * Uint32Array.BYTES_PER_ELEMENT;
    const header_bytes = table.byteLength >= Uint32Array.BYTES_PER_ELEMENT &&
        new DataView(table.buffer, table.byteOffset, table.byteLength).getUint32(0, true) ===
          tokenCount
      ? Uint32Array.BYTES_PER_ELEMENT
      : 0;
    if (table.byteLength <= header_bytes + offsetsBytes) {
      throw new QwenscriberError(STATUS.truncated, "qw_tokenizer_set", {
        message:
          `a ${tokenCount}-token table takes more than ${offsetsBytes} bytes of offsets, and the ` +
          `table is ${table.byteLength} bytes`,
        context: { token_count: tokenCount, table_bytes: table.byteLength },
      });
    }
    // One allocation holds both parts: the offsets array is 4-byte aligned at its own offset, and
    // the token bytes need no alignment at all.
    const bytes = this.alloc(table.byteLength, 4);
    const descriptor_buffer = this.alloc(TOKENIZER_DESCRIPTOR_BYTES, 4);
    const fresh: readonly Allocation[] = [bytes, descriptor_buffer];
    try {
      bytes.copyFrom(table);
      // The offsets are checked before the runtime sees the table: a shifted table would otherwise
      // be reported as a malformed vocabulary without saying which word is wrong.
      const offsets = bytes.u32(tokenCount + 1, header_bytes);
      const first_offset = offsets[0] ?? 0;
      if (first_offset !== 0) unsupportedTable(header_bytes, tokenCount, first_offset);
      const layout: TokenizerDescriptor = {
        offsets_ptr: bytes.pointer + header_bytes,
        offsets_len: offsetsBytes,
        bytes_ptr: bytes.pointer + header_bytes + offsetsBytes,
        bytes_len: bytes.size - header_bytes - offsetsBytes,
        token_count: tokenCount,
        reserved: 0,
      };
      writeTokenizerDescriptor(descriptor_buffer.dataView(), 0, layout);
      this.check(
        this.exports.qw_tokenizer_set(this.handleValue, descriptor_buffer.pointer),
        "qw_tokenizer_set",
        { token_count: tokenCount, bytes_len: layout.bytes_len },
      );
      // Byte length per id, so `detokenize` can size its output without asking the module twice.
      const id_bytes = new Uint32Array(tokenCount);
      for (let id = 0; id < tokenCount; id += 1) {
        id_bytes[id] = (offsets[id + 1] ?? 0) - (offsets[id] ?? 0);
      }
      // Accepted: the runtime now reads this memory, so only now may the previous vocabulary go.
      this.releaseVocabulary();
      this.vocabulary = { allocations: fresh, idBytes: id_bytes };
    } catch (error) {
      for (const allocation of fresh) allocation.dispose();
      throw error;
    }
  }

  /**
   * The model family, or a typed `not_implemented` naming the stage this build cannot run.
   *
   * The family is negotiated rather than assumed: a module built before it existed still loads and
   * still preprocesses, and asking for a stage it does not have says exactly that.
   */
  private modelExports(operation: string): ModelExports {
    const family = this.modelFamilyValue;
    if (family !== undefined) return family;
    const features = this.features();
    const missing = (features & FEATURE.model) === 0 ? "model_load" : "decode";
    throw new NotImplementedError(missing, operation, {
      context: { features },
    });
  }

  /** The greedy loop: step until the sequence ends, then copy the ids out. */
  private generateTokens(
    model: ModelExports,
    token: Allocation,
    ids: Allocation,
    maxTokens: number,
  ): Uint32Array {
    // Bounded by `maxTokens`: every iteration either produces a token or ends the sequence.
    for (let step = 0; step < maxTokens; step += 1) {
      const status = model.qw_decode_step(this.handleValue, token.pointer);
      if (status === 0) break;
      if (status !== 1) this.check(status, "qw_decode_step", { step, max_tokens: maxTokens });
    }
    const count = model.qw_decode_tokens(this.handleValue, ids.pointer, maxTokens);
    if (count < 0) this.check(count, "qw_decode_tokens", { capacity_tokens: maxTokens });
    return ids.u32(count).slice();
  }

  /** Releases the vocabulary buffers the runtime is still pointing at. */
  private releaseVocabulary(): void {
    if (this.vocabulary === undefined) return;
    for (const allocation of this.vocabulary.allocations) allocation.dispose();
    this.vocabulary = undefined;
  }

  /** Worst-case output bytes for these ids: the sum of the tokens' own byte lengths. */
  private detokenizedByteLength(ids: Uint32Array): number {
    if (this.vocabulary === undefined) {
      throw new QwenscriberError(STATUS.invalid_state, "qw_detokenize", {
        message: "detokenize needs a vocabulary; call setVocabulary first",
      });
    }
    let total = 0;
    for (const id of ids) total += this.vocabulary.idBytes[id] ?? 0;
    return Math.max(total, 1);
  }

  /** Destroys the runtime handle and releases the vocabulary. Idempotent. */
  dispose(): void {
    if (this.handleValue === 0) return;
    const status = this.exports.qw_destroy(this.handleValue);
    this.check(status, "qw_destroy", { handle: this.handleValue });
    this.handleValue = 0;
    this.releaseVocabulary();
  }

  private check(
    status: number,
    operation: string,
    context?: Readonly<Record<string, unknown>>,
  ): void {
    throwForStatus(status, operation, {
      context,
      readMessage: (code) => this.errorMessage(code),
    });
  }
}

function isSupportedAlignment(value: number): value is Alignment {
  return (SUPPORTED_ALIGNMENTS as readonly number[]).includes(value);
}
