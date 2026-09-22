//! Loading a converted model directory into the core, and decoding with it.
//!
//! This is the browser path end to end: fetch a model directory, move each file into the core's
//! linear memory once, and drive the ABI's model and decode family over it.
//!
//! Two properties are worth stating up front, because they shape every function here:
//!
//!   * **Weights are never in the JavaScript heap.** A shard is fetched, copied once into a
//!     16-byte-aligned linear-memory buffer, and handed to the runtime, which parses the container
//!     in place and points the model's tensors straight into it. That is why the buffers are kept
//!     alive in `WasmModel` and released only when the model is released.
//!   * **Nothing is decoded that the runtime can read itself.** `tokens.bin` goes in as the ABI's
//!     own table format, so a 150k-token vocabulary costs one copy instead of 150k JavaScript
//!     strings.
//!
//! The model directory's own manifest is the contract between the converter and this loader: it
//! names the configuration, the vocabulary, and every shard with its byte count, so a directory
//! that was regenerated with different options does not present itself as the one that was
//! requested.

import {
  MODEL_CONFIG_BYTES,
  SHARD_ALIGNMENT,
  STATUS,
  type ModelRequirements,
} from "./wasm/abi.ts";
import { QwenscriberError, SDK_STATUS } from "./errors.ts";
import type { Allocation, WasmCore } from "./wasm/runtime.ts";

/** Manifest every converted model directory carries. */
export const MODEL_MANIFEST_FILE = "manifest.json";

/** Format version this loader reads. */
export const MODEL_MANIFEST_FORMAT_VERSION = 1;

/** Shards one directory may hold, mirroring `model.zig`'s `shard_count_max`. */
export const MODEL_SHARDS_MAX = 64;

/** Per-utterance generation budget when the caller does not choose one. */
export const DECODE_TOKENS_DEFAULT = 256;

/** Where a model directory's files come from: one name in, the file's bytes out. */
export interface ModelDirectory {
  /** Absolute URL, path, or provider name, for error messages and logs. */
  readonly source: string;
  read(name: string): Promise<Uint8Array>;
}

export interface FetchModelDirectoryOptions {
  /** Overrides `globalThis.fetch`, for a caller with its own transport. */
  readonly fetch?: typeof globalThis.fetch;
  /** Aborts every read, including one in flight. */
  readonly signal?: AbortSignal;
}

/**
 * Reads a directory over HTTP, which is how a browser loads a model.
 *
 * `baseUrl` may be absolute (`https://cdn/qwen3-asr-0.6b-q4/`) or relative (`/models/qwen3-asr-0.6b-q4/`),
 * and either way each file is fetched with its own request, so a large shard streams into memory
 * without a second copy for a manifest or a bundle.
 */
export function fetchModelDirectory(
  baseUrl: string | URL,
  options: FetchModelDirectoryOptions = {},
): ModelDirectory {
  const base = typeof baseUrl === "string" ? baseUrl : baseUrl.href;
  const doFetch = options.fetch ?? globalThis.fetch;
  return {
    source: base,
    async read(name: string): Promise<Uint8Array> {
      const url = base.endsWith("/") ? `${base}${name}` : `${base}/${name}`;
      let response: Response;
      try {
        response = await doFetch(url, { signal: options.signal ?? null });
      } catch (error) {
        throw new QwenscriberError(SDK_STATUS.protocol, "loadModel", {
          message: `fetching ${url} failed`,
          context: { url, file: name },
          cause: error,
        });
      }
      if (!response.ok) {
        throw new QwenscriberError(SDK_STATUS.protocol, "loadModel", {
          message: `fetching ${url} returned ${response.status} ${response.statusText}`,
          context: { url, file: name, http_status: response.status },
        });
      }
      return new Uint8Array(await response.arrayBuffer());
    },
  };
}

export interface ManifestFile {
  readonly file: string;
  readonly bytes: number;
}

export interface ManifestShard {
  readonly name: string;
  readonly bytes: number;
}

/** The parts of `manifest.json` this loader reads. */
export interface ModelManifest {
  readonly format_version: number;
  readonly model_id: string;
  readonly quantization: string;
  readonly config: ManifestFile;
  readonly tokenizer: ManifestFile & { readonly count: number };
  readonly shards: readonly ManifestShard[];
}

function manifestError(message: string, context: Record<string, unknown>): QwenscriberError {
  return new QwenscriberError(SDK_STATUS.protocol, "loadModel", {
    message,
    context: { file: MODEL_MANIFEST_FILE, ...context },
  });
}

function manifestObject(value: unknown, what: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw manifestError(`the manifest's ${what} is not an object`, { [`${what}_type`]: typeof value });
  }
  return value as Record<string, unknown>;
}

/** A positive byte count, as the manifest records every size in bytes. */
function manifestBytes(value: unknown, what: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw manifestError(`the manifest's ${what} is not a positive byte count`, {
      [`${what}`]: String(value),
    });
  }
  return value;
}

function manifestName(value: unknown, what: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw manifestError(`the manifest's ${what} is not a file name`, {
      [`${what}`]: String(value),
    });
  }
  return value;
}

function manifestFile(value: unknown, what: string): ManifestFile {
  const file = manifestObject(value, what);
  return { file: manifestName(file["file"], `${what}.file`), bytes: manifestBytes(file["bytes"], `${what}.bytes`) };
}

/**
 * Validates a `manifest.json`, so a directory that does not match the runtime's own limits is
 * refused before any of it is fetched.
 */
export function parseManifest(text: string): ModelManifest {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw manifestError("the manifest is not valid JSON", { cause: String(error) });
  }
  const manifest = manifestObject(parsed, "document");
  if (manifest["format_version"] !== MODEL_MANIFEST_FORMAT_VERSION) {
    throw manifestError(
      `this loader reads manifest format ${MODEL_MANIFEST_FORMAT_VERSION}`,
      { format_version: String(manifest["format_version"]) },
    );
  }
  const config = manifestFile(manifest["config"], "config");
  if (config.bytes !== MODEL_CONFIG_BYTES) {
    throw manifestError(`config.bin is exactly ${MODEL_CONFIG_BYTES} bytes`, {
      config_bytes: config.bytes,
    });
  }
  const tokenizer_file = manifestObject(manifest["tokenizer"], "tokenizer");
  const tokenizer = manifestFile(tokenizer_file, "tokenizer");
  const token_count = tokenizer_file["count"];
  if (typeof token_count !== "number" || !Number.isInteger(token_count) || token_count <= 0) {
    throw manifestError("the manifest's tokenizer.count is not a positive token count", {
      count: String(token_count),
    });
  }
  const shards_value = manifest["shards"];
  if (!Array.isArray(shards_value) || shards_value.length === 0) {
    throw manifestError("the manifest lists no shards", {
      shards: Array.isArray(shards_value) ? shards_value.length : 0,
    });
  }
  if (shards_value.length > MODEL_SHARDS_MAX) {
    throw manifestError(`a model holds at most ${MODEL_SHARDS_MAX} shards`, {
      shards: shards_value.length,
    });
  }
  const shards = shards_value.map((entry, index) => {
    const shard = manifestObject(entry, `shards[${index}]`);
    return {
      name: manifestName(shard["name"], `shards[${index}].name`),
      bytes: manifestBytes(shard["bytes"], `shards[${index}].bytes`),
    };
  });
  return {
    format_version: MODEL_MANIFEST_FORMAT_VERSION,
    model_id: typeof manifest["model_id"] === "string" ? manifest["model_id"] : "",
    quantization: typeof manifest["quantization"] === "string" ? manifest["quantization"] : "",
    config,
    tokenizer: { ...tokenizer, count: token_count },
    shards,
  };
}

export interface ModelLoadProgress {
  /** Which part of the directory the loader is on: a file name, or `"load"` for the model build. */
  readonly stage: string;
  readonly files_completed: number;
  readonly files_total: number;
  readonly bytes_loaded: number;
  readonly bytes_total: number;
}

export interface LoadModelOptions {
  /** Receives one report per file, plus one for the model build itself. */
  readonly onProgress?: ((progress: ModelLoadProgress) => void) | undefined;
}

/** What a loaded model is, in a form that survives a worker boundary. */
export interface WasmModelInfo {
  readonly modelId: string;
  readonly quantization: string;
  /** Directory the model was read from. */
  readonly source: string;
  readonly requirements: ModelRequirements;
}

/** One utterance's output: the ids the model produced, and what they detokenize to. */
export interface Transcript {
  readonly tokens: Uint32Array;
  readonly text: string;
}

/**
 * The model family as a channel: the same four calls whether the core is in a worker or on this
 * thread.
 *
 * `RuntimeChannel` in `types.ts` covers the front end, which every channel has. A channel that also
 * implements this can load a model and decode with it, and the caller does not have to know where
 * the linear memory holding the weights lives.
 */
export interface ModelChannel {
  /**
   * Loads a converted model directory. A model already loaded is released first, so switching
   * models is one call.
   */
  loadModel(modelUrl: string | URL, options?: LoadModelOptions): Promise<WasmModelInfo>;
  /** Decodes one utterance from log-mel features. */
  decode(features: Float32Array, maxTokens?: number): Promise<Transcript>;
  /** Releases the model, leaving the front end and the tokenizer in place. */
  unloadModel(): Promise<void>;
}

/**
 * The generation budget for one utterance.
 *
 * The default is the smaller of the SDK's own cap and the model's configured budget, so a clip
 * generates a usable transcript without the caller having to read the configuration, and a model
 * configured for fewer tokens is never asked for more than it has.
 */
function decodeBudget(requested: number | undefined, requirements: ModelRequirements): number {
  const ceiling = Math.min(DECODE_TOKENS_DEFAULT, requirements.max_decode_tokens);
  if (requested === undefined) return ceiling;
  if (!Number.isInteger(requested) || requested <= 0 || requested > requirements.max_decode_tokens) {
    throw new QwenscriberError(STATUS.invalid_argument, "qw_decode_begin", {
      message:
        `max_tokens must be between 1 and this model's ${requirements.max_decode_tokens}, ` +
        `got ${requested}`,
      context: { max_tokens: requested, max_decode_tokens: requirements.max_decode_tokens },
    });
  }
  return requested;
}

/** A model resident in the core, with the linear-memory buffers it borrows from this side. */
export class WasmModel {
  private readonly core: WasmCore;
  /** The shard buffers the runtime points at. Released only when the model is. */
  private readonly buffers: readonly Allocation[];
  private released = false;

  readonly manifest: ModelManifest;
  readonly requirements: ModelRequirements;

  constructor(
    core: WasmCore,
    buffers: readonly Allocation[],
    manifest: ModelManifest,
    requirements: ModelRequirements,
  ) {
    this.core = core;
    this.buffers = buffers;
    this.manifest = manifest;
    this.requirements = requirements;
  }

  /** Model identifier from the manifest, for logs and for reporting what was transcribed. */
  get modelId(): string {
    return this.manifest.model_id;
  }

  /**
   * Decodes one clip's log-mel features into token ids and text.
   *
   * `features` is `[mel_bin][frame]` row major, exactly what `melCompute` (or `preprocess`)
   * returns. The returned text is what `qw_detokenize` produces from the ids the model generated:
   * the model's own `language X<asr_text>` framing is part of it, exactly as the native runner
   * prints it.
   */
  decode(
    features: Float32Array,
    options: { readonly maxTokens?: number | undefined } = {},
  ): Transcript {
    if (this.released) {
      throw new QwenscriberError(SDK_STATUS.protocol, "decode", {
        message: "this model was released; load it again",
        context: { model: this.modelId },
      });
    }
    const tokens = this.core.decodeUtterance(
      features,
      decodeBudget(options.maxTokens, this.requirements),
    );
    return { tokens, text: this.core.detokenize(tokens) };
  }

  /**
   * Releases the model: the runtime's arena, and the shard buffers it borrowed.
   *
   * The core stays usable -- its log-mel frontend, tokenizer, and self-test are unaffected -- but it
   * now holds no model. Idempotent.
   */
  dispose(): void {
    if (this.released) return;
    this.released = true;
    // Starting a model attempt releases whatever the handle holds, model included.
    this.core.modelBegin();
    for (const buffer of this.buffers) buffer.dispose();
  }
}

/**
 * Loads a converted model directory into `core` and returns it.
 *
 * The order is deliberate: the manifest is validated, then `config.bin`, then each shard, then the
 * model is built. A directory whose configuration or shard count is wrong fails before hundreds of
 * megabytes have moved, and a shard whose byte count disagrees with the manifest is refused before
 * the runtime parses it.
 *
 * On failure the shard buffers allocated here are released and the core is left ready to start
 * over: a load that failed does not leave a model's worth of memory behind.
 */
export async function loadModel(
  core: WasmCore,
  directory: ModelDirectory,
  options: LoadModelOptions = {},
): Promise<WasmModel> {
  const manifest = parseManifest(
    new TextDecoder("utf-8", { fatal: true }).decode(await directory.read(MODEL_MANIFEST_FILE)),
  );
  const files_total = manifest.shards.length + 2;
  const bytes_total =
    manifest.config.bytes +
    manifest.tokenizer.bytes +
    manifest.shards.reduce((total, shard) => total + shard.bytes, 0);
  let files_completed = 0;
  let bytes_loaded = 0;
  const report = (stage: string): void => {
    options.onProgress?.({
      stage,
      files_completed,
      files_total,
      bytes_loaded,
      bytes_total,
    });
  };

  // The configuration is 160 bytes and says whether the shards can be used at all.
  const config = await directory.read(manifest.config.file);
  if (config.byteLength !== manifest.config.bytes) {
    throw new QwenscriberError(STATUS.truncated, "loadModel", {
      message: `${manifest.config.file} is ${manifest.config.bytes} bytes in the manifest`,
      context: { file: manifest.config.file, read: config.byteLength },
    });
  }
  files_completed += 1;
  bytes_loaded += config.byteLength;
  report(manifest.config.file);

  core.modelBegin();
  const buffers: Allocation[] = [];
  try {
    for (const shard of manifest.shards) {
      const bytes = await directory.read(shard.name);
      if (bytes.byteLength !== shard.bytes) {
        throw new QwenscriberError(STATUS.truncated, "loadModel", {
          message: `${shard.name} is ${shard.bytes} bytes in the manifest`,
          context: { file: shard.name, read: bytes.byteLength },
        });
      }
      const buffer = core.alloc(bytes.byteLength, SHARD_ALIGNMENT);
      buffers.push(buffer);
      buffer.copyFrom(bytes);
      core.modelAddShard(buffer);
      files_completed += 1;
      bytes_loaded += bytes.byteLength;
      report(shard.name);
    }

    report("load");
    core.modelFinish(config);

    const table = await directory.read(manifest.tokenizer.file);
    if (table.byteLength !== manifest.tokenizer.bytes) {
      throw new QwenscriberError(STATUS.truncated, "loadModel", {
        message: `${manifest.tokenizer.file} is ${manifest.tokenizer.bytes} bytes in the manifest`,
        context: { file: manifest.tokenizer.file, read: table.byteLength },
      });
    }
    core.setVocabularyFromTable(table, manifest.tokenizer.count);
    files_completed += 1;
    bytes_loaded += table.byteLength;
    report(manifest.tokenizer.file);

    return new WasmModel(core, buffers, manifest, core.modelRequirements());
  } catch (error) {
    for (const buffer of buffers) buffer.dispose();
    throw error;
  }
}
