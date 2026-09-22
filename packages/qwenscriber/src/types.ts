//! Shared types for the SDK's public surface and the message channel between main thread and
//! worker.
//!
//! The ABI's own layouts live in `wasm/abi.ts` and keep the ABI's spelling. Everything here is SDK
//! vocabulary: camelCase, `Hz`/`Ms`/`Bytes` suffixes, and no reference to linear memory.

import type { SelfTestReport, WasmSource } from "./wasm/runtime.ts";

/** Which accelerator the encoder/decoder stages should use. */
export type Backend = "auto" | "webgpu" | "wasm";

/** A `Backend` with `"auto"` resolved against the capability probe. */
export type ResolvedBackend = "webgpu" | "wasm";

/** Storage format of the model's large matrices, mirroring `core/dtype.zig`. */
export type Quantization = "f32" | "f16" | "bf16" | "q4" | "q5" | "q8";

export interface QwenscriberOptions {
  /** Model id or path. Reserved: ABI v1 has no model-loading stage yet. */
  readonly model?: string | undefined;
  /** Accelerator for the stages that need one. Defaults to `"auto"`. */
  readonly backend?: Backend | undefined;
  /** Weight storage format the future loader will read. Defaults to `"q8"`. */
  readonly quantization?: Quantization | undefined;
  /** Run the core in a Web Worker. Defaults to true wherever `Worker` exists. */
  readonly worker?: boolean | undefined;
  /**
   * Where the WASM core comes from. Defaults to `./qwenscriber_core.wasm` next to this module,
   * which the package build puts in `dist/`. A compiled `WebAssembly.Module` can only be used with
   * `worker: false`: modules cannot cross the worker boundary portably.
   */
  readonly wasm?: WasmSource | undefined;
  /** Where the worker script comes from. Defaults to `./worker/inference.worker.js` next to this module. */
  readonly workerUrl?: string | URL | undefined;
  /** Progress for stages that report it. */
  readonly onProgress?: ((progress: PreprocessProgress) => void) | undefined;
}

export interface PreprocessProgress {
  readonly stage: "resample" | "mel";
  /** Items finished in this stage (input samples for `resample`, steps for `mel`). */
  readonly completed: number;
  readonly total: number;
}

/** What the mel stage produces, before the SDK attaches the audio's own metadata. */
export interface MelPayload {
  readonly frames: number;
  readonly features: Float32Array;
  readonly globalMaxLog: number;
  readonly paddingValue: number;
  readonly bytesWritten: number;
  readonly inputSampleRateHz: number;
}

/** The result of the whole front end: container/PCM decode, resample to 16 kHz, log-mel. */
export interface PreprocessedAudio extends MelPayload {
  /** Duration of the audio that produced `frames`, after resampling. */
  readonly audioDurationMs: number;
  readonly channelCount: number;
  readonly sourceKind: AudioSourceKind;
}

/** How the caller handed the audio over; recorded so a report can say what was actually read. */
export type AudioSourceKind =
  | "float32"
  | "int16"
  | "int32"
  | "pcm-bytes"
  | "wave"
  | "blob"
  | "audio-buffer-like";

/** A version report for the loaded core, so a page can print what it is running. */
export interface WasmVersions {
  readonly abiVersion: number;
  readonly coreVersion: number;
  /** Size of the compiled module in bytes. */
  readonly wasmBytes: number;
  /** Linear memory size after `qw_create`. */
  readonly runtimeBytes: number;
}

export interface QwenscriberVersions extends WasmVersions {
  readonly sdkVersion: string;
}

/**
 * Where inference runs.
 *
 * The SDK has two implementations of this: one that owns a `WasmCore` on the current thread and one
 * that owns it in a worker. Everything else in the SDK is written against this interface, so the
 * choice never leaks into call sites.
 */
export interface RuntimeChannel {
  readonly versions: WasmVersions;
  /** True when the core lives in a Worker. */
  readonly workerBacked: boolean;
  /** Where stage progress goes. Assigning replaces the previous listener; `undefined` is silent. */
  progress: ((progress: PreprocessProgress) => void) | undefined;
  preprocess(pcm: Float32Array, sampleRateHz: number): Promise<MelPayload>;
  /**
   * Runs the front end and then reports that the decode stage does not exist.
   *
   * The signature is `Promise<never>` rather than `Promise<Transcription>` because ABI v1 has no
   * encoder or decoder: a channel that returned text would have to invent it.
   */
  transcribe(pcm: Float32Array, sampleRateHz: number): Promise<never>;
  selfTest(): Promise<SelfTestReport>;
  setVocabulary(tokens: readonly string[]): Promise<void>;
  detokenize(ids: Uint32Array): Promise<string>;
  dispose(): Promise<void>;
}
