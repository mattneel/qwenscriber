//! The message protocol between the main thread and the inference worker.
//!
//! Shapes only: every request has a `kind` and a monotonically increasing `id`, and every response
//! carries the `id` of the request it answers, plus either the payload or a serialized error. That
//! pairing is what lets one worker serve concurrent calls without a second channel for failures.
//!
//! Buffers that are logically moved (audio in, mel features out) travel as transferables, so a
//! 51 KB mel frame set and a megabyte of PCM never get copied across the boundary. Control messages
//! (ids, vocabulary strings, progress) are structured-cloned, which is cheap for their size.
//!
//! Bigints are part of the protocol on purpose: the self-test reports FNV-1a hashes as u64, and
//! structured clone preserves `bigint` exactly, so no lossy string round trip is involved.

import {
  AbiMismatchError,
  NotImplementedError,
  QwenscriberError,
  SDK_STATUS,
  isQwenscriberError,
} from "../errors.ts";
import type { MelPayload, PreprocessProgress, WasmVersions } from "../types.ts";
import type { ModelLoadProgress, Transcript, WasmModelInfo } from "../decode.ts";
import type { SelfTestReport } from "../wasm/runtime.ts";

/**
 * Bumped whenever a message shape changes, or a request is added that a previously shipped worker
 * cannot serve.
 *
 * The main thread and the worker are compiled together, so a mismatch means a stale worker script
 * was fetched (a cached `inference.worker.js` next to a newer SDK). The worker refuses to run
 * against a version it does not know rather than misreading fields or answering a request kind it
 * has never heard of; version 2 added the model family (`loadModel`, `decode`, `unloadModel`) and
 * the `modelProgress` response.
 */
export const PROTOCOL_VERSION = 2;

/** Where the worker gets the core from: a URL it fetches, or bytes handed to it by transfer. */
export type WorkerWasmSource = { readonly url: string } | { readonly bytes: ArrayBuffer };

export interface InitRequest {
  readonly kind: "init";
  readonly id: number;
  readonly protocolVersion: number;
  readonly wasm: WorkerWasmSource;
}

export interface PreprocessRequest {
  readonly kind: "preprocess";
  readonly id: number;
  readonly pcm: ArrayBuffer;
  readonly sampleRateHz: number;
}

export interface TranscribeRequest {
  readonly kind: "transcribe";
  readonly id: number;
  readonly pcm: ArrayBuffer;
  readonly sampleRateHz: number;
}

/**
 * Loads a converted model directory, which the worker fetches itself.
 *
 * The worker owns a linear memory that can hold one model at a time, so the directory is named by
 * URL rather than transferred: pushing 426 MB through `postMessage` would copy every shard, while a
 * `fetch` inside the worker lands in exactly one buffer -- the one the runtime borrows.
 */
export interface LoadModelRequest {
  readonly kind: "loadModel";
  readonly id: number;
  /** Directory holding `manifest.json`, `config.bin`, `tokens.bin`, and the `.qw` shards. */
  readonly modelUrl: string;
}

/** Decodes one utterance from log-mel features the main thread already computed. */
export interface DecodeRequest {
  readonly kind: "decode";
  readonly id: number;
  /** `[mel_bin][frame]` row major, `128 * frames` values. Transferred, not copied. */
  readonly features: Float32Array;
  /** Generation budget; the model's own configuration caps it. */
  readonly maxTokens?: number | undefined;
}

/** Releases the model, leaving the runtime, its tokenizer, and its front end in place. */
export interface UnloadModelRequest {
  readonly kind: "unloadModel";
  readonly id: number;
}

export interface SelfTestRequest {
  readonly kind: "selfTest";
  readonly id: number;
}

export interface SetVocabularyRequest {
  readonly kind: "setVocabulary";
  readonly id: number;
  readonly tokens: readonly string[];
}

export interface DetokenizeRequest {
  readonly kind: "detokenize";
  readonly id: number;
  readonly ids: Uint32Array;
}

export interface DisposeRequest {
  readonly kind: "dispose";
  readonly id: number;
}

export type WorkerRequest =
  | InitRequest
  | PreprocessRequest
  | TranscribeRequest
  | LoadModelRequest
  | DecodeRequest
  | UnloadModelRequest
  | SelfTestRequest
  | SetVocabularyRequest
  | DetokenizeRequest
  | DisposeRequest;

export interface ReadyResponse {
  readonly kind: "ready";
  readonly id: number;
  readonly protocolVersion: number;
  readonly versions: WasmVersions;
}

/** Generic acknowledgement of a request that has no payload of its own. */
export interface OkResponse {
  readonly kind: "ok";
  readonly id: number;
}

export interface ProgressResponse {
  readonly kind: "progress";
  readonly id: number;
  readonly progress: PreprocessProgress;
}

/** One report from a model load, forwarded as it happens rather than after the whole directory. */
export interface ModelProgressResponse {
  readonly kind: "modelProgress";
  readonly id: number;
  readonly progress: ModelLoadProgress;
}

export interface ModelResponse {
  readonly kind: "model";
  readonly id: number;
  readonly model: WasmModelInfo;
}

export interface TranscriptResponse extends Transcript {
  readonly kind: "transcript";
  readonly id: number;
}

export interface MelResponse {
  readonly kind: "mel";
  readonly id: number;
  readonly payload: MelPayload;
}

export interface SelfTestResponse {
  readonly kind: "selfTest";
  readonly id: number;
  readonly report: SelfTestReport;
}

export interface TextResponse {
  readonly kind: "text";
  readonly id: number;
  readonly text: string;
}

export interface ErrorResponse {
  readonly kind: "error";
  readonly id: number;
  /** The ABI status integer, or an `SDK_STATUS` value. */
  readonly code: number;
  readonly status: string;
  readonly operation: string;
  readonly message: string;
  readonly context: Readonly<Record<string, unknown>> | undefined;
  /**
   * The error's constructor name, so the other side rebuilds the same class.
   *
   * Without it a typed error would arrive as a base `QwenscriberError`: the same code, a different
   * class, and `instanceof NotImplementedError` -- which callers reasonably write -- would be true
   * on the main thread and false across the worker boundary.
   */
  readonly errorClass: string;
}

export type WorkerResponse =
  | ReadyResponse
  | OkResponse
  | ProgressResponse
  | ModelProgressResponse
  | MelResponse
  | ModelResponse
  | TranscriptResponse
  | SelfTestResponse
  | TextResponse
  | ErrorResponse;

const REQUEST_KINDS: readonly string[] = [
  "init",
  "preprocess",
  "transcribe",
  "loadModel",
  "decode",
  "unloadModel",
  "selfTest",
  "setVocabulary",
  "detokenize",
  "dispose",
];

const RESPONSE_KINDS: readonly string[] = [
  "ready",
  "ok",
  "progress",
  "modelProgress",
  "mel",
  "model",
  "transcript",
  "selfTest",
  "text",
  "error",
];

function protocolError(message: string, context: Record<string, unknown>): QwenscriberError {
  return new QwenscriberError(SDK_STATUS.protocol, "channel", { message, context });
}

/**
 * Ids count up from 1 for requests; 0 is reserved for connection-level messages, such as a worker
 * reporting a request whose envelope it could not read (so there is no id to answer to).
 */
function asMessageEnvelope(value: unknown, kinds: readonly string[], side: string): { kind: string; id: number } {
  if (typeof value !== "object" || value === null) {
    throw protocolError(`${side} is not a message object`, { received: typeof value });
  }
  const envelope = value as { kind?: unknown; id?: unknown };
  if (typeof envelope.kind !== "string" || !kinds.includes(envelope.kind)) {
    throw protocolError(`${side} has an unknown kind`, { kind: String(envelope.kind) });
  }
  if (typeof envelope.id !== "number" || !Number.isInteger(envelope.id) || envelope.id < 0) {
    throw protocolError(`${side} has no request id`, { id: String(envelope.id) });
  }
  return { kind: envelope.kind, id: envelope.id };
}

/** Validates the envelope of an incoming request. The worker trusts only its own shapes. */
export function asWorkerRequest(value: unknown): WorkerRequest {
  asMessageEnvelope(value, REQUEST_KINDS, "request");
  return value as WorkerRequest;
}

/** Validates the envelope of an incoming response, so a stray message cannot settle a promise. */
export function asWorkerResponse(value: unknown): WorkerResponse {
  asMessageEnvelope(value, RESPONSE_KINDS, "response");
  return value as WorkerResponse;
}

/** Buffers a request moves rather than shares. */
export function requestTransferables(request: WorkerRequest): Transferable[] {
  if (request.kind === "preprocess") return [request.pcm];
  if (request.kind === "transcribe") return [request.pcm];
  if (request.kind === "decode") return [request.features.buffer as ArrayBuffer];
  if (request.kind === "detokenize") return [request.ids.buffer as ArrayBuffer];
  return [];
}

/** Buffers a response moves rather than shares. */
export function responseTransferables(response: WorkerResponse): Transferable[] {
  if (response.kind === "mel") return [response.payload.features.buffer as ArrayBuffer];
  if (response.kind === "transcript") return [response.tokens.buffer as ArrayBuffer];
  return [];
}

/**
 * Serializes a failure for the other side.
 *
 * A `QwenscriberError` keeps its code, status, operation, and class, so the caller that finally sees
 * it can branch on the same values -- and catch the same class -- it would have seen in-process.
 * Anything else is wrapped as `internal` with its text preserved, because an untyped throw still has
 * to cross the boundary.
 */
export function errorResponseFrom(error: unknown, operation: string, id: number): ErrorResponse {
  if (isQwenscriberError(error)) {
    return {
      kind: "error",
      id,
      code: error.code,
      status: error.status,
      operation: error.operation,
      message: error.message,
      context: error.context,
      errorClass: error.name,
    };
  }
  const text = error instanceof Error ? error.message : String(error);
  return {
    kind: "error",
    id,
    code: SDK_STATUS.internal,
    status: "internal",
    operation,
    message: `${operation}: ${text}`,
    context: undefined,
    errorClass: error instanceof Error ? error.name : "UnknownError",
  };
}

/** Rebuilds the error the other side reported, as the class it was. */
export function errorFromResponse(response: ErrorResponse): QwenscriberError {
  const options = { message: response.message, context: response.context };
  if (response.errorClass === "NotImplementedError") {
    const feature = response.context?.["feature"];
    return new NotImplementedError(
      typeof feature === "string" ? feature : "unknown",
      response.operation,
      options,
    );
  }
  if (response.errorClass === "AbiMismatchError") {
    const actual = response.context?.["actual_version"];
    const expected = response.context?.["expected_version"];
    if (typeof actual === "number" && typeof expected === "number") {
      return new AbiMismatchError(actual, expected, response.operation);
    }
  }
  return new QwenscriberError(response.code, response.operation, options);
}
