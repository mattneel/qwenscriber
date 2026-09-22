//! The worker entry point the SDK instantiates.
//!
//! It owns the WASM instance and the runtime handle, so mel computation happens off the main thread
//! and a long clip cannot freeze a page. It speaks exactly the protocol in `protocol.ts` and nothing
//! else: every reply is either the requested payload or a serialized `QwenscriberError`.
//!
//! Requests are answered strictly one at a time. The ABI has module-global state -- one handle, one
//! vocabulary -- and `qw_tokenizer_set` can only be reasoned about if no other call is in flight, so
//! a promise chain serializes messages instead of a lock. Progress messages are the one thing sent
//! out of band: they carry the id of the request being worked on and never settle anything.

import { MEL_SAMPLE_RATE_HZ, STATUS } from "../wasm/abi.ts";
import { NotImplementedError, QwenscriberError, SDK_STATUS } from "../errors.ts";
import { resample } from "../audio/resample.ts";
import { fetchModelDirectory, loadModel, type WasmModel } from "../decode.ts";
import { WasmCore } from "../wasm/runtime.ts";
import {
  PROTOCOL_VERSION,
  asWorkerRequest,
  errorResponseFrom,
  responseTransferables,
  type DisposeRequest,
  type InitRequest,
  type LoadModelRequest,
  type TranscribeRequest,
  type WorkerRequest,
  type WorkerResponse,
} from "./protocol.ts";
import type { MelPayload, PreprocessProgress, WasmVersions } from "../types.ts";

/** The only worker global this module touches. Structural, so no DOM type is required. */
interface WorkerScope {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  addEventListener(type: "message", listener: (event: { readonly data: unknown }) => void): void;
  close(): void;
}

const scope = globalThis as unknown as WorkerScope;

let core: WasmCore | undefined;
let model: WasmModel | undefined;
let queue: Promise<void> = Promise.resolve();

scope.addEventListener("message", (event) => {
  const answered = queue.then(() => answer(event.data));
  // The chain must survive a rejected step, or one malformed request would silence the worker.
  queue = answered.catch(() => undefined);
});

function post(response: WorkerResponse): void {
  scope.postMessage(response, responseTransferables(response));
}

async function answer(data: unknown): Promise<void> {
  let request: WorkerRequest;
  try {
    request = asWorkerRequest(data);
  } catch (error) {
    // The envelope was unreadable, so there is no id to answer to: id 0 is the connection-level
    // channel, and the main thread turns it into a failure of every pending call.
    post(errorResponseFrom(error, "worker", 0));
    return;
  }
  try {
    await dispatch(request);
  } catch (error) {
    post(errorResponseFrom(error, request.kind, request.id));
  }
}

async function dispatch(request: WorkerRequest): Promise<void> {
  if (request.kind === "init") {
    await initialize(request);
    return;
  }
  if (request.kind === "dispose") {
    dispose(request);
    return;
  }
  const active = requireCore();
  if (request.kind === "preprocess") {
    const payload = preprocess(active, request.pcm, request.sampleRateHz, request.id);
    post({ kind: "mel", id: request.id, payload });
    return;
  }
  if (request.kind === "transcribe") {
    transcribe(active, request);
    return;
  }
  if (request.kind === "loadModel") {
    await loadModelRequest(active, request);
    return;
  }
  if (request.kind === "decode") {
    const loaded = requireModel();
    const decoded = loaded.decode(request.features, { maxTokens: request.maxTokens });
    post({ kind: "transcript", id: request.id, tokens: decoded.tokens, text: decoded.text });
    return;
  }
  if (request.kind === "unloadModel") {
    // Releasing the model frees the shard buffers and the runtime's own arena; the front end, the
    // tokenizer, and the self-test stay usable.
    model?.dispose();
    model = undefined;
    post({ kind: "ok", id: request.id });
    return;
  }
  if (request.kind === "selfTest") {
    post({ kind: "selfTest", id: request.id, report: active.selfTest() });
    return;
  }
  if (request.kind === "setVocabulary") {
    active.setVocabulary({ tokens: request.tokens });
    post({ kind: "ok", id: request.id });
    return;
  }
  post({ kind: "text", id: request.id, text: active.detokenize(request.ids) });
}

function requireCore(): WasmCore {
  if (core === undefined) {
    throw new QwenscriberError(STATUS.invalid_state, "worker", {
      message: "the worker has no runtime yet; the init request must complete first",
      context: { worker_state: "uninitialised" },
    });
  }
  return core;
}

function requireModel(): WasmModel {
  if (model === undefined) {
    throw new QwenscriberError(STATUS.invalid_state, "decode", {
      message: "the worker holds no model; send a loadModel request first",
      context: { worker_state: "no-model" },
    });
  }
  return model;
}

/**
 * Loads a converted model directory into the worker's linear memory.
 *
 * The worker fetches the directory itself. A model is hundreds of megabytes, and transferring that
 * through `postMessage` would copy every shard on the way in and again on the way out; a `fetch`
 * here lands in exactly one buffer, the one the runtime borrows. Progress is reported per file,
 * because a load that runs for a minute with no output is indistinguishable from a hang.
 *
 * A model already in the worker is released first, so switching models is one request rather than a
 * strict unload-then-load sequence.
 */
async function loadModelRequest(active: WasmCore, request: LoadModelRequest): Promise<void> {
  model?.dispose();
  model = undefined;
  const loaded = await loadModel(active, fetchModelDirectory(request.modelUrl), {
    onProgress: (progress) => post({ kind: "modelProgress", id: request.id, progress }),
  });
  model = loaded;
  post({
    kind: "model",
    id: request.id,
    model: {
      modelId: loaded.modelId,
      quantization: loaded.manifest.quantization,
      source: request.modelUrl,
      requirements: loaded.requirements,
    },
  });
}

async function initialize(request: InitRequest): Promise<void> {
  if (request.protocolVersion !== PROTOCOL_VERSION) {
    // A cached worker script next to a newer SDK would otherwise be read with the wrong layout.
    throw new QwenscriberError(SDK_STATUS.protocol, "init", {
      message:
        `the page speaks protocol v${request.protocolVersion} but this worker script speaks ` +
        `v${PROTOCOL_VERSION}; a stale inference.worker.js is being served`,
      context: {
        page_protocol_version: request.protocolVersion,
        worker_protocol_version: PROTOCOL_VERSION,
      },
    });
  }
  if (core !== undefined) {
    throw new QwenscriberError(STATUS.invalid_state, "init", {
      message: "the worker already owns a runtime",
    });
  }
  const source =
    "url" in request.wasm ? request.wasm.url : new Uint8Array(request.wasm.bytes);
  const loaded = await WasmCore.load(source);
  core = loaded;
  const versions: WasmVersions = {
    abiVersion: loaded.abiVersion(),
    coreVersion: loaded.coreVersion(),
    wasmBytes: loaded.moduleBytes(),
    runtimeBytes: loaded.memoryBytes(),
  };
  post({ kind: "ready", id: request.id, protocolVersion: PROTOCOL_VERSION, versions });
}

function reportProgress(id: number, progress: PreprocessProgress): void {
  post({ kind: "progress", id, progress });
}

/** Resamples and computes log-mel, reporting each stage as it starts and finishes. */
function preprocess(
  active: WasmCore,
  pcm: ArrayBuffer,
  sampleRateHz: number,
  id: number,
): MelPayload {
  const input = new Float32Array(pcm);
  if (input.length === 0) {
    throw new QwenscriberError(STATUS.invalid_argument, "preprocess", {
      message: "the clip has no samples",
      context: { pcm_bytes: pcm.byteLength },
    });
  }
  reportProgress(id, { stage: "resample", completed: 0, total: input.length });
  const resampled = resample(input, sampleRateHz, MEL_SAMPLE_RATE_HZ);
  reportProgress(id, { stage: "resample", completed: input.length, total: input.length });
  reportProgress(id, { stage: "mel", completed: 0, total: 1 });
  const computed = active.melCompute(resampled);
  reportProgress(id, { stage: "mel", completed: 1, total: 1 });
  return {
    frames: computed.frames,
    features: computed.features,
    globalMaxLog: computed.globalMaxLog,
    paddingValue: computed.paddingValue,
    bytesWritten: computed.bytesWritten,
    inputSampleRateHz: sampleRateHz,
  };
}

/**
 * Runs the front end, then reports the stage that does not exist.
 *
 * Deliberately does the work before failing: a caller learns that preprocessing succeeded (frame
 * count and padding value travel in the error's context) and that the pipeline stops at the mel
 * boundary, instead of a blanket "not supported" that hides which half ran.
 */
function transcribe(active: WasmCore, request: TranscribeRequest): void {
  const payload = preprocess(active, request.pcm, request.sampleRateHz, request.id);
  throw new NotImplementedError("decode", "transcribe", {
    message:
      "transcribe ran the log-mel front end, and this worker holds no model to decode with: " +
      "send a loadModel request for a converted model directory, then decode its features.",
    context: {
      frames: payload.frames,
      padding_value: payload.paddingValue,
      input_sample_rate_hz: payload.inputSampleRateHz,
      stage: "mel",
    },
  });
}

function dispose(request: DisposeRequest): void {
  model?.dispose();
  model = undefined;
  core?.dispose();
  core = undefined;
  // Acknowledge before closing: `close()` takes effect once this task returns, so the reply is
  // already on its way out.
  post({ kind: "ok", id: request.id });
  scope.close();
}
