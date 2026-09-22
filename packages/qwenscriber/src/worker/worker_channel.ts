//! The main-thread half of the worker channel.
//!
//! Every request gets an id and a promise; the worker's reply settles it. Progress messages are the
//! exception and never settle anything: they belong to a request that is still running, so they are
//! forwarded to the listener and dropped if there is none.
//!
//! Failure handling is deliberately total. A worker that throws, sends a response nobody awaits, or
//! sends a message whose envelope cannot be read has broken the channel contract, and every pending
//! promise is rejected with a typed error instead of left hanging: a promise that never settles is
//! the one failure mode a caller cannot handle.

import { QwenscriberError, SDK_STATUS } from "../errors.ts";
import { STATUS } from "../wasm/abi.ts";
import {
  PROTOCOL_VERSION,
  asWorkerResponse,
  errorFromResponse,
  type WorkerRequest,
  type WorkerResponse,
  type WorkerWasmSource,
} from "./protocol.ts";
import type { MelPayload, PreprocessProgress, RuntimeChannel, WasmVersions } from "../types.ts";
import type { SelfTestReport } from "../wasm/runtime.ts";

/** Requests allowed in flight before new ones are refused. A bound, not a queue. */
const PENDING_MAX = 256;

interface InFlight {
  resolve(response: WorkerResponse): void;
  reject(error: QwenscriberError): void;
}

export interface WorkerChannelOptions {
  readonly workerUrl: string | URL;
  readonly wasm: WorkerWasmSource;
}

export class WorkerChannel implements RuntimeChannel {
  private readonly worker: Worker;
  private readonly inFlight = new Map<number, InFlight>();
  private versionsValue: WasmVersions | undefined;
  private nextIdValue = 1;
  private disposedValue = false;
  private disposal: Promise<void> | undefined;
  /** See `RuntimeChannel.progress`. */
  progress: ((progress: PreprocessProgress) => void) | undefined;

  private constructor(worker: Worker) {
    this.worker = worker;
    this.progress = undefined;
    worker.addEventListener("message", (event: MessageEvent<unknown>) => {
      this.onMessage(event.data);
    });
    worker.addEventListener("error", (event: ErrorEvent) => {
      this.failAll(
        new QwenscriberError(SDK_STATUS.protocol, "worker", {
          message: `the inference worker failed to start or crashed: ${event.message}`,
          context: { filename: event.filename, line: event.lineno },
        }),
      );
    });
    worker.addEventListener("messageerror", () => {
      this.failAll(
        new QwenscriberError(SDK_STATUS.protocol, "worker", {
          message: "a message could not be deserialized across the worker boundary",
        }),
      );
    });
  }

  /** Starts the worker and completes the `init` handshake. */
  static async start(options: WorkerChannelOptions): Promise<WorkerChannel> {
    const href = typeof options.workerUrl === "string" ? options.workerUrl : options.workerUrl.href;
    const worker = new Worker(href, { type: "module", name: "qwenscriber-inference" });
    const channel = new WorkerChannel(worker);
    try {
      const ready = await channel.expect(
        "ready",
        {
          kind: "init",
          id: channel.takeId(),
          protocolVersion: PROTOCOL_VERSION,
          wasm: options.wasm,
        },
        [],
      );
      channel.versionsValue = ready.versions;
    } catch (error) {
      // A worker that failed to load its module has nothing useful to say later.
      worker.terminate();
      throw error;
    }
    return channel;
  }

  get versions(): WasmVersions {
    if (this.versionsValue === undefined) {
      throw new QwenscriberError(SDK_STATUS.protocol, "worker", {
        message: "the channel has no versions: the init handshake never completed",
      });
    }
    return this.versionsValue;
  }

  get workerBacked(): boolean {
    return true;
  }

  async preprocess(pcm: Float32Array, sampleRateHz: number): Promise<MelPayload> {
    // The buffer is transferred, not copied. `decodeAudio` always hands back a fresh array, so
    // detaching it here cannot be observed by the caller.
    const buffer = pcm.buffer as ArrayBuffer;
    const response = await this.expect(
      "mel",
      { kind: "preprocess", id: this.takeId(), pcm: buffer, sampleRateHz },
      [buffer],
    );
    return response.payload;
  }

  async transcribe(pcm: Float32Array, sampleRateHz: number): Promise<never> {
    const buffer = pcm.buffer as ArrayBuffer;
    const response = await this.send(
      { kind: "transcribe", id: this.takeId(), pcm: buffer, sampleRateHz },
      [buffer],
    );
    // Unreachable by contract: the worker answers `transcribe` with an error response, which
    // `send` turns into a rejection. If it ever answers with a payload, the protocol is broken and
    // saying so is better than returning something the ABI cannot produce.
    throw new QwenscriberError(SDK_STATUS.protocol, "transcribe", {
      message: `the worker answered transcribe with ${response.kind} instead of an error`,
      context: { response_kind: response.kind },
    });
  }

  async selfTest(): Promise<SelfTestReport> {
    const response = await this.expect("selfTest", { kind: "selfTest", id: this.takeId() }, []);
    return response.report;
  }

  async setVocabulary(tokens: readonly string[]): Promise<void> {
    await this.expect("ok", { kind: "setVocabulary", id: this.takeId(), tokens }, []);
  }

  async detokenize(ids: Uint32Array): Promise<string> {
    // Copy first: sending transfers the buffer, and transfer detaches the source, so the caller's
    // array would come back empty.
    const copy = ids.slice();
    const response = await this.expect(
      "text",
      { kind: "detokenize", id: this.takeId(), ids: copy },
      [copy.buffer as ArrayBuffer],
    );
    return response.text;
  }

  dispose(): Promise<void> {
    if (this.disposal === undefined) this.disposal = this.disposeOnce();
    return this.disposal;
  }

  private async disposeOnce(): Promise<void> {
    try {
      await this.expect("ok", { kind: "dispose", id: this.takeId() }, []);
    } catch {
      // A worker that died before acknowledging has nothing left to release; terminate below is the
      // only remaining step, and dispose must stay idempotent rather than surface a second failure.
    }
    this.disposedValue = true;
    this.worker.terminate();
    this.failAll(
      new QwenscriberError(STATUS.invalid_state, "channel", { message: "the channel was disposed" }),
    );
  }

  private takeId(): number {
    const id = this.nextIdValue;
    this.nextIdValue += 1;
    return id;
  }

  private send(request: WorkerRequest, transfer: Transferable[]): Promise<WorkerResponse> {
    if (this.disposedValue) {
      return Promise.reject(
        new QwenscriberError(STATUS.invalid_state, "channel", {
          message: "the channel is disposed",
          context: { request: request.kind },
        }),
      );
    }
    if (this.inFlight.size >= PENDING_MAX) {
      return Promise.reject(
        new QwenscriberError(STATUS.limit_exceeded, "channel", {
          message: `at most ${PENDING_MAX} requests may be in flight`,
          context: { in_flight: this.inFlight.size, request: request.kind },
        }),
      );
    }
    return new Promise<WorkerResponse>((resolve, reject) => {
      // Executor form rather than `Promise.withResolvers` (ES2024) for the same reason as the
      // capability probe: the SDK declares Node 20 compatibility, and this file is also the one
      // path a bundler may evaluate in an older engine.
      this.inFlight.set(request.id, { resolve, reject });
      this.worker.postMessage(request, transfer);
    });
  }

  private async expect<TKind extends WorkerResponse["kind"]>(
    kind: TKind,
    request: WorkerRequest,
    transfer: Transferable[],
  ): Promise<Extract<WorkerResponse, { kind: TKind }>> {
    const response = await this.send(request, transfer);
    if (response.kind !== kind) {
      throw new QwenscriberError(SDK_STATUS.protocol, "worker", {
        message: `expected a ${kind} response to ${request.kind}, received ${response.kind}`,
        context: { expected: kind, received: response.kind, request: request.kind },
      });
    }
    // The check above is the narrowing; the compiler cannot see it through `TKind`.
    return response as Extract<WorkerResponse, { kind: TKind }>;
  }

  private onMessage(data: unknown): void {
    let response: WorkerResponse;
    try {
      response = asWorkerResponse(data);
    } catch (error) {
      this.failAll(
        error instanceof QwenscriberError
          ? error
          : new QwenscriberError(SDK_STATUS.protocol, "worker", {
              message: "the worker sent a message this SDK cannot read",
            }),
      );
      return;
    }
    if (response.kind === "progress") {
      this.progress?.(response.progress);
      return;
    }
    if (response.id === 0) {
      // Connection-level: the worker could not read a request, so no call can be settled by it.
      this.failAll(
        response.kind === "error"
          ? errorFromResponse(response)
          : new QwenscriberError(SDK_STATUS.protocol, "worker", {
              message: `the worker sent a connection-level ${response.kind} message`,
              context: { response_kind: response.kind },
            }),
      );
      return;
    }
    const pending = this.inFlight.get(response.id);
    if (pending === undefined) {
      this.failAll(
        new QwenscriberError(SDK_STATUS.protocol, "worker", {
          message: `no request is waiting for ${response.kind} #${response.id}`,
          context: { request_id: response.id, response_kind: response.kind },
        }),
      );
      return;
    }
    this.inFlight.delete(response.id);
    if (response.kind === "error") {
      pending.reject(errorFromResponse(response));
      return;
    }
    pending.resolve(response);
  }

  private failAll(error: QwenscriberError): void {
    const pending = [...this.inFlight.values()];
    this.inFlight.clear();
    for (const entry of pending) entry.reject(error);
  }
}
