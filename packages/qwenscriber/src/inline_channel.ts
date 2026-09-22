//! The in-process channel: the same stage pipeline as the worker, on the calling thread.
//!
//! Used when `Worker` is unavailable (Node, an environment without workers) or when the caller asks
//! for `worker: false`. It runs the identical sequence -- resample, then log-mel -- so a result does
//! not depend on which channel produced it; only where the work happens differs.
//!
//! It implements the model family too, so the same code drives a model whether or not a worker is
//! available: `loadModel` reads the directory from wherever `fetch` can reach it (a Node caller can
//! pass its own `ModelDirectory` instead), and decode runs the ABI in this thread.

import { MEL_SAMPLE_RATE_HZ, STATUS } from "./wasm/abi.ts";
import { NotImplementedError, QwenscriberError } from "./errors.ts";
import { WasmCore, type SelfTestReport, type WasmSource } from "./wasm/runtime.ts";
import {
  fetchModelDirectory,
  loadModel,
  type LoadModelOptions,
  type ModelChannel,
  type ModelDirectory,
  type Transcript,
  type WasmModel,
  type WasmModelInfo,
} from "./decode.ts";
import { resample } from "./audio/resample.ts";
import type {
  MelPayload,
  PreprocessProgress,
  RuntimeChannel,
  WasmVersions,
} from "./types.ts";

export class InlineChannel implements RuntimeChannel, ModelChannel {
  readonly versions: WasmVersions;
  /** See `RuntimeChannel.progress`. */
  progress: ((progress: PreprocessProgress) => void) | undefined;

  private readonly core: WasmCore;
  private model: WasmModel | undefined;

  private constructor(core: WasmCore) {
    this.core = core;
    this.progress = undefined;
    this.versions = {
      abiVersion: core.abiVersion(),
      coreVersion: core.coreVersion(),
      wasmBytes: core.moduleBytes(),
      runtimeBytes: core.memoryBytes(),
    };
  }

  /** Loads and creates the runtime; resolves once the core is ready to take calls. */
  static async start(wasm: WasmSource): Promise<InlineChannel> {
    return new InlineChannel(await WasmCore.load(wasm));
  }

  get workerBacked(): boolean {
    return false;
  }

  async preprocess(pcm: Float32Array, sampleRateHz: number): Promise<MelPayload> {
    // Identical stages and identical progress reports to the worker path, so a caller that switches
    // channels sees no difference other than the thread the work happened on.
    this.progress?.({ stage: "resample", completed: 0, total: pcm.length });
    const resampled = resample(pcm, sampleRateHz, MEL_SAMPLE_RATE_HZ);
    this.progress?.({ stage: "resample", completed: pcm.length, total: pcm.length });
    this.progress?.({ stage: "mel", completed: 0, total: 1 });
    const computed = this.core.melCompute(resampled);
    this.progress?.({ stage: "mel", completed: 1, total: 1 });
    return {
      frames: computed.frames,
      features: computed.features,
      globalMaxLog: computed.globalMaxLog,
      paddingValue: computed.paddingValue,
      bytesWritten: computed.bytesWritten,
      inputSampleRateHz: sampleRateHz,
    };
  }

  async transcribe(pcm: Float32Array, sampleRateHz: number): Promise<never> {
    const payload = await this.preprocess(pcm, sampleRateHz);
    throw new NotImplementedError("decode", "transcribe", {
      message:
        "transcribe ran the log-mel front end, and this channel holds no model to decode with: " +
        "load a converted model directory with this channel's loadModel, then decode its features.",
      context: {
        frames: payload.frames,
        padding_value: payload.paddingValue,
        input_sample_rate_hz: sampleRateHz,
        stage: "mel",
      },
    });
  }

  async selfTest(): Promise<SelfTestReport> {
    return this.core.selfTest();
  }

  async setVocabulary(tokens: readonly string[]): Promise<void> {
    this.core.setVocabulary({ tokens });
  }

  async detokenize(ids: Uint32Array): Promise<string> {
    return this.core.detokenize(ids);
  }

  /**
   * Loads a converted model directory into this thread's core.
   *
   * A URL or path is fetched through `fetchModelDirectory`; a `ModelDirectory` is used as it is,
   * which is how a Node caller reads a directory from disk without an HTTP server.
   */
  async loadModel(
    source: string | URL | ModelDirectory,
    options: LoadModelOptions = {},
  ): Promise<WasmModelInfo> {
    this.model?.dispose();
    this.model = undefined;
    const directory =
      typeof source === "string" || source instanceof URL ? fetchModelDirectory(source) : source;
    const loaded = await loadModel(this.core, directory, options);
    this.model = loaded;
    return {
      modelId: loaded.modelId,
      quantization: loaded.manifest.quantization,
      source: directory.source,
      requirements: loaded.requirements,
    };
  }

  async decode(features: Float32Array, maxTokens?: number): Promise<Transcript> {
    if (this.model === undefined) {
      throw new QwenscriberError(STATUS.invalid_state, "decode", {
        message: "this channel holds no model; call loadModel first",
      });
    }
    return this.model.decode(features, { maxTokens });
  }

  async unloadModel(): Promise<void> {
    this.model?.dispose();
    this.model = undefined;
  }

  async dispose(): Promise<void> {
    await this.unloadModel();
    this.core.dispose();
  }
}
