//! The in-process channel: the same stage pipeline as the worker, on the calling thread.
//!
//! Used when `Worker` is unavailable (Node, an environment without workers) or when the caller asks
//! for `worker: false`. It runs the identical sequence -- resample, then log-mel -- so a result does
//! not depend on which channel produced it; only where the work happens differs.

import { MEL_SAMPLE_RATE_HZ } from "./wasm/abi.ts";
import { NotImplementedError } from "./errors.ts";
import { WasmCore, type SelfTestReport, type WasmSource } from "./wasm/runtime.ts";
import { resample } from "./audio/resample.ts";
import type {
  MelPayload,
  PreprocessProgress,
  RuntimeChannel,
  WasmVersions,
} from "./types.ts";

export class InlineChannel implements RuntimeChannel {
  readonly versions: WasmVersions;
  /** See `RuntimeChannel.progress`. */
  progress: ((progress: PreprocessProgress) => void) | undefined;

  private readonly core: WasmCore;

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
        "transcribe needs the encoder and decode stages, which ABI v1 does not expose: the mel " +
        "front end ran, and there is nothing that can turn features into tokens yet.",
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

  async dispose(): Promise<void> {
    this.core.dispose();
  }
}
