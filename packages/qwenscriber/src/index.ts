//! The SDK's public entry point.
//!
//! What works today, end to end: decode audio to mono f32, resample it to the core's 16 kHz, and
//! compute Whisper log-mel features with the freestanding WASM core -- in a worker by default, so a
//! long clip never blocks a page. Detokenization and the core's self-test are exposed for the same
//! reason: they are stages ABI v1 actually implements.
//!
//! What does not work yet is stated rather than simulated. ABI v1 now loads a converted model and
//! decodes with it -- `WasmCore` exposes `qw_model_*`/`qw_decode_*` and `decode.ts` drives a whole
//! model directory -- but this facade still loads no model, so `transcribe()` runs the front end and
//! then throws `NotImplementedError` saying that no converted model directory is loaded, and
//! carrying the preprocessing that did happen. Nothing here ever returns invented text, and no
//! method silently degrades into a slower path.

import { STATUS } from "./wasm/abi.ts";
import { NotImplementedError, QwenscriberError, SDK_STATUS } from "./errors.ts";
import { capabilities, type Capabilities } from "./capabilities.ts";
import { decodeAudio, type AudioInput, type DecodeOptions } from "./audio/pcm.ts";
import { InlineChannel } from "./inline_channel.ts";
import { WorkerChannel } from "./worker/worker_channel.ts";
import type { WorkerWasmSource } from "./worker/protocol.ts";
import type {
  Backend,
  PreprocessProgress,
  PreprocessedAudio,
  QwenscriberOptions,
  QwenscriberVersions,
  Quantization,
  ResolvedBackend,
  RuntimeChannel,
} from "./types.ts";
import type { SelfTestReport, WasmSource } from "./wasm/runtime.ts";

export const SDK_VERSION = "0.1.0";
/** Model this build is aimed at. Reserved until ABI v1 grows a model-loading stage. */
export const DEFAULT_MODEL = "Qwen3-ASR-0.6B";

function toWorkerWasmSource(source: WasmSource): WorkerWasmSource {
  if (typeof source === "string") return { url: source };
  if (source instanceof URL) return { url: source.href };
  if (source instanceof WebAssembly.Module) {
    throw new QwenscriberError(SDK_STATUS.protocol, "create", {
      message:
        "a compiled WebAssembly.Module cannot cross the worker boundary; pass a URL or bytes, " +
        "or use worker: false",
    });
  }
  // Copied, because the bytes are transferred to the worker and transfer detaches the source:
  // a caller who passed a view of a larger buffer would otherwise find it emptied.
  const bytes =
    source instanceof ArrayBuffer
      ? source.slice(0)
      : (source.buffer.slice(
          source.byteOffset,
          source.byteOffset + source.byteLength,
        ) as ArrayBuffer);
  return { bytes };
}

function resolveBackend(requested: Backend, webgpuAvailable: boolean, reason?: string): ResolvedBackend {
  if (requested === "wasm") return "wasm";
  if (requested === "webgpu") {
    if (webgpuAvailable) return "webgpu";
    throw new QwenscriberError(STATUS.unsupported, "create", {
      message: `backend "webgpu" was requested but WebGPU is unavailable: ${reason ?? "no reason reported"}`,
      context: { requested_backend: requested, webgpu_reason: reason },
    });
  }
  return webgpuAvailable ? "webgpu" : "wasm";
}

export class Qwenscriber {
  private readonly channel: RuntimeChannel;
  private readonly backendValue: ResolvedBackend;
  private readonly modelValue: string;
  private readonly quantizationValue: Quantization;
  private disposedValue = false;

  private constructor(
    channel: RuntimeChannel,
    backend: ResolvedBackend,
    model: string,
    quantization: Quantization,
  ) {
    this.channel = channel;
    this.backendValue = backend;
    this.modelValue = model;
    this.quantizationValue = quantization;
  }

  /**
   * Loads the core, resolves the backend, and starts the channel.
   *
   * `worker` defaults to true wherever `Worker` exists, and the WASM core defaults to
   * `qwenscriber_core.wasm` next to this module, which the package build places in `dist/`.
   */
  static async create(options: QwenscriberOptions = {}): Promise<Qwenscriber> {
    const probe = await capabilities();
    const backend = resolveBackend(options.backend ?? "auto", probe.webgpu.available, probe.webgpu.reason);
    const wasm = options.wasm ?? new URL("./qwenscriber_core.wasm", import.meta.url);
    const model = options.model ?? DEFAULT_MODEL;
    const quantization = options.quantization ?? "q8";
    const worker_wanted = options.worker ?? probe.workers;
    let channel: RuntimeChannel;
    if (worker_wanted) {
      channel = await WorkerChannel.start({
        workerUrl: options.workerUrl ?? new URL("./worker/inference.worker.js", import.meta.url),
        wasm: toWorkerWasmSource(wasm),
      });
    } else {
      channel = await InlineChannel.start(wasm);
    }
    channel.progress = options.onProgress;
    return new Qwenscriber(channel, backend, model, quantization);
  }

  /**
   * Probes the host without creating an instance: WebGPU adapter and limits, WASM SIMD and threads,
   * `SharedArrayBuffer`, cross-origin isolation, and worker support.
   *
   * Static and instance-free, so a page can decide what to offer before paying for the core. It
   * never throws; an unavailable feature comes back with a `reason`.
   */
  static async capabilities(): Promise<Capabilities> {
    return capabilities();
  }

  /** The backend the encoder and decoder stages will use, with `"auto"` already resolved. */
  get backend(): ResolvedBackend {
    return this.backendValue;
  }

  /** The model this instance was created for. */
  get model(): string {
    return this.modelValue;
  }

  /** Weight storage format the future loader will read. */
  get quantization(): Quantization {
    return this.quantizationValue;
  }

  /** True when the core runs in a Worker rather than on this thread. */
  get workerBacked(): boolean {
    return this.channel.workerBacked;
  }

  /** Versions of the SDK and the loaded module, plus its compiled and resident sizes. */
  get versions(): QwenscriberVersions {
    return { sdkVersion: SDK_VERSION, ...this.channel.versions };
  }

  /** Receives stage progress for subsequent calls. Pass `undefined` to stop listening. */
  onProgress(listener: ((progress: PreprocessProgress) => void) | undefined): void {
    this.channel.progress = listener;
  }

  /**
   * Decodes audio, resamples it to 16 kHz, and computes log-mel features.
   *
   * Accepts a `Float32Array`/`Int16Array`/`Int32Array` of PCM, raw PCM bytes, WAVE bytes, a `Blob`,
   * or an `AudioBuffer`-like object; see `decodeAudio`. The returned features are `[mel_bin][frame]`
   * row-major, `128 * frames` values -- exactly what the encoder will consume once it exists.
   */
  async preprocess(audio: AudioInput, options: DecodeOptions = {}): Promise<PreprocessedAudio> {
    this.assertUsable("preprocess");
    const decoded = await decodeAudio(audio, options);
    const payload = await this.channel.preprocess(decoded.samples, decoded.sample_rate_hz);
    return {
      ...payload,
      audioDurationMs: decoded.duration_ms,
      channelCount: decoded.channel_count,
      sourceKind: decoded.source_kind,
    };
  }

  /**
   * Preprocesses the clip, then reports that no model is loaded.
   *
   * Always rejects with `NotImplementedError` (`code === SDK_STATUS.not_implemented`). The ABI can
   * decode now, but a transcript needs a converted model directory resident in the core's linear
   * memory, and this facade has no model to decode with. The front end genuinely ran first -- frame
   * count, padding value, and the audio metadata are in the error's `context` -- because "the half
   * that exists worked, the other half is missing" is a more useful answer than a blanket failure.
   * This method never returns a fabricated transcript.
   */
  async transcribe(audio: AudioInput, options: DecodeOptions = {}): Promise<never> {
    this.assertUsable("transcribe");
    const decoded = await decodeAudio(audio, options);
    try {
      await this.channel.transcribe(decoded.samples, decoded.sample_rate_hz);
    } catch (error) {
      if (error instanceof NotImplementedError) {
        throw new NotImplementedError(error.feature, "transcribe", {
          message:
            `transcribe preprocessed ${decoded.frames} samples at ${decoded.sample_rate_hz} Hz and ` +
            `stopped: no converted model directory is loaded, so there is nothing to decode with ` +
            `(model ${this.modelValue}, backend ${this.backendValue}, ` +
            `quantization ${this.quantizationValue})`,
          context: {
            ...error.context,
            model: this.modelValue,
            backend: this.backendValue,
            quantization: this.quantizationValue,
            source_kind: decoded.source_kind,
            audio_duration_ms: decoded.duration_ms,
          },
          cause: error,
        });
      }
      throw error;
    }
    throw new QwenscriberError(SDK_STATUS.internal, "transcribe", {
      message: "the channel returned from transcribe without a transcription",
    });
  }

  /**
   * Runs the core's self-test: arithmetic, quantization packing, tokenizer, SIMD, and log-mel,
   * hashed and compared against the constants the host build produces.
   *
   * `failures` is a bit mask; zero means this module computes exactly what the reference computes,
   * which is what makes it usable as the oracle for the GPU kernels. A page can show this without
   * loading a model, which is why the browser example leads with it.
   */
  async selfTest(): Promise<SelfTestReport> {
    this.assertUsable("selfTest");
    return this.channel.selfTest();
  }

  /** Points the runtime at a vocabulary, for `detokenize`. */
  async setVocabulary(tokens: readonly string[]): Promise<void> {
    this.assertUsable("setVocabulary");
    return this.channel.setVocabulary(tokens);
  }

  /** Decodes token ids into UTF-8 text using the vocabulary `setVocabulary` installed. */
  async detokenize(ids: Uint32Array): Promise<string> {
    this.assertUsable("detokenize");
    return this.channel.detokenize(ids);
  }

  /** Releases the runtime and, for the worker channel, terminates the worker. Idempotent. */
  async dispose(): Promise<void> {
    if (this.disposedValue) return;
    this.disposedValue = true;
    this.channel.progress = undefined;
    await this.channel.dispose();
  }

  private assertUsable(operation: string): void {
    if (this.disposedValue) {
      throw new QwenscriberError(STATUS.invalid_state, operation, {
        message: "this instance is disposed; create a new one",
      });
    }
  }
}

export { SDK_STATUS, QwenscriberError, AbiMismatchError, NotImplementedError, isQwenscriberError } from "./errors.ts";
export {
  capabilities,
  probeWebGpu,
  acquireAdapter,
  gpuEntryPoint,
  WEBGPU_PROBE_TIMEOUT_MS,
  type AdapterAcquisition,
  type Capabilities,
  type WebGpuAdapterInfo,
  type WebGpuCapability,
  type WebGpuLimits,
} from "./capabilities.ts";
export {
  WebGpuRuntime,
  type ShaderSource,
  type WebGpuDispatchGeometry,
  type WebGpuRuntimeCapability,
  type WebGpuRuntimeOptions,
} from "./gpu/runtime.ts";
export {
  matmulQ4,
  Q4_CODE_BYTES_PER_GROUP,
  Q4_GROUP_SIZE,
  type MatmulQ4Request,
  type MatmulQ4Result,
} from "./gpu/matmul_q4.ts";
export { shaderSourceFromBaseUrl } from "./gpu/shaders.ts";
export { TENSOR_KIND, type TensorKind, type TensorKindName } from "./gpu/tensor_kind.ts";
export {
  WEBGPU_KERNELS,
  bindGroupLayoutEntries,
  type WebGpuKernelBinding,
  type WebGpuKernelDescriptor,
  type WebGpuKernelName,
} from "./gpu/kernels.ts";
export {
  measuredLimits,
  requireLimits,
  shortfallsFor,
  type WebGpuRequirements,
} from "./gpu/limits.ts";
export {
  partCapacityOf,
  planUpload,
  type WebGpuBufferPlan,
  type WebGpuShardPlacement,
  type WebGpuShardPlan,
  type WebGpuUploadPlan,
  type WebGpuUploadPlanLimits,
} from "./gpu/upload_plan.ts";
export {
  GpuError,
  GpuDeviceError,
  GpuLimitsError,
  GpuShaderError,
  GpuSoftwareAdapterError,
  GpuUnavailableError,
  isGpuError,
  type GpuLimitShortfall,
} from "./gpu/errors.ts";
export {
  WasmCore,
  Allocation,
  type MelComputation,
  type SelfTestReport,
  type VocabularyDescriptor,
  type WasmSource,
} from "./wasm/runtime.ts";
export {
  ABI_MAJOR,
  ABI_MINOR,
  ABI_VERSION_EXPECTED,
  MEL_BINS,
  MEL_CAPACITY_FRAMES,
  MEL_MIN_SAMPLES,
  MEL_SAMPLE_RATE_HZ,
  STATUS,
  formatAbiVersion,
  melPaddingValueFromGlobalMax,
} from "./wasm/abi.ts";
export {
  decodeAudio,
  type AudioBufferLike,
  type AudioInput,
  type DecodeOptions,
  type DecodedAudio,
  type PcmFormat,
} from "./audio/pcm.ts";
export { resample } from "./audio/resample.ts";
export {
  CAPTURE_BLOCK_FRAMES,
  CAPTURE_BLOCKS_IN_FLIGHT_MAX,
  CAPTURE_SAMPLES_MAX,
  CAPTURE_WORKLET_SOURCE,
  CaptureQueue,
  MicrophoneCapture,
  type CaptureOptions,
  type CaptureState,
} from "./audio/capture.ts";
export { looksLikeWave, readWave, type WaveAudio } from "./audio/wav.ts";
export type {
  AudioSourceKind,
  Backend,
  MelPayload,
  PreprocessProgress,
  PreprocessedAudio,
  QwenscriberOptions,
  QwenscriberVersions,
  Quantization,
  ResolvedBackend,
  RuntimeChannel,
  WasmVersions,
} from "./types.ts";
