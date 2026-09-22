//! The WebGPU runtime: one adapter, one device, and the buffers and pipelines the kernels need.
//!
//! What this layer is responsible for, in order:
//!
//!   * getting an adapter that is worth using, and saying which request produced it;
//!   * reporting the *measured* limits, so a page can answer "why is my discrete GPU not used" from
//!     its own output instead of from guesswork;
//!   * refusing, with a typed error, anything the adapter cannot host -- a buffer, a binding, a
//!     workgroup, or a manifest that has to be split;
//!   * building pipelines from the explicit layouts in `kernels.ts`, one per kernel, no reflection.
//!
//! It does not own the shader sources. `shaderSource` is passed in because only the caller knows
//! where `gpu/shaders/` is relative to its own module: this package is served from
//! `packages/qwenscriber/dist/` in one deployment and from a bundler in another, so a default path
//! would be right in exactly one of them.

import {
  acquireAdapter,
  gpuEntryPoint,
  type WebGpuAdapterInfo,
  type WebGpuLimits,
} from "../capabilities.ts";
import {
  GpuDeviceError,
  GpuShaderError,
  GpuSoftwareAdapterError,
  GpuUnavailableError,
} from "./errors.ts";
import {
  WEBGPU_KERNELS,
  bindGroupLayoutEntries,
  type WebGpuKernelDescriptor,
  type WebGpuKernelName,
} from "./kernels.ts";
import { requireLimits, type WebGpuRequirements } from "./limits.ts";
import { planUpload, type WebGpuUploadPlan, type WebGpuUploadPlanLimits } from "./upload_plan.ts";
import { BUFFER_USAGE, MAP_MODE } from "./webgpu_flags.ts";

/** Reads a shader source by file name inside the shader directory. */
export type ShaderSource = (file: string) => Promise<string>;

export interface WebGpuRuntimeOptions {
  readonly shaderSource: ShaderSource;
  /**
   * Run on SwiftShader/llvmpipe/WARP anyway.
   *
   * Default `false`: those adapters produce correct numbers at unusable speed, so choosing one has
   * to be a decision the caller made on purpose.
   */
  readonly allowSoftwareAdapter?: boolean | undefined;
  /** Adapter request timeout, for the case where `requestAdapter()` never settles. */
  readonly timeoutMs?: number | undefined;
  /** Refuse at `create()` when the adapter cannot host these, instead of at first use. */
  readonly requiredLimits?: WebGpuRequirements | undefined;
}

/** The geometry a dispatch used, reported back so a caller can print or assert it. */
export interface WebGpuDispatchGeometry {
  readonly workgroupSize: readonly [number, number, number];
  readonly workgroupCounts: readonly [number, number, number];
}

/** Everything a report needs: which adapter, how it was asked for, and what it measured. */
export interface WebGpuRuntimeCapability {
  /** Which request succeeded; `"high-performance"` is the one that asks for a discrete GPU. */
  readonly adapterRequest: "high-performance" | "default";
  readonly adapterKind: "hardware" | "software";
  readonly adapterInfo: WebGpuAdapterInfo;
  /** A second GPU the other request answered with, when it was a different one. */
  readonly alternativeAdapterInfo?: WebGpuAdapterInfo | undefined;
  readonly limits: WebGpuLimits;
  readonly deviceFeatures: readonly string[];
}

interface BuiltPipeline {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/** The default WebGPU `minStorageBufferOffsetAlignment`, used only when an adapter omits it. */
const DEFAULT_ALIGNMENT_BYTES = 256;

function alignTo4(bytes: number): number {
  return Math.ceil(bytes / 4) * 4;
}

/** Reads the compile diagnostics, on engines that can report them. */
async function shaderDiagnostics(module: GPUShaderModule): Promise<string[]> {
  if (typeof module.getCompilationInfo !== "function") return [];
  const info = await module.getCompilationInfo();
  return info.messages
    .filter((message) => message.type === "error")
    .map((message) => `line ${message.lineNum}: ${message.message}`);
}

/**
 * A device plus the pipelines built on it.
 *
 * Construct with `await WebGpuRuntime.create(...)`; `create()` is where every failure that should be
 * fatal is raised, so a caller that holds a runtime holds a usable one.
 */
export class WebGpuRuntime {
  readonly #device: GPUDevice;
  readonly #capability: WebGpuRuntimeCapability;
  readonly #shaderSource: ShaderSource;
  // A `Partial<Record<...>>` and not a `Map`: the key set is the closed union of kernel names and
  // nothing is ever deleted, so a plain object is both the smaller and the more honest shape.
  #pipelines: Partial<Record<WebGpuKernelName, BuiltPipeline>> = {};
  #lostReason: string | undefined;

  private constructor(
    device: GPUDevice,
    capability: WebGpuRuntimeCapability,
    shaderSource: ShaderSource,
  ) {
    this.#device = device;
    this.#capability = capability;
    this.#shaderSource = shaderSource;
    // `device.lost` is a promise, not an event: it settles once and stays settled, which is exactly
    // the lifetime rule every later call needs.
    void device.lost.then((info) => {
      this.#lostReason = `${info.reason}${info.message === "" ? "" : `: ${info.message}`}`;
    });
  }

  /**
   * Asks for an adapter, then a device, then checks the measured limits against `requiredLimits`.
   *
   * The first adapter request is `{ powerPreference: "high-performance" }` and the second is the
   * no-argument one: on a laptop with two GPUs the default request is answered by whichever adapter
   * the browser prefers, which is how a discrete GPU ends up idle, and a page cannot tell after the
   * fact that it asked for the wrong one.
   */
  static async create(options: WebGpuRuntimeOptions): Promise<WebGpuRuntime> {
    const acquisition = await acquireAdapter(gpuEntryPoint(), options.timeoutMs);
    const capability = acquisition.capability;
    if (capability.available !== true || acquisition.adapter === undefined) {
      throw new GpuUnavailableError("gpu.create", capability.reason ?? "no adapter reported");
    }
    const adapter_info = capability.adapterInfo ?? {};
    const adapter_kind = capability.adapter ?? "hardware";
    const software = adapter_kind === "software";
    if (software && options.allowSoftwareAdapter !== true) {
      throw new GpuSoftwareAdapterError(
        "gpu.create",
        adapter_info,
        adapter_info.isFallbackAdapter ?? false,
        { context: { architecture: adapter_info.architecture, vendor: adapter_info.vendor } },
      );
    }

    const limits = capability.limits;
    if (limits === undefined) {
      throw new GpuUnavailableError(
        "gpu.create",
        "the adapter reported no limits, so nothing about its capacity can be checked",
      );
    }
    if (options.requiredLimits !== undefined) {
      requireLimits(limits, options.requiredLimits, "gpu.create");
    }

    const adapter = acquisition.adapter as GPUAdapter;
    // A device starts at the specification's defaults, not at the adapter's capability, and some
    // kernels need the raised ones: workgroup storage stays at 16384 bytes unless the request asks
    // for more, while the convolution kernel stages 18432 bytes of decoded 3x3 weights. Only `max`
    // limits are requested. They are the ones an adapter can hand over; a `min` limit requested
    // below what the adapter reports is an invalid request rather than a stricter device.
    // `requireLimits` above has already reported any shortfall, so this request is the declared
    // value clamped to what the adapter offers.
    const declared = (options.requiredLimits ?? {}) as unknown as Record<string, number | undefined>;
    const reported = limits as unknown as Record<string, number | undefined>;
    const required: Record<string, number> = {};
    for (const [name, value] of Object.entries(declared)) {
      const available = reported[name];
      if (name.startsWith("max") && typeof value === "number" && typeof available === "number") {
        required[name] = Math.min(value, available);
      }
    }

    let device: GPUDevice;
    try {
      device = await adapter.requestDevice(
        Object.keys(required).length === 0 ? {} : { requiredLimits: required },
      );
    } catch (error) {
      throw new GpuUnavailableError(
        "gpu.create",
        `requestDevice() failed: ${error instanceof Error ? error.message : String(error)}`,
        { cause: error },
      );
    }
    const features = typeof device.features === "undefined" ? [] : [...device.features];
    return new WebGpuRuntime(
      device,
      {
        adapterRequest: capability.adapterRequest ?? "default",
        adapterKind: adapter_kind,
        adapterInfo: adapter_info,
        ...(capability.alternativeAdapterInfo === undefined
          ? {}
          : { alternativeAdapterInfo: capability.alternativeAdapterInfo }),
        limits,
        deviceFeatures: features,
      },
      options.shaderSource,
    );
  }

  get capability(): WebGpuRuntimeCapability {
    return this.#capability;
  }

  /** The layout numbers the upload planner works from, taken from the measured limits. */
  get uploadLimits(): WebGpuUploadPlanLimits {
    const alignment = this.#capability.limits.minStorageBufferOffsetAlignment;
    return {
      maxBufferSizeBytes: this.#capability.limits.maxBufferSize,
      maxStorageBufferBindingSizeBytes: this.#capability.limits.maxStorageBufferBindingSize,
      alignmentBytes: alignment >= 1 ? alignment : DEFAULT_ALIGNMENT_BYTES,
    };
  }

  /** Throws `GpuLimitsError` when the measured limits cannot host `requirements`. */
  checkRequirements(requirements: WebGpuRequirements, operation = "gpu.requirements"): void {
    this.#throwIfLost(operation);
    requireLimits(this.#capability.limits, requirements, operation);
  }

  /**
   * Plans an upload of `shardLengthsBytes`, or throws.
   *
   * The plan is returned rather than acted on: a caller that has just been told its manifest needs
   * nine buffers of 512 MiB can decide not to allocate them, which is not a decision this layer can
   * make on its behalf.
   */
  planUpload(shardLengthsBytes: readonly number[], operation = "gpu.planUpload"): WebGpuUploadPlan {
    this.#throwIfLost(operation);
    return planUpload(shardLengthsBytes, this.uploadLimits, operation);
  }

  /** Uploads bytes into a storage buffer. */
  uploadBytes(bytes: ArrayBufferView, label: string): GPUBuffer {
    const size_bytes = alignTo4(bytes.byteLength);
    requireLimits(
      this.#capability.limits,
      { bufferBytes: size_bytes, storageBindingBytes: size_bytes },
      `gpu.upload(${label})`,
    );
    this.#throwIfLost(`gpu.upload(${label})`);
    const buffer = this.#device.createBuffer({
      label,
      size: size_bytes,
      usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC,
    });
    this.#device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  }

  /** A storage buffer for a kernel to write into, sized in bytes. */
  createOutputBuffer(sizeBytes: number, label: string): GPUBuffer {
    requireLimits(
      this.#capability.limits,
      { bufferBytes: sizeBytes },
      `gpu.output(${label})`,
    );
    this.#throwIfLost(`gpu.output(${label})`);
    return this.#device.createBuffer({
      label,
      size: alignTo4(sizeBytes),
      usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC,
    });
  }

  /** A uniform buffer from an already-packed block. */
  createUniformBuffer(bytes: ArrayBufferView, label: string): GPUBuffer {
    requireLimits(
      this.#capability.limits,
      { bufferBytes: bytes.byteLength, uniformBindingBytes: bytes.byteLength },
      `gpu.uniform(${label})`,
    );
    this.#throwIfLost(`gpu.uniform(${label})`);
    const buffer = this.#device.createBuffer({
      label,
      size: alignTo4(bytes.byteLength),
      usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
    });
    this.#device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  }

  /** Reads `elementCount` f32 values back from a buffer that has `COPY_SRC`. */
  async readFloats(
    buffer: GPUBuffer,
    elementCount: number,
    offsetBytes = 0,
  ): Promise<Float32Array> {
    this.#throwIfLost("gpu.read");
    const byte_length = elementCount * 4;
    const staging = this.#device.createBuffer({
      size: alignTo4(byte_length),
      usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
    });
    const encoder = this.#device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, offsetBytes, staging, 0, byte_length);
    this.#device.queue.submit([encoder.finish()]);
    await staging.mapAsync(MAP_MODE.READ);
    // Copied out of the mapped range before unmapping: the view is invalid the moment the mapping
    // ends, and a caller keeping a reference to it would read freed memory.
    const values = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return values;
  }

  /** Builds the pipeline for one kernel, once, with the layout its shader declares. */
  async pipeline(name: WebGpuKernelName): Promise<BuiltPipeline> {
    this.#throwIfLost(`gpu.pipeline(${name})`);
    const cached = this.#pipelines[name];
    if (cached !== undefined) return cached;
    const kernel = WEBGPU_KERNELS[name];
    requireLimits(
      this.#capability.limits,
      {
        workgroupStorageBytes: kernel.workgroupStorageBytes,
        invocationsPerWorkgroup:
          kernel.workgroupSize[0] * kernel.workgroupSize[1] * kernel.workgroupSize[2],
      },
      `gpu.pipeline(${name})`,
      { kernel: name },
    );

    const source = await this.#shaderSource(kernel.file);
    const module = this.#device.createShaderModule({ code: source, label: kernel.file });
    const errors = await shaderDiagnostics(module);
    if (errors.length > 0) throw new GpuShaderError(`gpu.pipeline(${name})`, kernel.file, errors);

    const bind_group_layout = this.#device.createBindGroupLayout({
      label: name,
      entries: bindGroupLayoutEntries(kernel),
    });
    const pipeline = this.#device.createComputePipeline({
      label: name,
      layout: this.#device.createPipelineLayout({ bindGroupLayouts: [bind_group_layout] }),
      compute: { module, entryPoint: kernel.entryPoint, constants: {} },
    });
    const built: BuiltPipeline = { pipeline, bindGroupLayout: bind_group_layout };
    this.#pipelines[name] = built;
    return built;
  }

  /**
   * Runs one kernel over `buffers`, which must be in the catalogue's binding order.
   *
   * The dispatch count is the caller's: only the kernel's own geometry rules say how many workgroups
   * a shape needs, and those rules live next to the shape, not here.
   */
  async dispatch(
    name: WebGpuKernelName,
    buffers: readonly GPUBuffer[],
    workgroupCounts: readonly [number, number, number],
  ): Promise<WebGpuDispatchGeometry> {
    this.#throwIfLost(`gpu.dispatch(${name})`);
    const kernel: WebGpuKernelDescriptor = WEBGPU_KERNELS[name];
    if (buffers.length !== kernel.bindings.length) {
      throw new GpuShaderError(
        `gpu.dispatch(${name})`,
        kernel.file,
        [`${buffers.length} buffers were bound but ${kernel.bindings.length} bindings are declared`],
      );
    }
    const built = await this.pipeline(name);
    const entries = buffers.map((buffer, index) => ({ binding: index, resource: { buffer } }));
    const bind_group = this.#device.createBindGroup({
      label: name,
      layout: built.bindGroupLayout,
      entries,
    });
    const encoder = this.#device.createCommandEncoder({ label: name });
    const pass = encoder.beginComputePass({ label: name });
    pass.setPipeline(built.pipeline);
    pass.setBindGroup(0, bind_group);
    pass.dispatchWorkgroups(...workgroupCounts);
    pass.end();
    this.#device.queue.submit([encoder.finish()]);
    return { workgroupSize: kernel.workgroupSize, workgroupCounts };
  }

  destroy(): void {
    this.#pipelines = {};
    this.#device.destroy();
  }

  #throwIfLost(operation: string): void {
    if (this.#lostReason !== undefined) throw new GpuDeviceError(operation, this.#lostReason);
  }
}
