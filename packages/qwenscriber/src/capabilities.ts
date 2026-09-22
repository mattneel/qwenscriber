//! Feature probes: what this browser can actually do, decided without assuming a browser.
//!
//! A probe never throws. Diagnosing an environment is exactly when the environment is odd, so every
//! failure mode -- no `navigator`, no `navigator.gpu`, no adapter, a `requestAdapter()` that rejects
//! or never settles -- comes back as `available: false` with a `reason` string that says which one
//! happened. A caller can print the reason; it cannot print a rejected promise.
//!
//! Adapters are asked for twice, high-performance first. A machine with a discrete GPU and an
//! integrated one answers the no-argument request with whichever the browser prefers, which is
//! routinely the integrated part, and a page that asked once cannot tell afterwards that it asked
//! for the wrong thing. `adapterRequest` in the report records which attempt succeeded, so "why is
//! my discrete GPU idle" is a question the page answers about itself.
//!
//! Nothing here touches globals directly: `globalThis` is read through a record-style lookup, so the
//! module behaves identically in a window, a worker, and Node.

/** Milliseconds to wait for `requestAdapter()`. A hung probe is worse than a negative one. */
export const WEBGPU_PROBE_TIMEOUT_MS = 5000;

/**
 * A module that uses SIMD, encoded by hand, so detection needs no toolchain.
 *
 * It declares `() -> v128`, then does `v128.const i32x4 0 0 0 1` -- enough for `WebAssembly.validate`
 * to accept it only where the fixed-width SIMD proposal is implemented. Validation does not compile
 * or instantiate, so this costs a parse and frees immediately. Detection is the point: SIMD is what
 * the core's `@Vector` kernels lower to, and a build without it would be several times slower.
 */
const SIMD_PROBE_MODULE = [
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
  0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00, 0x41, 0x00, 0xfd, 0x0f, 0xfd, 0x62, 0x0b,
];

/**
 * A module with a shared memory and an atomic load.
 *
 * Memory flags `0x03` request a `shared` memory with a maximum, and `i32.atomic.load` is the
 * threads proposal's opcode. A runtime without shared-memory support rejects the memory section
 * during validation. Browsers additionally gate *instantiating* shared memory behind
 * cross-origin isolation; `capabilities().crossOriginIsolated` reports that separately, so a
 * `wasmThreads: true` with `crossOriginIsolated: false` is a real combination rather than a bug.
 */
const THREADS_PROBE_MODULE = [
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x04, 0x01, 0x60, 0x00, 0x00, 0x03, 0x02,
  0x01, 0x00, 0x05, 0x04, 0x01, 0x03, 0x01, 0x01, 0x0a, 0x0b, 0x01, 0x09, 0x00, 0x41, 0x00, 0xfe,
  0x10, 0x02, 0x00, 0x1a, 0x0b,
];

/** Renderers that mean "a GPU API exists but no GPU is behind it". */
const SOFTWARE_ADAPTER_MARKERS = [
  "swiftshader",
  "llvmpipe",
  "lavapipe",
  "basic render",
  "software",
];

export interface WebGpuLimits {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxComputeWorkgroupStorageSize: number;
  readonly maxComputeInvocationsPerWorkgroup: number;
  /**
   * `minStorageBufferOffsetAlignment`: every binding offset is a multiple of this.
   *
   * Measured, or the WebGPU default of 256 when an adapter does not report it, because an upload
   * planner that assumed 1 would produce offsets the browser rejects.
   */
  readonly minStorageBufferOffsetAlignment: number;
  /**
   * `maxUniformBufferBindingSize`.
   *
   * Measured, or the WebGPU default of 65536 when an adapter does not report it. Every uniform block
   * here is 16 or 32 bytes, so this limit exists to be *reported*, not to be approached.
   */
  readonly maxUniformBufferBindingSize: number;
}

export interface WebGpuAdapterInfo {
  readonly vendor?: string | undefined;
  readonly architecture?: string | undefined;
  readonly device?: string | undefined;
  readonly description?: string | undefined;
  /** `adapter.isFallbackAdapter`: the browser's own "this is not a real GPU" flag. */
  readonly isFallbackAdapter?: boolean | undefined;
}

export interface WebGpuCapability {
  readonly available: boolean;
  /** `"software"` covers SwiftShader, llvmpipe, and WARP: usable, but not a benchmark target. */
  readonly adapter?: "hardware" | "software" | undefined;
  /**
   * Which request produced the adapter.
   *
   * `"high-performance"` is the one that asks for a discrete GPU; `"default"` means the adapter came
   * from the fallback request, which is worth knowing when the machine has two GPUs. Absent when
   * there is no adapter at all.
   */
  readonly adapterRequest?: "high-performance" | "default" | undefined;
  readonly adapterInfo?: WebGpuAdapterInfo | undefined;
  /**
   * The identity the *other* request answered with, when it was a different adapter.
   *
   * A page can only ever see the adapters the browser offers it, and on a machine with two GPUs the
   * two requests may answer with different ones. When they do, this is the only way the page can
   * show the second GPU at all -- which is what makes "my discrete GPU is idle" answerable rather
   * than mysterious. Absent when both requests produced the same adapter (the common case) or when
   * only one of them produced anything.
   */
  readonly alternativeAdapterInfo?: WebGpuAdapterInfo | undefined;
  readonly limits?: WebGpuLimits | undefined;
  /** Why WebGPU is unavailable. Absent when `available` is true. */
  readonly reason?: string | undefined;
}

export interface Capabilities {
  readonly webgpu: WebGpuCapability;
  /** Fixed-width SIMD (`v128`) validates, so the core's vector kernels lower natively. */
  readonly wasmSimd: boolean;
  /** A shared-memory + atomics module validates. Needed for a threaded core. */
  readonly wasmThreads: boolean;
  /** `SharedArrayBuffer` exists. Streaming may use it; the SDK never requires it. */
  readonly sharedArrayBuffer: boolean;
  /** `Worker` exists, so the core can run off the main thread. */
  readonly workers: boolean;
  /** The page is cross-origin isolated, which is what makes `SharedArrayBuffer` usable. */
  readonly crossOriginIsolated: boolean;
  /** The adapter class, or `"none"`: the short answer to "is there a GPU here". */
  readonly webgpuAdapter: "none" | "hardware" | "software";
}

/** Reads one field of an unknown value, or `undefined` when it is not an object. */
function field(source: unknown, key: string): unknown {
  if (typeof source !== "object" || source === null) return undefined;
  return (source as Record<string, unknown>)[key];
}

function validateInlineModule(bytes: readonly number[]): boolean {
  try {
    return WebAssembly.validate(new Uint8Array(bytes));
  } catch {
    // `WebAssembly.validate` is allowed to throw on a malformed prefix; the only honest answer then
    // is "this engine does not support it".
    return false;
  }
}

function adapterLimitsOf(adapter: unknown): WebGpuLimits | undefined {
  const limits = field(adapter, "limits");
  const max_buffer_size = field(limits, "maxBufferSize");
  if (typeof max_buffer_size !== "number") return undefined;
  const max_binding_size = field(limits, "maxStorageBufferBindingSize");
  const max_workgroup_storage = field(limits, "maxComputeWorkgroupStorageSize");
  const max_invocations = field(limits, "maxComputeInvocationsPerWorkgroup");
  const offset_alignment = field(limits, "minStorageBufferOffsetAlignment");
  const uniform_binding = field(limits, "maxUniformBufferBindingSize");
  return {
    maxBufferSize: max_buffer_size,
    maxStorageBufferBindingSize:
      typeof max_binding_size === "number" ? max_binding_size : 0,
    maxComputeWorkgroupStorageSize:
      typeof max_workgroup_storage === "number" ? max_workgroup_storage : 0,
    maxComputeInvocationsPerWorkgroup:
      typeof max_invocations === "number" ? max_invocations : 0,
    // The WebGPU default rather than 0: an off-by-default here would plan unaligned uploads, which
    // is a failure the browser reports far away from the number that caused it.
    minStorageBufferOffsetAlignment:
      typeof offset_alignment === "number" && offset_alignment >= 1 ? offset_alignment : 256,
    maxUniformBufferBindingSize:
      typeof uniform_binding === "number" && uniform_binding >= 1 ? uniform_binding : 65536,
  };
}

function adapterInfoOf(adapter: unknown): WebGpuAdapterInfo | undefined {
  const info = field(adapter, "info");
  if (typeof info !== "object" || info === null) return undefined;
  const vendor = field(info, "vendor");
  const architecture = field(info, "architecture");
  const device = field(info, "device");
  const description = field(info, "description");
  const is_fallback = field(adapter, "isFallbackAdapter");
  return {
    vendor: typeof vendor === "string" ? vendor : undefined,
    architecture: typeof architecture === "string" ? architecture : undefined,
    device: typeof device === "string" ? device : undefined,
    description: typeof description === "string" ? description : undefined,
    isFallbackAdapter: is_fallback === true,
  };
}

function adapterKindOf(adapter: unknown, info: WebGpuAdapterInfo | undefined): "hardware" | "software" {
  if (field(adapter, "isFallbackAdapter") === true) return "software";
  const haystack = `${info?.architecture ?? ""} ${info?.device ?? ""} ${info?.description ?? ""}`;
  const lowered = haystack.toLowerCase();
  for (const marker of SOFTWARE_ADAPTER_MARKERS) {
    if (lowered.includes(marker)) return "software";
  }
  return "hardware";
}

/** Resolves to `undefined` when `promise` has not settled within `timeoutMs`. */
async function settleWithin(promise: Promise<unknown>, timeoutMs: number): Promise<unknown> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  // The executor form rather than `Promise.withResolvers`, which is ES2024: `capabilities()` runs on
  // every `create()`, and the package declares Node 20, where `withResolvers` does not exist.
  const expiry = new Promise<undefined>((resolve) => {
    timer = setTimeout(() => resolve(undefined), timeoutMs);
  });
  try {
    return await Promise.race([promise, expiry]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

/**
 * The two adapter requests, in the order that finds a discrete GPU first.
 *
 * The arguments are written out per attempt rather than derived, because
 * `{ powerPreference: "high-performance" }` is the whole point of the first one and inlining it
 * here is what makes that visible.
 */
const ADAPTER_REQUESTS = [
  { preference: "high-performance", options: { powerPreference: "high-performance" } },
  { preference: "default", options: undefined },
] as const;

/** An adapter plus the report describing it. `adapter` is present exactly when `available` is true. */
export interface AdapterAcquisition {
  readonly capability: WebGpuCapability;
  readonly adapter?: unknown;
}

/** True when two adapter identities describe the same GPU, so the report can skip the duplicate. */
function sameAdapter(
  left: WebGpuAdapterInfo | undefined,
  right: WebGpuAdapterInfo | undefined,
): boolean {
  if (left === undefined || right === undefined) return true;
  return (
    left.vendor === right.vendor &&
    left.architecture === right.architecture &&
    left.device === right.device &&
    left.description === right.description &&
    (left.isFallbackAdapter ?? false) === (right.isFallbackAdapter ?? false)
  );
}

/**
 * Asks for an adapter, high-performance first, and describes what came back.
 *
 * Both requests are made, not just the first one that succeeds: an adapter the page cannot see is an
 * adapter it cannot report, and on a two-GPU machine the two requests may be answered by different
 * parts. The high-performance answer wins; the other one is kept only as `alternativeAdapterInfo`,
 * and only when it is genuinely a different GPU.
 *
 * The probe's own reason strings are produced here, so a failed acquisition and a failed probe
 * explain themselves with the same words. The caller that needs the adapter object (to request a
 * device) reads `adapter`; the caller that only needs to know whether there is a GPU reads
 * `capability`.
 */
export async function acquireAdapter(
  gpu: unknown,
  timeoutMs: number = WEBGPU_PROBE_TIMEOUT_MS,
): Promise<AdapterAcquisition> {
  const request_adapter = field(gpu, "requestAdapter");
  if (typeof request_adapter !== "function") {
    return {
      capability: {
        available: false,
        reason: "navigator.gpu is unavailable: the browser has no WebGPU entry point",
      },
    };
  }

  let failure: string | undefined;
  const answered: { preference: "high-performance" | "default"; adapter: unknown }[] = [];
  for (const request of ADAPTER_REQUESTS) {
    let adapter: unknown;
    try {
      const call = request.options === undefined
        ? (request_adapter as () => Promise<unknown>).call(gpu)
        : (request_adapter as (options: unknown) => Promise<unknown>).call(gpu, request.options);
      adapter = await settleWithin(call as Promise<unknown>, timeoutMs);
    } catch (error) {
      failure =
        `requestAdapter(${request.preference}) failed: ` +
        `${error instanceof Error ? error.message : String(error)}`;
      continue;
    }
    if (adapter === undefined) {
      failure = `requestAdapter(${request.preference}) did not settle within ${timeoutMs} ms`;
      continue;
    }
    if (adapter === null) {
      failure = "requestAdapter() returned null: no compatible adapter, or WebGPU is disabled";
      continue;
    }
    answered.push({ preference: request.preference, adapter });
  }

  const preferred = answered.find((entry) => entry.preference === "high-performance") ?? answered[0];
  if (preferred === undefined) {
    return {
      capability: {
        available: false,
        reason: failure ?? "requestAdapter() returned no adapter and gave no reason",
      },
    };
  }
  const info = adapterInfoOf(preferred.adapter);
  const other = answered.length > 1 && preferred === answered[0] ? answered[1] : undefined;
  const other_info = other === undefined ? undefined : adapterInfoOf(other.adapter);
  return {
    adapter: preferred.adapter,
    capability: {
      available: true,
      adapter: adapterKindOf(preferred.adapter, info),
      adapterRequest: preferred.preference,
      adapterInfo: info,
      ...(sameAdapter(info, other_info) ? {} : { alternativeAdapterInfo: other_info }),
      limits: adapterLimitsOf(preferred.adapter),
    },
  };
}

/**
 * Probes a `navigator.gpu`-shaped object.
 *
 * Exported as a seam: it takes the object rather than reaching for a global, so it can be exercised
 * against a fake that returns null, rejects, or exposes partial limits -- the cases that make
 * environment detection worth writing down.
 */
export async function probeWebGpu(
  gpu: unknown,
  timeoutMs: number = WEBGPU_PROBE_TIMEOUT_MS,
): Promise<WebGpuCapability> {
  return (await acquireAdapter(gpu, timeoutMs)).capability;
}

/** Reads globalThis as a record, so no probe depends on a window, a document, or a DOM type. */
function globalField(key: string): unknown {
  return (globalThis as unknown as Record<string, unknown>)[key];
}

/**
 * `navigator.gpu`, or `undefined` where there is no navigator at all.
 *
 * Exported because the runtime needs the same entry point the probe used, and reaching for it a
 * second way -- a cast here, a lookup there -- is how the two drift apart.
 */
export function gpuEntryPoint(): unknown {
  return field(globalField("navigator"), "gpu");
}

/**
 * Everything the SDK wants to know about its host, in one call.
 *
 * Cheap enough to call on every page load: it validates two tiny modules and asks for an adapter,
 * and caches nothing, so a second call after the user granted or lost a GPU reports the truth.
 */
export async function capabilities(): Promise<Capabilities> {
  const cross_origin_isolated = globalField("crossOriginIsolated") === true;
  const shared_array_buffer = typeof SharedArrayBuffer === "function";
  const wasm_threads = shared_array_buffer && validateInlineModule(THREADS_PROBE_MODULE);
  const webgpu = await probeWebGpu(gpuEntryPoint());
  return {
    webgpu,
    wasmSimd: validateInlineModule(SIMD_PROBE_MODULE),
    wasmThreads: wasm_threads,
    sharedArrayBuffer: shared_array_buffer,
    workers: typeof Worker === "function",
    crossOriginIsolated: cross_origin_isolated,
    webgpuAdapter: webgpu.available ? (webgpu.adapter ?? "hardware") : "none",
  };
}
