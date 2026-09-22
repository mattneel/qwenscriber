//! Feature probes: what this browser can actually do, decided without assuming a browser.
//!
//! A probe never throws. Diagnosing an environment is exactly when the environment is odd, so every
//! failure mode -- no `navigator`, no `navigator.gpu`, no adapter, a `requestAdapter()` that rejects
//! or never settles -- comes back as `available: false` with a `reason` string that says which one
//! happened. A caller can print the reason; it cannot print a rejected promise.
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
  readonly adapterInfo?: WebGpuAdapterInfo | undefined;
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
  return {
    maxBufferSize: max_buffer_size,
    maxStorageBufferBindingSize:
      typeof max_binding_size === "number" ? max_binding_size : 0,
    maxComputeWorkgroupStorageSize:
      typeof max_workgroup_storage === "number" ? max_workgroup_storage : 0,
    maxComputeInvocationsPerWorkgroup:
      typeof max_invocations === "number" ? max_invocations : 0,
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
  const request_adapter = field(gpu, "requestAdapter");
  if (typeof request_adapter !== "function") {
    return {
      available: false,
      reason: "navigator.gpu is unavailable: the browser has no WebGPU entry point",
    };
  }
  let adapter: unknown;
  try {
    const pending = (request_adapter as () => Promise<unknown>).call(gpu) as Promise<unknown>;
    adapter = await settleWithin(pending, timeoutMs);
  } catch (error) {
    return {
      available: false,
      reason: `requestAdapter() failed: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
  if (adapter === undefined) {
    return { available: false, reason: `requestAdapter() did not settle within ${timeoutMs} ms` };
  }
  if (adapter === null) {
    return {
      available: false,
      reason: "requestAdapter() returned null: no compatible adapter, or WebGPU is disabled",
    };
  }
  const info = adapterInfoOf(adapter);
  return {
    available: true,
    adapter: adapterKindOf(adapter, info),
    adapterInfo: info,
    limits: adapterLimitsOf(adapter),
  };
}

/** Reads globalThis as a record, so no probe depends on a window, a document, or a DOM type. */
function globalField(key: string): unknown {
  return (globalThis as unknown as Record<string, unknown>)[key];
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
  const webgpu = await probeWebGpu(field(globalField("navigator"), "gpu"));
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
