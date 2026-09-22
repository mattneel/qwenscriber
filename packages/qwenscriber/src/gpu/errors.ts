//! Typed errors for the WebGPU path.
//!
//! They extend the SDK's single error type rather than introducing a second one, and they reuse the
//! ABI's own status codes rather than starting a new numeric block: `statusLabel` knows the ABI's
//! `-1 .. -14` and the SDK's `-1000 .. -1003`, so a code outside both would report
//! `status: "unknown"` and throw away the machine-readable name a caller branches on. What a GPU
//! failure means therefore lives in the class, in `operation`, and in `context`, which carries the
//! numbers that made it fail.

import { QwenscriberError, type QwenscriberErrorOptions } from "../errors.ts";
import { STATUS } from "../wasm/abi.ts";
import type { WebGpuAdapterInfo } from "../capabilities.ts";

/** Why a measured limit cannot host a request: the number asked for, and the number the adapter has. */
export interface GpuLimitShortfall {
  /** The limit's name as WebGPU spells it: `"maxBufferSize"`. */
  readonly limit: string;
  readonly needed: number;
  readonly measured: number;
}

/** Base class for every GPU failure, so "was this the accelerator?" is one `instanceof` test. */
export class GpuError extends QwenscriberError {}

/**
 * No `navigator.gpu`, no adapter, or no device.
 *
 * The reason is the same string the capability probe reports, so a page can print one message for
 * both the probe and the failed attempt.
 */
export class GpuUnavailableError extends GpuError {
  readonly reason: string;

  constructor(operation: string, reason: string, options: QwenscriberErrorOptions = {}) {
    super(STATUS.unsupported, operation, {
      message: `no usable WebGPU device for ${operation}: ${reason}`,
      context: { ...options.context, reason },
      ...(options.cause === undefined ? {} : { cause: options.cause }),
    });
    this.name = "GpuUnavailableError";
    this.reason = reason;
  }
}

/**
 * The adapter cannot host the requested artifact: a buffer, a binding, a workgroup, or a shard that
 * has nowhere to go.
 *
 * `shortfalls` and `limits` carry the measured numbers, because "the GPU is too small" is not
 * actionable and "maxStorageBufferBindingSize is 134217728, the artifact needs 268435456" is.
 */
export class GpuLimitsError extends GpuError {
  readonly shortfalls: readonly GpuLimitShortfall[];
  readonly limits: Readonly<Record<string, number>>;

  constructor(
    operation: string,
    shortfalls: readonly GpuLimitShortfall[],
    limits: Readonly<Record<string, number>>,
    options: QwenscriberErrorOptions = {},
  ) {
    const text = shortfalls
      .map((entry) => `${entry.limit} is ${entry.measured}, ${entry.needed} is required`)
      .join("; ");
    super(STATUS.limit_exceeded, operation, {
      message: `${operation}: the adapter's limits are too small (${text})`,
      context: { ...options.context, shortfalls, limits },
      ...(options.cause === undefined ? {} : { cause: options.cause }),
    });
    this.name = "GpuLimitsError";
    this.shortfalls = shortfalls;
    this.limits = limits;
  }
}

/**
 * The only adapter is a software or fallback one.
 *
 * SwiftShader, llvmpipe, and WARP produce correct numbers and unusable latency, so running the
 * decoder on them is a silent performance failure rather than a correctness one. The caller has to
 * say it means it: `allowSoftwareAdapter: true`.
 */
export class GpuSoftwareAdapterError extends GpuError {
  readonly adapterInfo: WebGpuAdapterInfo;
  readonly isFallbackAdapter: boolean;

  constructor(
    operation: string,
    adapterInfo: WebGpuAdapterInfo,
    isFallbackAdapter: boolean,
    options: QwenscriberErrorOptions = {},
  ) {
    const identity =
      `vendor=${adapterInfo.vendor ?? "?"} architecture=${adapterInfo.architecture ?? "?"} ` +
      `isFallbackAdapter=${String(isFallbackAdapter)}`;
    super(STATUS.unsupported, operation, {
      message:
        `${operation}: the only available adapter is a software adapter (${identity}); ` +
        `pass allowSoftwareAdapter to run anyway`,
      context: { ...options.context, adapterInfo, isFallbackAdapter },
      ...(options.cause === undefined ? {} : { cause: options.cause }),
    });
    this.name = "GpuSoftwareAdapterError";
    this.adapterInfo = adapterInfo;
    this.isFallbackAdapter = isFallbackAdapter;
  }
}

/** A WGSL module did not compile. The diagnostics are the driver's own message text. */
export class GpuShaderError extends GpuError {
  readonly shaderFile: string;
  readonly diagnostics: readonly string[];

  constructor(
    operation: string,
    shaderFile: string,
    diagnostics: readonly string[],
    options: QwenscriberErrorOptions = {},
  ) {
    super(STATUS.invalid_argument, operation, {
      message: `${operation}: ${shaderFile} did not compile (${diagnostics.join(" | ")})`,
      context: { ...options.context, shaderFile, diagnostics },
      ...(options.cause === undefined ? {} : { cause: options.cause }),
    });
    this.name = "GpuShaderError";
    this.shaderFile = shaderFile;
    this.diagnostics = diagnostics;
  }
}

/** The device was lost, or the driver refused a command. Every later call on the runtime fails. */
export class GpuDeviceError extends GpuError {
  readonly reason: string;

  constructor(operation: string, reason: string, options: QwenscriberErrorOptions = {}) {
    super(STATUS.invalid_state, operation, {
      message: `${operation}: the WebGPU device is unusable (${reason})`,
      context: { ...options.context, reason },
      ...(options.cause === undefined ? {} : { cause: options.cause }),
    });
    this.name = "GpuDeviceError";
    this.reason = reason;
  }
}

/** True when `value` is one of this module's errors, for a caller that only cares "the GPU failed". */
export function isGpuError(value: unknown): value is GpuError {
  return value instanceof GpuError;
}
