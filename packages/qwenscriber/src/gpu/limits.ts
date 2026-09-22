//! What a request needs from an adapter, and whether the adapter has it.
//!
//! The comparison is made against the measured limits of the adapter this page actually got, never
//! against a table of "typical" numbers and never against a boolean. A caller that wants to know
//! whether a 2 GiB shard can be uploaded asks a question about numbers, and the answer it gets back
//! -- either silence or a `GpuLimitsError` carrying both sides of the comparison -- is the same
//! question answered.

import type { WebGpuLimits } from "../capabilities.ts";
import { GpuLimitsError, type GpuLimitShortfall } from "./errors.ts";

/**
 * Bytes and counts an artifact needs from the adapter.
 *
 * Every field is optional: a caller that only cares about binding size says so by not mentioning
 * buffers, and the check does not invent a requirement it was not given.
 */
export interface WebGpuRequirements {
  /** The largest single `GPUBuffer` the artifact needs. Defaults to `maxBufferSize`. */
  readonly bufferBytes?: number | undefined;
  /** The largest buffer bound to one shader as a storage binding. */
  readonly storageBindingBytes?: number | undefined;
  /** The largest buffer bound to one shader as a uniform binding. */
  readonly uniformBindingBytes?: number | undefined;
  /** Workgroup memory the kernel declares (all kernels here use at most 2 KiB). */
  readonly workgroupStorageBytes?: number | undefined;
  /** Invocations per workgroup the kernel dispatches with. */
  readonly invocationsPerWorkgroup?: number | undefined;
}

/**
 * Each requirement, the adapter limit it is measured against, and the name WebGPU spells that
 * limit with. Written out rather than derived by reflection: the ABI-facing names are part of what
 * a caller reads in an error, and a typo here would silently drop a check.
 */
const REQUIREMENT_CHECKS = [
  { requirement: "bufferBytes", limit: "maxBufferSize" },
  { requirement: "storageBindingBytes", limit: "maxStorageBufferBindingSize" },
  { requirement: "uniformBindingBytes", limit: "maxUniformBufferBindingSize" },
  { requirement: "workgroupStorageBytes", limit: "maxComputeWorkgroupStorageSize" },
  { requirement: "invocationsPerWorkgroup", limit: "maxComputeInvocationsPerWorkgroup" },
] as const satisfies readonly {
  readonly requirement: keyof WebGpuRequirements;
  readonly limit: keyof WebGpuLimits;
}[];

/** The measured limits as a plain record, which is the shape an error report carries. */
export function measuredLimits(limits: WebGpuLimits): Readonly<Record<string, number>> {
  const measured: Record<string, number> = {};
  for (const check of REQUIREMENT_CHECKS) measured[check.limit] = limits[check.limit];
  measured.minStorageBufferOffsetAlignment = limits.minStorageBufferOffsetAlignment;
  return measured;
}

/**
 * Every requirement the adapter does not satisfy, as pairs of numbers.
 *
 * Empty means the artifact fits. The caller decides what an empty list means; `requireLimits` turns
 * a non-empty one into the typed error.
 */
export function shortfallsFor(
  limits: WebGpuLimits,
  requirements: WebGpuRequirements,
): GpuLimitShortfall[] {
  const shortfalls: GpuLimitShortfall[] = [];
  for (const check of REQUIREMENT_CHECKS) {
    const needed = requirements[check.requirement];
    if (needed === undefined) continue;
    const measured = limits[check.limit];
    if (needed > measured) shortfalls.push({ limit: check.limit, needed, measured });
  }
  return shortfalls;
}

/** Throws a `GpuLimitsError` naming the artifact when any requirement exceeds its limit. */
export function requireLimits(
  limits: WebGpuLimits,
  requirements: WebGpuRequirements,
  operation: string,
  context: Readonly<Record<string, unknown>> = {},
): void {
  const shortfalls = shortfallsFor(limits, requirements);
  if (shortfalls.length === 0) return;
  throw new GpuLimitsError(operation, shortfalls, measuredLimits(limits), { context });
}
