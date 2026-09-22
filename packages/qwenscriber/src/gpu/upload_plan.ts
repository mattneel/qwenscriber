//! Upload planning: which shard goes into which `GPUBuffer`, at which offset.
//!
//! A model manifest describes its weights as a list of shard byte lengths. An adapter will not take
//! them all at once: a single buffer is capped by `maxBufferSize`, a single *binding* is capped by
//! `maxStorageBufferBindingSize`, and a binding's offset must be a multiple of
//! `minStorageBufferOffsetAlignment`. A loader that discovers those caps by having a write fail has
//! already lost the information it needed, so the plan is computed first, from the adapter's own
//! numbers, and it either fits or it raises `GpuLimitsError`.
//!
//! Nothing here truncates. A shard too large for one part is split into aligned parts, and a part
//! that cannot exist at all -- because the smaller of the two caps, rounded down to the alignment,
//! is zero -- is a rejection, not a smaller load.
//!
//! The plan is pure data: it never touches a `GPUDevice`, so the decision is testable with invented
//! limit sets (see `test/gpu_upload_plan.test.ts`), which is the only way to exercise a 2 GiB cap on
//! a machine whose adapter has other numbers.

import { QwenscriberError } from "../errors.ts";
import { STATUS } from "../wasm/abi.ts";
import { GpuLimitsError, type GpuLimitShortfall } from "./errors.ts";

/** The adapter numbers a plan is built from, in bytes. */
export interface WebGpuUploadPlanLimits {
  readonly maxBufferSizeBytes: number;
  readonly maxStorageBufferBindingSizeBytes: number;
  /** `minStorageBufferOffsetAlignment`: every placement offset is a multiple of this. */
  readonly alignmentBytes: number;
}

/** Where one contiguous run of one shard lives: a buffer index, an offset, a length. */
export interface WebGpuShardPlacement {
  readonly buffer: number;
  readonly offsetBytes: number;
  readonly lengthBytes: number;
}

/** One manifest shard, and the placements holding its bytes in order. */
export interface WebGpuShardPlan {
  /** Index in the manifest this plan was built from. */
  readonly index: number;
  /** Offset of this shard in the manifest's concatenated byte stream. */
  readonly offsetBytes: number;
  readonly lengthBytes: number;
  readonly placements: readonly WebGpuShardPlacement[];
}

/** One buffer to allocate, sized to exactly what it holds. */
export interface WebGpuBufferPlan {
  readonly index: number;
  readonly sizeBytes: number;
  /** Manifest shards with at least one placement in this buffer, ascending. */
  readonly shardIndices: readonly number[];
}

export interface WebGpuUploadPlan {
  readonly buffers: readonly WebGpuBufferPlan[];
  readonly shards: readonly WebGpuShardPlan[];
  /** Sum of the manifest's shard lengths: the bytes that have to arrive. */
  readonly payloadBytes: number;
  /** Sum of the buffer sizes: payload plus alignment padding. */
  readonly allocatedBytes: number;
  /** Shards that needed more than one placement, i.e. the splits the limits forced. */
  readonly splitShardCount: number;
  /** Largest single placement this plan may use; `0` only when the plan was rejected. */
  readonly partCapacityBytes: number;
}

/**
 * The largest placement the limits allow: the smaller of the buffer and binding caps, rounded down
 * to the alignment so every part after the first stays aligned.
 *
 * Rounding down matters. A part is both stored in a buffer and bound to a shader, so a size that
 * fits one cap and not the other is no use, and a size that is not a multiple of the alignment makes
 * the *next* part unplaceable.
 */
export function partCapacityOf(limits: WebGpuUploadPlanLimits): number {
  const smaller = Math.min(limits.maxBufferSizeBytes, limits.maxStorageBufferBindingSizeBytes);
  if (!Number.isFinite(smaller) || !Number.isFinite(limits.alignmentBytes)) return 0;
  if (limits.alignmentBytes < 1) return 0;
  return Math.floor(smaller / limits.alignmentBytes) * limits.alignmentBytes;
}

/** Rejects a manifest that is not a list of byte lengths, before any limit is consulted. */
function assertShardLengths(shardLengthsBytes: readonly number[], operation: string): void {
  for (let index = 0; index < shardLengthsBytes.length; index += 1) {
    const length_bytes = shardLengthsBytes[index] ?? 0;
    const plausible = Number.isSafeInteger(length_bytes) && length_bytes >= 0;
    if (!plausible) {
      throw new QwenscriberError(STATUS.invalid_argument, operation, {
        message: `${operation}: shard ${index} is not a byte length`,
        context: { shardIndex: index, lengthBytes: length_bytes },
      });
    }
  }
}

/**
 * Rejects a limit set no placement can satisfy, naming the numbers.
 *
 * This is the "one small set forces rejection" case: with a 128-byte binding cap and a 256-byte
 * alignment there is no offset a 256-byte part could start at, so the honest answer is a refusal
 * rather than an upload that would fail later inside the browser.
 */
function assertPlanLimits(limits: WebGpuUploadPlanLimits, operation: string): void {
  const measured: Record<string, number> = {
    maxBufferSize: limits.maxBufferSizeBytes,
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSizeBytes,
    minStorageBufferOffsetAlignment: limits.alignmentBytes,
  };
  const shortfalls: GpuLimitShortfall[] = [];
  if (!Number.isSafeInteger(limits.alignmentBytes) || limits.alignmentBytes < 1) {
    shortfalls.push({
      limit: "minStorageBufferOffsetAlignment",
      needed: 1,
      measured: limits.alignmentBytes,
    });
  }
  const capacity_bytes = partCapacityOf(limits);
  if (shortfalls.length === 0 && capacity_bytes < 1) {
    shortfalls.push({
      limit: "maxStorageBufferBindingSize",
      needed: limits.alignmentBytes,
      measured: Math.min(limits.maxBufferSizeBytes, limits.maxStorageBufferBindingSizeBytes),
    });
  }
  if (shortfalls.length > 0) throw new GpuLimitsError(operation, shortfalls, measured);
}

/**
 * Plans the upload of `shardLengthsBytes` under `limits`.
 *
 * Greedy first fit: each part goes into the open buffer while it fits, and a part that does not fit
 * opens the next one. Buffers are sized to what they hold rather than to the cap, so the plan also
 * says how much memory the upload really costs -- which is what a caller needs to compare against
 * the device's budget before allocating anything.
 */
export function planUpload(
  shardLengthsBytes: readonly number[],
  limits: WebGpuUploadPlanLimits,
  operation = "gpu.planUpload",
): WebGpuUploadPlan {
  assertShardLengths(shardLengthsBytes, operation);
  assertPlanLimits(limits, operation);

  const alignment_bytes = limits.alignmentBytes;
  const capacity_bytes = partCapacityOf(limits);
  const buffers: WebGpuBufferPlan[] = [];
  const shards: WebGpuShardPlan[] = [];
  let payload_bytes = 0;
  let manifest_offset_bytes = 0;
  let open_index = -1;
  let open_offset_bytes = 0;
  let open_shards: number[] = [];

  for (let index = 0; index < shardLengthsBytes.length; index += 1) {
    const length_bytes = shardLengthsBytes[index] ?? 0;
    const placements: WebGpuShardPlacement[] = [];
    let remaining_bytes = length_bytes;
    while (remaining_bytes > 0) {
      const part_bytes = Math.min(remaining_bytes, capacity_bytes);
      const fits_open = open_index >= 0 && open_offset_bytes + part_bytes <= limits.maxBufferSizeBytes;
      if (!fits_open) {
        if (open_index >= 0) {
          buffers.push({
            index: open_index,
            sizeBytes: open_offset_bytes,
            shardIndices: open_shards,
          });
        }
        open_index = buffers.length;
        open_offset_bytes = 0;
        open_shards = [];
      }
      if (!open_shards.includes(index)) open_shards.push(index);
      placements.push({
        buffer: open_index,
        offsetBytes: open_offset_bytes,
        lengthBytes: part_bytes,
      });
      // The next part starts on the alignment, never in the padding of the previous one.
      const consumed_bytes = open_offset_bytes + part_bytes;
      open_offset_bytes = Math.ceil(consumed_bytes / alignment_bytes) * alignment_bytes;
      remaining_bytes -= part_bytes;
    }
    shards.push({
      index,
      offsetBytes: manifest_offset_bytes,
      lengthBytes: length_bytes,
      placements,
    });
    manifest_offset_bytes += length_bytes;
    payload_bytes += length_bytes;
  }
  if (open_index >= 0) {
    buffers.push({ index: open_index, sizeBytes: open_offset_bytes, shardIndices: open_shards });
  }

  let allocated_bytes = 0;
  for (const buffer of buffers) allocated_bytes += buffer.sizeBytes;
  let split_shard_count = 0;
  for (const shard of shards) {
    if (shard.placements.length > 1) split_shard_count += 1;
  }
  return {
    buffers,
    shards,
    payloadBytes: payload_bytes,
    allocatedBytes: allocated_bytes,
    splitShardCount: split_shard_count,
    partCapacityBytes: capacity_bytes,
  };
}
