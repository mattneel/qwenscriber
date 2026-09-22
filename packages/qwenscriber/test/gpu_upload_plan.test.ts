//! Upload planning against invented limit sets.
//!
//! The adapter on the desk cannot be asked for a 1 GiB `maxStorageBufferBindingSize`, and the point
//! of planning the upload before allocating anything is precisely to make that decision when the
//! limits are known -- so the limits are the input, and the test supplies three of them: the caps
//! a 2 GiB adapter reports, the caps a 1 GiB adapter reports, and a set so small that no placement
//! exists at all.
//!
//! What is asserted is the decision, not the plumbing: how many parts a shard becomes, how many
//! buffers those parts occupy, that every offset is aligned, and that the parts of a shard add back
//! up to the shard. A planner that truncated a shard to make it fit would fail the last one.

import { test } from "node:test";
import assert from "node:assert/strict";

import { GpuLimitsError, isGpuError } from "../src/gpu/errors.ts";
import { STATUS } from "../src/wasm/abi.ts";
import { measuredLimits, requireLimits, shortfallsFor } from "../src/gpu/limits.ts";
import type { WebGpuLimits } from "../src/capabilities.ts";
import {
  partCapacityOf,
  planUpload,
  type WebGpuShardPlacement,
  type WebGpuUploadPlan,
  type WebGpuUploadPlanLimits,
} from "../src/gpu/upload_plan.ts";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const ALIGNMENT = 256;

/** The caps a 2 GiB adapter reports: a big buffer, a 128 MiB binding, 128 MiB of upload at a time. */
const LIMITS_2GIB: WebGpuUploadPlanLimits = {
  maxBufferSizeBytes: 2 * GIB,
  maxStorageBufferBindingSizeBytes: 128 * MIB,
  alignmentBytes: ALIGNMENT,
};

/** The same machine with a 1 GiB buffer cap: the binding cap must not be the only thing that splits. */
const LIMITS_1GIB: WebGpuUploadPlanLimits = {
  maxBufferSizeBytes: 1 * GIB,
  maxStorageBufferBindingSizeBytes: 128 * MIB,
  alignmentBytes: ALIGNMENT,
};

/** A set no placement can satisfy: 128 bytes of binding against a 256-byte alignment. */
const LIMITS_IMPOSSIBLE: WebGpuUploadPlanLimits = {
  maxBufferSizeBytes: 256,
  maxStorageBufferBindingSizeBytes: 128,
  alignmentBytes: ALIGNMENT,
};

const ADAPTER_LIMITS: WebGpuLimits = {
  maxBufferSize: 2 * GIB,
  maxStorageBufferBindingSize: 128 * MIB,
  maxComputeWorkgroupStorageSize: 32768,
  maxComputeInvocationsPerWorkgroup: 256,
  minStorageBufferOffsetAlignment: ALIGNMENT,
  maxUniformBufferBindingSize: 65536,
};

function placementsOf(plan: WebGpuUploadPlan, shard: number): readonly WebGpuShardPlacement[] {
  return plan.shards[shard]?.placements ?? [];
}

/** The invariants a plan has to hold whatever the limits were, so each case asserts the same thing. */
function assertPlanInvariants(
  plan: WebGpuUploadPlan,
  shardLengthsBytes: readonly number[],
  limits: WebGpuUploadPlanLimits,
): void {
  assert.equal(plan.shards.length, shardLengthsBytes.length);
  assert.ok(plan.partCapacityBytes <= limits.maxBufferSizeBytes, "a part must fit a buffer");
  assert.ok(
    plan.partCapacityBytes <= limits.maxStorageBufferBindingSizeBytes,
    "a part must fit a binding",
  );
  for (let index = 0; index < shardLengthsBytes.length; index += 1) {
    const shard = plan.shards[index];
    assert.ok(shard !== undefined);
    assert.equal(shard.lengthBytes, shardLengthsBytes[index]);
    let placed = 0;
    for (const placement of shard.placements) {
      assert.equal(
        placement.offsetBytes % limits.alignmentBytes,
        0,
        `shard ${index} has an unaligned offset`,
      );
      assert.ok(placement.lengthBytes > 0, "a placement with no bytes is not a placement");
      assert.ok(
        placement.lengthBytes <= plan.partCapacityBytes,
        `shard ${index} has a part larger than the capacity`,
      );
      const buffer = plan.buffers[placement.buffer];
      assert.ok(buffer !== undefined, `shard ${index} points at a buffer that was not planned`);
      assert.ok(
        placement.offsetBytes + placement.lengthBytes <= buffer.sizeBytes,
        `shard ${index} runs past the end of buffer ${buffer.index}`,
      );
      placed += placement.lengthBytes;
    }
    assert.equal(placed, shard.lengthBytes, `shard ${index} lost or gained bytes`);
  }
  let allocated = 0;
  for (const buffer of plan.buffers) allocated += buffer.sizeBytes;
  assert.equal(plan.allocatedBytes, allocated);
}

test("a 2 GiB adapter splits a 2.5 GiB shard on the binding cap, into two buffers", () => {
  const shard = 2.5 * GIB;
  const plan = planUpload([shard], LIMITS_2GIB);
  // 128 MiB per part, because binding size is the smaller of the two caps.
  assert.equal(plan.partCapacityBytes, 128 * MIB);
  assert.equal(placementsOf(plan, 0).length, Math.ceil(shard / (128 * MIB)));
  assert.equal(plan.splitShardCount, 1);
  // 16 parts of 128 MiB fill a 2 GiB buffer exactly, the remaining 4 go into the next one.
  assert.equal(plan.buffers.length, 2);
  assert.equal(plan.buffers[0]?.sizeBytes, 2 * GIB);
  assert.equal(plan.buffers[1]?.sizeBytes, 0.5 * GIB);
  assert.equal(plan.payloadBytes, shard);
  assert.equal(plan.allocatedBytes, shard);
  assertPlanInvariants(plan, [shard], LIMITS_2GIB);
});

test("halving the buffer cap changes the buffer count without changing the parts", () => {
  const shard = 2.5 * GIB;
  const plan = planUpload([shard], LIMITS_1GIB);
  assert.equal(plan.partCapacityBytes, 128 * MIB);
  assert.equal(placementsOf(plan, 0).length, Math.ceil(shard / (128 * MIB)));
  // 8 parts of 128 MiB fill a 1 GiB buffer, so the same 20 parts now need three buffers.
  assert.equal(plan.buffers.length, 3);
  assert.equal(plan.buffers[2]?.sizeBytes, 0.5 * GIB);
  assertPlanInvariants(plan, [shard], LIMITS_1GIB);
});

test("a binding cap smaller than the alignment is a refusal, not an empty upload", () => {
  assert.throws(
    () => planUpload([4 * MIB], LIMITS_IMPOSSIBLE),
    (error: unknown) => {
      assert.ok(isGpuError(error), "the refusal must be a typed GPU error");
      assert.ok(error instanceof GpuLimitsError);
      assert.equal(error.code, STATUS.limit_exceeded);
      assert.equal(error.status, "limit_exceeded");
      assert.equal(error.operation, "gpu.planUpload");
      // The numbers are in the error, because "the GPU is too small" is not actionable.
      const shortfall = error.shortfalls[0];
      assert.equal(shortfall?.limit, "maxStorageBufferBindingSize");
      assert.equal(shortfall?.needed, ALIGNMENT);
      assert.equal(shortfall?.measured, 128);
      assert.equal(error.limits.maxBufferSize, 256);
      return true;
    },
  );
  assert.throws(() => planUpload([4 * MIB], { ...LIMITS_2GIB, alignmentBytes: 0 }), GpuLimitsError);
  assert.equal(partCapacityOf(LIMITS_IMPOSSIBLE), 0);
});

test("a manifest of several shards packs into one buffer, and a zero-length shard takes none", () => {
  const shardLengths = [300 * MIB, 0, 64 * MIB];
  const plan = planUpload(shardLengths, LIMITS_2GIB);
  assert.equal(placementsOf(plan, 0).length, 3, "300 MiB is three 128 MiB-limited parts");
  assert.equal(placementsOf(plan, 1).length, 0, "a shard with no bytes has nothing to place");
  assert.equal(placementsOf(plan, 2).length, 1);
  assert.equal(plan.buffers.length, 1, "everything fits in one 2 GiB buffer");
  assert.deepEqual(plan.buffers[0]?.shardIndices, [0, 2]);
  assert.equal(plan.splitShardCount, 1);
  assert.equal(plan.payloadBytes, 364 * MIB);
  assertPlanInvariants(plan, shardLengths, LIMITS_2GIB);
});

test("a shard that is not a byte length is rejected before any limit is consulted", () => {
  for (const invalid of [[-1], [1.5], [Number.NaN], [Number.MAX_SAFE_INTEGER + 2]]) {
    assert.throws(() => planUpload(invalid, LIMITS_2GIB), (error: unknown) => {
      assert.ok(isGpuError(error) === false, "misuse is not a GPU limit failure");
      assert.equal((error as { code: number }).code, STATUS.invalid_argument);
      return true;
    });
  }
});

test("requirements are compared against the measured numbers, not against a boolean", () => {
  assert.deepEqual(shortfallsFor(ADAPTER_LIMITS, { storageBindingBytes: 64 * MIB }), []);
  assert.deepEqual(
    shortfallsFor(ADAPTER_LIMITS, { invocationsPerWorkgroup: 256, workgroupStorageBytes: 2048 }),
    [],
  );
  const shortfalls = shortfallsFor(ADAPTER_LIMITS, {
    bufferBytes: 3 * GIB,
    storageBindingBytes: 256 * MIB,
  });
  assert.deepEqual(shortfalls, [
    { limit: "maxBufferSize", needed: 3 * GIB, measured: 2 * GIB },
    { limit: "maxStorageBufferBindingSize", needed: 256 * MIB, measured: 128 * MIB },
  ]);
  assert.equal(measuredLimits(ADAPTER_LIMITS).maxComputeInvocationsPerWorkgroup, 256);
  assert.throws(
    () => requireLimits(ADAPTER_LIMITS, { bufferBytes: 3 * GIB }, "gpu.matmulQ4"),
    (error: unknown) => {
      assert.ok(error instanceof GpuLimitsError);
      assert.equal(error.operation, "gpu.matmulQ4");
      return true;
    },
  );
});
