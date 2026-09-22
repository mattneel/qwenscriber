//! Capability probes, including the hostile cases that make environment detection worth writing.
//!
//! `probeWebGpu` takes the `navigator.gpu`-shaped object as an argument rather than reaching for a
//! global, so a fake adapter can be handed to it directly: one that returns null, one that throws,
//! one that never settles, one whose limits are missing. Those are the paths a real page hits on
//! hardware this SDK's authors do not own, and they are the only reason the probe is a function.

import { test } from "node:test";
import assert from "node:assert/strict";

import { capabilities, probeWebGpu } from "../src/capabilities.ts";
import { Qwenscriber } from "../src/index.ts";

const ADAPTER_INFO = { vendor: "amd", architecture: "rdna-2", device: "", description: "" };
const ADAPTER_LIMITS = {
  maxBufferSize: 2147483648,
  maxStorageBufferBindingSize: 2147483648,
  maxComputeWorkgroupStorageSize: 32768,
  maxComputeInvocationsPerWorkgroup: 256,
};

test("capabilities() answers in Node, where there is no browser at all", async () => {
  const probe = await capabilities();
  assert.equal(typeof probe.wasmSimd, "boolean");
  assert.equal(typeof probe.wasmThreads, "boolean");
  assert.equal(typeof probe.sharedArrayBuffer, "boolean");
  assert.equal(typeof probe.workers, "boolean");
  assert.equal(typeof probe.crossOriginIsolated, "boolean");
  // Worker is not a Node global, so this is false here and true in a page.
  assert.equal(probe.workers, typeof Worker === "function");
  // V8 implements fixed-width SIMD in every shipping configuration.
  assert.equal(probe.wasmSimd, true);
  assert.equal(probe.webgpu.available, false);
  assert.equal(probe.webgpuAdapter, "none");
  assert.equal(typeof probe.webgpu.reason, "string");
  assert.ok((probe.webgpu.reason ?? "").length > 0, "an unavailable feature must say why");
});

test("the probe is stable and never throws, on repeated calls", async () => {
  const first = await capabilities();
  const second = await capabilities();
  assert.equal(first.wasmSimd, second.wasmSimd);
  assert.equal(first.wasmThreads, second.wasmThreads);
  assert.equal(first.workers, second.workers);
  assert.equal(first.webgpu.available, second.webgpu.available);
  assert.equal(first.webgpuAdapter, second.webgpuAdapter);
});

test("Qwenscriber.capabilities() is the same probe", async () => {
  const direct = await capabilities();
  const through_class = await Qwenscriber.capabilities();
  assert.equal(through_class.wasmSimd, direct.wasmSimd);
  assert.equal(through_class.workers, direct.workers);
  assert.equal(through_class.webgpu.available, direct.webgpu.available);
});

test("probeWebGpu reports a null adapter, a throwing one, and a hanging one", async () => {
  const missing = await probeWebGpu(undefined);
  assert.equal(missing.available, false);
  assert.match(missing.reason ?? "", /navigator\.gpu/);

  const refused = await probeWebGpu({ requestAdapter: async () => null });
  assert.equal(refused.available, false);
  assert.match(refused.reason ?? "", /null/);

  const throwing = await probeWebGpu({
    requestAdapter: () => {
      throw new Error("boom");
    },
  });
  assert.equal(throwing.available, false);
  assert.match(throwing.reason ?? "", /boom/);

  const rejecting = await probeWebGpu({
    requestAdapter: async () => {
      throw new Error("refused by policy");
    },
  });
  assert.equal(rejecting.available, false);
  assert.match(rejecting.reason ?? "", /refused by policy/);

  // A promise that never settles: the probe must time out rather than leave the page waiting.
  const hanging = await probeWebGpu({ requestAdapter: () => new Promise(() => {}) }, 25);
  assert.equal(hanging.available, false);
  assert.match(hanging.reason ?? "", /did not settle/);
});

test("probeWebGpu classifies a hardware adapter and a software one", async () => {
  const hardware = await probeWebGpu({
    requestAdapter: async () => ({ info: ADAPTER_INFO, limits: ADAPTER_LIMITS }),
  });
  assert.equal(hardware.available, true);
  assert.equal(hardware.adapter, "hardware");
  assert.equal(hardware.adapterInfo?.architecture, "rdna-2");
  assert.equal(hardware.adapterInfo?.vendor, "amd");
  assert.equal(hardware.limits?.maxBufferSize, ADAPTER_LIMITS.maxBufferSize);
  assert.equal(
    hardware.limits?.maxStorageBufferBindingSize,
    ADAPTER_LIMITS.maxStorageBufferBindingSize,
  );
  assert.equal(hardware.reason, undefined);

  // SwiftShader and llvmpipe are real adapters that are not real GPUs.
  const swiftshader = await probeWebGpu({
    requestAdapter: async () => ({
      info: { vendor: "google", architecture: "swiftshader", device: "", description: "" },
      limits: ADAPTER_LIMITS,
    }),
  });
  assert.equal(swiftshader.available, true);
  assert.equal(swiftshader.adapter, "software");

  const fallback = await probeWebGpu({
    requestAdapter: async () => ({ isFallbackAdapter: true, info: ADAPTER_INFO, limits: ADAPTER_LIMITS }),
  });
  assert.equal(fallback.adapter, "software");
});

test("an adapter with no limits or info is still reported as available", async () => {
  const partial = await probeWebGpu({ requestAdapter: async () => ({}) });
  assert.equal(partial.available, true);
  assert.equal(partial.limits, undefined);
  assert.equal(partial.adapter, "hardware");
});
