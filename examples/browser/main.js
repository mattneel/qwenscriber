//! Browser example driver.
//!
//! Serve the repository root over http and open `/examples/browser/index.html`:
//!
//!     python3 -m http.server 8766 --bind 127.0.0.1          # from the repository root
//!     cd packages/qwenscriber && npm install && npm run build
//!
//! No bundler and no build step for this file: it is a module script, and it imports the compiled
//! SDK by relative path (which is why the page has to be served from the repository root rather than
//! copied elsewhere). Every result is also written to `window.__result`, so a driver such as
//! `tools/browser/run-page.mjs` can use this page as a pass/fail gate.
//!
//! What it proves, in order: the host's capabilities, that the WASM core instantiated in a worker
//! and passed its own self-test, that audio decoding + resampling + log-mel ran end to end on real
//! numbers, that `transcribe()` fails with the typed "decode is not implemented" error instead of
//! inventing a transcript, and that one WebGPU kernel -- `matmul_q4` over the packed weight layout
//! -- reproduces the CPU reference within the tolerance the GPU harness sets.

const log_lines = [];

function log(message) {
  log_lines.push(message);
  const output = document.getElementById("log");
  if (output !== null) output.textContent = log_lines.join("\n");
  console.log(message);
}

function setField(id, value, tone = "") {
  const node = document.getElementById(id);
  if (node === null) return;
  node.textContent = value;
  node.className = tone;
}

function formatBytes(bytes) {
  if (bytes === 0) return "unknown (module was passed precompiled)";
  return `${bytes} bytes (${(bytes / 1024 / 1024).toFixed(2)} MiB)`;
}

function formatHash(value) {
  return `0x${value.toString(16)}`;
}

/** 1 second of a 1 kHz sine, the same signal the core's self-test measures. */
function tone(sample_count, frequency_hz, sample_rate_hz) {
  const samples = new Float32Array(sample_count);
  for (let index = 0; index < sample_count; index += 1) {
    samples[index] = Math.sin((2 * Math.PI * frequency_hz * index) / sample_rate_hz);
  }
  return samples;
}

function loudestBand(features, frames, frame) {
  let band = 0;
  let peak = -Infinity;
  for (let bin = 0; bin < 128; bin += 1) {
    const value = features[bin * frames + frame] ?? -Infinity;
    if (value > peak) {
      peak = value;
      band = bin;
    }
  }
  return band;
}

function renderCapabilities(caps) {
  const webgpu = caps.webgpu;
  setField(
    "cap-webgpu",
    webgpu.available ? "available" : `unavailable: ${webgpu.reason ?? "no reason reported"}`,
    webgpu.available ? "ok" : "warn",
  );
  if (webgpu.available) {
    const info = webgpu.adapterInfo ?? {};
    setField(
      "cap-adapter",
      `${caps.webgpuAdapter} - vendor=${info.vendor ?? "?"} architecture=${info.architecture ?? "?"}` +
        ` device=${info.device === undefined || info.device === "" ? "?" : info.device}` +
        ` fallback=${String(info.isFallbackAdapter ?? false)}`,
      caps.webgpuAdapter === "hardware" ? "ok" : "warn",
    );
    const limits = webgpu.limits;
    setField(
      "cap-limits",
      limits === undefined
        ? "not reported"
        : `maxBufferSize=${limits.maxBufferSize} maxStorageBufferBindingSize=` +
            `${limits.maxStorageBufferBindingSize} maxComputeWorkgroupStorageSize=` +
            `${limits.maxComputeWorkgroupStorageSize} maxComputeInvocationsPerWorkgroup=` +
            `${limits.maxComputeInvocationsPerWorkgroup}`,
    );
  } else {
    setField("cap-adapter", "none", "warn");
    setField("cap-limits", "none", "warn");
  }
  setField("cap-simd", caps.wasmSimd ? "supported" : "unsupported", caps.wasmSimd ? "ok" : "no");
  setField(
    "cap-threads",
    caps.wasmThreads ? "supported" : "unsupported",
    caps.wasmThreads ? "ok" : "warn",
  );
  setField(
    "cap-sab",
    caps.sharedArrayBuffer ? "available" : "unavailable",
    caps.sharedArrayBuffer ? "ok" : "warn",
  );
  setField("cap-workers", caps.workers ? "available" : "unavailable", caps.workers ? "ok" : "no");
  setField(
    "cap-coi",
    caps.crossOriginIsolated ? "yes" : "no",
    caps.crossOriginIsolated ? "ok" : "warn",
  );
}

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;

/** The fields a typed error carries, for a page whose job is to print them. */
function describeError(error) {
  return {
    name: error?.name ?? "Error",
    code: error?.code,
    status: error?.status,
    operation: error?.operation,
    message: error?.message ?? String(error),
  };
}

/**
 * The WebGPU section: acquire an adapter, plan an upload from its measured limits, run matmul_q4,
 * and compare the products against the CPU reference the GPU test harness uses.
 *
 * Both imports are test-side modules on purpose. `reference.mjs` is the module `node
 * tests/gpu/harness.mjs` compares against -- itself validated against the WASM core's `qw_selftest`
 * quantization hash -- and `harness.mjs` supplies the comparison and the tolerance, so this page
 * and that harness cannot disagree about either the expected numbers or what counts as close
 * enough. Neither is a dependency of the SDK: this page is the only thing that loads them.
 */
async function runGpuSection(sdk) {
  const [reference, harness] = await Promise.all([
    import("../../tests/gpu/reference.mjs"),
    import("../../tests/gpu/harness.mjs"),
  ]);
  const rows = 64;
  const cols = 256;
  const tokens = 24;
  const weights = reference.random_vector(rows * cols, 0x5091);
  const activations = reference.random_vector(tokens * cols, 0x5092);
  const packed = reference.pack_tensor("q4", rows, cols, weights);
  const expected = reference.matmul_quantized_reference(activations, packed, "q4", tokens, rows, cols);
  const shader_source = sdk.shaderSourceFromBaseUrl(new URL("../../gpu/shaders/", import.meta.url));

  // The SDK refuses a software adapter unless it is asked to allow one. This page shows the refusal
  // and then opts in explicitly, so the numbers below stay labelled with the adapter that produced
  // them instead of looking like they came from a real GPU.
  let refusal = null;
  let runtime;
  try {
    runtime = await sdk.WebGpuRuntime.create({ shaderSource: shader_source });
  } catch (error) {
    if (!(error instanceof sdk.GpuSoftwareAdapterError)) throw error;
    refusal = describeError(error);
    log(`gpu: refused the software adapter by default: ${error.message}`);
    runtime = await sdk.WebGpuRuntime.create({
      shaderSource: shader_source,
      allowSoftwareAdapter: true,
    });
  }

  const capability = runtime.capability;
  const plan = runtime.planUpload([packed.bytes.length]);
  // A manifest that cannot fit: the planner splits it against this adapter's real numbers.
  const hypothetical = runtime.planUpload([3 * GIB, 512 * MIB]);
  const dispatch = () =>
    sdk.matmulQ4(runtime, {
      packed: packed.bytes,
      dataOffsetBytes: packed.data_offset_bytes,
      rows,
      cols,
      activations,
      tokens,
    });
  const result = await dispatch();
  // Eleven more dispatches, median reported: one dispatch is dominated by
  // pipeline creation, and a mean would hide the warm number behind it.
  const timings = [];
  for (let index = 0; index < 11; index += 1) {
    const started = performance.now();
    await dispatch();
    timings.push(performance.now() - started);
  }
  timings.sort((left, right) => left - right);
  const dispatch_ms_median = timings[timings.length >> 1];
  const comparison = harness.compare(result.values, expected, harness.TOLERANCES.matmul);
  runtime.destroy();

  return {
    ok: comparison.failures === 0,
    dispatchMsMedian: dispatch_ms_median,
    refusal,
    capability: {
      adapterRequest: capability.adapterRequest,
      adapterKind: capability.adapterKind,
      adapterInfo: { ...capability.adapterInfo },
      alternativeAdapterInfo:
        capability.alternativeAdapterInfo === undefined
          ? undefined
          : { ...capability.alternativeAdapterInfo },
      limits: { ...capability.limits },
      deviceFeatures: [...capability.deviceFeatures],
    },
    uploadPlan: {
      buffers: plan.buffers.length,
      parts: plan.shards[0]?.placements.length ?? 0,
      payloadBytes: plan.payloadBytes,
      allocatedBytes: plan.allocatedBytes,
      partCapacityBytes: plan.partCapacityBytes,
    },
    hypotheticalPlan: {
      buffers: hypothetical.buffers.length,
      splitShardCount: hypothetical.splitShardCount,
      parts: hypothetical.shards.map((shard) => shard.placements.length),
      allocatedBytes: hypothetical.allocatedBytes,
    },
    matmul: {
      kernel: "matmul_q4",
      rows,
      cols,
      tokens,
      elements: comparison.elements,
      maxAbs: comparison.max_abs,
      meanAbs: comparison.mean_abs,
      maxRel: comparison.max_rel,
      failures: comparison.failures,
      tolerance: harness.TOLERANCES.matmul,
      dataOffsetBytes: result.dataOffsetBytes,
      packedBytes: result.packedBytes,
      groupCount: result.groupCount,
      workgroupSize: [...result.workgroupSize],
      workgroupCounts: [...result.workgroupCounts],
    },
  };
}

function renderGpu(report) {
  if (report.unavailable !== undefined) {
    setField("gpu-status", `no adapter: ${report.unavailable}`, "warn");
    setField("gpu-adapter", "none: no adapter", "warn");
    setField("gpu-limits", "none: the adapter is required first", "warn");
    setField("gpu-plan", "not planned: the adapter is required first", "warn");
    setField("gpu-matmul", "not run", "warn");
    setField("gpu-timing", "not run", "warn");
    return;
  }
  if (typeof report.dispatchMsMedian === "number") {
    setField(
      "gpu-timing",
      `median ${report.dispatchMsMedian.toFixed(3)} ms over 11 dispatches, ` +
        `${report.matmul.elements} elements`,
    );
  } else {
    setField("gpu-timing", "not run", "warn");
  }
  const capability = report.capability;
  const info = capability.adapterInfo;
  const alternative = capability.alternativeAdapterInfo;
  setField(
    "gpu-adapter-request",
    capability.adapterRequest === "high-performance"
      ? "high-performance (asked for a discrete GPU)"
      : "default (the browser chose the adapter)",
    capability.adapterRequest === "high-performance" ? "ok" : "warn",
  );
  setField(
    "gpu-adapter",
    `vendor=${info.vendor ?? "?"} architecture=${info.architecture ?? "?"} ` +
      `device=${info.device === undefined || info.device === "" ? "?" : info.device} ` +
      `description=${info.description === undefined || info.description === "" ? "?" : info.description} ` +
      `isFallbackAdapter=${String(info.isFallbackAdapter ?? false)}` +
      (alternative === undefined
        ? ""
        : `; the other request answered with vendor=${alternative.vendor ?? "?"} ` +
          `architecture=${alternative.architecture ?? "?"} -- this browser offers two adapters`),
  );
  setField(
    "gpu-degraded",
    report.refusal === null
      ? capability.adapterKind
      : `${capability.adapterKind}: run only because the page opted in`,
    capability.adapterKind === "hardware" ? "ok" : "warn",
  );
  const limits = capability.limits;
  setField(
    "gpu-limits",
    `maxBufferSize=${limits.maxBufferSize} ` +
      `maxStorageBufferBindingSize=${limits.maxStorageBufferBindingSize} ` +
      `maxComputeWorkgroupStorageSize=${limits.maxComputeWorkgroupStorageSize} ` +
      `maxComputeInvocationsPerWorkgroup=${limits.maxComputeInvocationsPerWorkgroup} ` +
      `minStorageBufferOffsetAlignment=${limits.minStorageBufferOffsetAlignment} ` +
      `maxUniformBufferBindingSize=${limits.maxUniformBufferBindingSize}`,
  );
  setField(
    "gpu-features",
    capability.deviceFeatures.length === 0 ? "none" : capability.deviceFeatures.join(", "),
  );
  const plan = report.uploadPlan;
  const hypothetical = report.hypotheticalPlan;
  setField(
    "gpu-plan",
    `this tensor: ${plan.buffers} buffer(s), ${plan.parts} part(s) of at most ` +
      `${plan.partCapacityBytes} bytes, ${plan.payloadBytes} bytes payload / ` +
      `${plan.allocatedBytes} allocated; hypothetical 3 GiB + 512 MiB manifest: ` +
      `${hypothetical.buffers} buffer(s), ${hypothetical.splitShardCount} split shard(s), ` +
      `parts ${hypothetical.parts.join("+")}`,
  );
  const matmul = report.matmul;
  setField(
    "gpu-matmul",
    `${matmul.tokens}x${matmul.cols} activations by ${matmul.rows}x${matmul.cols} q4 weights -> ` +
      `${matmul.elements} values; max_abs=${matmul.maxAbs} mean_abs=${matmul.meanAbs} ` +
      `max_rel=${matmul.maxRel} (tolerance ${matmul.tolerance.atol}+${matmul.tolerance.rtol}*|ref|)`,
    matmul.failures === 0 ? "ok" : "no",
  );
  setField(
    "gpu-status",
    report.ok
      ? "kernel output is within tolerance of the CPU reference"
      : `${matmul.failures} values outside tolerance`,
    report.ok ? "ok" : "no",
  );
}

function renderSelfTest(report) {
  setField(
    "selftest-result",
    report.failures === 0 ? "passed" : `FAILED: ${report.failedChecks.join(", ")}`,
    report.failures === 0 ? "ok" : "no",
  );
  setField("selftest-failures", String(report.failures));
  setField("selftest-band", String(report.loudestBand));
  setField("selftest-silence", String(report.silenceValue));
  setField("selftest-quant", formatHash(report.quantHash));
  setField("selftest-mel", formatHash(report.melHash));
}

function renderMel(mel) {
  setField("mel-shape", `${mel.frames} frames x 128 bins = ${mel.features.length} features`);
  setField("mel-max", String(mel.globalMaxLog));
  setField("mel-padding", String(mel.paddingValue));
  const band = loudestBand(mel.features, mel.frames, 40);
  setField("mel-band", String(band), band === 42 ? "ok" : "warn");
  return band;
}

/**
 * The preprocessing benchmark: the one stage ABI v1 implements completely, timed
 * over clip lengths that matter (a phrase, a sentence, a whole 30 s window). The
 * realtime factor is wall time divided by audio duration, so below 1.0 is faster
 * than real time on this machine.
 */
async function runBenchmark(qw) {
  const cases = [
    { seconds: 1, sample_rate_hz: 48000 },
    { seconds: 5, sample_rate_hz: 48000 },
    { seconds: 30, sample_rate_hz: 16000 },
  ];
  const rows = [];
  for (const entry of cases) {
    const audio = tone(entry.seconds * entry.sample_rate_hz, 440, entry.sample_rate_hz);
    const started = performance.now();
    const mel = await qw.preprocess(audio, { sample_rate_hz: entry.sample_rate_hz });
    const elapsed_ms = performance.now() - started;
    rows.push({
      seconds: entry.seconds,
      sample_rate_hz: entry.sample_rate_hz,
      frames: mel.frames,
      elapsed_ms,
      realtime_factor: elapsed_ms / (entry.seconds * 1000),
    });
    log(
      `benchmark: ${entry.seconds}s @ ${entry.sample_rate_hz}Hz -> ${mel.frames} frames in ` +
        `${elapsed_ms.toFixed(1)}ms (x${(entry.seconds * 1000 / elapsed_ms).toFixed(1)} realtime)`,
    );
  }
  return rows;
}

function renderBenchmark(rows) {
  const cases = rows.map((row) => `${row.seconds}s@${row.sample_rate_hz}Hz`).join(", ");
  const best = rows.reduce((left, right) => (right.realtime_factor < left.realtime_factor ? right : left));
  setField("bench-cases", cases);
  setField(
    "bench-best",
    `${best.elapsed_ms.toFixed(1)} ms for ${best.seconds}s of audio (${best.frames} frames)`,
  );
  const summary = `best ${best.elapsed_ms.toFixed(1)}ms, x${(best.seconds * 1000 / best.elapsed_ms).toFixed(1)} realtime`;
  setField(
    "bench-rtf",
    rows.map((row) => `${row.seconds}s x${(row.seconds * 1000 / row.elapsed_ms).toFixed(1)}`).join(", "),
    best.realtime_factor < 1 ? "ok" : "warn",
  );
  return summary;
}

async function main() {
  const sdk = await import("../../packages/qwenscriber/dist/index.js");
  log(`sdk: ${sdk.SDK_VERSION}`);

  const caps = await sdk.Qwenscriber.capabilities();
  renderCapabilities(caps);
  log(
    `capabilities: webgpu=${caps.webgpu.available ? caps.webgpuAdapter : "none"} ` +
      `simd=${caps.wasmSimd} threads=${caps.wasmThreads} workers=${caps.workers} ` +
      `sharedArrayBuffer=${caps.sharedArrayBuffer} crossOriginIsolated=${caps.crossOriginIsolated}`,
  );

  // Worker defaults to true wherever Worker exists, which is the whole point of the SDK.
  const qw = await sdk.Qwenscriber.create({ backend: "auto", quantization: "q8" });
  const versions = qw.versions;
  setField("sdk-version", versions.sdkVersion);
  setField(
    "core-version",
    `${sdk.formatAbiVersion(versions.abiVersion)} (raw 0x${versions.abiVersion.toString(16)}), ` +
      `core ${versions.coreVersion}`,
  );
  setField("module-bytes", `${formatBytes(versions.wasmBytes)}, ${formatBytes(versions.runtimeBytes)} resident`);
  setField("backend", `auto -> ${qw.backend}`);
  setField("worker", qw.workerBacked ? "Web Worker" : "main thread");
  log(
    `core: abi=${sdk.formatAbiVersion(versions.abiVersion)} core=${versions.coreVersion} ` +
      `wasm=${versions.wasmBytes}B runtime=${versions.runtimeBytes}B worker=${qw.workerBacked}`,
  );

  const stages = [];
  qw.onProgress((progress) => {
    stages.push(`${progress.stage} ${progress.completed}/${progress.total}`);
    log(`progress: ${progress.stage} ${progress.completed}/${progress.total}`);
  });

  const report = await qw.selfTest();
  renderSelfTest(report);
  log(
    `selftest: failures=${report.failures} loudestBand=${report.loudestBand} ` +
      `silence=${report.silenceValue} quant=${formatHash(report.quantHash)} ` +
      `mel=${formatHash(report.melHash)}`,
  );

  // 48 kHz in, so the page exercises the resampler rather than the pass-through path.
  const audio = tone(48000, 1000, 48000);
  const mel = await qw.preprocess(audio, { sample_rate_hz: 48000 });
  const band = renderMel(mel);
  log(
    `preprocess: source=${mel.sourceKind} frames=${mel.frames} durationMs=${mel.audioDurationMs.toFixed(1)} ` +
      `globalMaxLog=${mel.globalMaxLog} paddingValue=${mel.paddingValue} loudestBand@40=${band}`,
  );

  // ABI v1 also decodes ids, so the page exercises that too. The token strings below are the byte
  // alphabet's shape (U+0120 is the space marker); a real vocabulary comes from the model container,
  // and the container loader is part of the model-load stage that ABI v1 does not have yet.
  const vocabulary = ["H", "i", "\u0120", "there"];
  await qw.setVocabulary(vocabulary);
  const ids = Uint32Array.from([0, 1, 2, 3]);
  const text = await qw.detokenize(ids);
  setField("detokenize", `ids [${ids.join(", ")}] -> ${JSON.stringify(text)}`, text === "Hi there" ? "ok" : "no");
  log(`detokenize: ${vocabulary.length} tokens, ids=[${ids.join(",")}] -> ${JSON.stringify(text)}`);

  const benchmark = await runBenchmark(qw);
  const benchmarkSummary = renderBenchmark(benchmark);

  let transcribe_error = null;
  try {
    await qw.transcribe(audio, { sample_rate_hz: 48000 });
    setField("transcribe", "returned a transcript, which ABI v1 cannot do", "no");
  } catch (error) {
    transcribe_error = {
      name: error.name,
      code: error.code,
      status: error.status,
      operation: error.operation,
      feature: error.feature,
      message: error.message,
      context: error.context,
    };
    setField("transcribe", `expected typed error: ${error.name} ${error.message}`, "warn");
    log(`transcribe: ${error.name} code=${error.code} status=${error.status} feature=${error.feature}`);
    log(`transcribe message: ${error.message}`);
  }

  let gpu;
  if (!caps.webgpu.available) {
    // No adapter is an ordinary outcome, not a failure to recover from: the page
    // says so with the capability's own reason instead of letting acquisition
    // throw something untyped at a reader.
    gpu = { ok: false, unavailable: caps.webgpu.reason };
    renderGpu(gpu);
    log(`gpu: unavailable -- ${caps.webgpu.reason}`);
  } else try {
    gpu = await runGpuSection(sdk);
    renderGpu(gpu);
    log(
      `gpu: adapter=${gpu.capability.adapterKind} request=${gpu.capability.adapterRequest} ` +
        `vendor=${gpu.capability.adapterInfo.vendor ?? "?"} ` +
        `architecture=${gpu.capability.adapterInfo.architecture ?? "?"} ` +
        `maxBufferSize=${gpu.capability.limits.maxBufferSize}`,
    );
    log(`gpu plan: ${JSON.stringify(gpu.uploadPlan)}`);
    log(
      `gpu matmul_q4 dispatch: median ${gpu.dispatchMsMedian.toFixed(3)}ms over 11 dispatches ` +
        `(${gpu.matmul.elements} elements)`,
    );
    log(`gpu hypothetical manifest plan: ${JSON.stringify(gpu.hypotheticalPlan)}`);
    log(
      `gpu matmul_q4: elements=${gpu.matmul.elements} max_abs=${gpu.matmul.maxAbs} ` +
        `mean_abs=${gpu.matmul.meanAbs} max_rel=${gpu.matmul.maxRel} ` +
        `failures=${gpu.matmul.failures}`,
    );
  } catch (error) {
    const failure = describeError(error);
    gpu = { ok: false, error: failure };
    setField("gpu-status", `typed error: ${failure.name} ${failure.message}`, "warn");
    setField("gpu-adapter", "none: no adapter", "warn");
    setField("gpu-limits", "none", "warn");
    setField("gpu-plan", "not planned: the adapter is required first", "warn");
    setField("gpu-matmul", "not run", "warn");
    log(
      `gpu: ${failure.name} code=${failure.code} status=${failure.status} ` +
        `operation=${failure.operation}`,
    );
    log(`gpu message: ${failure.message}`);
  }

  document.getElementById("rerun-selftest").disabled = false;
  document.getElementById("rerun-selftest").addEventListener("click", async () => {
    const again = await qw.selfTest();
    renderSelfTest(again);
    log(`selftest again: failures=${again.failures}`);
  });
  document.getElementById("rerun-mel").disabled = false;
  document.getElementById("rerun-mel").addEventListener("click", async () => {
    const again = await qw.preprocess(tone(48000, 1000, 48000), { sample_rate_hz: 48000 });
    renderMel(again);
    log(`preprocess again: frames=${again.frames}`);
  });

  return {
    ok:
      report.failures === 0 &&
      mel.frames === 100 &&
      band === 42 &&
      transcribe_error !== null &&
      gpu.ok === true,
    sdkVersion: versions.sdkVersion,
    abiVersion: sdk.formatAbiVersion(versions.abiVersion),
    benchmark,
    coreVersion: versions.coreVersion,
    wasmBytes: versions.wasmBytes,
    runtimeBytes: versions.runtimeBytes,
    backend: qw.backend,
    workerBacked: qw.workerBacked,
    capabilities: caps,
    selfTest: {
      failures: report.failures,
      loudestBand: report.loudestBand,
      silenceValue: report.silenceValue,
      quantHash: formatHash(report.quantHash),
      melHash: formatHash(report.melHash),
      failedChecks: [...report.failedChecks],
    },
    preprocess: {
      sourceKind: mel.sourceKind,
      frames: mel.frames,
      features: mel.features.length,
      globalMaxLog: mel.globalMaxLog,
      paddingValue: mel.paddingValue,
      loudestBandAtFrame40: band,
      audioDurationMs: mel.audioDurationMs,
      stages,
    },
    detokenize: { vocabulary: vocabulary.length, ids: [...ids], text },
    transcribeError: transcribe_error,
    gpu,
  };
}

try {
  window.__result = await main();
} catch (error) {
  const failure = {
    name: error?.name ?? "Error",
    message: error?.message ?? String(error),
    code: error?.code,
    status: error?.status,
    operation: error?.operation,
  };
  log(`FAILED: ${failure.name}: ${failure.message}`);
  setField("transcribe", `page failed: ${failure.message}`, "no");
  window.__result = { ok: false, error: failure };
}
