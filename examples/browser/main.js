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
//! numbers, and that `transcribe()` fails with the typed "decode is not implemented" error instead
//! of inventing a transcript.

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
    ok: report.failures === 0 && mel.frames === 100 && band === 42 && transcribe_error !== null,
    sdkVersion: versions.sdkVersion,
    abiVersion: sdk.formatAbiVersion(versions.abiVersion),
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
