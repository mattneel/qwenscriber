//! The WASM path and the audio front end it depends on, exercised through the SDK's own API.
//!
//! Run with `node --test test/`; Node's type stripping executes the TypeScript directly (23.6+ does
//! this by default, 22.6+ behind a flag). If the wasm artifact is missing the suite builds it with
//! `zig build wasm` rather than skipping: a skipped ABI test and a passing one look the same in CI
//! output, and only one of them means anything.

import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  ABI_MAJOR,
  ABI_VERSION_EXPECTED,
  MEL_BINS,
  MEL_CAPACITY_FRAMES,
  MEL_CAPACITY_SAMPLES,
  STATUS,
  statusName,
} from "../src/wasm/abi.ts";
import { WasmCore } from "../src/wasm/runtime.ts";
import {
  AbiMismatchError,
  NotImplementedError,
  QwenscriberError,
  SDK_STATUS,
} from "../src/errors.ts";
import { Qwenscriber } from "../src/index.ts";
import { decodeAudio } from "../src/audio/pcm.ts";
import { resample } from "../src/audio/resample.ts";
import { readWave } from "../src/audio/wav.ts";
import { errorFromResponse, errorResponseFrom } from "../src/worker/protocol.ts";
import { withAudioFormat, withBlockAlign, writeWave } from "./wave_fixture.ts";

const WASM_URL = new URL("../../../zig-out/bin/qwenscriber_core.wasm", import.meta.url);
const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));

function wasmBytes(): Uint8Array<ArrayBuffer> {
  if (!existsSync(WASM_URL)) {
    execFileSync("zig", ["build", "wasm"], { cwd: REPO_ROOT, stdio: "inherit" });
  }
  if (!existsSync(WASM_URL)) {
    throw new Error(`zig build wasm did not produce ${WASM_URL.href}`);
  }
  return new Uint8Array(readFileSync(WASM_URL));
}

/** Compiled once: compiling a 963 KB module per test would dominate the suite's runtime. */
const MODULE = await WebAssembly.compile(wasmBytes());

async function freshCore(): Promise<WasmCore> {
  return WasmCore.load(MODULE);
}

function tone(sample_count: number, frequency_hz = 1000, sample_rate_hz = 16000): Float32Array {
  const samples = new Float32Array(sample_count);
  for (let index = 0; index < sample_count; index += 1) {
    samples[index] = Math.sin((2 * Math.PI * frequency_hz * index) / sample_rate_hz);
  }
  return samples;
}

/** The mel band with the most energy in one frame, the same measurement the core's self-test makes. */
function loudestBandAt(features: Float32Array, frames: number, frame: number): number {
  let band = 0;
  let peak = -Infinity;
  for (let bin = 0; bin < MEL_BINS; bin += 1) {
    const value = features[bin * frames + frame] ?? -Infinity;
    if (value > peak) {
      peak = value;
      band = bin;
    }
  }
  return band;
}

function rms(values: Float32Array): number {
  let total = 0;
  for (const value of values) total += value * value;
  return Math.sqrt(total / values.length);
}

function expectStatus(body: () => unknown, code: number, operation: string): QwenscriberError {
  let caught: unknown;
  try {
    body();
  } catch (error) {
    caught = error;
  }
  assert.ok(caught instanceof QwenscriberError, `expected a QwenscriberError from ${operation}`);
  assert.equal(caught.code, code, `${operation} reports code ${code}`);
  assert.equal(caught.status, statusName(code), `${operation} reports the name of ${code}`);
  assert.equal(caught.operation, operation);
  return caught;
}

async function rejection(promise: Promise<unknown>): Promise<QwenscriberError> {
  let caught: unknown;
  try {
    await promise;
  } catch (error) {
    caught = error;
  }
  assert.ok(caught instanceof QwenscriberError, "the call must reject with a QwenscriberError");
  return caught;
}

test("the module loads, validates ABI v1, and creates one handle", async () => {
  const core = await WasmCore.load(wasmBytes());
  try {
    assert.equal(core.abiVersion(), ABI_VERSION_EXPECTED);
    assert.equal(core.abiVersion() >>> 16, ABI_MAJOR);
    assert.ok(core.coreVersion() >= 1, `core version ${core.coreVersion()}`);
    assert.equal(core.handle, 1);
    assert.ok(core.moduleBytes() > 0);
    assert.ok(core.memoryBytes() >= 65536);
    assert.equal(core.disposed, false);
  } finally {
    core.dispose();
  }
});

test("the ABI major check refuses another major and accepts a newer minor", () => {
  let caught: unknown;
  try {
    WasmCore.assertAbiVersion(0x0002_0000);
  } catch (error) {
    caught = error;
  }
  assert.ok(caught instanceof AbiMismatchError, "a wrong major throws AbiMismatchError");
  assert.equal(caught.code, SDK_STATUS.abi_mismatch);
  assert.equal(caught.status, "abi_mismatch");
  assert.equal(caught.actualVersion, 0x0002_0000);
  assert.equal(caught.expectedVersion, ABI_VERSION_EXPECTED);
  assert.match(caught.message, /0x20000/);

  // Minor versions are additive by contract, so this must not throw.
  WasmCore.assertAbiVersion(ABI_VERSION_EXPECTED | 0x0003);
});

test("a module that is not the core is refused, naming the missing export", async () => {
  // A valid, empty wasm module: it instantiates, and exports nothing.
  const empty = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
  const error = await rejection(WasmCore.load(empty));
  assert.equal(error.code, SDK_STATUS.protocol);
  assert.equal(error.operation, "instantiate");
  // The linear memory is the first thing checked, so that is the name it reports here.
  assert.match(error.message, /does not export/);
});

test("the core's self-test reports zero failures and the reference geometry", async () => {
  const core = await freshCore();
  try {
    const report = core.selfTest();
    assert.equal(report.failures, 0, `failed checks: ${report.failedChecks.join(", ")}`);
    assert.deepEqual([...report.failedChecks], []);
    assert.equal(report.loudestBand, 42);
    assert.ok(Math.abs(report.silenceValue - -1.5) < 1e-6, `silence ${report.silenceValue}`);
    assert.equal(typeof report.quantHash, "bigint");
    assert.equal(typeof report.melHash, "bigint");
    assert.notEqual(report.quantHash, 0n);
    assert.notEqual(report.melHash, 0n);
  } finally {
    core.dispose();
  }
});

test("log-mel of a 1 kHz tone lands in band 42 and pads with the range floor", async () => {
  const core = await freshCore();
  try {
    const computed = core.melCompute(tone(16000));
    assert.equal(computed.frames, 100);
    assert.equal(computed.features.length, 100 * MEL_BINS);
    assert.equal(computed.bytesWritten, computed.features.length * 4);
    assert.ok(Number.isFinite(computed.globalMaxLog));
    assert.equal(loudestBandAt(computed.features, computed.frames, 40), 42);

    // Padding frames carry the dynamic-range floor, which is eight log10 units below the loudest
    // frame of this clip -- so a partially filled encoder chunk is padded with the reference value,
    // not with zero.
    const floor = (Math.max(computed.globalMaxLog - 8, -10) + 4) / 4;
    assert.ok(
      Math.abs(computed.paddingValue - floor) < 1e-5,
      `padding ${computed.paddingValue} should be ${floor}`,
    );
    const loudest_value = computed.features[42 * computed.frames + 40] ?? 0;
    assert.ok(computed.paddingValue < loudest_value);

    // Digital silence has no dynamic range at all: every bin clamps to log10(1e-10), which the
    // normalisation turns into exactly -1.5.
    const silence = core.melCompute(new Float32Array(8000));
    assert.equal(silence.frames, 50);
    assert.ok(Math.abs(silence.paddingValue - -1.5) < 1e-6, `silence padding ${silence.paddingValue}`);
    for (const value of silence.features) {
      assert.ok(Math.abs(value - -1.5) < 1e-6, `silence bin ${value}`);
    }
  } finally {
    core.dispose();
  }
});

test("audio longer than the 30-second capacity is refused before allocating", async () => {
  const core = await freshCore();
  try {
    const too_long = new Float32Array(MEL_CAPACITY_SAMPLES + 1);
    const error = expectStatus(
      () => core.melCompute(too_long),
      STATUS.audio_too_long,
      "qw_mel_compute",
    );
    assert.equal(error.context["frames"], MEL_CAPACITY_FRAMES + 1);
    assert.equal(error.context["frames_max"], MEL_CAPACITY_FRAMES);
  } finally {
    core.dispose();
  }
});

test("bad arguments become typed errors carrying the module's own text", async () => {
  const core = await freshCore();
  try {
    // Zero samples is refused by the SDK before the output allocation would fail.
    expectStatus(
      () => core.melCompute(new Float32Array(0)),
      STATUS.invalid_argument,
      "qw_mel_compute",
    );

    // A handle the ABI does not know: the status and the text both come from the module.
    const bogus_handle = expectStatus(
      () => core.melCompute(tone(1600), 42),
      STATUS.invalid_argument,
      "qw_mel_compute",
    );
    assert.match(bogus_handle.message, /invalid argument/);
    assert.equal(bogus_handle.context["sample_count"], 1600);

    // Decoding before a vocabulary is a state error, not an argument error.
    expectStatus(
      () => core.detokenize(Uint32Array.from([0])),
      STATUS.invalid_state,
      "qw_detokenize",
    );

    // Alignment is a closed set, and a zero-sized allocation is never valid.
    expectStatus(() => core.alloc(16, 3), STATUS.invalid_argument, "qw_alloc");
    expectStatus(() => core.alloc(0, 4), STATUS.invalid_argument, "qw_alloc");

    // A vocabulary needs at least one token.
    expectStatus(() => core.setVocabulary({ tokens: [] }), STATUS.invalid_argument, "qw_tokenizer_set");
  } finally {
    core.dispose();
  }
});

test("a vocabulary round trips through the ABI and bad ids are reported", async () => {
  const core = await freshCore();
  try {
    // The byte alphabet uses U+0120 for a space, and the two-byte token exercises UTF-8 assembly.
    core.setVocabulary({ tokens: ["H", "i", "\u0120", "there", "\u00e4\u00bd", "\u00a0"] });
    assert.equal(core.detokenize(Uint32Array.from([0, 1, 2, 3])), "Hi there");
    expectStatus(() => core.detokenize(Uint32Array.from([99])), STATUS.not_found, "qw_detokenize");
    expectStatus(
      () => core.detokenize(Uint32Array.from([4, 5])),
      STATUS.invalid_encoding,
      "qw_detokenize",
    );

    // Replacing the vocabulary must release the previous buffers and keep decoding correct.
    core.setVocabulary({ tokens: ["a", "b"] });
    assert.equal(core.detokenize(Uint32Array.from([0, 1, 1])), "abb");
  } finally {
    core.dispose();
  }
});

test("dispose releases the handle and tolerates a second call", async () => {
  const core = await freshCore();
  core.dispose();
  assert.equal(core.disposed, true);
  core.dispose();
  // Anything that takes the handle is refused once it is gone.
  expectStatus(
    () => core.melCompute(tone(1600)),
    STATUS.invalid_argument,
    "qw_mel_compute",
  );
  // `qw_selftest` takes no handle in ABI v1 and needs no instance, so it still answers; that is a
  // property of the ABI, and the reason `dispose()` releasing the handle is not the whole story.
  assert.equal(core.selfTest().failures, 0);
});

test("the facade preprocesses audio in-process and reports decode as unimplemented", async () => {
  const sdk = await Qwenscriber.create({ worker: false, wasm: wasmBytes() });
  try {
    assert.equal(sdk.workerBacked, false);
    // No WebGPU in Node, so "auto" resolves to the portable backend rather than throwing.
    assert.equal(sdk.backend, "wasm");
    assert.equal(sdk.versions.abiVersion, ABI_VERSION_EXPECTED);
    assert.ok(sdk.versions.wasmBytes > 0);

    const stages: string[] = [];
    sdk.onProgress((progress) => stages.push(`${progress.stage}:${progress.completed}/${progress.total}`));

    const pcm = Int16Array.from(tone(16000), (value) => Math.round(value * 32767));
    const mel = await sdk.preprocess(pcm);
    assert.equal(mel.frames, 100);
    assert.equal(mel.features.length, 100 * MEL_BINS);
    assert.equal(mel.sourceKind, "int16");
    assert.equal(mel.channelCount, 1);
    assert.equal(mel.inputSampleRateHz, 16000);
    assert.ok(Math.abs(mel.audioDurationMs - 1000) < 1);
    assert.deepEqual(stages, ["resample:0/16000", "resample:16000/16000", "mel:0/1", "mel:1/1"]);

    const report = await sdk.selfTest();
    assert.equal(report.failures, 0);

    const error = await rejection(sdk.transcribe(pcm));
    assert.ok(error instanceof NotImplementedError);
    assert.equal(error.code, SDK_STATUS.not_implemented);
    assert.equal(error.feature, "decode");
    assert.equal(error.operation, "transcribe");
    // The front end really ran: the error carries what preprocessing produced.
    assert.equal(error.context["frames"], 100);
    assert.equal(error.context["model"], sdk.model);
    assert.equal(error.context["backend"], "wasm");
  } finally {
    await sdk.dispose();
  }

  const disposed = await rejection(sdk.preprocess(Int16Array.from([0, 1, 2])));
  assert.equal(disposed.code, STATUS.invalid_state);
});

test("a serialized error comes back as the class it was", () => {
  const original = new NotImplementedError("decode", "transcribe", { context: { frames: 100 } });
  const wire = errorResponseFrom(original, "transcribe", 7);
  assert.equal(wire.errorClass, "NotImplementedError");
  assert.equal(wire.code, SDK_STATUS.not_implemented);

  const rebuilt = errorFromResponse(wire);
  assert.ok(rebuilt instanceof NotImplementedError, "the boundary must not flatten the class");
  assert.equal(rebuilt.feature, "decode");
  assert.equal(rebuilt.code, SDK_STATUS.not_implemented);
  assert.equal(rebuilt.operation, "transcribe");
  assert.equal(rebuilt.context["frames"], 100);
  assert.equal(rebuilt.message, original.message);

  const mismatch = errorFromResponse(
    errorResponseFrom(new AbiMismatchError(0x0002_0000, ABI_VERSION_EXPECTED), "qw_abi_version", 8),
  );
  assert.ok(mismatch instanceof AbiMismatchError);
  assert.equal(mismatch.actualVersion, 0x0002_0000);

  // An untyped throw still crosses as a typed error with its text intact.
  const wrapped = errorFromResponse(errorResponseFrom(new TypeError("nope"), "init", 9));
  assert.equal(wrapped.code, SDK_STATUS.internal);
  assert.match(wrapped.message, /nope/);
});

test("resampling to 16 kHz is deterministic, in range, and preserves level", () => {
  const input = tone(48000, 1000, 48000);
  const first = resample(input, 48000, 16000);
  const second = resample(input, 48000, 16000);
  assert.equal(first.length, 16000);
  assert.deepEqual(Array.from(first), Array.from(second));

  let peak = 0;
  for (const value of first) {
    assert.ok(Number.isFinite(value));
    peak = Math.max(peak, Math.abs(value));
  }
  assert.ok(peak <= 1, `peak ${peak} must stay inside [-1, 1]`);

  // Unity gain: a band-limited interpolator must not change the level of a steady tone.
  const ratio = rms(first) / rms(input);
  assert.ok(Math.abs(ratio - 1) < 0.02, `RMS ratio ${ratio} should be close to 1`);

  // A pass-through rate is a copy, not a shared buffer.
  const same = resample(input, 16000, 16000);
  assert.equal(same.length, input.length);
  assert.notEqual(same, input);
});

test("the WAVE reader accepts the containers it documents and refuses the rest", () => {
  const values = tone(1600, 1000, 16000);
  const i16 = writeWave(values, 16000);
  const parsed = readWave(i16);
  assert.equal(parsed.sample_rate_hz, 16000);
  assert.equal(parsed.channel_count, 1);
  assert.equal(parsed.bits_per_sample, 16);
  assert.equal(parsed.is_float, false);
  assert.equal(parsed.frame_count, values.length);
  assert.ok(Math.abs((parsed.samples[10] ?? 0) - (values[10] ?? 0)) < 1e-4);

  const float = readWave(writeWave(values, 16000, "f32"));
  assert.equal(float.is_float, true);
  assert.ok(Math.abs((float.samples[10] ?? 0) - (values[10] ?? 0)) < 1e-7);

  // IMA ADPCM: a real format this reader must not pretend to decode.
  const compressed = expectStatus(
    () => readWave(withAudioFormat(i16, 0x0011)),
    STATUS.unsupported,
    "readWave",
  );
  assert.equal(compressed.context["audio_format"], 0x0011);
  expectStatus(() => readWave(writeWave(values, 16000, "i16", 2)), STATUS.unsupported, "readWave");
  expectStatus(() => readWave(withBlockAlign(i16, 7)), STATUS.shape_mismatch, "readWave");
  expectStatus(() => readWave(i16.slice(0, 20)), STATUS.truncated, "readWave");
  expectStatus(() => readWave(new Uint8Array(64)), STATUS.bad_magic, "readWave");
});

test("decodeAudio scales PCM into range, mixes channels, and rejects NaN", async () => {
  const values = tone(1600, 1000, 16000);
  const decoded = await decodeAudio(Int16Array.from(values, (value) => Math.round(value * 32767)));
  assert.equal(decoded.sample_rate_hz, 16000);
  assert.equal(decoded.source_kind, "int16");
  assert.equal(decoded.frames, 1600);
  assert.ok(Math.abs(decoded.duration_ms - 100) < 1e-9);
  assert.ok(Math.abs((decoded.samples[3] ?? 0) - (values[3] ?? 0)) < 1e-3);

  const from_container = await decodeAudio(writeWave(values, 16000));
  assert.equal(from_container.source_kind, "wave");
  assert.equal(from_container.frames, 1600);

  // A Blob's bytes are sniffed, so a WAVE inside one is reported as what it is.
  const from_blob = await decodeAudio(new Blob([writeWave(values, 16000)]));
  assert.equal(from_blob.source_kind, "wave");
  const raw_blob = await decodeAudio(new Blob([Int16Array.from(values, (v) => Math.round(v * 32767))]));
  assert.equal(raw_blob.source_kind, "blob");

  // Averaging two identical channels must not change the samples.
  const stereo = await decodeAudio({
    sampleRate: 16000,
    numberOfChannels: 2,
    getChannelData: () => values,
  });
  assert.equal(stereo.channel_count, 2);
  assert.equal(stereo.source_kind, "audio-buffer-like");
  assert.ok(Math.abs((stereo.samples[3] ?? 0) - (values[3] ?? 0)) < 1e-6);

  const nan = await rejection(decodeAudio(new Float32Array([0, Number.NaN])));
  assert.equal(nan.code, STATUS.invalid_argument);
  const unknown = await rejection(decodeAudio({} as Float32Array));
  assert.equal(unknown.code, STATUS.invalid_argument);
});
