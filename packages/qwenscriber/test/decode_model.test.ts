//! The whole browser path against the real 0.6B model: fetch a converted directory, move it into
//! the core's linear memory, compute log-mel with `qw_mel_compute`, decode, and detokenize.
//!
//! This is the test that says the runtime works rather than that it is wired: it needs a converted
//! model directory, so it is skipped -- with the path it looked for -- when none is present, and
//! `QWENSCRIBER_MODEL_DIR` points it at one. Nothing here is a fixture of the repository; weights
//! are never committed (see models/README.md).
//!
//! Run it on its own with:
//!
//!     node --test packages/qwenscriber/test/decode_model.test.ts

import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { WasmCore } from "../src/wasm/runtime.ts";
import { QwenscriberError } from "../src/errors.ts";
import { loadModel, type ModelDirectory } from "../src/decode.ts";
import { readWave } from "../src/audio/wav.ts";
import { FEATURE, formatAbiVersion, STATUS } from "../src/wasm/abi.ts";

const WASM_URL = new URL("../../../zig-out/bin/qwenscriber_core.wasm", import.meta.url);
const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const WAVE_PATH = fileURLToPath(new URL("../../../tests/fixtures/audio/asr_zh.wav", import.meta.url));

/**
 * Converted directory to run against. 2048 positions is what fits a 2 GiB instance, and q5 rather
 * than q4 because the 0.6B q4 conversion loses the first decision on this clip and produces no
 * tokens at all — see the quantization row in `docs/src/status.md`. Build it with:
 *
 *     qwenscriber-convert --input models/Qwen3-ASR-0.6B --output /tmp/qw-0.6b-q5-2k --quant q5 \
 *       --max-positions 2048
 */
const MODEL_DIR = process.env["QWENSCRIBER_MODEL_DIR"] ?? "/tmp/qw-0.6b-q5-2k";

/** Generation budget. The fixture is one short Chinese sentence, so this is generous. */
const MAX_TOKENS = 64;

function wasmBytes(): Uint8Array<ArrayBuffer> {
  if (!existsSync(WASM_URL)) {
    execFileSync("zig", ["build", "wasm"], { cwd: REPO_ROOT, stdio: "inherit" });
  }
  if (!existsSync(WASM_URL)) {
    throw new Error(`zig build wasm did not produce ${WASM_URL.href}`);
  }
  return new Uint8Array(readFileSync(WASM_URL));
}

/** A directory read straight off disk: the same loader the browser drives, without an HTTP server. */
function diskDirectory(base: string): ModelDirectory {
  return {
    source: base,
    async read(name: string): Promise<Uint8Array> {
      return new Uint8Array(readFileSync(`${base}/${name}`));
    },
  };
}

test(
  "the 0.6B model transcribes the Chinese fixture through the ABI",
  {
    timeout: 30 * 60 * 1000,
    skip: existsSync(`${MODEL_DIR}/manifest.json`)
      ? false
      : `no converted model directory at ${MODEL_DIR}; set QWENSCRIBER_MODEL_DIR (see models/README.md)`,
  },
  async () => {
    const core = await WasmCore.load(wasmBytes());
    const model = await loadModel(core, diskDirectory(MODEL_DIR));

    const requirements = model.requirements;
    const shardBytes = model.manifest.shards.reduce((total, shard) => total + shard.bytes, 0);
    assert.equal(requirements.weight_bytes, shardBytes, "the weights are the shard bytes supplied");
    assert.equal(
      requirements.total_bytes,
      requirements.weight_bytes + requirements.cache_bytes + requirements.scratch_bytes,
      "the reported plan adds up",
    );
    console.log(
      `abi ${formatAbiVersion(core.abiVersion())}, features 0x${core.features().toString(16)}, ` +
        `model ${model.modelId} (${model.manifest.quantization})`,
    );
    console.log(
      `plan: weights ${requirements.weight_bytes} B, cache ${requirements.cache_bytes} B ` +
        `(${requirements.max_positions} positions), scratch ${requirements.scratch_bytes} B, ` +
        `total ${requirements.total_bytes} B = ${(requirements.total_bytes / 2 ** 30).toFixed(3)} GiB; ` +
        `linear memory after load ${(core.memoryBytes() / 2 ** 30).toFixed(3)} GiB`,
    );
    assert.equal(
      core.features() & (FEATURE.model | FEATURE.decode),
      FEATURE.model | FEATURE.decode,
      "the module advertises the families this test needs",
    );

    const wave = readWave(new Uint8Array(readFileSync(WAVE_PATH)));
    const mel = core.melCompute(wave.samples);
    console.log(
      `audio: ${wave.samples.length} samples at ${wave.sample_rate_hz} Hz -> ${mel.frames} mel frames`,
    );
    // The reference reports 55 encoder steps and a 70-token prompt for this file, from 420 mel
    // frames at a 160-sample hop: the count the released processor produces for this clip, which the
    // stage comparison checks against its `input_features` of 128x420. A different frame count here
    // means the front end changed, not the model.
    assert.equal(mel.frames, 420, "the fixture is 420 mel frames at a 160-sample hop");

    const started = Date.now();
    const { tokens, text } = model.decode(mel.features, { maxTokens: MAX_TOKENS });
    console.log(
      `decode: ${tokens.length} tokens in ${Date.now() - started} ms -> ${JSON.stringify(text)}`,
    );
    console.log(`ids: ${Array.from(tokens).join(",")}`);

    // The fixture is Mandarin speech, so a transcript that came out of the model has to contain
    // Chinese characters. Anything else means the pipeline ran but produced the wrong text.
    assert.ok(tokens.length > 0, "the model produced at least one token");
    assert.match(text, /[\u4e00-\u9fff]/u, "the transcript contains Chinese characters");

    // The utterance is over, and the model is still resident for the next clip.
    const again = model.decode(mel.features, { maxTokens: 4 });
    assert.equal(again.tokens.length > 0, true, "a second utterance decodes too");

    model.dispose();
    assert.throws(
      () => model.decode(mel.features, { maxTokens: 4 }),
      /released/,
      "a released model refuses to decode",
    );
    // Releasing the model leaves the instance usable: the front end still computes, and the model
    // family reports that there is no model rather than anything being half-released.
    assert.equal(core.features() & FEATURE.mel, FEATURE.mel, "the front end is still advertised");
    assert.throws(
      () => core.modelRequirements(),
      (error: unknown) => {
        assert.ok(error instanceof QwenscriberError);
        assert.equal(error.code, STATUS.invalid_state);
        return true;
      },
      "the model family reports that no model is loaded",
    );
    assert.equal(core.melCompute(wave.samples).frames, mel.frames, "mel still computes");
    core.dispose();
  },
);
