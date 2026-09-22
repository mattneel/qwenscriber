//! The model family through the SDK: negotiation, the directory contract, the vocabulary table, and
//! the typed errors every call produces when it is made in the wrong state.
//!
//! What can be checked without weights lives here. The end-to-end transcript needs a converted
//! directory and lives in `decode_model.test.ts`. Run with `node --test test/`; if the wasm
//! artifact is missing the suite builds it, because a skipped ABI test and a passing one look the
//! same in CI output and only one of them means anything.

import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  FEATURE,
  FEATURE_ALL,
  FEATURE_NAMES,
  MODEL_CONFIG_BYTES,
  MODEL_REQUIREMENTS_BYTES,
  MODEL_REQUIREMENTS_OFFSET,
  SHARD_ALIGNMENT,
  STATUS,
  readModelRequirements,
} from "../src/wasm/abi.ts";
import { Allocation, WasmCore } from "../src/wasm/runtime.ts";
import { QwenscriberError } from "../src/errors.ts";
import {
  MODEL_MANIFEST_FORMAT_VERSION,
  MODEL_SHARDS_MAX,
  fetchModelDirectory,
  loadModel,
  parseManifest,
  type ModelDirectory,
} from "../src/decode.ts";

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

const MODULE = await WebAssembly.compile(wasmBytes());

async function freshCore(): Promise<WasmCore> {
  return WasmCore.load(MODULE);
}

/** A manifest shaped like the converter's, with every field overridable for a rejection case. */
function manifest(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({
    format_version: MODEL_MANIFEST_FORMAT_VERSION,
    model_id: "Qwen3-ASR-0.6B",
    quantization: "q4",
    config: { file: "config.bin", bytes: MODEL_CONFIG_BYTES },
    tokenizer: { file: "tokens.bin", bytes: 4096, count: 3 },
    shards: [{ name: "shard-000.qw", bytes: 8192 }],
    ...overrides,
  });
}

/** A directory backed by an in-memory map, so a loader test needs no filesystem and no network. */
function memoryDirectory(files: Record<string, Uint8Array>): ModelDirectory {
  return {
    source: "memory:",
    async read(name: string): Promise<Uint8Array> {
      const bytes = files[name];
      if (bytes === undefined) throw new Error(`no such file: ${name}`);
      return bytes;
    },
  };
}

/** An ABI vocabulary table: `count + 1` u32 offsets then the token bytes, either layout. */
function vocabularyTable(tokens: readonly string[], withCountPrefix: boolean): Uint8Array {
  const encoded = tokens.map((token) => new TextEncoder().encode(token));
  const pool = encoded.reduce((total, token) => total + token.length, 0);
  const offsetsBytes = (tokens.length + 1) * 4;
  const header = withCountPrefix ? 4 : 0;
  const bytes = new Uint8Array(header + offsetsBytes + pool);
  const view = new DataView(bytes.buffer);
  if (withCountPrefix) view.setUint32(0, tokens.length, true);
  let cursor = 0;
  encoded.forEach((token, index) => {
    view.setUint32(header + index * 4, cursor, true);
    bytes.set(token, header + offsetsBytes + cursor);
    cursor += token.length;
  });
  view.setUint32(header + tokens.length * 4, cursor, true);
  return bytes;
}

test("the feature mask names every family once", () => {
  assert.equal(FEATURE_ALL, 0b1_1111, "five families, five bits");
  assert.deepEqual(FEATURE_NAMES, ["mel", "tokenizer", "selftest", "model", "decode"]);
  const seen = new Set<number>();
  for (const name of FEATURE_NAMES) {
    const bit = FEATURE[name];
    assert.equal(Math.log2(bit) % 1, 0, `${name} is a single bit`);
    assert.equal(seen.has(bit), false, `${name} does not share a bit`);
    seen.add(bit);
  }
});

test("the requirements layout is the one abi.zig documents", () => {
  assert.equal(MODEL_REQUIREMENTS_BYTES, 48);
  const offsets = Object.entries(MODEL_REQUIREMENTS_OFFSET);
  assert.equal(offsets.length, 8);
  for (const [name, offset] of offsets) {
    assert.ok(offset + 4 <= MODEL_REQUIREMENTS_BYTES, `${name} fits in the struct`);
  }
  // The last two words are the token budget and the reserved word the module zeroes.
  assert.equal(MODEL_REQUIREMENTS_OFFSET.max_decode_tokens, 40);
  assert.equal(MODEL_REQUIREMENTS_OFFSET.reserved, 44);

  const bytes = new ArrayBuffer(MODEL_REQUIREMENTS_BYTES);
  const view = new DataView(bytes);
  view.setBigUint64(0, 1n, true);
  view.setBigUint64(8, 2n, true);
  view.setBigUint64(16, 3n, true);
  view.setBigUint64(24, 6n, true);
  view.setUint32(32, 2048, true);
  view.setUint32(36, 3000, true);
  view.setUint32(40, 256, true);
  const parsed = readModelRequirements(view, 0);
  assert.deepEqual(parsed, {
    weight_bytes: 1,
    cache_bytes: 2,
    scratch_bytes: 3,
    total_bytes: 6,
    max_positions: 2048,
    max_audio_frames: 3000,
    max_decode_tokens: 256,
    reserved: 0,
  });
});

test("a manifest is validated before any of it is fetched", () => {
  const parsed = parseManifest(manifest());
  assert.equal(parsed.model_id, "Qwen3-ASR-0.6B");
  assert.equal(parsed.tokenizer.count, 3);
  assert.deepEqual(parsed.shards, [{ name: "shard-000.qw", bytes: 8192 }]);

  const rejections: readonly [string, string][] = [
    ["not JSON at all", "valid JSON"],
    [manifest({ format_version: 99 }), "manifest format"],
    [manifest({ config: { file: "config.bin", bytes: 159 } }), "exactly 160 bytes"],
    [manifest({ tokenizer: { file: "tokens.bin", bytes: 4096 } }), "token count"],
    [manifest({ shards: [] }), "lists no shards"],
    [manifest({ shards: new Array(MODEL_SHARDS_MAX + 1).fill({ name: "s.qw", bytes: 1 }) }), "at most"],
    [manifest({ shards: [{ name: "", bytes: 8192 }] }), "file name"],
    [manifest({ shards: [{ name: "s.qw", bytes: 0 }] }), "byte count"],
  ];
  for (const [text, expected] of rejections) {
    assert.throws(
      () => parseManifest(text),
      (error: unknown) => {
        assert.ok(error instanceof QwenscriberError, `${expected}: typed error`);
        assert.equal(error.operation, "loadModel");
        assert.match(error.message, new RegExp(expected));
        return true;
      },
    );
  }
});

test("a directory fetch reports the URL it failed on", async () => {
  const requested: string[] = [];
  const directory = fetchModelDirectory("https://example.invalid/models/q4", {
    fetch: (async (url: string | URL | Request) => {
      requested.push(String(url));
      return new Response(new Uint8Array([1, 2, 3]), { status: 200 });
    }) as typeof globalThis.fetch,
  });
  assert.equal(directory.source, "https://example.invalid/models/q4");
  assert.equal((await directory.read("config.bin")).length, 3);
  assert.deepEqual(requested, ["https://example.invalid/models/q4/config.bin"]);

  const trailing = fetchModelDirectory(new URL("https://example.invalid/models/q4/"), {
    fetch: (async (url: string | URL | Request) => {
      requested.push(String(url));
      return new Response(new Uint8Array([0]), { status: 404, statusText: "Not Found" });
    }) as typeof globalThis.fetch,
  });
  await assert.rejects(
    () => trailing.read("manifest.json"),
    (error: unknown) => {
      assert.ok(error instanceof QwenscriberError);
      assert.equal(error.operation, "loadModel");
      assert.equal(error.context["http_status"], 404);
      return true;
    },
  );
  assert.equal(requested[1], "https://example.invalid/models/q4/manifest.json");
});

test("the module negotiates the model family instead of guessing", async () => {
  const core = await freshCore();
  try {
    const features = core.features();
    assert.equal(features & FEATURE.mel, FEATURE.mel);
    assert.equal(features & FEATURE.model, FEATURE.model, "the model family is advertised");
    assert.equal(features & FEATURE.decode, FEATURE.decode, "the decode family is advertised");
    // `SHARD_ALIGNMENT` is the alignment `container.File.parse` requires of every shard buffer.
    assert.equal(SHARD_ALIGNMENT, 16);
    assert.throws(
      () => core.modelRequirements(),
      (error: unknown) => {
        assert.ok(error instanceof QwenscriberError);
        assert.equal(error.code, STATUS.invalid_state);
        assert.equal(error.operation, "qw_model_requirements");
        return true;
      },
    );
  } finally {
    core.dispose();
  }
});

test("every model-family call fails typed, never with a trap", async () => {
  const core = await freshCore();
  try {
    core.modelBegin();
    const garbage = core.alloc(64, SHARD_ALIGNMENT);
    assert.throws(
      () => core.modelAddShard(garbage),
      (error: unknown) => {
        assert.ok(error instanceof QwenscriberError);
        assert.equal(error.code, STATUS.bad_magic, "a buffer that is not a shard");
        assert.equal(error.operation, "qw_model_add_shard");
        return true;
      },
    );

    // The SDK refuses a buffer the runtime could not read safely, before the call.
    const misaligned = core.alloc(64, SHARD_ALIGNMENT);
    // An offset pointer is the case the runtime would read as a corrupt shard; the SDK says what is
    // actually wrong before the call.
    const shiftedPointer = new Allocation(
      core,
      misaligned.pointer + 8,
      misaligned.size,
      SHARD_ALIGNMENT,
    );
    assert.throws(
      () => core.modelAddShard(shiftedPointer),
      (error: unknown) => {
        assert.ok(error instanceof QwenscriberError);
        assert.equal(error.code, STATUS.invalid_argument);
        assert.equal(error.context["alignment"], SHARD_ALIGNMENT);
        return true;
      },
    );
    const released = core.alloc(64, SHARD_ALIGNMENT);
    released.dispose();
    assert.throws(() => core.modelAddShard(released), /released/);

    const config = new Uint8Array(MODEL_CONFIG_BYTES);
    assert.throws(
      () => core.modelFinish(config),
      (error: unknown) => {
        assert.ok(error instanceof QwenscriberError);
        assert.equal(error.code, STATUS.bad_magic);
        assert.equal(error.operation, "qw_model_finish");
        return true;
      },
    );
    assert.throws(() => core.modelFinish(new Uint8Array(8)), /exactly 160 bytes/);

    // Decoding without a model is a state error, and it names the export that refused it.
    assert.throws(
      () => core.decodeUtterance(new Float32Array(128 * 2), 8),
      (error: unknown) => {
        assert.ok(error instanceof QwenscriberError);
        assert.equal(error.code, STATUS.invalid_state);
        return true;
      },
    );

    garbage.dispose();
    misaligned.dispose();
  } finally {
    core.dispose();
  }
});

test("a vocabulary table is installed from either layout, and a shifted one is refused", async () => {
  const core = await freshCore();
  try {
    // The byte alphabet: a space is U+0120, which is what the table stores.
    const tokens = ["H", "i", "\u0120", "there"];
    const ids = Uint32Array.from([0, 1, 2, 3]);
    for (const withCountPrefix of [false, true]) {
      core.setVocabularyFromTable(vocabularyTable(tokens, withCountPrefix), tokens.length);
      assert.equal(
        core.detokenize(ids),
        "Hi there",
        `the table ${withCountPrefix ? "with" : "without"} a count prefix decodes`,
      );
    }

    // A table whose first offset is not zero is not a vocabulary, and saying so beats decoding it
    // into the middle of every token.
    const shifted = vocabularyTable(tokens, false);
    new DataView(shifted.buffer).setUint32(0, 7, true);
    assert.throws(
      () => core.setVocabularyFromTable(shifted, tokens.length),
      (error: unknown) => {
        assert.ok(error instanceof QwenscriberError);
        assert.equal(error.code, STATUS.invalid_encoding);
        assert.equal(error.context["first_offset"], 7);
        return true;
      },
    );
    assert.throws(() => core.setVocabularyFromTable(shifted, 0), /1 to/);
  } finally {
    core.dispose();
  }
});

test("a failed model load leaves the instance usable", async () => {
  const core = await freshCore();
  try {
    // The manifest promises more bytes than the shard holds, which is what a truncated download
    // looks like. The loader must refuse it without leaving the runtime mid-load.
    const directory = memoryDirectory({
      "manifest.json": new TextEncoder().encode(manifest()),
      "config.bin": new Uint8Array(MODEL_CONFIG_BYTES),
      "shard-000.qw": new Uint8Array(16),
      "tokens.bin": new Uint8Array(4096),
    });
    await assert.rejects(
      () => loadModel(core, directory),
      (error: unknown) => {
        assert.ok(error instanceof QwenscriberError);
        assert.equal(error.code, STATUS.truncated);
        assert.equal(error.operation, "loadModel");
        assert.equal(error.context["file"], "shard-000.qw");
        return true;
      },
    );

    // Nothing is half-built: the next call sees a fresh model attempt, not a stale one.
    core.modelBegin();
    assert.throws(
      () => core.modelRequirements(),
      (error: unknown) => {
        assert.ok(error instanceof QwenscriberError);
        assert.equal(error.code, STATUS.invalid_state);
        return true;
      },
    );
    assert.throws(() => core.modelFinish(new Uint8Array(MODEL_CONFIG_BYTES)), /bad magic/);
  } finally {
    core.dispose();
  }
});
