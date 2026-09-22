#!/usr/bin/env node
// Exercises the freestanding WASM module through its public ABI.
//
// This is not a unit test of the Zig code -- that is `zig build test`. It checks
// the things only a real JavaScript engine can check: that the module
// instantiates, that the exported surface matches ABI v1, that argument
// validation rejects bad pointers and handles with the documented status codes,
// and that the freestanding build computes the same bits as the host build
// (the golden hashes below come from `zig build selftest`).
//
// Usage: node tools/wasm_selftest.mjs zig-out/bin/qwenscriber_core.wasm

import { readFileSync } from "node:fs";

// Values printed by `zig build selftest` from the host build. The freestanding
// build must produce the same bits; if a numeric routine changes, both move
// together and this diff is the review signal.
const GOLDEN = {
    abiVersion: 0x00010000,
    coreVersion: 1,
    failures: 0,
    quantHash: 0xf247917b05691fd1n,
    melHash: 0xfec7aacfeea768d7n,
    loudestBand: 42,
    silenceValue: -1.5,
    /** `abi.features`: mel | tokenizer | selftest | model | decode. */
    features: 0b1_1111,
};

const STATUS = {
    ok: 0,
    invalidArgument: -1,
    outOfMemory: -2,
    unsupported: -3,
    badMagic: -4,
    badVersion: -5,
    checksumMismatch: -6,
    shapeMismatch: -7,
    invalidState: -8,
    limitExceeded: -9,
    notFound: -10,
    truncated: -11,
    invalidEncoding: -12,
    unsupportedModel: -13,
    audioTooLong: -14,
};

let failures = 0;

function check(condition, description, detail = "") {
    if (condition) {
        console.log(`ok   ${description}`);
    } else {
        failures += 1;
        console.log(`FAIL ${description}${detail ? `: ${detail}` : ""}`);
    }
}

function equal(actual, expected, description) {
    check(actual === expected, description, `expected ${expected}, got ${actual}`);
}

const modulePath = process.argv[2];
if (!modulePath) {
    console.error("usage: node tools/wasm_selftest.mjs <module.wasm> [model dir]");
    process.exit(2);
}

const bytes = readFileSync(modulePath);
// No imports: a genuine wasm32-freestanding module resolves everything itself.
const { instance } = await WebAssembly.instantiate(bytes, {});
const e = instance.exports;

// Growing linear memory detaches every ArrayBuffer view over it, and
// `qw_alloc` grows memory, so views must be rebuilt from `e.memory.buffer`
// after every allocation. This is the single most common WASM integration bug,
// so the harness never caches a view.
const mem = () => new Uint8Array(e.memory.buffer);

// Every pointer an export returns is a `u32` linear-memory offset, and a wasm result arrives in
// JavaScript as a signed i32: above 2 GiB -- which is where a model's weights and cache put the
// allocator -- the raw value is negative. Every `qw_alloc` result is masked back with `>>> 0`.

console.log(`module: ${modulePath} (${bytes.length} bytes)`);

// --- exported surface -------------------------------------------------------

const required = [
    "qw_abi_version",
    "qw_core_version",
    "qw_features",
    "qw_create",
    "qw_destroy",
    "qw_alloc",
    "qw_free",
    "qw_memory_bytes",
    "qw_error_message_ptr",
    "qw_error_message_len",
    "qw_mel_frames_for_samples",
    "qw_mel_min_samples",
    "qw_mel_compute",
    "qw_mel_padding_value",
    "qw_tokenizer_set",
    "qw_detokenize",
    "qw_selftest",
    "qw_model_begin",
    "qw_model_add_shard",
    "qw_model_finish",
    "qw_model_set_cache_format",
    "qw_model_requirements",
    "qw_model_audio_config",
    "qw_decode_begin",
    "qw_decode_step",
    "qw_decode_tokens",
    "qw_decode_end",
];
for (const name of required) {
    check(typeof e[name] === "function", `exports ${name}`);
}

equal(e.qw_abi_version(), GOLDEN.abiVersion, "ABI version is v1.0");
equal(e.qw_core_version(), GOLDEN.coreVersion, "core version matches the host build");

// --- feature negotiation ----------------------------------------------------

{
    // The mask is the SDK's contract for which families it may call. A bit that
    // is set without exports, or exports without a bit, is a module that lies
    // about itself, so both directions are checked here.
    const features = e.qw_features();
    equal(features, GOLDEN.features, "feature mask names every compiled family");
    for (const [name, bit] of [
        ["mel", 1 << 0],
        ["tokenizer", 1 << 1],
        ["selftest", 1 << 2],
        ["model", 1 << 3],
        ["decode", 1 << 4],
    ]) {
        check((features & bit) !== 0, `feature bit: ${name}`);
    }
}

const melBins = 128;

// --- lifecycle --------------------------------------------------------------

equal(e.qw_destroy(1), STATUS.invalidArgument, "destroy before create is rejected");
const handle = e.qw_create();
equal(handle, 1, "create returns handle 1");
equal(e.qw_create(), 0, "a second create is refused");
equal(e.qw_detokenize(handle, 0, 0, 0, 0, 0), STATUS.invalidState, "detokenize needs a vocabulary");

// --- error messages ---------------------------------------------------------

{
    const code = STATUS.badMagic;
    const pointer = e.qw_error_message_ptr(code);
    const length = e.qw_error_message_len(code);
    const text = new TextDecoder().decode(mem().subarray(pointer, pointer + length));
    equal(text, "bad magic", "error messages are readable through linear memory");
    const unknown = e.qw_error_message_len(99999);
    check(unknown > 0, "unknown status codes still produce a message");
}

// --- self-test --------------------------------------------------------------

{
    const resultPtr = e.qw_alloc(32, 8);
    check(resultPtr !== 0, "allocated the self-test result buffer");
    equal(e.qw_selftest(resultPtr), STATUS.ok, "selftest runs");
    const view = new DataView(e.memory.buffer, resultPtr, 32);
    const report = {
        failures: view.getUint32(0, true),
        loudestBand: view.getUint32(4, true),
        silenceValue: view.getFloat32(8, true),
        quantHash: view.getBigUint64(16, true),
        melHash: view.getBigUint64(24, true),
    };
    equal(report.failures, GOLDEN.failures, "no self-test check failed");
    equal(report.loudestBand, GOLDEN.loudestBand, "1 kHz tone lands in the expected mel band");
    check(
        Math.abs(report.silenceValue - GOLDEN.silenceValue) < 1e-6,
        "silence produces the documented log-mel floor",
        `got ${report.silenceValue}`,
    );
    equal(
        report.quantHash,
        GOLDEN.quantHash,
        "quantization packing matches the host build",
    );
    equal(report.melHash, GOLDEN.melHash, "log-mel output matches the host build");
    e.qw_free(resultPtr, 32, 8);
}

// --- log-mel ----------------------------------------------------------------

{
    const sampleCount = 16000;
    const samplesPtr = e.qw_alloc(sampleCount * 4, 4) >>> 0;
    const frames = e.qw_mel_frames_for_samples(sampleCount);
    equal(frames, 100, "frame count for one second of audio");
    const outPtr = e.qw_alloc(frames * melBins * 4, 4);
    const resultPtr = e.qw_alloc(16, 4);

    const samples = new Float32Array(e.memory.buffer, samplesPtr, sampleCount);
    for (let index = 0; index < sampleCount; index += 1) {
        samples[index] = Math.sin((2 * Math.PI * 1000 * index) / 16000);
    }

    equal(
        e.qw_mel_compute(handle, samplesPtr, sampleCount, outPtr, frames * melBins * 4, resultPtr),
        STATUS.ok,
        "mel compute succeeds",
    );
    const view = new DataView(e.memory.buffer, resultPtr, 16);
    equal(view.getUint32(0, true), frames, "reported frame count");
    equal(view.getUint32(4, true), frames * melBins * 4, "reported byte count");
    const globalMax = view.getFloat32(8, true);
    check(Number.isFinite(globalMax), "global maximum is finite", `got ${globalMax}`);

    const features = new Float32Array(e.memory.buffer, outPtr, frames * melBins);
    let loudest = 0;
    let loudestValue = -Infinity;
    for (let bin = 0; bin < melBins; bin += 1) {
        const value = features[bin * frames + 40];
        if (value > loudestValue) {
            loudestValue = value;
            loudest = bin;
        }
    }
    equal(loudest, GOLDEN.loudestBand, "1 kHz tone lands in the loudest mel band from JavaScript");

    const padding = e.qw_mel_padding_value(globalMax);
    check(Number.isFinite(padding), "padding value is finite");

    // Argument validation.
    equal(
        e.qw_mel_compute(handle, samplesPtr, sampleCount, outPtr, frames * melBins * 4 - 4, resultPtr),
        STATUS.shapeMismatch,
        "an undersized output buffer is refused",
    );
    equal(
        e.qw_mel_compute(handle, samplesPtr + 1, sampleCount, outPtr, frames * melBins * 4, resultPtr),
        STATUS.invalidArgument,
        "a misaligned sample pointer is refused",
    );
    equal(
        e.qw_mel_compute(handle, e.qw_memory_bytes(), 16, outPtr, frames * melBins * 4, resultPtr),
        STATUS.invalidArgument,
        "a pointer past the end of memory is refused",
    );
    equal(
        e.qw_mel_compute(0, samplesPtr, sampleCount, outPtr, frames * melBins * 4, resultPtr),
        STATUS.invalidArgument,
        "a bogus handle is refused",
    );
    equal(
        e.qw_mel_compute(handle, samplesPtr, 0, outPtr, frames * melBins * 4, resultPtr),
        STATUS.invalidArgument,
        "zero samples is refused",
    );

    e.qw_free(resultPtr, 16, 4);
    e.qw_free(outPtr, frames * melBins * 4, 4);
    e.qw_free(samplesPtr, sampleCount * 4, 4);
}

// --- tokenizer --------------------------------------------------------------

{
    // A miniature vocabulary: "H", "i", the byte-alphabet space, "there", and a
    // two-byte UTF-8 character split across tokens.
    const tokens = ["H", "i", "\u0120", "there", "\u00e4\u00bd", "\u00a0"];
    const encoded = tokens.map((token) => new TextEncoder().encode(token));
    const bytesLength = encoded.reduce((total, token) => total + token.length, 0);

    const bytesPtr = e.qw_alloc(bytesLength, 1);
    const offsetsPtr = e.qw_alloc((tokens.length + 1) * 4, 4);
    const descriptorPtr = e.qw_alloc(24, 4);

    const byteView = new Uint8Array(e.memory.buffer, bytesPtr, bytesLength);
    const offsetView = new DataView(e.memory.buffer, offsetsPtr, (tokens.length + 1) * 4);
    let cursor = 0;
    encoded.forEach((token, index) => {
        offsetView.setUint32(index * 4, cursor, true);
        byteView.set(token, cursor);
        cursor += token.length;
    });
    offsetView.setUint32(tokens.length * 4, cursor, true);

    const descriptor = new DataView(e.memory.buffer, descriptorPtr, 24);
    descriptor.setUint32(0, offsetsPtr, true);
    descriptor.setUint32(4, (tokens.length + 1) * 4, true);
    descriptor.setUint32(8, bytesPtr, true);
    descriptor.setUint32(12, bytesLength, true);
    descriptor.setUint32(16, tokens.length, true);
    descriptor.setUint32(20, 0, true);

    equal(e.qw_tokenizer_set(handle, descriptorPtr), STATUS.ok, "vocabulary accepted");

    const idsPtr = e.qw_alloc(4 * 4, 4);
    const idView = new DataView(e.memory.buffer, idsPtr, 16);
    [0, 1, 2, 3].forEach((id, index) => idView.setUint32(index * 4, id, true));
    const outPtr = e.qw_alloc(64, 1);
    const writtenPtr = e.qw_alloc(4, 4);
    equal(e.qw_detokenize(handle, idsPtr, 4, outPtr, 64, writtenPtr), STATUS.ok, "detokenize");
    const written = new DataView(e.memory.buffer, writtenPtr, 4).getUint32(0, true);
    equal(written, 8, "detokenized byte count");
    const text = new TextDecoder().decode(new Uint8Array(e.memory.buffer, outPtr, written));
    equal(text, "Hi there", "detokenized text");

    // A token outside the byte alphabet must be reported, not skipped.
    [4, 5].forEach((id, index) => idView.setUint32(index * 4, id, true));
    equal(
        e.qw_detokenize(handle, idsPtr, 2, outPtr, 64, writtenPtr),
        STATUS.invalidEncoding,
        "code points outside the alphabet are rejected",
    );

    // Unknown ids and undersized outputs are reported too.
    idView.setUint32(0, 99, true);
    equal(e.qw_detokenize(handle, idsPtr, 1, outPtr, 64, writtenPtr), STATUS.notFound, "unknown id");
    idView.setUint32(0, 0, true);
    equal(
        e.qw_detokenize(handle, idsPtr, 4, outPtr, 3, writtenPtr),
        STATUS.invalidEncoding,
        "an undersized output buffer is refused",
    );

    for (const [ptr, size, align] of [
        [writtenPtr, 4, 4],
        [outPtr, 64, 1],
        [idsPtr, 16, 4],
        [descriptorPtr, 24, 4],
        [offsetsPtr, (tokens.length + 1) * 4, 4],
        [bytesPtr, bytesLength, 1],
    ]) {
        e.qw_free(ptr, size, align);
    }
}

// --- model family: state machine and shard validation -----------------------

/** Offsets of the fields this harness reads out of a `model_config.Config` (160 bytes). */
const CONFIG_OFFSET = {
    audio_d_model: 16,
    audio_layers: 20,
    audio_attention_heads: 24,
    audio_ffn_dim: 28,
    audio_downsample_hidden: 32,
    audio_n_window: 36,
    audio_n_window_infer: 40,
    audio_max_position_steps: 44,
    audio_output_dim: 48,
    audio_mel_bins: 52,
    audio_layer_norm_eps: 56,
    text_layers: 64,
    text_key_value_heads: 72,
    text_head_dim: 76,
    max_positions: 96,
    max_decode_tokens: 100,
};

/** FNV-1a 64 over `bytes`, truncated to 32 bits: what the container records. */
function fnv1a32(bytes) {
    const prime = 0x100000001b3n;
    const mask = (1n << 64n) - 1n;
    let hash = 0xcbf29ce484222325n;
    for (const byte of bytes) hash = ((hash ^ BigInt(byte)) * prime) & mask;
    return Number(hash & 0xffffffffn);
}

/**
 * A one-tensor f32 shard: 32-byte header, one 40-byte index entry, then the
 * payload at a 256-byte boundary. The smallest container `File.parse` accepts,
 * built here so the shard checks have a positive case as well as negative ones.
 */
function buildShard(values) {
    const payloadOffset = 256;
    const payloadBytes = values.length * 4;
    const storage = new Uint8Array(payloadOffset + payloadBytes);
    const view = new DataView(storage.buffer);
    storage.set(new TextEncoder().encode("QWSHARD1"), 0);
    view.setUint32(8, 1, true); // format_version
    view.setUint32(12, 40, true); // index_len_bytes
    view.setUint32(16, payloadOffset, true);
    view.setUint32(20, payloadBytes, true);
    view.setUint32(24, 1, true); // tensor_count
    view.setUint16(32, 0, true); // kind
    view.setUint16(34, 0, true); // layer
    view.setUint8(36, 0); // dtype.Format.f32
    view.setUint8(37, 1); // rank
    view.setUint32(40, values.length, true); // dims[0]
    view.setBigUint64(56, 0n, true); // offset_bytes
    view.setBigUint64(64, BigInt(payloadBytes), true); // len_bytes
    new Float32Array(storage.buffer, payloadOffset, values.length).set(values);
    view.setUint32(28, fnv1a32(new Uint8Array(storage.buffer, payloadOffset, payloadBytes)), true);
    return storage;
}

{
    const requirementsPtr = e.qw_alloc(48, 8) >>> 0;
    const configPtr = e.qw_alloc(160, 4) >>> 0;
    check(requirementsPtr !== 0 && configPtr !== 0, "allocated the model-family buffers");

    // Every call in the model family is refused before its state, with the
    // documented status rather than a trap.
    equal(
        e.qw_model_requirements(handle, requirementsPtr),
        STATUS.invalidState,
        "requirements before begin are refused",
    );
    const audioConfigPtr = e.qw_alloc(72, 8) >>> 0;
    check(audioConfigPtr !== 0, "allocated the audio config buffer");
    equal(
        e.qw_model_audio_config(handle, audioConfigPtr),
        STATUS.invalidState,
        "audio config before begin is refused",
    );
    equal(
        e.qw_model_add_shard(handle, requirementsPtr, 16),
        STATUS.invalidState,
        "a shard before begin is refused",
    );
    equal(
        e.qw_model_finish(handle, configPtr, 160),
        STATUS.invalidState,
        "finish before begin is refused",
    );
    equal(
        e.qw_decode_begin(handle, requirementsPtr, 512, 4),
        STATUS.invalidState,
        "decode before a model exists is refused",
    );
    equal(
        e.qw_decode_step(handle, requirementsPtr),
        STATUS.invalidState,
        "decode step before decode begin is refused",
    );
    equal(
        e.qw_decode_tokens(handle, requirementsPtr, 4),
        STATUS.invalidState,
        "decode tokens before decode begin are refused",
    );
    equal(e.qw_decode_end(handle), STATUS.invalidState, "decode end before decode begin is refused");

    equal(e.qw_model_begin(handle), STATUS.ok, "model begin");
    // Arguments are validated before state, so a config that is not a
    // configuration is reported as such rather than as a state error.
    equal(
        e.qw_model_finish(handle, configPtr, 160),
        STATUS.badMagic,
        "a config that is not a configuration is refused",
    );
    equal(
        e.qw_model_requirements(handle, requirementsPtr),
        STATUS.invalidState,
        "requirements need a loaded model",
    );
    equal(
        e.qw_model_finish(handle, e.qw_memory_bytes(), 160),
        STATUS.invalidArgument,
        "a config outside linear memory is refused",
    );
    equal(
        e.qw_model_finish(handle, configPtr, 4),
        STATUS.truncated,
        "a config that is not 160 bytes is refused",
    );

    // Malformed shards, one per rejection path.
    const garbage = e.qw_alloc(64, 16) >>> 0;
    new Uint8Array(e.memory.buffer, garbage, 64).fill(0);
    equal(
        e.qw_model_add_shard(handle, garbage, 64),
        STATUS.badMagic,
        "a shard with no magic is refused",
    );
    equal(
        e.qw_model_add_shard(handle, garbage + 1, 64),
        STATUS.invalidArgument,
        "a misaligned shard is refused",
    );
    equal(
        e.qw_model_add_shard(handle, e.qw_memory_bytes(), 64),
        STATUS.invalidArgument,
        "a shard past the end of memory is refused",
    );
    equal(
        e.qw_model_add_shard(handle, garbage, 0),
        STATUS.truncated,
        "an empty shard is refused",
    );

    // The positive space: a well-formed shard, then both ways of duplicating it.
    const good = buildShard(new Float32Array([1.5, -2.25]));
    const goodPtr = e.qw_alloc(good.length, 16) >>> 0;
    const duplicatePtr = e.qw_alloc(good.length, 16) >>> 0;
    check(goodPtr !== 0 && duplicatePtr !== 0, "allocated the shard buffers");
    new Uint8Array(e.memory.buffer, goodPtr, good.length).set(good);
    new Uint8Array(e.memory.buffer, duplicatePtr, good.length).set(good);
    equal(
        e.qw_model_add_shard(handle, goodPtr, good.length),
        STATUS.ok,
        "a well-formed shard is accepted",
    );
    equal(
        e.qw_model_add_shard(handle, goodPtr, good.length),
        STATUS.invalidArgument,
        "the same buffer twice is refused",
    );
    equal(
        e.qw_model_add_shard(handle, duplicatePtr, good.length),
        STATUS.invalidArgument,
        "the same payload in another buffer is refused",
    );

    const corrupt = Uint8Array.from(good);
    corrupt[corrupt.length - 1] ^= 0xff;
    const corruptPtr = e.qw_alloc(corrupt.length, 16) >>> 0;
    new Uint8Array(e.memory.buffer, corruptPtr, corrupt.length).set(corrupt);
    equal(
        e.qw_model_add_shard(handle, corruptPtr, corrupt.length),
        STATUS.checksumMismatch,
        "a shard whose payload does not match its checksum is refused",
    );
    equal(
        e.qw_model_add_shard(handle, goodPtr, good.length - 8),
        STATUS.truncated,
        "a shard cut short is refused",
    );

    // Beginning again releases the partial model, so a load that failed can be
    // retried without destroying the instance.
    equal(e.qw_model_begin(handle), STATUS.ok, "begin releases the partial model");
    equal(
        e.qw_model_requirements(handle, requirementsPtr),
        STATUS.invalidState,
        "the restarted model has nothing to report",
    );
    equal(
        e.qw_model_add_shard(handle, goodPtr, good.length),
        STATUS.ok,
        "a shard may be added again after a restart",
    );

    e.qw_free(garbage, 64, 16);
    e.qw_free(corruptPtr, corrupt.length, 16);
    e.qw_free(duplicatePtr, good.length, 16);
    e.qw_free(goodPtr, good.length, 16);
    e.qw_free(configPtr, 160, 4);
    e.qw_free(requirementsPtr, 48, 8);
}

// --- model family: a real converted model directory, when one is given ------

/**
 * Loads a converted model directory through the ABI and checks what the model
 * reports and computes.
 *
 * This is the part of the harness that needs weights, so it runs only when one
 * is named on the command line; `zig build test-wasm` passes a module and
 * nothing else, and a skip is printed rather than silently passing.
 */
function exerciseModel(modelDir) {
    console.log(`\nmodel: ${modelDir}`);
    const manifest = JSON.parse(readFileSync(`${modelDir}/manifest.json`, "utf8"));
    const config = new Uint8Array(readFileSync(`${modelDir}/${manifest.config.file}`));
    if (config.length !== 160) {
        check(false, "config.bin is 160 bytes", `got ${config.length}`);
        return;
    }
    const configView = new DataView(config.buffer, config.byteOffset, config.byteLength);
    // A wasm32 instance tops out at 2 GiB of linear memory, and a configuration's own position
    // budget can ask for more than fits at f32: 8192 positions of key/value cache need 1.88 GB on
    // top of the weights, which `qw_model_finish` refuses with `out_of_memory`. `--max-positions`
    // exists for the host tools for this reason; here the budget is lowered before the load rather
    // than the checks skipped, so every expectation below reads the value the load actually used.
    const declaredPositions = configView.getUint32(CONFIG_OFFSET.max_positions, true);
    const positions = Math.min(declaredPositions, 2048);
    if (positions !== declaredPositions) {
        configView.setUint32(CONFIG_OFFSET.max_positions, positions, true);
    }
    console.log(`positions: ${declaredPositions} declared, ${positions} loaded`);
    const shardBuffers = manifest.shards.map((shard) => {
        const bytes = new Uint8Array(readFileSync(`${modelDir}/${shard.name}`));
        if (bytes.length !== shard.bytes) {
            check(false, `${shard.name} matches the manifest byte count`, `got ${bytes.length}`);
        }
        return bytes;
    });

    equal(e.qw_model_begin(handle), STATUS.ok, "model begin");
    const configPtr = e.qw_alloc(config.length, 4) >>> 0;
    new Uint8Array(e.memory.buffer, configPtr, config.length).set(config);
    const shardPtrs = [];
    for (const [index, bytes] of shardBuffers.entries()) {
        const ptr = e.qw_alloc(bytes.length, 16) >>> 0;
        check(ptr !== 0, `allocated shard ${index} (${bytes.length} bytes)`);
        new Uint8Array(e.memory.buffer, ptr, bytes.length).set(bytes);
        shardPtrs.push(ptr);
        equal(e.qw_model_add_shard(handle, ptr, bytes.length), STATUS.ok, `shard ${index} accepted`);
    }
    equal(e.qw_model_finish(handle, configPtr, config.length), STATUS.ok, "the model loads");

    // The same calls that are legal after `begin` are illegal after `finish`.
    equal(
        e.qw_model_add_shard(handle, shardPtrs[0], shardBuffers[0].length),
        STATUS.invalidState,
        "a shard added after finish is refused",
    );
    equal(
        e.qw_model_finish(handle, configPtr, config.length),
        STATUS.invalidState,
        "finishing twice is refused",
    );
    equal(e.qw_model_begin(handle), STATUS.ok, "begin releases the loaded model");
    equal(
        e.qw_model_finish(handle, configPtr, config.length),
        STATUS.invalidState,
        "finish with no shard is refused",
    );
    equal(e.qw_model_begin(handle), STATUS.ok, "model begin again");
    for (const [index, ptr] of shardPtrs.entries()) {
        equal(
            e.qw_model_add_shard(handle, ptr, shardBuffers[index].length),
            STATUS.ok,
            `shard ${index} accepted again`,
        );
    }
    equal(
        e.qw_model_finish(handle, configPtr, config.length),
        STATUS.ok,
        "the model loads again after being released",
    );

    // Requirements: every number is checked against the arithmetic it claims.
    const statePtr = e.qw_alloc(48, 8) >>> 0;
    equal(
        e.qw_model_requirements(handle, statePtr),
        STATUS.ok,
        "requirements for the loaded model",
    );
    const view = new DataView(e.memory.buffer, statePtr, 48);
    const requirements = {
        weightBytes: Number(view.getBigUint64(0, true)),
        cacheBytes: Number(view.getBigUint64(8, true)),
        scratchBytes: Number(view.getBigUint64(16, true)),
        totalBytes: Number(view.getBigUint64(24, true)),
        maxPositions: view.getUint32(32, true),
        maxAudioFrames: view.getUint32(36, true),
        maxDecodeTokens: view.getUint32(40, true),
    };
    const weightBytes = shardBuffers.reduce((total, bytes) => total + bytes.length, 0);
    const layers = configView.getUint32(CONFIG_OFFSET.text_layers, true);
    const keyValueWidth = configView.getUint32(CONFIG_OFFSET.text_key_value_heads, true) *
        configView.getUint32(CONFIG_OFFSET.text_head_dim, true);
    const maxPositions = configView.getUint32(CONFIG_OFFSET.max_positions, true);
    equal(requirements.weightBytes, weightBytes, "weight bytes are the shard bytes supplied");
    equal(
        requirements.cacheBytes,
        2 * layers * keyValueWidth * maxPositions * 4,
        "key/value cache bytes match 2 x layers x key/value width x positions x 4",
    );
    equal(
        requirements.totalBytes,
        requirements.weightBytes + requirements.cacheBytes + requirements.scratchBytes,
        "total is weight + cache + scratch",
    );
    equal(requirements.maxPositions, maxPositions, "max positions is the configuration's");
    equal(
        requirements.maxDecodeTokens,
        configView.getUint32(CONFIG_OFFSET.max_decode_tokens, true),
        "max decode tokens is the configuration's",
    );
    equal(requirements.maxAudioFrames, 3000, "max audio frames is the mel frontend's capacity");
    check(requirements.scratchBytes > 0, "scratch is reported", `${requirements.scratchBytes}`);
    check(
        requirements.scratchBytes < 64 * 1024 * 1024,
        "scratch stays under 64 MiB",
        `${requirements.scratchBytes} bytes`,
    );
    console.log(
        `requirements: weights ${requirements.weightBytes} B, cache ${requirements.cacheBytes} B, ` +
        `scratch ${requirements.scratchBytes} B, total ${requirements.totalBytes} B ` +
        `(${(requirements.totalBytes / 2 ** 30).toFixed(3)} GiB), positions ` +
        `${requirements.maxPositions}, audio frames ${requirements.maxAudioFrames}`,
    );

    // The audio tower's geometry, which a caller dispatching the tower itself reads. Each value is
    // checked against the configuration the manifest names, and each derived value against an
    // independent derivation from it: a core helper that changed its mind would fail here rather
    // than silently moving a GPU dispatch off the path the reference transcript came from.
    const audioConfigPtr = e.qw_alloc(72, 8) >>> 0;
    check(audioConfigPtr !== 0, "allocated the loaded-model audio config buffer");
    equal(
        e.qw_model_audio_config(handle, audioConfigPtr),
        STATUS.ok,
        "audio config for the loaded model",
    );
    const audioView = new DataView(e.memory.buffer, audioConfigPtr, 72);
    const audio = {
        dModel: audioView.getUint32(0, true),
        layers: audioView.getUint32(4, true),
        attentionHeads: audioView.getUint32(8, true),
        headDim: audioView.getUint32(12, true),
        ffnDim: audioView.getUint32(16, true),
        downsampleHidden: audioView.getUint32(20, true),
        nWindow: audioView.getUint32(24, true),
        nWindowInfer: audioView.getUint32(28, true),
        chunkFrames: audioView.getUint32(32, true),
        frequencyBins: audioView.getUint32(36, true),
        convOutInputFeatures: audioView.getUint32(40, true),
        chunkSteps: audioView.getUint32(44, true),
        maxPositionSteps: audioView.getUint32(48, true),
        outputDim: audioView.getUint32(52, true),
        melBins: audioView.getUint32(56, true),
        layerNormEps: audioView.getFloat32(60, true),
        reserved0: audioView.getUint32(64, true),
        reserved1: audioView.getUint32(68, true),
    };
    const configAudio = (name) => configView.getUint32(CONFIG_OFFSET[name], true);
    equal(audio.dModel, configAudio("audio_d_model"), "audio d_model is the configuration's");
    equal(audio.layers, configAudio("audio_layers"), "audio layers are the configuration's");
    equal(audio.attentionHeads, configAudio("audio_attention_heads"), "audio heads are the configuration's");
    equal(audio.ffnDim, configAudio("audio_ffn_dim"), "audio feed-forward width is the configuration's");
    equal(audio.downsampleHidden, configAudio("audio_downsample_hidden"), "downsample width is the configuration's");
    equal(audio.nWindow, configAudio("audio_n_window"), "window length is the configuration's");
    equal(audio.nWindowInfer, configAudio("audio_n_window_infer"), "inference window is the configuration's");
    equal(audio.maxPositionSteps, configAudio("audio_max_position_steps"), "position rows are the configuration's");
    equal(audio.outputDim, configAudio("audio_output_dim"), "projector width is the configuration's");
    equal(audio.melBins, configAudio("audio_mel_bins"), "mel bins are the configuration's");
    check(
        audio.layerNormEps === configView.getFloat32(CONFIG_OFFSET.audio_layer_norm_eps, true),
        "layer norm epsilon is the configuration's",
        `got ${audio.layerNormEps}`,
    );
    equal(audio.headDim, audio.dModel / audio.attentionHeads, "head width is d_model over heads");
    equal(audio.chunkFrames, 2 * audio.nWindow, "a chunk is two windows of mel frames");
    equal(
        audio.frequencyBins,
        audio.melBins / 8,
        "three stride-two convolutions leave an eighth of the mel bins",
    );
    equal(
        audio.convOutInputFeatures,
        audio.downsampleHidden * audio.frequencyBins,
        "the downsample input is its width times the remaining bins",
    );
    // The convolution stack's own arithmetic -- stride two, padding one, a 3x3 kernel -- applied
    // three times to the frames one chunk consumes.
    let chunkSteps = audio.chunkFrames;
    for (let index = 0; index < 3; index += 1) chunkSteps = Math.floor((chunkSteps - 1) / 2) + 1;
    equal(audio.chunkSteps, chunkSteps, "chunk steps is the stack's output for one chunk");
    equal(
        audio.maxPositionSteps >= audio.chunkSteps,
        true,
        "the sinusoidal table covers a whole chunk",
    );
    equal(audio.reserved0, 0, "reserved_0 is written as zero");
    equal(audio.reserved1, 0, "reserved_1 is written as zero");
    console.log(
        `audio tower: d_model ${audio.dModel}, ${audio.layers} layers, ${audio.attentionHeads} heads ` +
        `x ${audio.headDim}, chunk ${audio.chunkFrames} frames -> ${audio.chunkSteps} steps, ` +
        `${audio.frequencyBins} bins left, downsample input ${audio.convOutInputFeatures}, ` +
        `projector ${audio.outputDim}`,
    );

    // The cache width is chosen before a load, and the requirements say what it costs. The exact
    // plane arithmetic belongs to the core and is tested there; the contract here is that an unknown
    // format is refused, that a chosen format cannot change under a loaded model, and that the choice
    // is what the reported bytes follow.
    const cacheBytesF32 = requirements.cacheBytes;
    equal(
        e.qw_model_set_cache_format(handle, 7),
        STATUS.invalidArgument,
        "an unknown cache format is refused",
    );
    equal(
        e.qw_model_set_cache_format(handle, 1),
        STATUS.invalidState,
        "the cache format cannot change once a model is loaded",
    );
    equal(e.qw_model_begin(handle), STATUS.ok, "begin releases the loaded model");
    for (const [index, ptr] of shardPtrs.entries()) {
        equal(
            e.qw_model_add_shard(handle, ptr, shardBuffers[index].length),
            STATUS.ok,
            `shard ${index} accepted for the q8 load`,
        );
    }
    equal(e.qw_model_set_cache_format(handle, 1), STATUS.ok, "q8 is accepted before a load");
    equal(
        e.qw_model_finish(handle, configPtr, config.length),
        STATUS.ok,
        "the model loads with a q8 cache",
    );
    equal(e.qw_model_requirements(handle, statePtr), STATUS.ok, "requirements for the q8 model");
    // A fresh view: loading the q8 model can grow the instance's memory, and a view taken before a
    // growth is detached. The configuration numbers above are already plain values, so they survive.
    const q8View = new DataView(e.memory.buffer, statePtr, 48);
    const cacheBytesQ8 = Number(q8View.getBigUint64(8, true));
    const totalBytesQ8 = Number(q8View.getBigUint64(24, true));
    check(
        cacheBytesQ8 < cacheBytesF32 / 3,
        "a q8 cache is under a third of the f32 one",
        `${cacheBytesQ8} vs ${cacheBytesF32}`,
    );
    check(
        cacheBytesQ8 >= layers * keyValueWidth * maxPositions,
        "a q8 cache is at least one byte per element",
        `${cacheBytesQ8}`,
    );
    console.log(
        `q8 cache: ${cacheBytesQ8} B (${(cacheBytesQ8 / 2 ** 20).toFixed(1)} MiB) against ` +
        `${cacheBytesF32} B f32, total ${(totalBytesQ8 / 2 ** 30).toFixed(3)} GiB`,
    );

    // The vocabulary, so the prompt can be built and ids detokenized. The table carries its own
    // count as a leading word; the manifest repeats it, and the two must agree.
    const tableBytes = new Uint8Array(readFileSync(`${modelDir}/${manifest.tokenizer.file}`));
    const tokenCount = manifest.tokenizer.count;
    const tableFirstWord = new DataView(
        tableBytes.buffer,
        tableBytes.byteOffset,
        4,
    ).getUint32(0, true);
    equal(tableFirstWord, tokenCount, "the vocabulary table carries the manifest's token count");
    const headerBytes = tableFirstWord === tokenCount ? 4 : 0;
    const offsetsBytes = (tokenCount + 1) * 4;
    check(
        tableBytes.length > headerBytes + offsetsBytes,
        "the vocabulary table holds more than its offsets",
        `${tableBytes.length} bytes`,
    );
    const tablePtr = e.qw_alloc(tableBytes.length, 4) >>> 0;
    const descriptorPtr = e.qw_alloc(24, 4) >>> 0;
    new Uint8Array(e.memory.buffer, tablePtr, tableBytes.length).set(tableBytes);
    const descriptor = new DataView(e.memory.buffer, descriptorPtr, 24);
    descriptor.setUint32(0, tablePtr + headerBytes, true);
    descriptor.setUint32(4, offsetsBytes, true);
    descriptor.setUint32(8, tablePtr + headerBytes + offsetsBytes, true);
    descriptor.setUint32(12, tableBytes.length - headerBytes - offsetsBytes, true);
    descriptor.setUint32(16, tokenCount, true);
    descriptor.setUint32(20, 0, true);
    equal(e.qw_tokenizer_set(handle, descriptorPtr), STATUS.ok, "vocabulary accepted");

    // Decoding: one second of a tone, so the encoder runs a single chunk and
    // the token budget stays tiny.
    const sampleCount = 16000;
    const samplesPtr = e.qw_alloc(sampleCount * 4, 4) >>> 0;
    const samples = new Float32Array(e.memory.buffer, samplesPtr, sampleCount);
    for (let index = 0; index < sampleCount; index += 1) {
        samples[index] = Math.sin((2 * Math.PI * 1000 * index) / 16000);
    }
    const frames = e.qw_mel_frames_for_samples(sampleCount);
    const featuresPtr = e.qw_alloc(frames * melBins * 4, 4) >>> 0;
    const melResultPtr = e.qw_alloc(16, 4) >>> 0;
    equal(
        e.qw_mel_compute(
            handle,
            samplesPtr,
            sampleCount,
            featuresPtr,
            frames * melBins * 4,
            melResultPtr,
        ),
        STATUS.ok,
        "mel compute for the decode check",
    );
    const featureBytes = frames * melBins * 4;

    equal(
        e.qw_decode_begin(handle, featuresPtr, featureBytes, 0),
        STATUS.limitExceeded,
        "a zero token budget is refused",
    );
    equal(
        e.qw_decode_begin(handle, featuresPtr, featureBytes, requirements.maxDecodeTokens + 1),
        STATUS.limitExceeded,
        "a budget beyond the configuration's is refused",
    );
    equal(
        e.qw_decode_begin(handle, featuresPtr, featureBytes - 4, 8),
        STATUS.shapeMismatch,
        "a partial mel frame is refused",
    );
    equal(
        e.qw_decode_begin(handle, featuresPtr, featureBytes, 8),
        STATUS.ok,
        "decode begin",
    );

    const tokenPtr = e.qw_alloc(4, 4) >>> 0;
    const idsPtr = e.qw_alloc(8 * 4, 4) >>> 0;
    let produced = 0;
    let stepFailure = 0;
    while (produced < 8) {
        const status = e.qw_decode_step(handle, tokenPtr);
        if (status === 0) break;
        if (status !== 1) {
            stepFailure = status;
            break;
        }
        produced += 1;
    }
    equal(stepFailure, STATUS.ok, "decode step reports no failure");
    check(produced > 0, "decode step produced at least one token", `${produced}`);

    const copied = e.qw_decode_tokens(handle, idsPtr, 8);
    if (copied < 0) {
        // Reported rather than used as a length: a negative status here would
        // otherwise become a typed-array length and hide which call failed.
        check(false, "decode tokens copies what was produced", `status ${copied}`);
    } else {
        equal(copied, produced, "decode tokens copies what was produced");
        equal(
            e.qw_decode_tokens(handle, idsPtr, produced - 1),
            STATUS.limitExceeded,
            "an undersized token buffer is refused",
        );
        const ids = new Uint32Array(e.memory.buffer, idsPtr, copied);
        let inVocabulary = true;
        for (const id of ids) if (id >= manifest.tokenizer.count) inVocabulary = false;
        check(inVocabulary, "every produced id is inside the vocabulary", `${ids.join(",")}`);
        const textPtr = e.qw_alloc(copied * 8 + 64, 1) >>> 0;
        const writtenPtr = e.qw_alloc(4, 4) >>> 0;
        equal(
            e.qw_detokenize(handle, idsPtr, copied, textPtr, copied * 8 + 64, writtenPtr),
            STATUS.ok,
            "the produced ids detokenize",
        );
        const written = new DataView(e.memory.buffer, writtenPtr, 4).getUint32(0, true);
        const text = new TextDecoder().decode(new Uint8Array(e.memory.buffer, textPtr, written));
        check(written > 0, "the decoded text is not empty");
        console.log(`decode: ${copied} tokens ${ids.join(",")} -> ${JSON.stringify(text)}`);
    }

    equal(e.qw_decode_end(handle), STATUS.ok, "decode end");
    equal(e.qw_decode_end(handle), STATUS.invalidState, "ending twice is refused");
    equal(
        e.qw_decode_step(handle, tokenPtr),
        STATUS.invalidState,
        "a step after decode end is refused",
    );
    // The model stays resident, so a second utterance is one begin away.
    equal(e.qw_decode_begin(handle, featuresPtr, featureBytes, 4), STATUS.ok, "decode begin again");
    equal(e.qw_decode_end(handle), STATUS.ok, "the second utterance ends");
}

const modelDir = process.argv[3];
if (modelDir === undefined) {
    console.log(
        "\nnote: no model directory given, so the model-loading checks are skipped\n" +
        "      (pass one to run them: node tools/wasm_selftest.mjs <module.wasm> <model-dir>)",
    );
} else {
    exerciseModel(modelDir);
}

// --- allocator --------------------------------------------------------------

{
    const pointer = e.qw_alloc(4096, 64);
    check(pointer !== 0, "aligned allocation succeeds");
    equal(pointer % 64, 0, "allocation honours the requested alignment");
    equal(e.qw_alloc(0, 4), 0, "zero-sized allocation fails");
    equal(e.qw_alloc(16, 3), 0, "non-power-of-two alignment fails");
    e.qw_free(pointer, 4096, 64);
}

equal(e.qw_destroy(handle), STATUS.ok, "destroy");
equal(e.qw_destroy(handle), STATUS.invalidArgument, "double destroy is rejected");

console.log(failures === 0 ? "\nall ABI checks passed" : `\n${failures} ABI check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
