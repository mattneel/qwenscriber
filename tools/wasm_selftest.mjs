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
    console.error("usage: node tools/wasm_selftest.mjs <module.wasm>");
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

console.log(`module: ${modulePath} (${bytes.length} bytes)`);

// --- exported surface -------------------------------------------------------

const required = [
    "qw_abi_version",
    "qw_core_version",
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
];
for (const name of required) {
    check(typeof e[name] === "function", `exports ${name}`);
}

equal(e.qw_abi_version(), GOLDEN.abiVersion, "ABI version is v1.0");
equal(e.qw_core_version(), GOLDEN.coreVersion, "core version matches the host build");

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
    const samplesPtr = e.qw_alloc(sampleCount * 4, 4);
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
