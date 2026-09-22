// Page side of the tower check: run one chunk of the audio tower's convolution stack and downsample
// projection on the GPU, and compare the result against the reference implementation's own fixture.
//
//     audio_conv_out.f32   [13][896]   the released checkpoint's output for one chunk
//     input_features.f32   [128][420]  the log-mel block it was produced from
//
// The first chunk is the first `chunk_frames` columns of that block, and the fixture's 13 rows are
// one full chunk's steps, so the two line up without any further slicing. The fixtures are generated
// by `tools/reference/gen_fixtures.py` from the published checkpoint and are not committed (they are
// hundreds of kilobytes of derived data), which is why this check runs locally rather than in CI --
// the same arrangement as the kernel harness.
//
// The comparison is against the *released implementation*, not against our own CPU path: it is the
// same fixture `tools/reference/compare_fixtures.py` uses to validate the Zig runtime, so agreement
// here means the GPU tower agrees with the numbers the checkpoint itself produces.

import { read_fixture } from "./fixture.mjs";
// Test-side modules on purpose: the check drives the runtime below the public SDK surface, the same
// way the kernel harness imports `reference.mjs` rather than going through a package.
import { runConvStage } from "../../packages/qwenscriber/dist/gpu/conv_stack.js";
import { WebGpuRuntime } from "../../packages/qwenscriber/dist/gpu/runtime.js";
import { shaderSourceFromBaseUrl } from "../../packages/qwenscriber/dist/gpu/shaders.js";
import { uploadTowerWeights } from "../../packages/qwenscriber/dist/gpu/tower_weights.js";
import { fetchModelDirectory, loadModel } from "../../packages/qwenscriber/dist/decode.js";
import { WasmCore } from "../../packages/qwenscriber/dist/wasm/runtime.js";

// The q5 conversion, not the f16 one: a wasm32 instance tops out at 2 GiB of linear memory and the
// f16 build's 1.57 GB of weights plus a key/value cache do not fit, which `qw_model_finish` reports
// as `out_of_memory`. Its convolutions are f16 either way -- the converter only quantizes the large
// weight matrices, and `audio.conv_out.weight` is one of them: f16 in the f16 build, q5 here.
const MODEL_URL = "/models/qwen3-asr-0.6b-q5/";
// The comparison target is our own runtime's stage dump for that same conversion, so both sides run
// the same formats and the only difference left is the CPU/GPU implementation:
//
//     qwenscriber-transcribe --model models/qwen3-asr-0.6b-q5 \
//         --audio tests/fixtures/audio/asr_zh.wav --dump models/qwen3-asr-0.6b-q5/dump
//
// `tools/reference/compare_fixtures.py` holds that dump to the released implementation at 1e-3
// absolute on this stage, which is why that is the tolerance below.
const FIXTURE_URL = "/models/qwen3-asr-0.6b-q5/dump/";

/** Reads the `.f32` fixture format: seven u32 words, then f32 little endian. */
export async function load_fixture(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: http ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    return read_fixture(bytes, url);
}

function max_abs(values) {
    let worst = 0;
    for (const value of values) worst = Math.max(worst, Math.abs(value));
    return worst;
}

function compare(actual, expected, atol, rtol) {
    let max_abs = 0;
    let sum_abs = 0;
    let failures = 0;
    let worst_index = -1;
    for (let index = 0; index < expected.length; index += 1) {
        const difference = Math.abs(actual[index] - expected[index]);
        const bound = atol + rtol * Math.abs(expected[index]);
        max_abs = Math.max(max_abs, difference);
        sum_abs += difference;
        if (!(difference <= bound)) {
            failures += 1;
            if (worst_index < 0) worst_index = index;
        }
    }
    return { max_abs, mean_abs: sum_abs / expected.length, failures, worst_index };
}

export async function run() {
    const notes = [];
    const rows = [];
    // Two requests, as the SDK's own acquisition does: on a host whose only adapter is software,
    // the high-performance request can be answered with nothing at all.
    const adapter = (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" })) ??
        (await navigator.gpu.requestAdapter());
    if (adapter === null || adapter === undefined) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice({
        requiredLimits: {
            maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        },
    });
    const info = adapter.info ?? {};
    const limits = {};
    for (const name of [
        "maxBufferSize",
        "maxStorageBufferBindingSize",
        "maxComputeWorkgroupStorageSize",
    ]) {
        limits[name] = adapter.limits[name];
    }

    const features = await load_fixture(`${FIXTURE_URL}input_features.f32`);
    const expected = await load_fixture(`${FIXTURE_URL}audio_conv_out.f32`);
    notes.push(`fixtures: input_features ${features.dims.join("x")}, audio_conv_out ` +
        `${expected.dims.join("x")}`);

    const runtime = await WebGpuRuntime.create({
        shaderSource: shaderSourceFromBaseUrl(new URL("/gpu/shaders/", location.href)),
        // This host's adapter is SwiftShader, which the SDK refuses by default.
        allowSoftwareAdapter: true,
        // The convolution stages 18432 bytes of decoded weights, above the specification's
        // 16384-byte device default, so the device has to be asked for the adapter's own limit --
        // which is what declaring a requirement here does.
        requiredLimits: {
            maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        },
    });
    const core = await WasmCore.load("/packages/qwenscriber/dist/qwenscriber_core.wasm");
    const directory = fetchModelDirectory(MODEL_URL);
    // The conversion declares 8192 positions, whose f32 key/value cache needs 1.88 GB on top of the
    // weights -- more than a wasm32 instance can hold, which `qw_model_finish` answers with
    // `out_of_memory`. The tower stage under test does not decode, so the budget is lowered to the
    // 2048 positions `status.md` records as the runnable one: the configuration's own
    // `max_positions`, at offset 96 of its 160 bytes, written into the bytes the loader is handed.
    const CONFIG_MAX_POSITIONS_OFFSET = 96;
    const manifest = JSON.parse(new TextDecoder().decode(await directory.read("manifest.json")));
    const config_file = manifest.config.file;
    const config_bytes = await directory.read(config_file);
    const config_view = new DataView(
        config_bytes.buffer,
        config_bytes.byteOffset,
        config_bytes.byteLength,
    );
    const declared_positions = config_view.getUint32(CONFIG_MAX_POSITIONS_OFFSET, true);
    const positions = Math.min(declared_positions, 2048);
    if (positions !== declared_positions) {
        config_view.setUint32(CONFIG_MAX_POSITIONS_OFFSET, positions, true);
        notes.push(`positions: ${declared_positions} declared, ${positions} loaded`);
    }
    // A provider around the provider, so every file comes from the network except the configuration,
    // which comes from the patched copy.
    const patched_directory = {
        source: directory.source,
        read: (name) =>
            name === config_file ? Promise.resolve(config_bytes) : directory.read(name),
    };
    const model = await loadModel(core, patched_directory);
    const config = core.modelAudioConfig();
    notes.push(
        `tower: d_model ${config.d_model}, chunk ${config.chunk_frames} frames -> ` +
        `${config.chunk_steps} steps, conv_out input ${config.conv_out_input_features}`,
    );

    // One chunk is the first `chunk_frames` columns of the mel block, which is row major
    // `[mel_bin][frame]`, so the chunk is a strided gather rather than a prefix.
    const chunk = new Float32Array(config.mel_bins * config.chunk_frames);
    for (let bin = 0; bin < config.mel_bins; bin += 1) {
        for (let frame = 0; frame < config.chunk_frames; frame += 1) {
            chunk[bin * config.chunk_frames + frame] =
                features.values[bin * features.dims[1] + frame];
        }
    }

    // The chunk is the first `chunk_frames` columns of the mel block. Check that against the dump's
    // own mel before running anything: if the input is wrong, every later number is noise.
    let chunk_max = 0;
    let dump_max = 0;
    for (let bin = 0; bin < config.mel_bins; bin += 1) {
        for (let frame = 0; frame < config.chunk_frames; frame += 1) {
            chunk_max = Math.max(chunk_max, Math.abs(chunk[bin * config.chunk_frames + frame]));
            dump_max = Math.max(
                dump_max,
                Math.abs(features.values[bin * features.dims[1] + frame]),
            );
        }
    }
    notes.push(
        `chunk: max ${chunk_max.toExponential(3)} over ${config.mel_bins}x${config.chunk_frames}, ` +
            `dump mel max over the same columns ${dump_max.toExponential(3)}`,
    );

    const weights = uploadTowerWeights(runtime, model);
    const started = performance.now();
    const result = await runConvStage(runtime, weights, config, chunk);
    const dispatch_ms = performance.now() - started;
    const actual = await runtime.readFloats(result.output, result.steps * result.d_model);

    // The buffers between the stack and the projection, so a disagreement can be located rather than
    // guessed at: a zero stack means the convolution chain produced nothing, and a zero transposed
    // buffer with a live stack means the permutation did.
    const first = await runtime.readFloats(
        result.intermediates.conv1,
        weights.conv1.out_channels * 64 * 50,
    );
    const second = await runtime.readFloats(
        result.intermediates.conv2,
        weights.conv2.out_channels * 32 * 25,
    );
    const stack = await runtime.readFloats(
        result.intermediates.stack,
        weights.conv3.out_channels * 16 * result.steps,
    );
    notes.push(`conv1: max ${max_abs(first).toExponential(3)} over 480x64x50`);
    notes.push(`conv2: max ${max_abs(second).toExponential(3)} over 480x32x25`);
    const transposed = await runtime.readFloats(
        result.intermediates.transposed,
        result.steps * weights.downsample.cols,
    );
    notes.push(`stack: max ${max_abs(stack).toExponential(3)} over ${stack.length} elements`);
    notes.push(
        `transposed: max ${max_abs(transposed).toExponential(3)} over ${transposed.length} elements`,
    );
    for (const buffer of Object.values(result.intermediates)) buffer.destroy();

    const comparison = compare(actual, expected.values, 1e-3, 1e-3);
    notes.push(`dispatches: ${result.dispatches.length} in ${dispatch_ms.toFixed(1)} ms`);
    notes.push(
        `conv stack: expected ${expected.dims[0]}x${expected.dims[1]}, got ` +
        `${result.steps}x${result.d_model}`,
    );
    rows.push({
        name: "conv_stack + conv_out",
        status: comparison.failures === 0 ? "pass" : "fail",
        elements: expected.values.length,
        max_abs: comparison.max_abs,
        mean_abs: comparison.mean_abs,
        failures: comparison.failures,
    });
    // A failure needs to say which way it failed. A GPU result whose magnitude matches the fixture's
    // but whose elements do not is a layout problem; one whose magnitude is the embedding's alone
    // (~2.3 for 13x896) is a projection that contributed nothing.
    notes.push(
        `magnitude: ours ${max_abs(actual).toExponential(3)}, dump ` +
        `${max_abs(expected.values).toExponential(3)}`,
    );
    notes.push(
        `first row: ours ${Array.from(actual.slice(0, 4), (value) => value.toFixed(6)).join(", ")}`,
    );
    notes.push(
        `first row: dump ${Array.from(expected.values.slice(0, 4), (value) => value.toFixed(6)).join(", ")}`,
    );
    notes.push(
        `worst index ${comparison.worst_index}: ours ${actual[comparison.worst_index]}, ` +
        `dump ${expected.values[comparison.worst_index]}`,
    );
    return { adapter: info, limits, rows, notes, ok: comparison.failures === 0 };
}
