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
import { runAudioTower, uploadAudioTowerWeights, windowSteps } from "../../packages/qwenscriber/dist/gpu/audio_tower.js";
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

/**
 * The steps one chunk contributes: the convolution stack's depth applied to the chunk's *valid*
 * frames, which is what `packedStepCount` does per chunk in the configuration.
 */
function validSteps(frames) {
    let steps = frames;
    for (let index = 0; index < 3; index += 1) steps = Math.floor((steps - 1) / 2) + 1;
    return steps;
}

/**
 * The comparison `tools/reference/compare_fixtures.py` sanctions for `audio_encoded`: the worst
 * absolute difference against the tensor's own scale, not element by element.
 *
 * The reason is in that script: one element of the reference near zero turns a per-element relative
 * error into a huge number while the tensor as a whole agrees to a millionth.
 */
function scaleRelative(actual, expected) {
    let worst = 0;
    for (let index = 0; index < expected.length; index += 1) {
        worst = Math.max(worst, Math.abs(actual[index] - expected[index]));
    }
    let scale = 0;
    for (const value of expected) scale = Math.max(scale, Math.abs(value));
    return { worst, scale, relative: scale === 0 ? worst : worst / scale };
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
    const encoded = await load_fixture(`${FIXTURE_URL}audio_encoded.f32`);
    notes.push(
        `fixtures: input_features ${features.dims.join("x")}, audio_conv_out ` +
            `${expected.dims.join("x")}, audio_encoded ${encoded.dims.join("x")}`,
    );

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
    // The blocks read the whole packed sequence, so the convolution stage runs once per chunk and
    // only the steps that belong to valid mel frames are kept.
    // The tail of a short chunk is zeros, not the mel padding value: the core's `convolveAndPack`
    // memsets the chunk before copying the frames it has, and its comment says why -- the reference
    // pads with zeros in the mel domain, and the padding *value* belongs to the frontend's buffer
    // past the audio, where the zeros live in the audio itself.
    const chunk_count = Math.ceil(features.dims[1] / config.chunk_frames);
    // Counted per chunk, not over the whole clip: the convolution depth is applied to each chunk's
    // valid frames, so 420 frames are 13+13+13+13+3 = 55 steps rather than the 53 a single
    // application of the same arithmetic to 420 would give. The dump's 55 rows say which is right.
    let declared_steps = 0;
    for (let index = 0; index < chunk_count; index += 1) {
        const valid = Math.min(config.chunk_frames, features.dims[1] - index * config.chunk_frames);
        declared_steps += validSteps(valid);
    }
    const packed = new Float32Array(declared_steps * config.d_model);
    let packed_steps = 0;
    for (let index = 0; index < chunk_count; index += 1) {
        const first_frame = index * config.chunk_frames;
        const valid = Math.min(config.chunk_frames, features.dims[1] - first_frame);
        const block = new Float32Array(config.mel_bins * config.chunk_frames);
        for (let bin = 0; bin < config.mel_bins; bin += 1) {
            for (let frame = 0; frame < valid; frame += 1) {
                block[bin * config.chunk_frames + frame] =
                    features.values[bin * features.dims[1] + first_frame + frame];
            }
        }
        const chunk_result = await runConvStage(runtime, weights, config, block);
        const values = await runtime.readFloats(
            chunk_result.output,
            chunk_result.steps * result.d_model,
        );
        chunk_result.output.destroy();
        const keep = validSteps(valid);
        packed.set(values.subarray(0, keep * result.d_model), packed_steps * result.d_model);
        packed_steps += keep;
    }
    notes.push(
        `packing: ${chunk_count} chunks over ${features.dims[1]} mel frames -> ${packed_steps} ` +
            `steps (the dump has ${encoded.dims[0]})`,
    );

    const tower_weights = uploadAudioTowerWeights(runtime, model, config);
    const tower_input = runtime.uploadBytes(new Uint8Array(packed.buffer), "tower.encoder.input");
    const tower_started = performance.now();
    const tower = await runAudioTower(runtime, tower_weights, config, tower_input, packed_steps);
    const tower_ms = performance.now() - tower_started;
    const encoded_values = await runtime.readFloats(tower.output, packed_steps * config.d_model);
    tower.output.destroy();
    const blocks_relative = scaleRelative(encoded_values, encoded.values);
    // The released implementation's own `audio_encoded` for the same clip, which the q5 dump was
    // already held to by `tools/reference/compare_fixtures.py`. Comparing both distances says
    // whether a disagreement is this tower's arithmetic or the quantization the dump itself carries:
    // an output closer to the dump than the dump is to the reference is as good as this check can be.
    let reference_gap = null;
    try {
        const reference = await load_fixture("/tests/fixtures/reference/audio_encoded.f32");
        const dump_to_reference = scaleRelative(encoded.values, reference.values);
        const ours_to_reference = scaleRelative(encoded_values, reference.values);
        reference_gap = { dump_to_reference, ours_to_reference };
        notes.push(
            `against the released implementation: dump ${dump_to_reference.relative.toExponential(2)}, ` +
                `ours ${ours_to_reference.relative.toExponential(2)}`,
        );
    } catch (error) {
        notes.push(`no released-implementation fixture to compare against: ${error.message}`);
    }
    const blocks = compare(encoded_values, encoded.values, 0, 2e-2);
    notes.push(
        `blocks (scale relative): max|d| ${blocks_relative.worst.toExponential(2)} over scale ` +
            `${blocks_relative.scale.toExponential(2)} = ${blocks_relative.relative.toExponential(2)}`,
    );
    notes.push(
        `blocks: ${tower.dispatches.length} dispatches in ${tower_ms.toFixed(1)} ms, one window ` +
            `of ${windowSteps(config)} steps`,
    );
    notes.push(
        `magnitude: encoded ours ${max_abs(encoded_values).toExponential(3)}, dump ` +
            `${max_abs(encoded.values).toExponential(3)}`,
    );
    if (blocks.failures > 0) {
        notes.push(
            `blocks worst index ${blocks.worst_index}: ours ${encoded_values[blocks.worst_index]}, ` +
                `dump ${encoded.values[blocks.worst_index]}`,
        );
        // Which steps disagree, and by how much: a per-step profile says whether the difference is
        // one wrong region -- a chunk boundary, a padded tail -- or spread evenly, which would be a
        // pass applied differently rather than an input being wrong.
        const per_step = [];
        let step_starts = 0;
        for (let index = 0; index < chunk_count; index += 1) {
            const valid = Math.min(
                config.chunk_frames,
                features.dims[1] - index * config.chunk_frames,
            );
            const keep = validSteps(valid);
            let worst = 0;
            let failed = 0;
            // The same scale as the whole-tensor comparison, so a chunk's share of the error is
            // readable on its own rather than against its own local scale.
            for (let step = step_starts; step < step_starts + keep; step += 1) {
                for (let column = 0; column < config.d_model; column += 1) {
                    const at = step * config.d_model + column;
                    const difference = Math.abs(encoded_values[at] - encoded.values[at]);
                    worst = Math.max(worst, difference);
                    if (difference > 2e-2 * blocks_relative.scale) failed += 1;
                }
            }
            per_step.push(
                `chunk ${index}: steps ${step_starts}..${step_starts + keep - 1} ` +
                    `max|d| ${worst.toExponential(2)} over-tolerance ${failed}/${keep * config.d_model}`,
            );
            step_starts += keep;
        }
        notes.push(...per_step);
    }

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
    rows.push({
        name: "audio_tower blocks",
        status: blocks_relative.relative <= 2e-2 ? "pass" : "fail",
        elements: encoded.values.length,
        max_abs: blocks_relative.worst,
        mean_abs: blocks.mean_abs,
        failures: blocks.failures,
    });
    const ok = comparison.failures === 0 && blocks_relative.relative <= 2e-2;
    return { adapter: info, limits, rows, notes, ok };
}
