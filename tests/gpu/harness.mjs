// Numeric comparison harness for the kernels in gpu/shaders/.
//
// It generates deterministic inputs, computes the CPU reference results with
// `reference.mjs` (which mirrors `src/core/{math,quant}.zig` and is itself
// validated against `qw_selftest` inside the WASM module), uploads the inputs,
// runs each kernel, reads the results back, and prints a per-kernel table of
// maximum and mean absolute error plus the worst relative error against the
// tolerance that kernel is allowed.
//
// Running it
// ----------
//
//   node tests/gpu/harness.mjs                 # starts its own http server and a
//                                             # headless Chrome, prints the table
//   node tests/gpu/harness.mjs --url <url>     # drive a page from a server that
//                                             # is already serving the repository
//   node tests/gpu/harness.mjs --keep-open      # leave Chrome running for a look
//
// The page side (`harness.html`) can also be opened by hand: it renders the same
// table and sets `window.__result`. That is how the repository's Windows-side
// driver (`tools/browser/run-page.mjs`) runs it against a real adapter:
//
//   cmd.exe /c 'cd /d C:\Users\<you>\omp-browser && node.exe run-page.mjs \
//       http://127.0.0.1:8766/tests/gpu/harness.html --settle-ms 60000'
//
// On a WSL2 host the Linux side has no GPU, so the fallback is Chrome's
// SwiftShader adapter, which is correct but slow:
//
//   --enable-unsafe-webgpu --enable-unsafe-swiftshader \
//   --use-webgpu-adapter=swiftshader --no-sandbox --disable-dev-shm-usage
//
// `tests/gpu/driver.mjs` passes that set; `--no-swiftshader` drops the two
// software-adapter flags for a machine with a real adapter. A cold headless
// start can answer the first `requestAdapter()` with null while its GPU process
// is still coming up, so the harness retries before it concludes that WebGPU is
// unavailable.
//
// No dependencies: the browser side is plain ES modules and the Node side drives
// Chrome over the DevTools protocol with Node's built-in WebSocket.

import * as ref from "./reference.mjs";

const SHADER_BASE = new URL("../../gpu/shaders/", import.meta.url);
const WASM_URL = new URL("../../zig-out/bin/qwenscriber_core.wasm", import.meta.url);

// Tolerances, in the form `|actual - reference| <= atol + rtol * |reference|`.
//
// Each bound answers "what could differ between these two implementations",
// measured against the largest term each kernel produces. They are deliberately
// much looser than the observed errors (the table prints both) and much tighter
// than any real bug: a swapped nibble, a dropped high bit, or a wrong bias moves
// a weight by its own magnitude, which is orders of magnitude above these
// bounds.
export const TOLERANCES = {
    // f32 accumulation on both sides in the same order per output element, so
    // only the multiply-add fusion and reassociation inside a slab can differ.
    // Sums of 256 products of O(1) operands reach |sum| ~ 20, where one ULP is
    // ~2e-6.
    matmul: { atol: 1e-3, rtol: 1e-4 },
    // The mean of squares is reduced as a tree on the GPU and sequentially on
    // the CPU, so the divisor moves in its last few bits before the row is
    // scaled by a value near 1.
    rmsnorm: { atol: 1e-4, rtol: 1e-4 },
    // `pow`, `sin`, and `cos` are the adapter's implementations, not the JS
    // library's. Measured on this repository's two adapters:
    //   AMD RDNA-2 (Windows Chrome):   worst error 9.5e-7 over the whole case
    //   SwiftShader (WSL headless):    worst error 5.0e-4
    // and a direct probe shows why: SwiftShader's `pow(1e6, -2i/128)` is accurate
    // to 8e-7 relative, but its `cos`/`sin` are off by up to 1.9e-4 absolute near
    // argument 1, which a rotation amplifies by the operand magnitude (~3 sigma
    // of the generator). The bound below is that, with margin; a wrong theta or
    // interleaved (rather than rotate_half) pairs errs by 1e-2 or O(1).
    rope: { atol: 2e-3, rtol: 1e-3 },
    // exp() differs between the adapter and JS, and the softmax divides by a sum
    // of up to `window` exponentials. The probabilities stay in [0, 1] and the
    // output is a convex combination of O(1) value vectors, so an absolute bound
    // is the meaningful one.
    attention: { atol: 1e-4, rtol: 1e-4 },
    // One exp() and one division per element, values in [-8, 8].
    silu_mul: { atol: 1e-5, rtol: 1e-5 },
    // GELU is one `exp` and the `erfc` polynomial, both the adapter's arithmetic. The polynomial's
    // own error is 1.2e-7, four orders below this bound.
    gelu: { atol: 1e-5, rtol: 1e-5 },
    // Same shape of reduction as rmsnorm, one more pass for the mean: accumulating 256 f32 values
    // per step in a different order from the reference is where the difference comes from.
    layernorm: { atol: 1e-4, rtol: 1e-4 },
    // The accumulation order matches the reference tap for tap and the f16 decode is exact, so the
    // bound is GELU's `exp` plus the adapter's arithmetic.
    conv3x3: { atol: 1e-4, rtol: 1e-4 },
    // Both sides multiply f16 weights that decode exactly to the same f32 values, in the same
    // accumulation order, so the difference is the adapter's arithmetic alone.
    matmul_f16: { atol: 1e-3, rtol: 1e-4 },
    // A permutation of the same values: expected bit exact.
    transpose: { atol: 0, rtol: 0 },
    // One addition per element: expected bit exact.
    add: { atol: 0, rtol: 0 },
    add_bias: { atol: 0, rtol: 0 },
    // Expected bit exact: both sides read the same integers and multiply by the
    // same f16 scale. The case deliberately includes a group whose f16 scale is
    // subnormal (~1.2e-6, where the decoded weights are ~6e-7), so the bound is
    // there to absorb denormal flushing rather than any real slack.
    dequant: { atol: 1e-6, rtol: 1e-6 },
};

// Shapes. Small on purpose: this runs on SwiftShader as well as on a GPU, and
// every bug worth catching is visible at these sizes.
const SHAPE = {
    tokens: 24,
    rows: 64,
    cols: 256,
    rms_rows: 32,
    rms_cols: 256,
    rms_eps: 1e-6,
    rope_tokens: 8,
    rope_heads: 4,
    head_dim: 128,
    rope_theta: 1e6,
    attention_tokens: 16,
    attention_heads: 4,
    attention_keys: 16,
    attention_window: 16,
    attention_causal_window: 8,
    dequant_rows: 8,
    dequant_cols: 128,
    silu_count: 1024,
    gelu_count: 1024,
    layernorm_rows: 32,
    layernorm_cols: 256,
    layernorm_eps: 1e-5,
    // Odd channel counts on purpose: nine weights per input channel put every second channel's
    // block at an odd offset in the packed f16 plane, which is where a base-relative half index
    // goes wrong.
    conv_out_channels: 5,
    conv_in_channels: 3,
    conv_in_height: 8,
    conv_in_width: 8,
};

// ---------------------------------------------------------------------------
// Uniform packing helpers
// ---------------------------------------------------------------------------

function align_up(value, alignment) {
    return Math.ceil(value / alignment) * alignment;
}

export function f32_field(value) {
    return { f: value };
}

// Packs one uniform block in declaration order, padded to 16 bytes so the same
// helper works for every struct in gpu/shaders/.
export function pack_uniform(fields) {
    const bytes = new ArrayBuffer(align_up(fields.length * 4, 16));
    const view = new DataView(bytes);
    fields.forEach((field, index) => {
        if (field !== null && typeof field === "object" && "f" in field) {
            view.setFloat32(index * 4, field.f, true);
        } else {
            view.setUint32(index * 4, field, true);
        }
    });
    return new Uint8Array(bytes);
}

// Storage buffers are addressed as u32 words in WGSL, and their size must be a
// multiple of 4 bytes.
export function pack_words(bytes) {
    if (bytes.length % 4 === 0) return bytes;
    const padded = new Uint8Array(align_up(bytes.length, 4));
    padded.set(bytes);
    return padded;
}

function ceil_div(value, divisor) {
    return Math.ceil(value / divisor);
}

// ---------------------------------------------------------------------------
// Kernel cases
// ---------------------------------------------------------------------------

// The dense kernel and the two quantized kernels share inputs, shapes, tiling
// and accumulation order, so their three rows in the table can be read together:
// the quantized rows differ from the dense row only by the quantization error
// the format allows.
function build_matmul_cases() {
    const { tokens, rows, cols } = SHAPE;
    const activations = ref.random_vector(tokens * cols, 0x5eed_0001);
    const weights = ref.random_vector(rows * cols, 0x5eed_0002);
    const expected_f32 = ref.matmul_f32_reference(activations, weights, tokens, rows, cols);
    const cases = [
        {
            name: "matmul_f32",
            shader: "matmul_f32.wgsl",
            entry_point: "matmul_f32_main",
            workgroup: [16, 16, 1],
            bindings: [
                { uniform: pack_uniform([tokens, rows, cols, 0]) },
                { input: activations },
                { input: weights },
                { output: tokens * rows },
            ],
            dispatch: [ceil_div(rows, 16), ceil_div(tokens, 16), 1],
            expected: expected_f32,
            tolerance: TOLERANCES.matmul,
            detail: `tokens ${tokens}, rows ${rows}, cols ${cols}`,
        },
        {
            // The same weights, rounded to f16 and read back, which is what the kernel decodes:
            // the reference then sees exactly the values the shader does.
            name: "matmul_f16",
            shader: "matmul_f16.wgsl",
            entry_point: "matmul_f16_main",
            workgroup: [16, 16, 1],
            bindings: [
                { uniform: pack_uniform([tokens, rows, cols, 0]) },
                { input: activations },
                { input: ref.pack_f16_pairs(weights) },
                { output: tokens * rows },
            ],
            dispatch: [ceil_div(rows, 16), ceil_div(tokens, 16), 1],
            expected: ref.matmul_f32_reference(
                activations,
                Float32Array.from(weights, (value) => ref.from_f16_bits(ref.to_f16_bits(value))),
                tokens,
                rows,
                cols,
            ),
            tolerance: TOLERANCES.matmul_f16,
            detail: `tokens ${tokens}, rows ${rows}, cols ${cols}, f16 weights`,
        },
    ];

    for (const format of ["q4", "q5"]) {
        const tensor = ref.pack_tensor(format, rows, cols, weights);
        cases.push({
            name: `matmul_${format}`,
            shader: `matmul_${format}.wgsl`,
            entry_point: `matmul_${format}_main`,
            workgroup: [16, 16, 1],
            bindings: [
                { uniform: pack_uniform([tokens, rows, cols, tensor.data_offset_bytes]) },
                { input: activations },
                { input: pack_words(tensor.bytes) },
                { output: tokens * rows },
            ],
            dispatch: [ceil_div(rows, 16), ceil_div(tokens, 16), 1],
            expected: ref.matmul_quantized_reference(
                activations, tensor, format, tokens, rows, cols,
            ),
            tolerance: TOLERANCES.matmul,
            detail: `tokens ${tokens}, rows ${rows}, cols ${cols}, packed ${tensor.bytes.length} B`,
        });
    }
    return cases;
}

function build_normalization_cases() {
    const activations = ref.random_vector(SHAPE.rms_rows * SHAPE.rms_cols, 0x5eed_0003);
    const norm_weight = ref.random_vector(SHAPE.rms_cols, 0x5eed_0004);
    const rope_input = ref.random_vector(
        SHAPE.rope_tokens * SHAPE.rope_heads * SHAPE.head_dim,
        0x5eed_0005,
    );
    // Deliberately not square, and not a multiple of the tile: both guards are exercised.
    const transpose_rows = 480;
    const transpose_cols = 13;
    const transpose_input = ref.random_vector(transpose_rows * transpose_cols, 0x5eed_000f);
    // 7 columns over 9 rows: neither divides the 256-lane workgroup, so the guard and the
    // broadcast both run on a partial row.
    const bias_rows = 9;
    const bias_cols = 7;
    const bias_target = ref.random_vector(bias_rows * bias_cols, 0x5eed_0012);
    const bias_values = ref.random_vector(bias_cols, 0x5eed_0013);
    const add_left = ref.random_vector(SHAPE.silu_count, 0x5eed_0010);
    const add_right = ref.random_vector(SHAPE.silu_count, 0x5eed_0011);
    const gate = ref.random_vector(SHAPE.silu_count, 0x5eed_0006);
    const up = ref.random_vector(SHAPE.silu_count, 0x5eed_0007);
    const gelu_input = ref.random_vector(SHAPE.gelu_count, 0x5eed_0008);
    const layernorm_input = ref.random_vector(
        SHAPE.layernorm_rows * SHAPE.layernorm_cols,
        0x5eed_0009,
    );
    const layernorm_weight = ref.random_vector(SHAPE.layernorm_cols, 0x5eed_000a);
    const layernorm_bias = ref.random_vector(SHAPE.layernorm_cols, 0x5eed_000b);
    const conv_input = ref.random_vector(
        SHAPE.conv_in_channels * SHAPE.conv_in_height * SHAPE.conv_in_width,
        0x5eed_000c,
    );
    const conv_weights = ref.pack_f16_pairs(ref.random_vector(
        SHAPE.conv_out_channels * SHAPE.conv_in_channels * 9,
        0x5eed_000d,
        // Small weights: nine taps times three channels of unit-scale input would otherwise leave
        // GELU's saturated tails, where every implementation agrees and nothing is being tested.
        ref.random_uniform,
    ));
    const conv_bias = ref.random_vector(SHAPE.conv_out_channels, 0x5eed_000e);

    return [
        {
            name: "rmsnorm",
            shader: "rmsnorm.wgsl",
            entry_point: "rmsnorm_main",
            workgroup: [256, 1, 1],
            bindings: [
                {
                    uniform: pack_uniform([
                        SHAPE.rms_rows, SHAPE.rms_cols, f32_field(SHAPE.rms_eps), 0,
                    ]),
                },
                { input: activations },
                { input: norm_weight },
                { output: SHAPE.rms_rows * SHAPE.rms_cols },
            ],
            dispatch: [SHAPE.rms_rows, 1, 1],
            expected: ref.rmsnorm_reference(
                activations, norm_weight, SHAPE.rms_rows, SHAPE.rms_cols, SHAPE.rms_eps,
            ),
            tolerance: TOLERANCES.rmsnorm,
            detail: `rows ${SHAPE.rms_rows}, cols ${SHAPE.rms_cols}, eps ${SHAPE.rms_eps}`,
        },
        {
            name: "rope",
            shader: "rope.wgsl",
            entry_point: "rope_main",
            workgroup: [64, 1, 1],
            bindings: [
                {
                    uniform: pack_uniform([
                        SHAPE.rope_tokens, SHAPE.rope_heads, SHAPE.head_dim,
                        f32_field(SHAPE.rope_theta),
                    ]),
                },
                { input: rope_input },
                { output: rope_input.length },
            ],
            dispatch: [ceil_div(SHAPE.head_dim / 2, 64), SHAPE.rope_heads, SHAPE.rope_tokens],
            expected: ref.rope_reference(
                rope_input, SHAPE.rope_tokens, SHAPE.rope_heads, SHAPE.head_dim, SHAPE.rope_theta,
            ),
            tolerance: TOLERANCES.rope,
            detail: `tokens ${SHAPE.rope_tokens}, heads ${SHAPE.rope_heads}, ` +
                `head_dim ${SHAPE.head_dim}, theta ${SHAPE.rope_theta}`,
        },
        {
            name: "silu_mul",
            shader: "silu_mul.wgsl",
            entry_point: "silu_mul_main",
            workgroup: [256, 1, 1],
            bindings: [
                { uniform: pack_uniform([SHAPE.silu_count, 0, 0, 0]) },
                { input: gate },
                { input: up },
                { output: SHAPE.silu_count },
            ],
            dispatch: [ceil_div(SHAPE.silu_count, 256), 1, 1],
            expected: ref.silu_mul_reference(gate, up, SHAPE.silu_count),
            tolerance: TOLERANCES.silu_mul,
            detail: `count ${SHAPE.silu_count}`,
        },
        {
            name: "gelu",
            shader: "gelu.wgsl",
            entry_point: "gelu_main",
            workgroup: [256, 1, 1],
            bindings: [
                { uniform: pack_uniform([SHAPE.gelu_count, 0, 0, 0]) },
                { input: gelu_input },
                { output: SHAPE.gelu_count },
            ],
            dispatch: [ceil_div(SHAPE.gelu_count, 256), 1, 1],
            expected: ref.gelu_reference(gelu_input, SHAPE.gelu_count),
            tolerance: TOLERANCES.gelu,
            detail: `count ${SHAPE.gelu_count}`,
        },
        {
            name: "layernorm",
            shader: "layernorm.wgsl",
            entry_point: "layernorm_main",
            workgroup: [256, 1, 1],
            bindings: [
                {
                    uniform: pack_uniform([
                        SHAPE.layernorm_rows,
                        SHAPE.layernorm_cols,
                        f32_field(SHAPE.layernorm_eps),
                        0,
                    ]),
                },
                { input: layernorm_input },
                { input: layernorm_weight },
                { input: layernorm_bias },
                { output: SHAPE.layernorm_rows * SHAPE.layernorm_cols },
            ],
            dispatch: [SHAPE.layernorm_rows, 1, 1],
            expected: ref.layernorm_reference(
                layernorm_input,
                layernorm_weight,
                layernorm_bias,
                SHAPE.layernorm_rows,
                SHAPE.layernorm_cols,
                SHAPE.layernorm_eps,
            ),
            tolerance: TOLERANCES.layernorm,
            detail: `rows ${SHAPE.layernorm_rows}, cols ${SHAPE.layernorm_cols}, ` +
                `eps ${SHAPE.layernorm_eps}`,
        },
        {
            name: "add_f32",
            shader: "add_f32.wgsl",
            entry_point: "add_f32_main",
            workgroup: [256, 1, 1],
            bindings: [
                { uniform: pack_uniform([SHAPE.silu_count, 0, 0, 0]) },
                { input: add_left },
                { input: add_right },
                { output: SHAPE.silu_count },
            ],
            dispatch: [ceil_div(SHAPE.silu_count, 256), 1, 1],
            expected: Float32Array.from(add_left, (value, index) => value + add_right[index]),
            tolerance: TOLERANCES.add,
            detail: `count ${SHAPE.silu_count}`,
        },
        {
            name: "add_bias_f32",
            shader: "add_bias_f32.wgsl",
            entry_point: "add_bias_f32_main",
            workgroup: [256, 1, 1],
            bindings: [
                { uniform: pack_uniform([bias_rows, bias_cols, 0, 0]) },
                { input: bias_values },
                // The kernel adds in place, so the buffer it writes is the one it read: binding 2 is
                // seeded with the values the addition starts from.
                { output: bias_rows * bias_cols, seed: bias_target },
            ],
            dispatch: [ceil_div(bias_rows * bias_cols, 256), 1, 1],
            expected: Float32Array.from(
                bias_target,
                (value, index) => value + bias_values[index % bias_cols],
            ),
            tolerance: TOLERANCES.add_bias,
            detail: `target ${bias_rows}x${bias_cols}, bias ${bias_cols}, in place`,
        },
        {
            name: "transpose_f32",
            shader: "transpose_f32.wgsl",
            entry_point: "transpose_f32_main",
            workgroup: [16, 16, 1],
            bindings: [
                { uniform: pack_uniform([transpose_rows, transpose_cols, 0, 0]) },
                { input: transpose_input },
                { output: transpose_rows * transpose_cols },
            ],
            dispatch: [ceil_div(transpose_cols, 16), ceil_div(transpose_rows, 16), 1],
            expected: ref.transpose_reference(transpose_input, transpose_rows, transpose_cols),
            tolerance: TOLERANCES.transpose,
            detail: `[${transpose_rows}][${transpose_cols}] -> [${transpose_cols}][${transpose_rows}]`,
        },
        {
            name: "conv3x3_stride2_gelu",
            shader: "conv3x3_stride2_gelu.wgsl",
            entry_point: "conv3x3_stride2_gelu_main",
            workgroup: [256, 1, 1],
            bindings: [
                {
                    uniform: pack_uniform([
                        SHAPE.conv_out_channels,
                        SHAPE.conv_in_channels,
                        SHAPE.conv_in_height,
                        SHAPE.conv_in_width,
                    ]),
                },
                { input: conv_input },
                { input: conv_weights },
                { input: conv_bias },
                {
                    output: SHAPE.conv_out_channels *
                        (Math.floor((SHAPE.conv_in_height - 1) / 2) + 1) *
                        (Math.floor((SHAPE.conv_in_width - 1) / 2) + 1),
                },
            ],
            dispatch: [
                ceil_div(
                    (Math.floor((SHAPE.conv_in_height - 1) / 2) + 1) *
                        (Math.floor((SHAPE.conv_in_width - 1) / 2) + 1),
                    256,
                ),
                SHAPE.conv_out_channels,
                1,
            ],
            expected: ref.conv3x3_stride2_gelu_reference(
                conv_input,
                conv_weights,
                conv_bias,
                SHAPE.conv_out_channels,
                SHAPE.conv_in_channels,
                SHAPE.conv_in_height,
                SHAPE.conv_in_width,
            ),
            tolerance: TOLERANCES.conv3x3,
            detail: `out ${SHAPE.conv_out_channels}, in ${SHAPE.conv_in_channels}, ` +
                `${SHAPE.conv_in_height}x${SHAPE.conv_in_width} -> ` +
                `${Math.floor((SHAPE.conv_in_height - 1) / 2) + 1}x` +
                `${Math.floor((SHAPE.conv_in_width - 1) / 2) + 1}, stride 2, pad 1`,
        },
    ];
}

// Two rows, one per mode: the plain windowed (audio encoder) and the causal
// (decoder) one.
function build_attention_cases() {
    const { attention_tokens: tokens, attention_heads: heads } = SHAPE;
    const keys = SHAPE.attention_keys;
    const head_dim = SHAPE.head_dim;
    const scale = ref.f32(1 / Math.sqrt(head_dim));
    const queries = ref.random_vector(tokens * heads * head_dim, 0x5eed_0008);
    const key_vectors = ref.random_vector(keys * heads * head_dim, 0x5eed_0009);
    const value_vectors = ref.random_vector(keys * heads * head_dim, 0x5eed_000a);

    const modes = [
        { name: "attention_windowed", window: SHAPE.attention_window, causal: 0 },
        { name: "attention_causal", window: SHAPE.attention_causal_window, causal: 1 },
    ];
    return modes.map((mode) => ({
        name: mode.name,
        shader: "attention.wgsl",
        entry_point: "attention_main",
        workgroup: [128, 1, 1],
        bindings: [
            {
                uniform: pack_uniform([
                    tokens, heads, keys, head_dim,
                    f32_field(scale), mode.window, mode.causal, 0,
                ]),
            },
            { input: queries },
            { input: key_vectors },
            { input: value_vectors },
            { output: tokens * heads * head_dim },
        ],
        dispatch: [heads, tokens, 1],
        expected: ref.attention_reference(queries, key_vectors, value_vectors, {
            tokens,
            heads,
            keys,
            head_dim,
            scale,
            window: mode.window,
            causal: mode.causal === 1,
        }),
        tolerance: TOLERANCES.attention,
        detail: `tokens ${tokens}, heads ${heads}, keys ${keys}, window ${mode.window}, ` +
            `causal ${mode.causal}`,
    }));
}

// The packing check, one row per format. The first group of every row is scaled
// to ~1e-5 so that its f16 scale is subnormal: a scale plane read with the wrong
// shift, or a f16 decode that assumes a normal exponent, shows up here and
// nowhere else.
function build_dequant_cases() {
    const rows = SHAPE.dequant_rows;
    const cols = SHAPE.dequant_cols;
    const base = ref.random_vector(rows * cols, 0x5eed_000b);
    const weights = new Float32Array(base.length);
    for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < cols; column += 1) {
            const index = row * cols + column;
            weights[index] = column < ref.GROUP_SIZE ? ref.f32(base[index] * 1e-5) : base[index];
        }
    }

    return ["q4", "q5", "q8"].map((format) => {
        const tensor = ref.pack_tensor(format, rows, cols, weights);
        const planes = ref.plane_views(tensor, rows, cols);
        return {
            name: `dequant_${format}`,
            shader: "dequant_reference.wgsl",
            entry_point: "dequant_reference_main",
            workgroup: [64, 1, 1],
            bindings: [
                {
                    uniform: pack_uniform([
                        rows, cols, tensor.data_offset_bytes, ref.FORMAT_ID[format],
                    ]),
                },
                { input: pack_words(tensor.bytes) },
                { output: rows * cols },
            ],
            dispatch: [ceil_div(cols, 64), rows, 1],
            expected: ref.dequant_reference(planes, format, rows, cols),
            tolerance: TOLERANCES.dequant,
            detail: `format ${format} (id ${ref.FORMAT_ID[format]}), rows ${rows}, cols ${cols}, ` +
                `one subnormal-scale group per row`,
        };
    });
}

export function build_cases() {
    return [
        ...build_matmul_cases(),
        ...build_normalization_cases(),
        ...build_attention_cases(),
        ...build_dequant_cases(),
    ];
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

// Every element must satisfy `|actual - reference| <= atol + rtol * |reference|`.
// A NaN difference counts as a failure, which is what makes the comparison a
// real gate: NaN comparisons are false, so the check is written as a negated
// inequality rather than `difference > bound`.
export function compare(actual, expected, tolerance) {
    let max_abs = 0;
    let sum_abs = 0;
    let max_rel = 0;
    let failures = 0;
    let worst_index = -1;
    for (let index = 0; index < expected.length; index += 1) {
        const reference = expected[index];
        const difference = Math.abs(actual[index] - reference);
        const bound = tolerance.atol + tolerance.rtol * Math.abs(reference);
        if (!(difference <= bound)) {
            failures += 1;
            if (worst_index < 0) worst_index = index;
        }
        if (Number.isNaN(difference)) {
            return { max_abs: NaN, mean_abs: NaN, max_rel: NaN, failures, worst_index: index };
        }
        if (difference > max_abs) max_abs = difference;
        sum_abs += difference;
        // The atol floor keeps the ratio meaningful for a reference near zero,
        // where a relative error would otherwise explode for an error that is
        // still inside the tolerance.
        const relative = difference / Math.max(Math.abs(reference), tolerance.atol);
        if (relative > max_rel) max_rel = relative;
    }
    return {
        max_abs,
        mean_abs: sum_abs / expected.length,
        max_rel,
        failures,
        worst_index,
        elements: expected.length,
    };
}

// ---------------------------------------------------------------------------
// WebGPU
// ---------------------------------------------------------------------------

// `requestAdapter()` resolves null when the browser has not finished bringing
// its GPU process up, which is exactly what a cold headless start on a software
// adapter looks like: the first call loses the race and a later one succeeds.
// A few retries are worth more than a one-shot verdict, and a genuinely absent
// adapter still fails after five seconds.
async function request_adapter() {
    let adapter = null;
    const attempts = 10;
    for (let attempt = 0; attempt < attempts; attempt += 1) {
        adapter = await navigator.gpu.requestAdapter();
        if (adapter !== null) return adapter;
        await new Promise((done) => setTimeout(done, 500));
    }
    throw new Error(
        `requestAdapter() returned null on ${attempts} attempts: no WebGPU adapter is available`,
    );
}

async function create_device() {
    if (typeof navigator === "undefined" || navigator.gpu === undefined) {
        throw new Error("navigator.gpu is unavailable: this browser has no WebGPU");
    }
    const adapter = await request_adapter();
    // The device starts at the specification's defaults, which cap workgroup storage at 16384
    // bytes; the convolution kernel stages 18432 for its decoded weights and this host's adapter
    // offers 32768. The harness asks for the adapter's own maximum, which is what a runtime has to
    // do for a kernel like that to be creatable at all.
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
        "maxComputeInvocationsPerWorkgroup",
        "maxComputeWorkgroupSizeX",
        "minUniformBufferOffsetAlignment",
    ]) {
        limits[name] = adapter.limits[name];
    }
    return {
        adapter,
        device,
        info: {
            vendor: info.vendor ?? "",
            architecture: info.architecture ?? "",
            device: info.device ?? "",
            description: info.description ?? "",
            limits,
        },
    };
}

const module_cache = new Map();

// Fetches one of the harness's own files, retrying a transient failure. The
// shaders and the WASM oracle come from whatever http server the caller started,
// and a browser that is still opening its first connections fails attempts that
// succeed a moment later; a flaky fetch must not look like a kernel bug.
async function fetch_with_retry(url, attempts = 4) {
    let failure = null;
    for (let attempt = 0; attempt < attempts; attempt += 1) {
        try {
            const response = await fetch(url);
            if (response.ok) return response;
            // A 404 will not become a 200; report it as soon as it is certain.
            if (response.status === 404) throw new Error(`http 404 for ${url.pathname}`);
            failure = new Error(`http ${response.status} for ${url.pathname}`);
        } catch (error) {
            failure = error;
        }
        await new Promise((done) => setTimeout(done, 250));
    }
    throw failure;
}

// Compiles a shader and reports what the compiler thought of it: a WGSL error is
// a harness failure, not something to discover from a silently empty readback.
async function load_module(gpu, file) {
    if (module_cache.has(file)) return module_cache.get(file);
    const bytes = await (await fetch_with_retry(new URL(file, SHADER_BASE))).text();
    const module = gpu.device.createShaderModule({ code: bytes, label: file });
    const info = await module.getCompilationInfo();
    const messages = info.messages.map((message) => ({
        severity: message.type === "error" ? "error" : "warning",
        line: message.lineNum,
        text: message.message,
    }));
    const errors = messages.filter((message) => message.severity === "error");
    const entry = { module, messages, errors };
    module_cache.set(file, entry);
    return entry;
}

function create_binding_buffer(device, binding) {
    if (binding.uniform !== undefined) {
        const data = binding.uniform;
        const buffer = device.createBuffer({
            size: align_up(data.byteLength, 4),
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    }
    if (binding.input !== undefined) {
        const data = binding.input;
        const bytes = data instanceof Float32Array
            ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
            : data;
        const buffer = device.createBuffer({
            size: align_up(bytes.byteLength, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, bytes);
        return buffer;
    }
    const output = device.createBuffer({
        size: align_up(binding.output * 4, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // A kernel that reads and writes the same buffer -- an in-place bias add, a residual -- needs
    // its output seeded, because a fresh buffer holds whatever the driver left there.
    if (binding.seed !== undefined) {
        device.queue.writeBuffer(output, 0, binding.seed);
    }
    return output;
}

// Runs one case and compares it against its reference. `keep_values` attaches the
// read-back and reference arrays to the row, which is what makes a failure
// debuggable from the browser console:
//
//     const m = await import("/tests/gpu/harness.mjs");
//     const gpu = { device: await (await navigator.gpu.requestAdapter()).requestDevice() };
//     const row = await m.run_case(gpu, m.build_cases().find(c => c.name === "rmsnorm"), true);
export async function run_case(gpu, kernel_case, keep_values = false) {
    const row = {
        name: kernel_case.name,
        detail: kernel_case.detail,
        tolerance: kernel_case.tolerance,
    };
    const loaded = await load_module(gpu, kernel_case.shader);
    if (loaded.errors.length > 0) {
        row.status = "error";
        row.message = loaded.errors.map((error) => `line ${error.line}: ${error.text}`).join("; ");
        return row;
    }

    gpu.device.pushErrorScope("validation");
    gpu.device.pushErrorScope("internal");
    const pipeline = gpu.device.createComputePipeline({
        layout: "auto",
        compute: { module: loaded.module, entryPoint: kernel_case.entry_point },
    });
    const buffers = kernel_case.bindings.map(
        (binding) => create_binding_buffer(gpu.device, binding),
    );
    const entries = buffers.map((buffer, index) => ({ binding: index, resource: { buffer } }));
    const bind_group = gpu.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
    });
    const output_index = kernel_case.bindings.findIndex((binding) => binding.output !== undefined);
    const output_bytes = kernel_case.expected.length * 4;
    const readback = gpu.device.createBuffer({
        size: align_up(output_bytes, 4),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = gpu.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind_group);
    pass.dispatchWorkgroups(...kernel_case.dispatch);
    pass.end();
    encoder.copyBufferToBuffer(buffers[output_index], 0, readback, 0, output_bytes);
    gpu.device.queue.submit([encoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    // A copy out of the mapped range, so the comparison never touches memory the
    // driver is free to invalidate on unmap.
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();

    const internal_error = await gpu.device.popErrorScope();
    const validation_error = await gpu.device.popErrorScope();
    for (const buffer of buffers) buffer.destroy();
    readback.destroy();

    if (internal_error !== null || validation_error !== null) {
        row.status = "error";
        // Name the scope and prefer the message: a WebGPU error object stringifies to
        // "[object GPUValidationError]" and hides the one useful line.
        const scope = internal_error !== null ? "internal" : "validation";
        const error = internal_error ?? validation_error;
        row.message = `${scope} error: ${error?.message ?? String(error)}`;
        return row;
    }
    if (actual.length !== kernel_case.expected.length) {
        row.status = "error";
        row.message = `read back ${actual.length} values, expected ${kernel_case.expected.length}`;
        return row;
    }

    const comparison = compare(actual, kernel_case.expected, kernel_case.tolerance);
    row.status = comparison.failures === 0 ? "pass" : "fail";
    row.elements = comparison.elements;
    row.max_abs = comparison.max_abs;
    row.mean_abs = comparison.mean_abs;
    row.max_rel = comparison.max_rel;
    row.failures = comparison.failures;
    if (keep_values && actual.length <= 1 << 20) {
        row.actual = Array.from(actual);
        row.expected = Array.from(kernel_case.expected);
    }
    if (comparison.failures > 0) {
        row.message = `${comparison.failures} element(s) outside tolerance, first at index ` +
            `${comparison.worst_index} (actual ${actual[comparison.worst_index]}, ` +
            `reference ${kernel_case.expected[comparison.worst_index]})`;
    }
    return row;
}

// ---------------------------------------------------------------------------
// The WASM oracle
// ---------------------------------------------------------------------------

// Loads the freestanding core, checks the ABI major version, runs its self test,
// and compares the quantization hash it reports with the hash of the same fixed
// pattern computed by `reference.mjs`. A match means the packing this harness
// measures the GPU against is the packing the real implementation writes.
async function check_wasm_oracle() {
    const oracle = { present: false, ok: false };
    let response = null;
    try {
        response = await fetch_with_retry(WASM_URL);
    } catch (error) {
        oracle.note = String(error).includes("404")
            ? `not found at ${WASM_URL.pathname}; run 'zig build wasm'`
            : `not fetched (${String(error)})`;
        return oracle;
    }
    oracle.present = true;
    const bytes = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const exports = instance.exports;
    oracle.abi_version = exports.qw_abi_version();
    if (oracle.abi_version >>> 16 !== 1) {
        oracle.note = `ABI major ${oracle.abi_version >>> 16} is not the v1 this harness speaks`;
        return oracle;
    }
    oracle.core_version = exports.qw_core_version();
    // `qw_alloc` grows linear memory, and growing it detaches every view over
    // the old buffer, so the result view is built after the allocation and never
    // cached.
    const result_pointer = exports.qw_alloc(32, 8);
    if (result_pointer === 0) {
        oracle.note = "qw_alloc failed";
        return oracle;
    }
    const status = exports.qw_selftest(result_pointer);
    const view = new DataView(exports.memory.buffer);
    oracle.selftest_status = status;
    oracle.failures = view.getUint32(result_pointer + 0, true);
    oracle.quant_hash = view.getBigUint64(result_pointer + 16, true).toString(16);
    oracle.mel_hash = view.getBigUint64(result_pointer + 24, true).toString(16);
    exports.qw_free(result_pointer, 32, 8);

    const reference_hash = ref.quant_hash().toString(16);
    oracle.reference_quant_hash = reference_hash;
    oracle.ok = status === 0 && oracle.failures === 0 && oracle.quant_hash === reference_hash;
    if (!oracle.ok) {
        oracle.note = `self test status ${status}, failures ${oracle.failures}, ` +
            `quant hash ${oracle.quant_hash} vs reference ${reference_hash}`;
    }
    return oracle;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

function format_number(value, digits = 3) {
    if (value === undefined) return "-";
    if (Number.isNaN(value)) return "NaN";
    if (value === 0) return "0";
    // Small numbers would print as 0.000 and hide the margin the tolerance
    // actually left; large ones lose the exponent that makes them comparable.
    if (Math.abs(value) < 1e-3 || Math.abs(value) >= 1e6) return value.toExponential(2);
    return value.toFixed(digits);
}

export function format_table(result) {
    const lines = [];
    if (result.adapter !== undefined) {
        const adapter = result.adapter;
        lines.push(
            `adapter: vendor=${adapter.vendor} architecture=${adapter.architecture} ` +
                `device=${adapter.device} description=${adapter.description}`,
        );
        lines.push(
            `limits: maxBufferSize=${adapter.limits.maxBufferSize} ` +
                `maxStorageBufferBindingSize=${adapter.limits.maxStorageBufferBindingSize} ` +
                `maxComputeWorkgroupStorageSize=${adapter.limits.maxComputeWorkgroupStorageSize} ` +
                `maxComputeInvocationsPerWorkgroup=` +
                    `${adapter.limits.maxComputeInvocationsPerWorkgroup}`,
        );
    }
    const header = ["kernel", "status", "elements", "max_abs", "mean_abs", "max_rel", "tolerance"];
    const widths = [19, 7, 9, 11, 11, 11, 17];
    lines.push(header.map((cell, index) => cell.padEnd(widths[index])).join(" "));
    lines.push(widths.map((width) => "-".repeat(width)).join(" "));
    for (const row of result.rows) {
        const cells = [
            row.name,
            row.status,
            String(row.elements ?? "-"),
            format_number(row.max_abs),
            format_number(row.mean_abs),
            format_number(row.max_rel),
            `<${row.tolerance.atol}+${row.tolerance.rtol}*|ref|`,
        ];
        lines.push(cells.map((cell, index) => cell.padEnd(widths[index])).join(" "));
    }
    lines.push("");
    for (const row of result.rows) {
        lines.push(`${row.name} [${row.status}]: ${row.detail}`);
        if (row.message !== undefined) lines.push(`  ${row.message}`);
        if (row.messages !== undefined) {
            for (const message of row.messages) lines.push(`  ${message}`);
        }
    }
    for (const note of result.notes) lines.push(`note: ${note}`);
    lines.push(result.ok ? "harness: PASS" : "harness: FAIL");
    return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

export async function run_harness() {
    const result = { ok: false, rows: [], notes: [] };
    let gpu;
    try {
        gpu = await create_device();
        result.adapter = gpu.info;
    } catch (error) {
        result.notes.push(`WebGPU is unavailable: ${String(error)}`);
        result.rows.push({
            name: "webgpu",
            status: "error",
            tolerance: { atol: 0, rtol: 0 },
            detail: "adapter and device request",
            message: String(error),
        });
        return result;
    }

    const oracle = await check_wasm_oracle();
    result.oracle = oracle;
    result.rows.push({
        name: "wasm_oracle",
        status: oracle.present ? (oracle.ok ? "pass" : "fail") : "skip",
        tolerance: { atol: 0, rtol: 0 },
        detail: oracle.present
            ? `qw_selftest quant_hash ${oracle.quant_hash} vs ` +
                `reference.mjs ${oracle.reference_quant_hash}`
            : `zig-out/bin/qwenscriber_core.wasm ${oracle.note ?? "unavailable"}`,
        message: oracle.ok ? undefined : oracle.note,
    });
    if (!oracle.present) {
        result.notes.push(
            "the WASM oracle is missing, so the GPU results are compared against " +
                "reference.mjs alone; run 'zig build wasm' to enable the check",
        );
    }

    for (const kernel_case of build_cases()) {
        let row;
        try {
            row = await run_case(gpu, kernel_case);
        } catch (error) {
            row = {
                name: kernel_case.name,
                status: "error",
                tolerance: kernel_case.tolerance,
                detail: kernel_case.detail,
                message: error?.message ? String(error.message) : String(error),
            };
        }
        if (row.status === "error") {
            const loaded = module_cache.get(kernel_case.shader);
            if (loaded !== undefined && loaded.messages.length > 0) {
                row.messages = loaded.messages.map(
                    (message) => `${message.severity} at line ${message.line}: ${message.text}`,
                );
            }
        }
        result.rows.push(row);
    }

    result.ok = result.rows.every((row) => row.status === "pass" || row.status === "skip");
    result.notes.push(
        `tolerances (atol+rtol*|ref|): ${Object.entries(TOLERANCES)
            .map(([name, tolerance]) => `${name} ${tolerance.atol}+${tolerance.rtol}`)
            .join(", ")}`,
    );
    return result;
}

// ---------------------------------------------------------------------------
// Browser entry point
// ---------------------------------------------------------------------------

// Runs the suite, renders the table into `#table` when the page has one, prints
// it to the console, and publishes `window.__result` for the Node driver
// (`tests/gpu/driver.mjs`) and for `tools/browser/run-page.mjs`. Everything in
// `window.__result` is JSON serializable: no BigInt, no typed arrays.
export async function run_page() {
    const result = await run_harness();
    result.table_text = format_table(result);
    if (typeof document !== "undefined") {
        const output = document.querySelector("#table");
        if (output !== null) output.textContent = result.table_text;
        document.title = `qwenscriber gpu harness: ${result.ok ? "PASS" : "FAIL"}`;
    }
    console.log(result.table_text);
    if (typeof window !== "undefined") window.__result = result;
    return result;
}

// ---------------------------------------------------------------------------
// Node driver
// ---------------------------------------------------------------------------

function is_node_main() {
    return (
        typeof process !== "undefined" &&
        process.versions !== undefined &&
        process.versions.node !== undefined &&
        process.argv[1] !== undefined &&
        process.argv[1].endsWith("harness.mjs")
    );
}

if (is_node_main()) {
    const { run_driver } = await import("./driver.mjs");
    process.exit(await run_driver(process.argv.slice(2)));
}
