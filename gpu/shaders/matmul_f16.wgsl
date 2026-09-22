// Dense f16-weight matmul: the element-format counterpart of `matmul_f32.wgsl`.
//
//   out[token][row] = dot(activation[token], weight[row])
//
// Tiling, workgroup shape, and per-output accumulation order are `matmul_f32.wgsl`'s, deliberately:
// a comparison between the two is then about the weight datatype and nothing else. The weights keep
// the container's storage: contiguous f16 values, two per u32 element, little endian, so weight `i`
// is the low half of element `i / 2` when `i` is even. WGSL has no f16 storage format without the
// `shader-f16` feature, and the packing is what the runtime reads anyway, so the widening is written
// out here -- the same function as in `conv3x3_stride2_gelu.wgsl`, duplicated for the reason
// WGSL gives every shader its own copy: no include mechanism exists.
//
// The decode happens once per staged weight, in the K-slab staging step, rather than once per
// output element: a 16x16 tile reuses each staged weight 16 times.
//
// # Memory layout
//
//   activations   tokens * cols f32, row major, stride cols  (activation[token][k])
//   weights       rows * cols f16 values, two per u32        (weight[row][k])
//   out           tokens * rows f32, row major, stride rows  (out[token][row])
//
// # Bind group 0
//
//   binding 0  uniform    MatmulParams { tokens, rows, cols, reserved }  (16 bytes)
//   binding 1  read       activations
//   binding 2  read       weights
//   binding 3  read_write out
//
// # Workgroup
//
//   workgroup_size(16, 16, 1): dispatch (ceil(rows / 16), ceil(tokens / 16), 1). Workgroup storage:
//   2 * 16 * 16 f32 = 2 KiB. See `matmul_f32.wgsl` for why the staging index is `lid.x * 16 + lid.y`.

const MATMUL_TILE_K: u32 = 16u;
const MATMUL_TILE: u32 = 16u;

struct MatmulParams {
    tokens: u32,
    rows: u32,
    cols: u32,
    reserved: u32,
};

@group(0) @binding(0) var<uniform> params: MatmulParams;
@group(0) @binding(1) var<storage, read> activations: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> activation_tile: array<f32, MATMUL_TILE_K * MATMUL_TILE>;
var<workgroup> weight_tile: array<f32, MATMUL_TILE_K * MATMUL_TILE>;

/// f16 bits (in the low half of a u32) to f32, as `half_float.fromF16` widens them.
///
/// Subnormals are `mantissa * 2^-24` and normals are `(1024 + mantissa) * 2^(exponent - 25)`, both
/// scaled by a power of two, so neither path rounds. The special values are assembled from the
/// value's own bits because WGSL rejects a constant expression that evaluates to inf or NaN.
fn f16_bits_to_f32(bits: u32) -> f32 {
    let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
    let exponent = (bits >> 10u) & 0x1Fu;
    let mantissa = bits & 0x3FFu;
    if (exponent == 0u) {
        return sign * f32(mantissa) * exp2(-24.0);
    }
    if (exponent == 31u) {
        let magnitude = select(0x7F800000u, 0x7FC00000u, mantissa != 0u);
        return bitcast<f32>(magnitude | ((bits & 0x8000u) << 16u));
    }
    return sign * f32(1024u + mantissa) * exp2(f32(exponent) - 25.0);
}

/// Weight `index` of the packed plane, as f32.
fn weight_value(index: u32) -> f32 {
    let packed = weights[index / 2u];
    let half_shift = (index & 1u) * 16u;
    return f16_bits_to_f32((packed >> half_shift) & 0xFFFFu);
}

@compute @workgroup_size(16, 16, 1)
fn matmul_f16_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let token = gid.y;
    let row = gid.x;
    let staging_row = gid.x - lid.x + lid.y;

    var accumulator = 0.0;
    for (var k_start = 0u; k_start < params.cols; k_start = k_start + MATMUL_TILE_K) {
        let slab_column = k_start + lid.x;
        var activation = 0.0;
        if (slab_column < params.cols) {
            if (token < params.tokens) {
                activation = activations[token * params.cols + slab_column];
            }
        }
        activation_tile[lid.x * MATMUL_TILE + lid.y] = activation;

        var weight = 0.0;
        if (slab_column < params.cols) {
            if (staging_row < params.rows) {
                weight = weight_value(staging_row * params.cols + slab_column);
            }
        }
        weight_tile[lid.x * MATMUL_TILE + lid.y] = weight;
        workgroupBarrier();

        for (var k = 0u; k < MATMUL_TILE_K; k = k + 1u) {
            accumulator = accumulator + activation_tile[k * MATMUL_TILE + lid.y] *
                weight_tile[k * MATMUL_TILE + lid.x];
        }
        workgroupBarrier();
    }

    if (token < params.tokens) {
        if (row < params.rows) {
            out[token * params.rows + row] = accumulator;
        }
    }
}
