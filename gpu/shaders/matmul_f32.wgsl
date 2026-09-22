// Dense f32 matmul: the reference case for the quantized kernels.
//
//   out[token][row] = dot(activation[token], weight[row])
//
// `matmul_q4.wgsl` and `matmul_q5.wgsl` deliberately share this file's tiling,
// workgroup shape, and per-output accumulation order. The comparison in
// `tests/gpu/harness.mjs` is then about the quantization path and nothing else:
// different tiling or a different K order would show up as accumulation noise
// and mask what is being measured.
//
// # Memory layout
//
//   activations   tokens * cols f32, row major, stride cols  (activation[token][k])
//   weights       rows * cols f32, row major, stride cols    (weight[row][k])
//   out           tokens * rows f32, row major, stride rows  (out[token][row])
//
// `cols` is a multiple of MATMUL_TILE_K in every shape this runtime serves; the
// kernel guards the tail anyway so a ragged shape reads zeros rather than past
// the end of a buffer.
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
//   workgroup_size(16, 16, 1): local x indexes the output row within the tile,
//   local y the token within the tile, so one workgroup computes a 16x16 output
//   tile in 16 K slabs. dispatch: (ceil(rows / 16), ceil(tokens / 16), 1).
//   Workgroup storage: 2 * MATMUL_TILE_K * 16 f32 = 2 KiB.
//
//   Each slab stages a 16x16 operand tile as `tile[k * 16 + lane]`, so the
//   accumulation loop reads both tiles contiguously along their lane axis. The
//   tiles are written with the k axis fastest (`index = lid.x * 16 + lid.y`),
//   which keeps every invocation's global read contiguous in k at the cost of
//   strided workgroup writes -- one store per slab against 16 loads.
//
// `cols` arrives in a uniform, so every invocation in the workgroup walks the
// same K slabs and reaches the barriers inside the loop together.

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
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> activation_tile: array<f32, MATMUL_TILE_K * MATMUL_TILE>;
var<workgroup> weight_tile: array<f32, MATMUL_TILE_K * MATMUL_TILE>;

@compute @workgroup_size(16, 16, 1)
fn matmul_f32_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let token = gid.y;
    let row = gid.x;
    // The tile row this invocation stages is the tile's lane index (`lid.y`),
    // not the row of its own output (`lid.x`). Staging the (lane = lid.y,
    // k = lid.x) element keeps each warp's global reads contiguous in k.
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
                weight = weights[staging_row * params.cols + slab_column];
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
