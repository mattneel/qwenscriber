// Per-row RMSNorm with a weight vector.
//
//   mean_square = sum(x[k] * x[k]) / cols
//   out[r][k]   = x[r][k] * (1 / sqrt(mean_square + eps)) * weight[k]
//
// Mirrors `rmsNormInPlace` in `src/core/math.zig`: mean of squares with no mean
// subtraction, `rsqrt(mean_square + eps)` written as `1.0 / sqrt(...)`, then the
// elementwise weight, everything in f32.
//
// # Memory layout
//
//   x       rows * cols f32, row major, stride cols   (x[row][k])
//   weight  cols f32
//   out     rows * cols f32, row major, stride cols
//
// # Bind group 0
//
//   binding 0  uniform    RmsNormParams { rows, cols, eps, reserved }  (16 bytes)
//   binding 1  read       x
//   binding 2  read       weight
//   binding 3  read_write out
//
// # Workgroup
//
//   workgroup_size(256, 1, 1): one workgroup per row, dispatch (rows, 1, 1).
//   The row index is `workgroup_id.x`, not `global_invocation_id.x`: the latter
//   counts invocations, so with 256 invocations per workgroup it would run off
//   the end of the matrix after the first workgroup.
//   Each invocation walks the row with a 256 stride and accumulates a partial
//   sum of squares, which a 256-lane tree reduction in workgroup memory folds
//   into `partials[0]` in 8 steps. Every invocation takes part in every barrier,
//   so the tree's `if (lane < stride)` body never wraps a barrier.
//   Workgroup storage: 256 f32 = 1 KiB.
//
// The row walk is a bounded loop: RMSNORM_COLS_MAX / RMSNORM_LANES steps, with a
// per-step column guard. A row wider than RMSNORM_COLS_MAX is not a shape this
// runtime accepts, and returning leaves the output untouched for the caller to
// notice instead of normalizing half a row.

const RMSNORM_LANES: u32 = 256u;
const RMSNORM_COLS_MAX: u32 = 8192u;
const RMSNORM_REDUCE_STEPS: u32 = 8u;
const RMSNORM_STEPS_MAX: u32 = RMSNORM_COLS_MAX / RMSNORM_LANES;

struct RmsNormParams {
    rows: u32,
    cols: u32,
    eps: f32,
    reserved: u32,
};

@group(0) @binding(0) var<uniform> params: RmsNormParams;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> partials: array<f32, RMSNORM_LANES>;

@compute @workgroup_size(RMSNORM_LANES, 1, 1)
fn rmsnorm_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    // Both guards depend only on the uniform block, so the whole workgroup takes
    // the same branch and the barriers below stay in uniform control flow.
    if (params.cols > RMSNORM_COLS_MAX) {
        return;
    }
    if (params.rows == 0u) {
        return;
    }
    if (params.cols == 0u) {
        return;
    }
    // The row is the *workgroup* index: this kernel dispatches one workgroup per
    // row with 256 invocations in it, so `global_invocation_id.x` would count
    // invocations, not rows. `workgroup_id` is uniform across the workgroup and
    // the row count comes from the uniform block, so the clamping below only
    // covers a dispatch that is larger than the data.
    let row = min(wid.x, params.rows - 1u);
    let in_range = wid.x < params.rows;
    let row_base = row * params.cols;

    var sum_squares = 0.0;
    for (var step = 0u; step < RMSNORM_STEPS_MAX; step = step + 1u) {
        let column = lane + step * RMSNORM_LANES;
        if (column < params.cols) {
            let value = x[row_base + column];
            sum_squares = sum_squares + value * value;
        }
    }

    partials[lane] = sum_squares;
    workgroupBarrier();
    var stride = RMSNORM_LANES / 2u;
    for (var step = 0u; step < RMSNORM_REDUCE_STEPS; step = step + 1u) {
        if (lane < stride) {
            partials[lane] = partials[lane] + partials[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let mean_square = partials[0] / f32(params.cols);
    let inverse_rms = 1.0 / sqrt(mean_square + params.eps);

    for (var step = 0u; step < RMSNORM_STEPS_MAX; step = step + 1u) {
        let column = lane + step * RMSNORM_LANES;
        if (column < params.cols) {
            let value = x[row_base + column];
            if (in_range) {
                out[row_base + column] = value * inverse_rms * weight[column];
            }
        }
    }
}
