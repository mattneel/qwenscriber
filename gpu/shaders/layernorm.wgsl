// Per-row LayerNorm with an affine weight and bias: the audio tower's normalization.
//
//   mean      = sum(x[k]) / cols
//   variance  = sum((x[k] - mean)^2) / cols
//   out[r][k] = (x[r][k] - mean) / sqrt(variance + eps) * weight[k] + bias[k]
//
// Mirrors `layerNormInPlace` in `src/core/math.zig`: the variance is biased, as `torch.nn.LayerNorm`
// computes it, and the statistics are gathered in the same two passes the core uses — the mean first,
// then the squared deviations from it. Accumulating a sum and a sum of squares in one pass would be
// algebraically equal and numerically different, which is the kind of difference this repository
// compares against the reference rather than argues about.
//
// Unlike `rmsnorm.wgsl` there is no reciprocal to take: LayerNorm subtracts the mean, so its affine
// parameter is a pair.
//
// # Memory layout
//
//   x       rows * cols f32, row major, stride cols   (x[row][k])
//   weight  cols f32
//   bias    cols f32
//   out     rows * cols f32, row major, stride cols
//
// # Bind group 0
//
//   binding 0  uniform    LayerNormParams { rows, cols, eps, reserved }  (16 bytes)
//   binding 1  read       x
//   binding 2  read       weight
//   binding 3  read       bias
//   binding 4  read_write out
//
// # Workgroup
//
//   workgroup_size(256, 1, 1): one workgroup per row, dispatch (rows, 1, 1), and the row index is
//   `workgroup_id.x` for the reason `rmsnorm.wgsl` documents — `global_invocation_id.x` counts
//   invocations, not rows. Each invocation walks the row with a 256 stride; two 256-lane tree
//   reductions in workgroup memory fold the partials, with every invocation taking part in every
//   barrier so the tree's `if (lane < stride)` body never wraps one. Workgroup storage: 256 f32.
//
// The row walk is bounded by LAYERNORM_COLS_MAX / LAYERNORM_LANES steps with a per-step column guard.
// A row wider than the bound is not a shape this runtime accepts, and returning leaves the output
// untouched for the caller to notice rather than normalizing half a row.

const LAYERNORM_LANES: u32 = 256u;
const LAYERNORM_COLS_MAX: u32 = 8192u;
const LAYERNORM_REDUCE_STEPS: u32 = 8u;
const LAYERNORM_STEPS_MAX: u32 = LAYERNORM_COLS_MAX / LAYERNORM_LANES;

struct LayerNormParams {
    rows: u32,
    cols: u32,
    eps: f32,
    reserved: u32,
};

@group(0) @binding(0) var<uniform> params: LayerNormParams;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> partials: array<f32, LAYERNORM_LANES>;

/// Folds this workgroup's partials into `partials[0]`, with the barriers outside the conditional.
///
/// The trailing barrier is load bearing, and is the one difference from the same helper in
/// `rmsnorm.wgsl`. That kernel reads its total once and never touches workgroup memory again, but
/// this one reduces twice through the same array: without a barrier here, lane 0's first write of
/// the second reduction can land before a slower lane has read the mean out of `partials[0]`, and
/// that lane then normalizes its columns against a partial sum of squares instead of the mean. The
/// harness caught exactly that as a few hundredths of a unit of drift in most elements.
fn reduce_partials(lane: u32) -> f32 {
    var stride = LAYERNORM_LANES / 2u;
    for (var step = 0u; step < LAYERNORM_REDUCE_STEPS; step = step + 1u) {
        if (lane < stride) {
            partials[lane] = partials[lane] + partials[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let total = partials[0];
    workgroupBarrier();
    return total;
}

@compute @workgroup_size(LAYERNORM_LANES, 1, 1)
fn layernorm_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    // The guards depend only on the uniform block, so the whole workgroup takes the same branch and
    // the barriers inside `reduce_partials` stay in uniform control flow.
    if (params.cols == 0u || params.cols > LAYERNORM_COLS_MAX || params.rows == 0u) {
        return;
    }
    let row = min(wid.x, params.rows - 1u);
    let in_range = wid.x < params.rows;
    let row_base = row * params.cols;

    var sum = 0.0;
    for (var step = 0u; step < LAYERNORM_STEPS_MAX; step = step + 1u) {
        let column = lane + step * LAYERNORM_LANES;
        if (column < params.cols) {
            sum = sum + x[row_base + column];
        }
    }
    partials[lane] = sum;
    workgroupBarrier();
    let mean = reduce_partials(lane) / f32(params.cols);

    var squared = 0.0;
    for (var step = 0u; step < LAYERNORM_STEPS_MAX; step = step + 1u) {
        let column = lane + step * LAYERNORM_LANES;
        if (column < params.cols) {
            let centered = x[row_base + column] - mean;
            squared = squared + centered * centered;
        }
    }
    partials[lane] = squared;
    workgroupBarrier();
    let variance = reduce_partials(lane) / f32(params.cols);
    let inverse_std = 1.0 / sqrt(variance + params.eps);

    for (var step = 0u; step < LAYERNORM_STEPS_MAX; step = step + 1u) {
        let column = lane + step * LAYERNORM_LANES;
        if (column < params.cols) {
            if (in_range) {
                out[row_base + column] =
                    (x[row_base + column] - mean) * inverse_std * weight[column] + bias[column];
            }
        }
    }
}
