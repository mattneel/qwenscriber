// Index of the largest value, ties going to the lowest index.
//
// The decoder's last step is an argmax over the vocabulary -- 151936 values for the released
// checkpoints -- and reading those back to pick one would move 600 KB per token across the boundary.
// The core does the same scan in `argmaxLogits`, with `value > best_value` so that the first of equal
// maxima wins, which is what this reproduces.
//
// # Memory layout
//
//   values  count f32
//   out     one u32: the index of the maximum
//
// # Bind group 0
//
//   binding 0  uniform    ArgmaxParams { count, reserved_0, reserved_1, reserved_2 }
//   binding 1  read       values
//   binding 2  read_write out
//
// # Workgroup
//
//   workgroup_size(256, 1, 1): one workgroup for the whole array, dispatch (1, 1, 1). Each lane
//   walks the values with a 256 stride, keeping its own best index, and a 256-lane tree reduction in
//   workgroup memory folds the partials -- pairs, so that the lower index wins a tie at every level
//   and the result does not depend on the reduction order.
//
//   `count` is bounded by ARGMAX_COUNT_MAX, which is the vocabulary this runtime accepts. A longer
//   array is not a shape a checkpoint can have; returning leaves the output untouched rather than
//   reporting a maximum over part of it.

const ARGMAX_LANES: u32 = 256u;
const ARGMAX_COUNT_MAX: u32 = 1u << 20u;
const ARGMAX_STEPS_MAX: u32 = ARGMAX_COUNT_MAX / ARGMAX_LANES;
const ARGMAX_REDUCE_STEPS: u32 = 8u;

struct ArgmaxParams {
    count: u32,
    reserved_0: u32,
    reserved_1: u32,
    reserved_2: u32,
};

@group(0) @binding(0) var<uniform> params: ArgmaxParams;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

var<workgroup> best_values: array<f32, ARGMAX_LANES>;
var<workgroup> best_indices: array<u32, ARGMAX_LANES>;

@compute @workgroup_size(ARGMAX_LANES, 1, 1)
fn argmax_f32_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    if (params.count == 0u || params.count > ARGMAX_COUNT_MAX) {
        return;
    }

    var best_value = values[0];
    var best_index = 0u;
    for (var step = 0u; step < ARGMAX_STEPS_MAX; step = step + 1u) {
        let index = lane + step * ARGMAX_LANES;
        if (index < params.count) {
            let value = values[index];
            // `>` rather than `>=`: the first of equal maxima wins, as `argmaxLogits` does.
            if (value > best_value) {
                best_value = value;
                best_index = index;
            }
        }
    }
    best_values[lane] = best_value;
    best_indices[lane] = best_index;
    workgroupBarrier();

    var stride = ARGMAX_LANES / 2u;
    for (var step = 0u; step < ARGMAX_REDUCE_STEPS; step = step + 1u) {
        if (lane < stride) {
            let other_value = best_values[lane + stride];
            let other_index = best_indices[lane + stride];
            if (other_value > best_values[lane] ||
                (other_value == best_values[lane] && other_index < best_indices[lane])) {
                best_values[lane] = other_value;
                best_indices[lane] = other_index;
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lane == 0u) {
        out[0] = best_indices[0];
    }
}
