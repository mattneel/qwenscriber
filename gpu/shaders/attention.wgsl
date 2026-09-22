// Attention over a window, causal or not, selected by a uniform flag.
//
//   score[j] = dot(q[token][head], k[j][head]) * scale
//   p        = softmax(score)                          (maximum subtracted first)
//   out[token][head][d] = sum over visible j of p[j] * v[j][head][d]
//
// The window is the last `window` keys ending at the visible end:
//
//   non causal (audio encoder):  visible keys are [max(0, keys - window), keys)
//   causal (text decoder):       visible_end = min(token + 1, keys)
//                                visible keys are [max(0, visible_end - window), visible_end)
//
// A query attends to at most `window` keys, none of them in its future. With
// `window >= keys` the causal mode degenerates to causal attention and the
// non-causal mode to full attention.
//
// The softmax subtracts the maximum score before exponentiating, like
// `softmaxInPlace` in `src/core/math.zig`, so a window of large logits cannot
// overflow: the largest probability is exactly 1.0 before normalization.
//
// # Memory layout
//
//   queries   tokens * heads * head_dim f32, row major  (q[token][head][d])
//   keys      keys * heads * head_dim f32, row major
//   values    keys * heads * head_dim f32, row major
//   out       tokens * heads * head_dim f32, row major
//
// `scale` is the usual `1 / sqrt(head_dim)` and arrives in the uniform so the
// kernel does not recompute a transcendental per invocation.
//
// # Bind group 0
//
//   binding 0  uniform    AttentionParams { tokens, heads, keys, head_dim,
//                                           scale, window, causal, reserved }
//                         32 bytes
//   binding 1  read       queries
//   binding 2  read       keys
//   binding 3  read       values
//   binding 4  read_write out
//
// # Workgroup
//
//   workgroup_size(128, 1, 1): one workgroup per (token, head). dispatch
//   (heads, tokens, 1). `workgroup_id.x` is the head and `workgroup_id.y` the
//   query token -- not `global_invocation_id`, which counts invocations and would
//   walk off the end of the head dimension after the first workgroup. lid.x is
//   the lane; lane d < head_dim owns output dimension d.
//   Workgroup storage: 128 + 256 + 128 f32 = 2 KiB.
//
// The visible range depends only on (token, head), which every invocation of the
// workgroup shares, so the barriers below are reached uniformly. Requirements
// the harness enforces: head_dim <= ATTENTION_HEAD_DIM_MAX, 1 <= window <=
// ATTENTION_WINDOW_MAX, keys >= 1.

const ATTENTION_LANES: u32 = 128u;
const ATTENTION_HEAD_DIM_MAX: u32 = 128u;
const ATTENTION_WINDOW_MAX: u32 = 256u;
const ATTENTION_REDUCE_STEPS: u32 = 7u;
// The most negative finite f32 (0xFF7FFFFF). Written out in full because Tint
// rejects a literal that is not exactly representable as f32, and -3.4028235e38
// rounds past the negative maximum.
const ATTENTION_SCORE_MIN: f32 = -3.4028234663852886e38;

struct AttentionParams {
    tokens: u32,
    heads: u32,
    keys: u32,
    head_dim: u32,
    scale: f32,
    window: u32,
    causal: u32,
    reserved: u32,
};

@group(0) @binding(0) var<uniform> params: AttentionParams;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read> keys: array<f32>;
@group(0) @binding(3) var<storage, read> values: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> query_shared: array<f32, ATTENTION_HEAD_DIM_MAX>;
var<workgroup> score_shared: array<f32, ATTENTION_WINDOW_MAX>;
var<workgroup> reduce_shared: array<f32, ATTENTION_LANES>;

@compute @workgroup_size(ATTENTION_LANES, 1, 1)
fn attention_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    // Every guard before the first barrier depends only on the uniform block, so
    // the whole workgroup takes the same branch. `workgroup_id` is uniform too,
    // which is why the head and token come from it and not from
    // `global_invocation_id` (that counts invocations, not workgroups).
    if (params.head_dim > ATTENTION_HEAD_DIM_MAX) {
        return;
    }
    if (params.window == 0u) {
        return;
    }
    if (params.window > ATTENTION_WINDOW_MAX) {
        return;
    }
    if (params.keys == 0u) {
        return;
    }
    if (params.heads == 0u) {
        return;
    }
    if (params.tokens == 0u) {
        return;
    }

    let head = min(wid.x, params.heads - 1u);
    let token = min(wid.y, params.tokens - 1u);
    let head_in_range = wid.x < params.heads;
    let token_in_range = wid.y < params.tokens;
    var visible_end = params.keys;
    if (params.causal != 0u) {
        visible_end = min(token + 1u, params.keys);
    }
    var window_start = 0u;
    if (visible_end > params.window) {
        window_start = visible_end - params.window;
    }
    let visible = visible_end - window_start;

    let query_base = (token * params.heads + head) * params.head_dim;
    if (lane < params.head_dim) {
        query_shared[lane] = queries[query_base + lane];
    }
    workgroupBarrier();

    for (var r = lane; r < visible; r = r + ATTENTION_LANES) {
        let key_base = ((window_start + r) * params.heads + head) * params.head_dim;
        var dot_product = 0.0;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            dot_product = dot_product + query_shared[d] * keys[key_base + d];
        }
        score_shared[r] = dot_product * params.scale;
    }
    workgroupBarrier();

    var partial_max = ATTENTION_SCORE_MIN;
    for (var r = lane; r < visible; r = r + ATTENTION_LANES) {
        partial_max = max(partial_max, score_shared[r]);
    }
    reduce_shared[lane] = partial_max;
    workgroupBarrier();
    var stride = ATTENTION_LANES / 2u;
    for (var step = 0u; step < ATTENTION_REDUCE_STEPS; step = step + 1u) {
        if (lane < stride) {
            reduce_shared[lane] = max(reduce_shared[lane], reduce_shared[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let score_max = reduce_shared[0];
    // Read before overwrite: the exponentiation pass below reuses reduce_shared
    // for the probability sum.
    workgroupBarrier();

    var partial_sum = 0.0;
    for (var r = lane; r < visible; r = r + ATTENTION_LANES) {
        score_shared[r] = exp(score_shared[r] - score_max);
        partial_sum = partial_sum + score_shared[r];
    }
    reduce_shared[lane] = partial_sum;
    workgroupBarrier();
    stride = ATTENTION_LANES / 2u;
    for (var step = 0u; step < ATTENTION_REDUCE_STEPS; step = step + 1u) {
        if (lane < stride) {
            reduce_shared[lane] = reduce_shared[lane] + reduce_shared[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inverse_total = 1.0 / reduce_shared[0];

    // Lane d owns output dimension d and walks every visible key, so the
    // weighted sum needs no cross-lane reduction.
    if (lane < params.head_dim) {
        var accumulator = 0.0;
        for (var r = 0u; r < visible; r = r + 1u) {
            let value_base = ((window_start + r) * params.heads + head) * params.head_dim;
            accumulator = accumulator + score_shared[r] * inverse_total * values[value_base + lane];
        }
        if (head_in_range) {
            if (token_in_range) {
                out[query_base + lane] = accumulator;
            }
        }
    }
}
