// One query's attention over a q8 key/value cache, dequantized as it is read.
//
//   score[j] = dot(query[head], key[j][kv_head]) * scale
//   p        = softmax(score)
//   out[d]   = sum over j of p[j] * value[j][kv_head][d]
//
// The cache holds no f32 copy: every cached row is an f16 scale per 64 values followed by one code
// per value, so a reader decodes `scale * (code - 128)` in place. That is the point of the format --
// 462 MiB at 8192 positions against 1792 MiB in f32 -- and a decode step reads every cached position,
// so the decode has to be fused with the arithmetic rather than run as a separate pass.
//
// Grouped attention: query head `h` reads the key/value head `h / (heads / kv_heads)`.
//
// # Memory layout
//
//   query      heads * head_dim f32                     (query[head][d])
//   k_scales   one f16 per group per cached position, keys
//   k_codes    one byte per value per cached position, keys
//   v_scales   the same for values
//   v_codes
//   out        heads * head_dim f32
//
// A cached position's row is `row_width = kv_heads * head_dim` values wide -- all of a position's
// heads together -- so the group a (position, column) belongs to is `column / 64` of that row, and a
// plane's byte for it is `base + (position * groups_per_row + column / 64) * 2` for a scale or
// `base + position * row_width + column` for a code.
//
// # Bind group 0
//
//   binding 0  uniform    DecodeAttentionParams { heads, kv_heads, head_dim, positions,
//                                                 groups_per_row, scale_base, code_base, reserved }
//                         32 bytes
//   binding 1  read       query
//   binding 2  read       k_scales
//   binding 3  read       k_codes
//   binding 4  read       v_scales
//   binding 5  read       v_codes
//   binding 6  read_write out
//
// # Workgroup
//
//   workgroup_size(128, 1, 1): one workgroup per head, dispatch (heads, 1, 1). Phase 1 gives each
//   lane whole scores -- lane `l` scores the positions `l, l + 128, ...`, each a head_dim-long dot --
//   so scoring needs no reduction at all. Phase 2 reduces the maximum and then the sum over the
//   positions, with the barriers outside every conditional. Phase 3 gives lane `l` output dimension
//   `l`, accumulating over the positions. Workgroup storage: `scores` (DECODE_POSITIONS_MAX f32) plus
//   a 128-lane reduction array, 4096 * 4 + 128 * 4 = 16896 bytes.
//
// A sequence longer than DECODE_POSITIONS_MAX is refused by the caller rather than scored in part:
// the bound exists because the scores have to live in workgroup storage, whose size is a device
// limit.

const DECODE_LANES: u32 = 128u;
const DECODE_GROUP: u32 = 64u;
const DECODE_BIAS: f32 = 128.0;
const DECODE_POSITIONS_MAX: u32 = 4096u;
const DECODE_REDUCE_STEPS: u32 = 7u;
const DECODE_SCORE_MIN: f32 = -3.4028234663852886e38;

struct DecodeAttentionParams {
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    positions: u32,
    groups_per_row: u32,
    scale_base: u32,
    code_base: u32,
    reserved: u32,
};

@group(0) @binding(0) var<uniform> params: DecodeAttentionParams;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> k_scales: array<u32>;
@group(0) @binding(3) var<storage, read> k_codes: array<u32>;
@group(0) @binding(4) var<storage, read> v_scales: array<u32>;
@group(0) @binding(5) var<storage, read> v_codes: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

var<workgroup> scores: array<f32, DECODE_POSITIONS_MAX>;
var<workgroup> reduce_values: array<f32, DECODE_LANES>;

/// f16 bits (in the low half of a u32) to f32, as `half_float.fromF16` widens them.
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

/// Byte `index` of a u32 word: little endian, so byte 0 is the low eight bits.
fn byte_of(word: u32, index: u32) -> u32 {
    return (word >> ((index & 3u) * 8u)) & 0xFFu;
}

/// The f16 scale covering (position, column) of the key cache.
fn key_scale_at(position: u32, column: u32) -> f32 {
    let group = position * params.groups_per_row + column / DECODE_GROUP;
    let byte_index = params.scale_base + group * 2u;
    let low = byte_of(k_scales[byte_index / 4u], byte_index);
    let high_byte = byte_index + 1u;
    let high = byte_of(k_scales[high_byte / 4u], high_byte);
    return f16_bits_to_f32(low | (high << 8u));
}

/// The decoded key at (position, column).
fn key_at(position: u32, column: u32) -> f32 {
    let code_index = params.code_base + position * params.groups_per_row * DECODE_GROUP + column;
    let code = byte_of(k_codes[code_index / 4u], code_index);
    return key_scale_at(position, column) * (f32(code) - DECODE_BIAS);
}

/// The decoded value at (position, column). The value planes use the same layout as the key planes.
fn value_at(position: u32, column: u32) -> f32 {
    let group = position * params.groups_per_row + column / DECODE_GROUP;
    let scale_byte = params.scale_base + group * 2u;
    let low = byte_of(v_scales[scale_byte / 4u], scale_byte);
    let high_byte = scale_byte + 1u;
    let high = byte_of(v_scales[high_byte / 4u], high_byte);
    let scale = f16_bits_to_f32(low | (high << 8u));
    let code_index = params.code_base + position * params.groups_per_row * DECODE_GROUP + column;
    let code = byte_of(v_codes[code_index / 4u], code_index);
    return scale * (f32(code) - DECODE_BIAS);
}

/// Folds the lanes' partials into `reduce_values[0]` and returns it, with every barrier outside a
/// conditional. `lane` must be each invocation's own id.
fn reduce_max(lane: u32, value: f32) -> f32 {
    reduce_values[lane] = value;
    workgroupBarrier();
    var stride = DECODE_LANES / 2u;
    for (var step = 0u; step < DECODE_REDUCE_STEPS; step = step + 1u) {
        if (lane < stride) {
            reduce_values[lane] = max(reduce_values[lane], reduce_values[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let total = reduce_values[0];
    workgroupBarrier();
    return total;
}

/// The sum counterpart; kept separate because WGSL has no function pointers and the operator is the
/// only difference.
fn reduce_sum(lane: u32, value: f32) -> f32 {
    reduce_values[lane] = value;
    workgroupBarrier();
    var stride = DECODE_LANES / 2u;
    for (var step = 0u; step < DECODE_REDUCE_STEPS; step = step + 1u) {
        if (lane < stride) {
            reduce_values[lane] = reduce_values[lane] + reduce_values[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let total = reduce_values[0];
    workgroupBarrier();
    return total;
}

@compute @workgroup_size(DECODE_LANES, 1, 1)
fn decode_attention_q8_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    let head = wid.x;
    if (params.heads == 0u || params.kv_heads == 0u || params.head_dim == 0u ||
        params.positions == 0u || params.positions > DECODE_POSITIONS_MAX ||
        params.heads % params.kv_heads != 0u) {
        return;
    }
    let group_size = params.heads / params.kv_heads;
    let kv_head = head / group_size;
    let column_base = kv_head * params.head_dim;
    let scale = 1.0 / sqrt(f32(params.head_dim));

    // Phase 1: lane `l` scores positions l, l + 128, ... Each is a whole head_dim-long dot, so the
    // result lands in workgroup memory without a reduction.
    for (var step = 0u; step < DECODE_POSITIONS_MAX; step = step + 1u) {
        let position = lane + step * DECODE_LANES;
        if (position >= params.positions) {
            break;
        }
        var accumulator = 0.0;
        for (var index = 0u; index < params.head_dim; index = index + 1u) {
            accumulator = accumulator +
                query[head * params.head_dim + index] * key_at(position, column_base + index);
        }
        scores[position] = accumulator * scale;
    }
    workgroupBarrier();

    // Phase 2: the maximum, then the sum of the exponentials. Subtracting the maximum is what keeps a
    // window of large scores from overflowing, and it is what `softmaxInPlace` does in the core.
    var partial_max = DECODE_SCORE_MIN;
    for (var step = 0u; step < DECODE_POSITIONS_MAX; step = step + 1u) {
        let position = lane + step * DECODE_LANES;
        if (position >= params.positions) {
            break;
        }
        partial_max = max(partial_max, scores[position]);
    }
    let maximum = reduce_max(lane, partial_max);

    var partial_sum = 0.0;
    for (var step = 0u; step < DECODE_POSITIONS_MAX; step = step + 1u) {
        let position = lane + step * DECODE_LANES;
        if (position >= params.positions) {
            break;
        }
        partial_sum = partial_sum + exp(scores[position] - maximum);
    }
    let inverse_total = 1.0 / reduce_sum(lane, partial_sum);

    // The probabilities replace the scores in place, so phase 3 is a weighted sum and does not
    // recompute an exponential per output dimension.
    for (var step = 0u; step < DECODE_POSITIONS_MAX; step = step + 1u) {
        let position = lane + step * DECODE_LANES;
        if (position >= params.positions) {
            break;
        }
        scores[position] = exp(scores[position] - maximum) * inverse_total;
    }
    workgroupBarrier();

    // Phase 3: lane `l` owns output dimension `l`.
    if (lane < params.head_dim) {
        var accumulator = 0.0;
        for (var position = 0u; position < params.positions; position = position + 1u) {
            accumulator = accumulator + scores[position] * value_at(position, column_base + lane);
        }
        out[head * params.head_dim + lane] = accumulator;
    }
}
