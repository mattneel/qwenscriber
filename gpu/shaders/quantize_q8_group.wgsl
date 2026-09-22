// Quantizes one 64-value group into the q8 cache layout: an f16 scale, then one code per value.
//
// Mirrors `quantizeRow`/`quantizeGroup` in `src/core/quant.zig`, where the decoder's key/value cache
// is written:
//
//   max_abs   = max |value| over the group
//   scale     = f16(max_abs / 128)                       (q8's bias, the code midpoint)
//   code      = clamp(round(value / scale), -128, 127) + 128
//
// Two details are load bearing. The codes are computed against the *stored* f16 scale, not the f32
// quotient, so a group can never decode above its true maximum; and rounding is half away from zero,
// which `floor(x + 0.5)` and `ceil(x - 0.5)` give for each sign. A group of zeros stores a zero scale
// and codes of 128, which decode back to zero without a special case in the reader.
//
// The scale plane and the code plane are separate regions of one cache buffer, at byte offsets the
// caller passes in, because the cache is layer-major and each layer's planes are strided. Writing
// inline here is what keeps the append to one dispatch per layer per token.
//
// # Memory layout
//
//   values  rows * cols f32, row major        (the row being appended, cols a multiple of 64)
//   scales  bytes; group `g` of row `r` stores its f16 at   scale_base   + 2 * (r * groups_per_row + g)
//   codes   bytes; group `g` of row `r` stores code `c` at  code_base    + (r * groups_per_row + g) * 64 + c
//
// # Bind group 0
//
//   binding 0  uniform    QuantizeParams { rows, cols, scale_base, code_base }  (16 bytes)
//   binding 1  read       values
//   binding 2  read_write scales, bound as u32 words
//   binding 3  read_write codes, bound as u32 words
//
// # Workgroup
//
//   workgroup_size(64, 1, 1): one workgroup per (group, row), dispatch (groups_per_row, rows, 1).
//   The 64 lanes reduce the group's maximum in workgroup memory, then lane 0 stores the scale and
//   lanes 0..15 each pack four codes into one 32-bit word -- four codes to a word is why the lane
//   count and the group size are both 64 and the store is 16 wide.

const QUANTIZE_GROUP: u32 = 64u;
const QUANTIZE_BIAS: f32 = 128.0;
const QUANTIZE_REDUCE_STEPS: u32 = 6u;
const QUANTIZE_CODES_PER_WORD: u32 = 4u;
const QUANTIZE_CODE_WORDS: u32 = QUANTIZE_GROUP / QUANTIZE_CODES_PER_WORD;

struct QuantizeParams {
    rows: u32,
    cols: u32,
    scale_base: u32,
    code_base: u32,
};

@group(0) @binding(0) var<uniform> params: QuantizeParams;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read_write> scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> codes: array<u32>;

var<workgroup> maximum: array<f32, QUANTIZE_GROUP>;

/// Round half away from zero, matching `quantizeOne`'s `floor(x + 0.5)` / `ceil(x - 0.5)`.
fn round_half_away(value: f32) -> f32 {
    if (value >= 0.0) {
        return floor(value + 0.5);
    }
    return ceil(value - 0.5);
}

/// One value's code, shifted into the unsigned range the plane stores.
fn code_of(value: f32, scale: f32) -> u32 {
    if (scale == 0.0) {
        // A zero scale stores the midpoint for every value, which decodes to zero.
        return u32(QUANTIZE_BIAS);
    }
    let scaled = round_half_away(value / scale);
    let clamped = clamp(scaled, -QUANTIZE_BIAS, QUANTIZE_BIAS - 1.0);
    return u32(clamped + QUANTIZE_BIAS);
}

@compute @workgroup_size(QUANTIZE_GROUP, 1, 1)
fn quantize_q8_group_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    let group = wid.x;
    let row = wid.y;
    if (params.cols % QUANTIZE_GROUP != 0u) {
        return;
    }
    let groups_per_row = params.cols / QUANTIZE_GROUP;
    if (group >= groups_per_row || row >= params.rows) {
        return;
    }
    let base = row * params.cols + group * QUANTIZE_GROUP;

    // No guard on the value: the core asserts finiteness rather than branching on it, and a branch
    // before the barrier would make the reduction's control flow depend on the data.
    maximum[lane] = abs(values[base + lane]);
    workgroupBarrier();

    var stride = QUANTIZE_GROUP / 2u;
    for (var step = 0u; step < QUANTIZE_REDUCE_STEPS; step = step + 1u) {
        if (lane < stride) {
            maximum[lane] = max(maximum[lane], maximum[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // The scale the codes are computed against is the f16 the plane will hold, not the f32 quotient.
    let scale = unpack2x16float(pack2x16float(vec2(maximum[0] / QUANTIZE_BIAS, 0.0))).x;
    if (lane == 0u) {
        let index = row * groups_per_row + group;
        let half = index % 2u;
        let packed = pack2x16float(vec2(scale, 0.0));
        if (half == 0u) {
            scales[(params.scale_base / 4u) + index / 2u] =
                (scales[(params.scale_base / 4u) + index / 2u] & 0xFFFF0000u) | (packed & 0xFFFFu);
        } else {
            scales[(params.scale_base / 4u) + index / 2u] =
                (scales[(params.scale_base / 4u) + index / 2u] & 0xFFFFu) | (packed << 16u);
        }
    }

    // Four codes to a word: lane `l` packs values `4l .. 4l + 3`, so sixteen lanes cover the group.
    if (lane < QUANTIZE_CODE_WORDS) {
        var word = 0u;
        for (var index = 0u; index < QUANTIZE_CODES_PER_WORD; index = index + 1u) {
            let offset = lane * QUANTIZE_CODES_PER_WORD + index;
            word = word | (code_of(values[base + offset], scale) << (index * 8u));
        }
        let code_index = (row * groups_per_row + group) * QUANTIZE_GROUP;
        codes[(params.code_base / 4u) + code_index / 4u + lane] = word;
    }
}
