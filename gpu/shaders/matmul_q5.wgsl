// Fused q5 dequantize + matmul. The weights are decoded inside the kernel.
//
//   out[token][row] = dot(activation[token], weight[row])
//
// Structurally identical to `matmul_q4.wgsl`; only the code layout differs. No
// dense copy of the weights exists at any level: the packed two-plane layout is
// decoded a 16x16 slab at a time into workgroup memory, where it lives for one K
// slab.
//
// # Memory layout
//
//   activations   tokens * cols f32, row major, stride cols
//   packed        the quantized weight tensor: one buffer, two planes
//                   byte 0                     scale plane: group_count f16 scales
//                   align16(2 * group_count)   data plane:  group_count * 40 bytes
//                 groups are row major: the group owning column k of row r is
//                 `r * (cols / QW_GROUP_SIZE) + k / QW_GROUP_SIZE`.
//   out           tokens * rows f32, row major, stride rows
//
// # Codes
//
//   weight = (unsigned_code - QW_Q5_BIAS) * scale
//
// A group of 64 q5 codes occupies 40 bytes: 32 bytes of low nibbles (weight 2i
// in the low nibble of byte i, weight 2i+1 in the high nibble, exactly as in
// q4) followed by 8 bytes holding the fifth bit of weight j in bit `j % 8` of
// byte `j / 8`. The fifth bit is what separates q5 from q4: the nibble plane is
// the *low* four bits, so a weight's unsigned code is `low | (high << 4)`.
//
// # Bind group 0
//
//   binding 0  uniform    MatmulQ5Params { tokens, rows, cols, data_offset_bytes }
//   binding 1  read       activations
//   binding 2  read       packed
//   binding 3  read_write out
//
// # Workgroup
//
//   workgroup_size(16, 16, 1); dispatch (ceil(rows / 16), ceil(tokens / 16), 1).
//   local x indexes the output row within the tile, local y the token within the
//   tile. Workgroup storage: 2 * 16 * 16 f32 = 2 KiB.
//
// `cols` arrives in a uniform, so every invocation walks the same K slabs and
// reaches the barriers inside the loop together.

const MATMUL_TILE_K: u32 = 16u;
const MATMUL_TILE: u32 = 16u;

// Mirrors of gpu/shaders/quant_layout.wgsl. Copy the lines verbatim:
// tests/gpu/layout_drift.mjs fails when a mirror and the layout disagree.
const QW_GROUP_SIZE: u32 = 64u;
const QW_Q5_DATA_BYTES_PER_GROUP: u32 = 40u;
const QW_Q5_BIAS: i32 = 16;
const QW_NIBBLE_HIGH_SHIFT: u32 = 4u;
const QW_Q5_HIGH_BIT_PLANE_OFFSET_BYTES: u32 = 32u;

struct MatmulQ5Params {
    tokens: u32,
    rows: u32,
    cols: u32,
    data_offset_bytes: u32,
};

@group(0) @binding(0) var<uniform> params: MatmulQ5Params;
@group(0) @binding(1) var<storage, read> activations: array<f32>;
@group(0) @binding(2) var<storage, read> packed: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> activation_tile: array<f32, MATMUL_TILE_K * MATMUL_TILE>;
var<workgroup> weight_tile: array<f32, MATMUL_TILE_K * MATMUL_TILE>;

// Byte `byte_index` of the packed tensor; see matmul_q4.wgsl.
fn packed_u8(byte_index: u32) -> u32 {
    return (packed[byte_index >> 2u] >> ((byte_index & 3u) * 8u)) & 0xFFu;
}

// f16 bit pattern -> f32, without the optional `f16` language extension.
//   normal     exponent + 112 rebases the 5-bit f16 exponent onto the 8-bit f32
//              exponent; the 10-bit mantissa shifts up by 13.
//   subnormal  value = mantissa * 2^-24, exact in f32.
fn f32_from_f16_bits(bits: u32) -> f32 {
    let exponent = (bits >> 10u) & 0x1Fu;
    let mantissa = bits & 0x3FFu;
    var magnitude = 0.0;
    if (exponent == 0u) {
        magnitude = f32(mantissa) * 0.000000059604644775390625;
    } else if (exponent == 31u) {
        magnitude = bitcast<f32>(0x7F800000u | (mantissa << 13u));
    } else {
        magnitude = bitcast<f32>(((exponent + 112u) << 23u) | (mantissa << 13u));
    }
    if ((bits & 0x8000u) == 0u) {
        return magnitude;
    }
    return -magnitude;
}

// Decodes one q5 weight of `row` at `column` from the packed planes.
fn decode_q5(row: u32, column: u32, groups_per_row: u32) -> f32 {
    let group_index = row * groups_per_row + column / QW_GROUP_SIZE;
    let within_group = column % QW_GROUP_SIZE;

    let scale_low = packed_u8(group_index * 2u);
    let scale_high = packed_u8(group_index * 2u + 1u);
    let scale = f32_from_f16_bits(scale_low | (scale_high << 8u));

    let group_base = params.data_offset_bytes + group_index * QW_Q5_DATA_BYTES_PER_GROUP;
    let byte = packed_u8(group_base + within_group / 2u);
    var low = byte & 0x0Fu;
    if ((within_group & 1u) != 0u) {
        low = byte >> QW_NIBBLE_HIGH_SHIFT;
    }
    let high_byte = packed_u8(group_base + QW_Q5_HIGH_BIT_PLANE_OFFSET_BYTES + within_group / 8u);
    let high = (high_byte >> (within_group % 8u)) & 1u;
    return f32(i32(low | (high << 4u)) - QW_Q5_BIAS) * scale;
}

@compute @workgroup_size(16, 16, 1)
fn matmul_q5_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let token = gid.y;
    let row = gid.x;
    // The tile row this invocation stages is the tile's lane index (`lid.y`),
    // not the row of its own output (`lid.x`); see matmul_q4.wgsl.
    let staging_row = gid.x - lid.x + lid.y;
    let groups_per_row = params.cols / QW_GROUP_SIZE;

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
                weight = decode_q5(staging_row, slab_column, groups_per_row);
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
