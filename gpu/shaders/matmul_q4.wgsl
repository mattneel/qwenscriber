// Fused q4 dequantize + matmul. The weights are decoded inside the kernel.
//
//   out[token][row] = dot(activation[token], weight[row])
//
// No dense copy of the weights exists at any level: the kernel reads the packed
// two-plane layout (`src/core/quant.zig`, mirrored in `quant_layout.wgsl`) and
// decodes one 16x16 slab of weights into workgroup memory, where it lives for
// the duration of one K slab. A dense f32 copy would be eight times the packed
// bytes and would defeat the point of the format; staging a slab instead of a
// whole matrix is what makes the fusion possible without one.
//
// Tiling, workgroup shape, and per-output accumulation order are identical to
// `matmul_f32.wgsl`, so a difference in the harness comparison is a difference
// in the quantization path and nothing else.
//
// # Memory layout
//
//   activations   tokens * cols f32, row major, stride cols
//   packed        the quantized weight tensor: one buffer, two planes
//                   byte 0                     scale plane: group_count f16 scales
//                   align16(2 * group_count)   data plane:  group_count * 32 bytes
//                 groups are row major, so the group owning column k of row r is
//                 `r * (cols / QW_GROUP_SIZE) + k / QW_GROUP_SIZE`, and both
//                 planes are indexed by that group index.
//   out           tokens * rows f32, row major, stride rows
//
// `cols` is a multiple of QW_GROUP_SIZE in every shape this runtime serves; the
// kernel guards the tail anyway, so a ragged shape is well defined rather than
// out of bounds.
//
// # Codes
//
//   weight = (unsigned_code - QW_Q4_BIAS) * scale
//   weight 2i is the low nibble of byte i of the group's 32 data bytes, weight
//   2i+1 the high nibble. One group is 64 weights, so the scale is one f16 per
//   64 weights and the codes are 32 bytes.
//
// # Bind group 0
//
//   binding 0  uniform    MatmulQ4Params { tokens, rows, cols, data_offset_bytes }
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
const QW_Q4_DATA_BYTES_PER_GROUP: u32 = 32u;
const QW_Q4_BIAS: i32 = 8;
const QW_NIBBLE_HIGH_SHIFT: u32 = 4u;

struct MatmulQ4Params {
    tokens: u32,
    rows: u32,
    cols: u32,
    data_offset_bytes: u32,
};

@group(0) @binding(0) var<uniform> params: MatmulQ4Params;
@group(0) @binding(1) var<storage, read> activations: array<f32>;
@group(0) @binding(2) var<storage, read> packed: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> activation_tile: array<f32, MATMUL_TILE_K * MATMUL_TILE>;
var<workgroup> weight_tile: array<f32, MATMUL_TILE_K * MATMUL_TILE>;

// Byte `byte_index` of the packed tensor. WGSL has no byte granular storage
// type, so the tensor is bound as u32 words and bytes are shifted out of them.
fn packed_u8(byte_index: u32) -> u32 {
    return (packed[byte_index >> 2u] >> ((byte_index & 3u) * 8u)) & 0xFFu;
}

// f16 bit pattern -> f32, without the optional `f16` language extension.
//
// `unpack2x16float` is gated behind `enable f16` on some implementations (and
// behind nothing on others), and the scale plane has to be readable on a plain
// adapter, so the conversion is integer bit moves:
//
//   normal     exponent + 112 rebases the 5-bit f16 exponent onto the 8-bit f32
//              exponent; the 10-bit mantissa shifts up by 13.
//   subnormal  value = mantissa * 2^-24, which is exact in f32.
//   infinity / NaN keeps a NaN a NaN rather than turning it into infinity.
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

// Decodes one q4 weight of `row` at `column` from the packed planes.
fn decode_q4(row: u32, column: u32, groups_per_row: u32) -> f32 {
    let group_index = row * groups_per_row + column / QW_GROUP_SIZE;
    let within_group = column % QW_GROUP_SIZE;

    let scale_low = packed_u8(group_index * 2u);
    let scale_high = packed_u8(group_index * 2u + 1u);
    let scale = f32_from_f16_bits(scale_low | (scale_high << 8u));

    let byte = packed_u8(params.data_offset_bytes + group_index * QW_Q4_DATA_BYTES_PER_GROUP +
        within_group / 2u);
    var nibble = byte & 0x0Fu;
    if ((within_group & 1u) != 0u) {
        nibble = byte >> QW_NIBBLE_HIGH_SHIFT;
    }
    return f32(i32(nibble) - QW_Q4_BIAS) * scale;
}

@compute @workgroup_size(16, 16, 1)
fn matmul_q4_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let token = gid.y;
    let row = gid.x;
    // The tile row this invocation stages is the tile's lane index (`lid.y`),
    // not the row of its own output (`lid.x`). Staging the (lane = lid.y,
    // k = lid.x) element keeps each warp's global reads contiguous in k.
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
                weight = decode_q4(staging_row, slab_column, groups_per_row);
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
