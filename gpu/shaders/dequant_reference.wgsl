// Dequantization instrument: writes the packed weights of a tile back out as
// dense f32.
//
// This kernel exists only so `tests/gpu/harness.mjs` can compare a GPU
// dequantization against the CPU one element by element, which is the check that
// pins the packing (nibble order, the q5 high bit plane, the f16 scale) rather
// than only the matmul that consumes it. The runtime never calls it: materializing
// dense weights would be eight times the packed bytes for q4 and exactly the copy
// the fused kernels avoid.
//
// # Memory layout
//
//   packed   the quantized weight tensor: one buffer, two planes
//              byte 0                     scale plane: group_count f16 scales
//              align16(2 * group_count)   data plane:  group_count * data bytes
//            with `group_count = rows * (cols / QW_GROUP_SIZE)`, groups row major
//   out      rows * cols f32, row major   (out[row][k])
//
// # Codes
//
//   q4: (nibble - QW_Q4_BIAS) * scale; weight 2i is the low nibble of byte i.
//   q5: ((low | high << 4) - QW_Q5_BIAS) * scale; the high bit of weight j is
//       bit `j % 8` of byte `QW_Q5_HIGH_BIT_PLANE_OFFSET_BYTES + j / 8`.
//   q8: (byte - QW_Q8_BIAS) * scale.
//
// # Bind group 0
//
//   binding 0  uniform    DequantParams { rows, cols, data_offset_bytes, format }
//                         format is a `Format` id from src/core/dtype.zig
//                         (q4 = 16, q5 = 17, q8 = 18)
//   binding 1  read       packed
//   binding 2  read_write out
//
// # Workgroup
//
//   workgroup_size(64, 1, 1); dispatch (ceil(cols / 64), rows, 1).
//   gid.x is the column, gid.y the row. One weight per invocation, no workgroup
//   memory, no barriers.

const DEQUANT_LANES: u32 = 64u;

// Mirrors of gpu/shaders/quant_layout.wgsl. Copy the lines verbatim:
// tests/gpu/layout_drift.mjs fails when a mirror and the layout disagree.
const QW_GROUP_SIZE: u32 = 64u;
const QW_Q4_DATA_BYTES_PER_GROUP: u32 = 32u;
const QW_Q5_DATA_BYTES_PER_GROUP: u32 = 40u;
const QW_Q8_DATA_BYTES_PER_GROUP: u32 = 64u;
const QW_Q4_BIAS: i32 = 8;
const QW_Q5_BIAS: i32 = 16;
const QW_Q8_BIAS: i32 = 128;
const QW_NIBBLE_HIGH_SHIFT: u32 = 4u;
const QW_Q5_HIGH_BIT_PLANE_OFFSET_BYTES: u32 = 32u;
const QW_FORMAT_Q4: u32 = 16u;
const QW_FORMAT_Q5: u32 = 17u;
const QW_FORMAT_Q8: u32 = 18u;

struct DequantParams {
    rows: u32,
    cols: u32,
    data_offset_bytes: u32,
    format: u32,
};

@group(0) @binding(0) var<uniform> params: DequantParams;
@group(0) @binding(1) var<storage, read> packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

// Byte `byte_index` of the packed tensor. WGSL has no byte granular storage
// type, so the tensor is bound as u32 words and bytes are shifted out of them.
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

// The group index of (row, column), shared by all three formats.
fn group_index_of(row: u32, column: u32) -> u32 {
    let groups_per_row = params.cols / QW_GROUP_SIZE;
    return row * groups_per_row + column / QW_GROUP_SIZE;
}

fn scale_of(row: u32, column: u32) -> f32 {
    let group_index = group_index_of(row, column);
    let low = packed_u8(group_index * 2u);
    let high = packed_u8(group_index * 2u + 1u);
    return f32_from_f16_bits(low | (high << 8u));
}

fn decode_q4(row: u32, column: u32) -> f32 {
    let group_index = group_index_of(row, column);
    let within_group = column % QW_GROUP_SIZE;
    let byte = packed_u8(params.data_offset_bytes + group_index * QW_Q4_DATA_BYTES_PER_GROUP +
        within_group / 2u);
    var code = byte & 0x0Fu;
    if ((within_group & 1u) != 0u) {
        code = byte >> QW_NIBBLE_HIGH_SHIFT;
    }
    return f32(i32(code) - QW_Q4_BIAS) * scale_of(row, column);
}

fn decode_q5(row: u32, column: u32) -> f32 {
    let group_index = group_index_of(row, column);
    let within_group = column % QW_GROUP_SIZE;
    let group_base = params.data_offset_bytes + group_index * QW_Q5_DATA_BYTES_PER_GROUP;
    let byte = packed_u8(group_base + within_group / 2u);
    var low = byte & 0x0Fu;
    if ((within_group & 1u) != 0u) {
        low = byte >> QW_NIBBLE_HIGH_SHIFT;
    }
    let high_byte = packed_u8(group_base + QW_Q5_HIGH_BIT_PLANE_OFFSET_BYTES + within_group / 8u);
    let high = (high_byte >> (within_group % 8u)) & 1u;
    return f32(i32(low | (high << 4u)) - QW_Q5_BIAS) * scale_of(row, column);
}

fn decode_q8(row: u32, column: u32) -> f32 {
    let group_index = group_index_of(row, column);
    let byte = packed_u8(params.data_offset_bytes + group_index * QW_Q8_DATA_BYTES_PER_GROUP +
        column % QW_GROUP_SIZE);
    return f32(i32(byte) - QW_Q8_BIAS) * scale_of(row, column);
}

@compute @workgroup_size(DEQUANT_LANES, 1, 1)
fn dequant_reference_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.y >= params.rows) {
        return;
    }
    if (gid.x >= params.cols) {
        return;
    }

    // The format decides the code width, so an id this kernel does not know
    // cannot be decoded. The check reads only the uniform block, so it is the
    // same for every invocation.
    if (params.format != QW_FORMAT_Q4) {
        if (params.format != QW_FORMAT_Q5) {
            if (params.format != QW_FORMAT_Q8) {
                return;
            }
        }
    }

    var value = 0.0;
    switch params.format {
        case QW_FORMAT_Q4: {
            value = decode_q4(gid.y, gid.x);
        }
        case QW_FORMAT_Q5: {
            value = decode_q5(gid.y, gid.x);
        }
        case QW_FORMAT_Q8: {
            value = decode_q8(gid.y, gid.x);
        }
        default: {
            // Unreachable: the guard above returns for any other format id.
            value = 0.0;
        }
    }
    out[gid.y * params.cols + gid.x] = value;
}
