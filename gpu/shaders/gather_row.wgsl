// Reads one row of a quantized matrix as f32.
//
// The decoder's first step is an embedding lookup: a token id names a row of
// `decoder.embed_tokens.weight`, and that row becomes the hidden state the prompt is prefilled from.
// The matrix is quantized like any other large weight, so the lookup is a dequantization of one row
// rather than a copy, and doing it in JavaScript would mean a second implementation of the plane
// layout. The decoders below are `dequant_reference.wgsl`'s, which the conformance harness checks
// against the Zig packing; the constants they use are the mirrors that
// `tests/gpu/layout_drift.mjs` holds to `src/core/quant.zig`.
//
// # Memory layout
//
//   packed  the tensor's payload, scales first then codes, bound as u32 words
//   out     cols f32: the decoded row
//
// # Bind group 0
//
//   binding 0  uniform    GatherParams { row, cols, data_offset_bytes, format }  (16 bytes)
//   binding 1  read       packed
//   binding 2  read_write out
//
// # Workgroup
//
//   workgroup_size(64, 1, 1): dispatch (ceil(cols / 64), 1, 1). One column per invocation, and the
//   lane count matches the group size so a whole group is decoded by one workgroup.
//
// The format ids are `dtype.Format` values: q4 is 16, q5 17, q8 18. An element format has no planes
// to read and is refused, which is not a shape this runtime serves: the browser can only load a
// quantized build, because the f16 conversion's weights do not fit a wasm32 instance.

const QW_GROUP_SIZE: u32 = 64u;
const QW_Q4_DATA_BYTES_PER_GROUP: u32 = 32u;
const QW_Q5_DATA_BYTES_PER_GROUP: u32 = 40u;
const QW_Q8_DATA_BYTES_PER_GROUP: u32 = 64u;
const QW_Q5_HIGH_BIT_PLANE_OFFSET_BYTES: u32 = 32u;
const QW_NIBBLE_HIGH_SHIFT: u32 = 4u;
const QW_Q4_BIAS: i32 = 8;
const QW_Q5_BIAS: i32 = 16;
const QW_Q8_BIAS: i32 = 128;
const QW_FORMAT_Q4: u32 = 16u;
const QW_FORMAT_Q5: u32 = 17u;
const QW_FORMAT_Q8: u32 = 18u;

const GATHER_LANES: u32 = 64u;

struct GatherParams {
    row: u32,
    cols: u32,
    data_offset_bytes: u32,
    format: u32,
};

@group(0) @binding(0) var<uniform> params: GatherParams;
@group(0) @binding(1) var<storage, read> packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

/// Byte `byte_index` of the packed tensor: WGSL has no byte granular storage type, so bytes are
/// shifted out of the words the tensor is bound as.
fn packed_u8(byte_index: u32) -> u32 {
    return (packed[byte_index >> 2u] >> ((byte_index & 3u) * 8u)) & 0xFFu;
}

/// f16 bit pattern to f32, without the optional `f16` language extension.
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

fn group_index_of(column: u32) -> u32 {
    return params.row * (params.cols / QW_GROUP_SIZE) + column / QW_GROUP_SIZE;
}

fn scale_of(column: u32) -> f32 {
    let group_index = group_index_of(column);
    let low = packed_u8(group_index * 2u);
    let high = packed_u8(group_index * 2u + 1u);
    return f32_from_f16_bits(low | (high << 8u));
}

fn decode_q4(column: u32) -> f32 {
    let group_index = group_index_of(column);
    let within_group = column % QW_GROUP_SIZE;
    let byte_index = params.data_offset_bytes +
        group_index * QW_Q4_DATA_BYTES_PER_GROUP + within_group / 2u;
    var code = packed_u8(byte_index) & 0x0Fu;
    if ((within_group & 1u) != 0u) {
        code = packed_u8(byte_index) >> QW_NIBBLE_HIGH_SHIFT;
    }
    return f32(i32(code) - QW_Q4_BIAS) * scale_of(column);
}

fn decode_q5(column: u32) -> f32 {
    let group_index = group_index_of(column);
    let within_group = column % QW_GROUP_SIZE;
    let group_base = params.data_offset_bytes + group_index * QW_Q5_DATA_BYTES_PER_GROUP;
    let byte = packed_u8(group_base + within_group / 2u);
    var low = byte & 0x0Fu;
    if ((within_group & 1u) != 0u) {
        low = byte >> QW_NIBBLE_HIGH_SHIFT;
    }
    let high_byte = packed_u8(group_base + QW_Q5_HIGH_BIT_PLANE_OFFSET_BYTES + within_group / 8u);
    let high = (high_byte >> (within_group % 8u)) & 1u;
    return f32(i32(low | (high << 4u)) - QW_Q5_BIAS) * scale_of(column);
}

fn decode_q8(column: u32) -> f32 {
    let group_index = group_index_of(column);
    let byte_index = params.data_offset_bytes +
        group_index * QW_Q8_DATA_BYTES_PER_GROUP + column % QW_GROUP_SIZE;
    return f32(i32(packed_u8(byte_index)) - QW_Q8_BIAS) * scale_of(column);
}

@compute @workgroup_size(GATHER_LANES, 1, 1)
fn gather_row_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let column = gid.x;
    if (column >= params.cols || params.cols % QW_GROUP_SIZE != 0u) {
        return;
    }
    var value = 0.0;
    if (params.format == QW_FORMAT_Q4) {
        value = decode_q4(column);
    } else if (params.format == QW_FORMAT_Q5) {
        value = decode_q5(column);
    } else if (params.format == QW_FORMAT_Q8) {
        value = decode_q8(column);
    }
    out[column] = value;
}
