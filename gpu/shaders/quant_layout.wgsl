// Quantization layout constants, mirrored from `src/core/quant.zig`.
//
// WGSL has no `#include`, so this file is compiled into no pipeline: it is the
// written-down copy of the numbers. `tests/gpu/layout_drift.mjs` extracts the
// constants from `src/core/quant.zig`, from this file, and from every kernel in
// this directory, and fails when any of the three disagree. The layout drifts
// because nothing checks it; this file exists to be checked.
//
// A kernel that needs one of these declares its own `const QW_*` with the exact
// name used here. Copy the line; never retype the number.
//
// # Tensor layout
//
// A `rows x cols` weight matrix, quantized along its rows, `cols % GROUP_SIZE ==
// 0`. Each tensor is two planes inside one buffer:
//
//     byte 0                        scale plane: group_count f16 scales
//     ...                           group_count = rows * (cols / QW_GROUP_SIZE)
//     align_up(2 * group_count, 16) data plane: group_count * data_bytes/group
//
// Groups are ordered row major, so the group that owns column k of row r is
// `r * (cols / QW_GROUP_SIZE) + k / QW_GROUP_SIZE`, and the scale plane is
// indexed by the same group index. The padding before the data plane exists so
// both planes are 16-byte aligned relative to the buffer: a browser uploads the
// whole payload to one `GPUBuffer` and a kernel can read the data plane with
// 16-byte vector loads.
//
// # Codes
//
//   value = (unsigned_code - bias) * scale
//
//   q4: unsigned 0..15   (4 data bits per weight)   bias 8    range [-8, 7]
//   q5: unsigned 0..31   (5 data bits per weight)   bias 16   range [-16, 15]
//   q8: unsigned 0..127  (8 data bits per weight)   bias 128  range [-128, 127]
//
// The per-group scale is `max_abs / bias` stored as f16, so the most negative
// code reaches `-max_abs` and the most positive `(bias - 1) / bias * max_abs`.
//
// # Packing inside one group of QW_GROUP_SIZE codes
//
//   q4: weight 2i is the low nibble of byte i, weight 2i+1 the high nibble
//       (little endian within the byte).
//   q5: the first QW_Q4_DATA_BYTES_PER_GROUP bytes are the low four bits with
//       the q4 packing; the last QW_Q5_HIGH_BIT_BYTES_PER_GROUP bytes hold the
//       fifth bit of weight j in bit `j % 8` of byte `j / 8`.
//   q8: weight j is byte j.
//
// `src/core/quant.zig` is the single source of truth for all of the above; see
// its module comment for why codes are stored unsigned around a bias.

const QW_GROUP_SIZE: u32 = 64u;
const QW_Q4_SCALE_BYTES_PER_GROUP: u32 = 2u;
const QW_Q4_DATA_BYTES_PER_GROUP: u32 = 32u;
const QW_Q5_SCALE_BYTES_PER_GROUP: u32 = 2u;
const QW_Q5_DATA_BYTES_PER_GROUP: u32 = 40u;
const QW_Q8_SCALE_BYTES_PER_GROUP: u32 = 2u;
const QW_Q8_DATA_BYTES_PER_GROUP: u32 = 64u;

// Alignment of a quantized tensor payload inside a container, and therefore the
// alignment of the data plane relative to the tensor base.
const QW_TENSOR_ALIGNMENT_BYTES: u32 = 16u;

const QW_Q4_BIAS: i32 = 8;
const QW_Q5_BIAS: i32 = 16;
const QW_Q8_BIAS: i32 = 128;

// Nibble positions inside a packed byte: weight 2i is the low nibble, weight
// 2i+1 the high one.
const QW_NIBBLE_LOW_SHIFT: u32 = 0u;
const QW_NIBBLE_HIGH_SHIFT: u32 = 4u;

// q5 stores its fifth bit in a separate plane inside the group's data: byte
// offset of that plane, and its size. The offset equals the q4 data size
// because the low four bits of q5 use the q4 packing.
const QW_Q5_HIGH_BIT_PLANE_OFFSET_BYTES: u32 = 32u;
const QW_Q5_HIGH_BIT_BYTES_PER_GROUP: u32 = 8u;

// Numeric format ids from `src/core/dtype.zig` (`Format`), passed to the
// dequantization instrument in `dequant_reference.wgsl` through a uniform.
const QW_FORMAT_Q4: u32 = 16u;
const QW_FORMAT_Q5: u32 = 17u;
const QW_FORMAT_Q8: u32 = 18u;
