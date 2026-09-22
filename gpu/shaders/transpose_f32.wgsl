// Dense f32 transpose: out[column][row] = input[row][column].
//
// The audio tower needs it between its convolutions and the downsample projection. The convolution
// stack emits `[channels][bins][steps]`, and the projection wants `[steps][channels * bins]`: since
// `channel * bins + bin` is the row-major flatten of the first two axes, that is a plain 2-D
// transpose of `[channels * bins][steps]`. Doing it in JavaScript would mean reading a few hundred
// kilobytes back from the GPU and uploading them again, once per chunk.
//
// # Memory layout
//
//   input     rows * cols f32, row major, stride cols
//   out       cols * rows f32, row major, stride rows
//
// # Bind group 0
//
//   binding 0  uniform    TransposeParams { rows, cols, reserved_0, reserved_1 }  (16 bytes)
//   binding 1  read       input
//   binding 2  read_write out
//
// # Workgroup
//
//   workgroup_size(16, 16, 1): one 16x16 tile per workgroup, dispatch
//   (ceil(cols / 16), ceil(rows / 16), 1). Workgroup storage: 16 * 17 f32 = 1088 bytes, the extra
//   column keeping the tile's rows on separate shared-memory banks so the read side of the transpose
//   does not serialize.

const TRANSPOSE_TILE: u32 = 16u;

struct TransposeParams {
    rows: u32,
    cols: u32,
    reserved_0: u32,
    reserved_1: u32,
};

@group(0) @binding(0) var<uniform> params: TransposeParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> tile: array<f32, TRANSPOSE_TILE * (TRANSPOSE_TILE + 1u)>;

@compute @workgroup_size(16, 16, 1)
fn transpose_f32_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let column = gid.x;
    // Read side: contiguous along the row, which is the input's fast axis.
    if (row < params.rows) {
        if (column < params.cols) {
            tile[lid.y * (TRANSPOSE_TILE + 1u) + lid.x] = input[row * params.cols + column];
        }
    }
    workgroupBarrier();
    // Write side: the tile's axes swapped, so the stores are contiguous along the output's fast axis.
    let out_row = gid.x - lid.x + lid.y;
    let out_column = gid.y - lid.y + lid.x;
    if (out_row < params.cols) {
        if (out_column < params.rows) {
            out[out_row * params.rows + out_column] =
                tile[lid.x * (TRANSPOSE_TILE + 1u) + lid.y];
        }
    }
}
