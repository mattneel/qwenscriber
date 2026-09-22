// Row-broadcast bias add: values[row][column] += bias[column].
//
// Every projection in the runtime is `x * weight + bias`, and the matmul kernels compute the product
// alone -- their accumulation is the thing the conformance harness compares, so a bias term folded
// into them would be one more difference to explain. The core adds it separately for the same reason
// (`kernels.addBias`), and this is that pass on the GPU: in place, because the values is a scratch
// buffer whose product is never needed once the bias is in.
//
// # Memory layout
//
//   values  rows * cols f32, row major, stride cols
//   bias    cols f32
//
// # Bind group 0
//
//   binding 0  uniform    AddBiasParams { rows, cols, reserved_0, reserved_1 }  (16 bytes)
//   binding 1  read       bias
//   binding 2  read_write values
//
// # Workgroup
//
//   workgroup_size(256, 1, 1); dispatch (ceil(rows * cols / 256), 1, 1). One element per invocation,
//   no workgroup memory, no barriers.

const ADD_BIAS_LANES: u32 = 256u;

struct AddBiasParams {
    rows: u32,
    cols: u32,
    reserved_0: u32,
    reserved_1: u32,
};

@group(0) @binding(0) var<uniform> params: AddBiasParams;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;

@compute @workgroup_size(ADD_BIAS_LANES, 1, 1)
fn add_bias_f32_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let count = params.rows * params.cols;
    if (gid.x >= count) {
        return;
    }
    // `gid.x` walks the matrix row major, so the column is the remainder of the division.
    let column = gid.x % params.cols;
    values[gid.x] = values[gid.x] + bias[column];
}
