// Elementwise f32 addition: out[i] = a[i] + b[i].
//
// The audio tower adds its sinusoidal position embedding to the projected convolution output, which
// is one row of the embedding per time step over a `[steps][d_model]` matrix -- the same shape as
// the output, so the addition is elementwise. The decoder's residual adds want the same kernel, and
// nothing about this one is tower-specific.
//
// # Memory layout
//
//   a     count f32
//   b     count f32
//   out   count f32
//
// # Bind group 0
//
//   binding 0  uniform    AddParams { count, reserved_0, reserved_1, reserved_2 }
//   binding 1  read       a
//   binding 2  read       b
//   binding 3  read_write out
//
// # Workgroup
//
//   workgroup_size(256, 1, 1); dispatch (ceil(count / 256), 1, 1). One element per invocation, no
//   workgroup memory, no barriers.

const ADD_LANES: u32 = 256u;

struct AddParams {
    count: u32,
    reserved_0: u32,
    reserved_1: u32,
    reserved_2: u32,
};

@group(0) @binding(0) var<uniform> params: AddParams;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(ADD_LANES, 1, 1)
fn add_f32_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) {
        return;
    }
    out[gid.x] = a[gid.x] + b[gid.x];
}
