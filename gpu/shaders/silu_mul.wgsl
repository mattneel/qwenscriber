// Fused gated MLP activation: out[i] = silu(gate[i]) * up[i].
//
// `silu(x) = x / (1 + exp(-x))`, matching `silu` in `src/core/math.zig`. The
// decoder MLP computes this on the two halves of one projection, which is why
// the two operands are separate buffers here rather than one interleaved one:
// the caller already has them apart.
//
// # Memory layout
//
//   gate   count f32
//   up     count f32
//   out    count f32
//
// # Bind group 0
//
//   binding 0  uniform    SiluMulParams { count, reserved_0, reserved_1, reserved_2 }
//   binding 1  read       gate
//   binding 2  read       up
//   binding 3  read_write out
//
// # Workgroup
//
//   workgroup_size(256, 1, 1); dispatch (ceil(count / 256), 1, 1). One element
//   per invocation, no workgroup memory, no barriers.

const SILU_MUL_LANES: u32 = 256u;

struct SiluMulParams {
    count: u32,
    reserved_0: u32,
    reserved_1: u32,
    reserved_2: u32,
};

@group(0) @binding(0) var<uniform> params: SiluMulParams;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(SILU_MUL_LANES, 1, 1)
fn silu_mul_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) {
        return;
    }
    let gate_value = gate[gid.x];
    let up_value = up[gid.x];
    out[gid.x] = (gate_value / (1.0 + exp(-gate_value))) * up_value;
}
