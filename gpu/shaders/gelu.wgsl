// GELU, the audio tower's activation.
//
//   out[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2)))
//
// Mirrors `gelu` in `src/core/math.zig`, which matches `torch.nn.functional.gelu` defaults. The
// error function is not a WGSL builtin, so the same Numerical Recipes `erfc` fit the core uses is
// written out here, coefficient for coefficient: a different approximation would be a different
// activation, and the harness compares these values against the core's for exactly that reason.
//
// # Memory layout
//
//   x     count f32
//   out   count f32
//
// # Bind group 0
//
//   binding 0  uniform    GeluParams { count, reserved_0, reserved_1, reserved_2 }
//   binding 1  read       x
//   binding 2  read_write out
//
// # Workgroup
//
//   workgroup_size(256, 1, 1); dispatch (ceil(count / 256), 1, 1). One element per
//   invocation, no workgroup memory, no barriers.

const GELU_LANES: u32 = 256u;

struct GeluParams {
    count: u32,
    reserved_0: u32,
    reserved_1: u32,
    reserved_2: u32,
};

@group(0) @binding(0) var<uniform> params: GeluParams;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

/// `erf` by the published `erfc` rational fit, |error| < 1.2e-7 over the whole real line.
///
/// The core comments why this fit and not a series: GELU feeds on `x / sqrt(2)` for arguments that
/// routinely exceed 3, where a Taylor series loses its significant digits.
fn erf_approximation(value: f32) -> f32 {
    let sign = select(-1.0, 1.0, value >= 0.0);
    let magnitude = abs(value);
    let t = 1.0 / (1.0 + 0.5 * magnitude);
    let tau = t * exp(-magnitude * magnitude - 1.26551223 +
        t * (1.00002368 +
            t * (0.37409196 +
                t * (0.09678418 +
                    t * (-0.18628806 +
                        t * (0.27886807 +
                            t * (-1.13520398 +
                                t * (1.48851587 +
                                    t * (-0.82215223 + t * 0.17087277)))))))));
    return sign * (1.0 - tau);
}

@compute @workgroup_size(GELU_LANES, 1, 1)
fn gelu_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) {
        return;
    }
    let value = x[gid.x];
    out[gid.x] = 0.5 * value * (1.0 + erf_approximation(value * 0.70710678));
}
