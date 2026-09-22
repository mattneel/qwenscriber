// Three-by-three, stride two, padding one convolution with a bias add and GELU: the audio tower's
// only convolution.
//
// Mirrors `conv3x3Stride2Gelu` in `src/core/qwen3_asr/kernels.zig`, including the accumulation
// order: the output element starts at its channel's bias, each input channel adds its nine taps in
// row-major kernel order, and GELU is applied once at the end rather than per channel. Weights are
// `[out_channels][in_channels][3][3]` as f16 bits, input is `[in_channels][in_height][in_width]`,
// output is `[out_channels][out_height][out_width]` -- the layouts the checkpoint already has, so
// nothing is repacked.
//
// The kernel decodes the nine weights of each `(out, in)` pair into workgroup memory once per
// workgroup and sweeps output positions from there, which is the point the core's `convolvePlane`
// makes in the same words: at 480 channels, re-reading and re-decoding the weights per output
// position would dominate the arithmetic. An invocation per position without that staging would do
// nine f16 decodes for every nine multiply-adds.
//
// # Memory layout
//
//   input     in_channels * in_height * in_width f32, channel planes contiguous
//   weights   out_channels * in_channels * 9 f16 bits, little endian, two per u32
//   bias      out_channels f32
//   out       out_channels * out_height * out_width f32
//   out_height = (in_height - 1) / 2 + 1, out_width = (in_width - 1) / 2 + 1   (integer division)
//
// f16 weights arrive as u16 pairs packed into u32 storage elements the way the container stores
// them, little endian, so weight `i` is the low half of element `i / 2` when `i` is even. WGSL has
// no f16 storage format without the `shader-f16` feature, and the packing is what the runtime
// reads anyway, so the decode is written out here. It matches `half_float.fromF16` in
// `src/core/half_float.zig`, a lossless f16-to-f32 widening.
//
// # Bind group 0
//
//   binding 0  uniform    ConvParams { out_channels, in_channels, in_height, in_width }
//   binding 1  read       input
//   binding 2  read       weights
//   binding 3  read       bias
//   binding 4  read_write out
//
// # Workgroup
//
//   workgroup_size(256, 1, 1): one invocation per output element, dispatch
//   (ceil(out_plane / 256), out_channels, 1), where out_plane = out_height * out_width. The channel
//   is `workgroup_id.y` and the position within its plane is `global_invocation_id.x`. Every
//   invocation of a workgroup stages part of the same weight block and then reads all of it, so the
//   barriers sit outside every conditional and the guards depend only on the uniform block.
//   Workgroup storage: CONV_IN_CHANNELS_MAX * 9 f32 = 18432 bytes, leaving room inside the 32768
//   byte limit. A model whose input channel count exceeds the bound is refused rather than
//   truncated: returning leaves the output untouched, and the caller reads a zero it can notice.

const CONV_LANES: u32 = 256u;
const CONV_KERNEL: u32 = 3u;
const CONV_KERNEL_VALUES: u32 = CONV_KERNEL * CONV_KERNEL;
const CONV_IN_CHANNELS_MAX: u32 = 512u;
const CONV_WEIGHT_VALUES_MAX: u32 = CONV_IN_CHANNELS_MAX * CONV_KERNEL_VALUES;
const CONV_WEIGHT_STEPS_MAX: u32 = (CONV_WEIGHT_VALUES_MAX + CONV_LANES - 1u) / CONV_LANES;

struct ConvParams {
    out_channels: u32,
    in_channels: u32,
    in_height: u32,
    in_width: u32,
};

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> kernel_weights: array<f32, CONV_WEIGHT_VALUES_MAX>;

/// f16 bits (in the low half of a u32) to f32, as `half_float.fromF16` widens them.
///
/// Subnormals are `mantissa * 2^-24` and normals are `(1024 + mantissa) * 2^(exponent - 25)`, both
/// scaled by a power of two, so no rounding is involved on either path.
fn f16_bits_to_f32(bits: u32) -> f32 {
    let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
    let exponent = (bits >> 10u) & 0x1Fu;
    let mantissa = bits & 0x3FFu;
    if (exponent == 0u) {
        return sign * f32(mantissa) * exp2(-24.0);
    }
    if (exponent == 31u) {
        // WGSL rejects an f32 const-expression that evaluates to inf or NaN, so the two special
        // values are assembled from the value's own bits: mixing the runtime sign bit in keeps the
        // bitcast out of constant folding, and avoids the `1.0 / 0.0` the compiler also refuses. An
        // f16 infinity in a weight plane means the checkpoint is corrupt, but widening it
        // faithfully costs nothing and inventing a finite substitute would hide that.
        let magnitude = select(0x7F800000u, 0x7FC00000u, mantissa != 0u);
        return bitcast<f32>(magnitude | ((bits & 0x8000u) << 16u));
    }
    return sign * f32(1024u + mantissa) * exp2(f32(exponent) - 25.0);
}

/// GELU, as `gelu.wgsl` computes it and `src/core/math.zig` defines it.
///
/// The duplicate is deliberate: WGSL has no include mechanism, and the repository's convention for
/// shared shader code is a copy with a pointer back (see `matmul_q5.wgsl`'s `packed_u8`). A change
/// to the polynomial belongs in both files and in the harness that compares them.
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

fn gelu(value: f32) -> f32 {
    return 0.5 * value * (1.0 + erf_approximation(value * 0.70710678));
}

@compute @workgroup_size(CONV_LANES, 1, 1)
fn conv3x3_stride2_gelu_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    // Uniform guards: every invocation of the workgroup takes the same branch, so the staging
    // barrier below is in uniform control flow.
    if (params.in_channels == 0u || params.in_channels > CONV_IN_CHANNELS_MAX ||
        params.in_height == 0u || params.in_width == 0u || params.out_channels == 0u) {
        return;
    }
    let out_height = (params.in_height - 1u) / 2u + 1u;
    let out_width = (params.in_width - 1u) / 2u + 1u;
    let out_plane = out_height * out_width;
    let in_plane = params.in_height * params.in_width;
    let channel = wid.y;
    let weight_count = params.in_channels * CONV_KERNEL_VALUES;
    let weight_base = channel * weight_count;

    // Stage this channel's weights, decoded once for the whole plane. Weight `index` is the low
    // half of storage element `index / 2` when `index` is even, the high half otherwise.
    for (var step = 0u; step < CONV_WEIGHT_STEPS_MAX; step = step + 1u) {
        let index = lane + step * CONV_LANES;
        if (index < weight_count) {
            // The element and half come from the channel's global weight index, not from its base:
            // nine weights per input channel make the base odd whenever the channel count is, and a
            // base-relative index would then read the neighbouring channel's halves.
            let weight_index = weight_base + index;
            let packed = weights[weight_index / 2u];
            let half_shift = (weight_index & 1u) * 16u;
            kernel_weights[index] = f16_bits_to_f32((packed >> half_shift) & 0xFFFFu);
        }
    }
    workgroupBarrier();

    if (channel >= params.out_channels || gid.x >= out_plane) {
        return;
    }
    let out_row = gid.x / out_width;
    let out_column = gid.x % out_width;
    var accumulator = bias[channel];
    for (var in_channel = 0u; in_channel < params.in_channels; in_channel = in_channel + 1u) {
        let channel_input = in_channel * in_plane;
        let channel_weights = in_channel * CONV_KERNEL_VALUES;
        for (var kernel_row = 0u; kernel_row < CONV_KERNEL; kernel_row = kernel_row + 1u) {
            let in_row = i32(out_row * 2u) + i32(kernel_row) - 1;
            if (in_row < 0 || in_row >= i32(params.in_height)) {
                continue;
            }
            for (var kernel_column = 0u; kernel_column < CONV_KERNEL; kernel_column = kernel_column + 1u) {
                let in_column = i32(out_column * 2u) + i32(kernel_column) - 1;
                if (in_column < 0 || in_column >= i32(params.in_width)) {
                    continue;
                }
                let sample = input[channel_input + u32(in_row) * params.in_width + u32(in_column)];
                let weight = kernel_weights[channel_weights + kernel_row * CONV_KERNEL + kernel_column];
                accumulator = accumulator + weight * sample;
            }
        }
    }
    out[channel * out_plane + gid.x] = gelu(accumulator);
}
