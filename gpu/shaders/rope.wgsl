// Rotary position embedding, non-interleaved ("rotate_half") layout.
//
// For pair i in [0, head_dim / 2), with `half = head_dim / 2` and
// `inv_freq[i] = theta ^ (-2i / head_dim)`:
//
//   angle = position * inv_freq[i]
//   out[..., i]        = x[..., i]        * cos(angle) - x[..., i + half] * sin(angle)
//   out[..., i + half] = x[..., i + half] * cos(angle) + x[..., i]        * sin(angle)
//
// The pair is (i, i + head_dim / 2), not (2i, 2i + 1): the frequency vector is
// duplicated along the head (`emb = cat(freqs, freqs)`) and applied to
// `rotate_half(x)`. This is the layout the HF/Qwen code path uses when
// `rope_scaling` is null, and it is why the frequencies only cover the first
// half of the head.
//
// head_dim is 128 and theta is 1e6 in the Qwen3-ASR configuration this runtime
// serves; both arrive through the uniform, and the kernel tolerates any
// head_dim <= ROPE_HEAD_DIM_MAX so a smaller head still runs. The harness only
// exercises 128.
//
// Position is the token index: a batch is one sequence laid out contiguously
// from position 0.
//
// # Memory layout
//
//   x     tokens * heads * head_dim f32, row major   (x[token][head][d])
//   out   same shape, written by this kernel; x and out are separate buffers so
//         the kernel has no read-after-write hazard inside one dispatch
//
// # Bind group 0
//
//   binding 0  uniform    RopeParams { tokens, heads, head_dim, theta }  (16 bytes)
//   binding 1  read       x
//   binding 2  read_write out
//
// # Workgroup
//
//   workgroup_size(64, 1, 1): one invocation per (token, head, pair).
//   dispatch: (ceil(head_dim / 2 / 64), heads, tokens), i.e. (1, heads, tokens)
//   at head_dim 128. gid.x is the pair index, gid.y the head, gid.z the token.
//   No workgroup memory and no barriers: invocations are independent.

const ROPE_HEAD_DIM_MAX: u32 = 128u;

struct RopeParams {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    theta: f32,
};

@group(0) @binding(0) var<uniform> params: RopeParams;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn rope_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.head_dim > ROPE_HEAD_DIM_MAX) {
        return;
    }
    let half = params.head_dim / 2u;
    if (gid.x >= half) {
        return;
    }
    if (gid.y >= params.heads) {
        return;
    }
    if (gid.z >= params.tokens) {
        return;
    }

    let position = f32(gid.z);
    let inverse_frequency = pow(params.theta, -2.0 * f32(gid.x) / f32(params.head_dim));
    let angle = position * inverse_frequency;
    let cos_angle = cos(angle);
    let sin_angle = sin(angle);

    let base = (gid.z * params.heads + gid.y) * params.head_dim;
    let first = x[base + gid.x];
    let second = x[base + gid.x + half];
    out[base + gid.x] = first * cos_angle - second * sin_angle;
    out[base + gid.x + half] = second * cos_angle + first * sin_angle;
}
