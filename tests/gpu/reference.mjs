// CPU reference for the WebGPU kernels: deterministic inputs, the quantized
// packing writer, and the same arithmetic the kernels are supposed to perform.
//
// Everything here mirrors `src/core/math.zig` and `src/core/quant.zig` line for
// line and stays in f32: every step goes through `f32()` (Math.fround), which is
// what a scalar f32 implementation does, so a difference against a GPU kernel is
// the GPU's, not the reference's sloppiness.
//
// The packing is not trusted on faith. `harness.mjs` hashes a fixed pattern with
// `quant_hash()` below and compares that hash with the one `qw_selftest` reports
// from zig-out/bin/qwenscriber_core.wasm, which is the FNV-1a of the same
// pattern quantized by `src/core/quant.zig`. Matching hashes mean this file's f16
// rounding, nibble packing, q5 high-bit plane, and code biasing are the Zig
// implementation's, bit for bit.
//
// No dependencies, no DOM: this module runs in the page and in Node.

export const GROUP_SIZE = 64;
export const GROUP_CODE_BYTES = { q4: 32, q5: 40, q8: 64 };
export const GROUP_SCALE_BYTES = 2;
export const GROUP_BIAS = { q4: 8, q5: 16, q8: 128 };
export const TENSOR_ALIGNMENT_BYTES = 16;
// Format ids from src/core/dtype.zig; they are part of the container format.
export const FORMAT_ID = { q4: 16, q5: 17, q8: 18 };

export const f32 = Math.fround;

// ---------------------------------------------------------------------------
// Deterministic inputs
// ---------------------------------------------------------------------------

// mulberry32: 32 bits of state, one multiply-xor round per draw, identical on
// every engine and platform. A PRNG that could differ between the two runs of a
// comparison harness would make the comparison meaningless.
export function make_random(seed) {
    let state = seed >>> 0;
    return function next_u32() {
        state = (state + 0x6d2b79f5) >>> 0;
        let value = state;
        value = Math.imul(value ^ (value >>> 15), value | 1) >>> 0;
        value = (value ^ (value + Math.imul(value ^ (value >>> 7), value | 61))) >>> 0;
        return (value ^ (value >>> 14)) >>> 0;
    };
}

// Uniform in [-1, 1): the weights and activations the kernels see are signed and
// bounded, and keeping them bounded keeps the matmul sums inside f32 range.
export function random_uniform(next_u32) {
    const fraction = next_u32() / 4294967296;
    return f32(2 * fraction - 1);
}

// Box-Muller. The transform needs a non-zero uniform, hence `1 - fraction`.
export function random_normal(next_u32) {
    const first = 1 - next_u32() / 4294967296;
    const second = next_u32() / 4294967296;
    return f32(Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second));
}

export function random_vector(length, seed, generator = random_normal) {
    const next_u32 = make_random(seed);
    const values = new Float32Array(length);
    for (let index = 0; index < length; index += 1) values[index] = generator(next_u32);
    return values;
}

// ---------------------------------------------------------------------------
// f16, as quant.zig stores its scales
// ---------------------------------------------------------------------------

// f32 -> f16 bits, round to nearest even, saturating to infinity on overflow.
export function to_f16_bits(value) {
    const view = new DataView(new ArrayBuffer(4));
    view.setFloat32(0, value, true);
    const bits = view.getUint32(0, true);
    const sign = (bits >>> 16) & 0x8000;
    const exponent = (bits >>> 23) & 0xff;
    let mantissa = bits & 0x7fffff;
    if (exponent === 0xff) return sign | 0x7c00 | (mantissa === 0 ? 0 : 0x200);
    const half_exponent = exponent - 127 + 15;
    if (half_exponent >= 0x1f) return sign | 0x7c00;
    if (half_exponent <= 0) {
        // Subnormal: the implicit leading bit moves into the mantissa, and the
        // shift grows as the exponent falls.
        if (half_exponent < -10) return sign;
        mantissa |= 0x800000;
        return sign | round_shift_right(mantissa, 14 - half_exponent);
    }
    const rounded = round_shift_right(mantissa, 13);
    if (rounded > 0x3ff) return sign | ((half_exponent + 1) << 10);
    return sign | (half_exponent << 10) | rounded;
}

// Shift right with round to nearest even on the discarded bits.
function round_shift_right(value, shift) {
    const kept = value >>> shift;
    const remainder = value & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (remainder > halfway) return kept + 1;
    if (remainder < halfway) return kept;
    return kept + (kept & 1);
}

export function from_f16_bits(bits) {
    const sign = (bits & 0x8000) === 0 ? 1 : -1;
    const exponent = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    if (exponent === 0) return f32(sign * mantissa * 2 ** -24);
    if (exponent === 31) return mantissa === 0 ? sign * Infinity : NaN;
    return f32(sign * (1 + mantissa / 1024) * 2 ** (exponent - 15));
}

// ---------------------------------------------------------------------------
// Quantization, mirroring src/core/quant.zig
// ---------------------------------------------------------------------------

// Quantizes one row of `values` (a multiple of GROUP_SIZE long) the way
// `quant.zig` does: the group scale is max_abs / bias rounded to f16, and the
// codes are quantized against the *stored* f16 scale so a group never decodes
// above its true maximum.
export function quantize_row(format, values) {
    const group_count = values.length / GROUP_SIZE;
    const scales = new Uint8Array(group_count * GROUP_SCALE_BYTES);
    const data = new Uint8Array(group_count * GROUP_CODE_BYTES[format]);
    for (let group = 0; group < group_count; group += 1) {
        quantize_group(format, values, group, scales, data);
    }
    return { scales, data };
}

function quantize_group(format, values, group, scales, data) {
    const bias = GROUP_BIAS[format];
    const base = group * GROUP_SIZE;
    let max_abs = 0;
    for (let index = 0; index < GROUP_SIZE; index += 1) {
        max_abs = Math.max(max_abs, Math.abs(values[base + index]));
    }
    const scale_bits = to_f16_bits(f32(f32(max_abs) / f32(bias)));
    scales[group * 2] = scale_bits & 0xff;
    scales[group * 2 + 1] = (scale_bits >>> 8) & 0xff;
    const stored_scale = from_f16_bits(scale_bits);

    const code_base = group * GROUP_CODE_BYTES[format];
    for (let index = 0; index < GROUP_SIZE; index += 1) {
        const code = quantize_one(values[base + index], stored_scale, -bias, bias - 1);
        const unsigned = code + bias;
        if (format === "q4" || format === "q5") {
            if (index % 2 === 0) data[code_base + (index >> 1)] |= unsigned & 0x0f;
            else data[code_base + (index >> 1)] |= (unsigned & 0x0f) << 4;
            if (format === "q5" && (unsigned & 0x10) !== 0) {
                data[code_base + 32 + (index >> 3)] |= 1 << (index % 8);
            }
        } else {
            data[code_base + index] = unsigned;
        }
    }
}

// Round half away from zero, then clamp into the code range.
function quantize_one(value, stored_scale, min_code, max_code) {
    if (stored_scale === 0) return 0;
    const scaled = f32(f32(value) / f32(stored_scale));
    const rounded = scaled >= 0 ? f32(Math.floor(scaled + 0.5)) : f32(Math.ceil(scaled - 0.5));
    if (rounded <= min_code) return min_code;
    if (rounded >= max_code) return max_code;
    return rounded;
}

// Writes a `rows x cols` weight matrix as the two-plane tensor `planeLayout`
// describes with a base offset of zero: the scale plane first, then padding to
// TENSOR_ALIGNMENT_BYTES, then the packed codes. Groups are row major.
export function pack_tensor(format, rows, cols, weights) {
    if (cols % GROUP_SIZE !== 0) throw new Error(`cols ${cols} is not a multiple of ${GROUP_SIZE}`);
    const groups_per_row = cols / GROUP_SIZE;
    const group_count = rows * groups_per_row;
    const data_offset_bytes = align_up(group_count * GROUP_SCALE_BYTES, TENSOR_ALIGNMENT_BYTES);
    const bytes = new Uint8Array(
        data_offset_bytes + group_count * GROUP_CODE_BYTES[format],
    );
    const scales = bytes.subarray(0, group_count * GROUP_SCALE_BYTES);
    for (let row = 0; row < rows; row += 1) {
        const row_values = weights.subarray(row * cols, (row + 1) * cols);
        const quantized = quantize_row(format, row_values);
        scales.set(quantized.scales, row * groups_per_row * GROUP_SCALE_BYTES);
        const row_offset = row * groups_per_row * GROUP_CODE_BYTES[format];
        bytes.set(quantized.data, data_offset_bytes + row_offset);
    }
    return { bytes, data_offset_bytes };
}

function align_up(value, alignment) {
    return Math.ceil(value / alignment) * alignment;
}

// Decodes one weight of `row` at `column`, referencing the packed planes a group
// at a time exactly like `quant.dequantizeGroup` + `quant.dotRow`.
function decode_weight(format, planes, groups_per_row, row, column) {
    const group_index = row * groups_per_row + Math.floor(column / GROUP_SIZE);
    const within_group = column % GROUP_SIZE;
    const bias = GROUP_BIAS[format];
    const scale = from_f16_bits(
        planes.scales[group_index * 2] | (planes.scales[group_index * 2 + 1] << 8),
    );
    let unsigned;
    if (format === "q4" || format === "q5") {
        const byte = planes.data[group_index * GROUP_CODE_BYTES[format] + (within_group >> 1)];
        unsigned = within_group % 2 === 0 ? byte & 0x0f : (byte >> 4) & 0x0f;
        if (format === "q5") {
            const high_bit_base = group_index * GROUP_CODE_BYTES[format] + 32;
            const high_byte = planes.data[high_bit_base + (within_group >> 3)];
            unsigned |= ((high_byte >> (within_group % 8)) & 1) << 4;
        }
    } else {
        unsigned = planes.data[group_index * GROUP_CODE_BYTES[format] + within_group];
    }
    return f32((unsigned - bias) * scale);
}

// ---------------------------------------------------------------------------
// Kernel references
// ---------------------------------------------------------------------------

// Dense f32 matmul: out[token][row] = dot(activation[token], weight[row]),
// accumulated in f32 with one rounding per multiply and per add, like a scalar
// implementation of the tiled kernel.
export function matmul_f32_reference(activations, weights, tokens, rows, cols) {
    const out = new Float32Array(tokens * rows);
    for (let token = 0; token < tokens; token += 1) {
        for (let row = 0; row < rows; row += 1) {
            let accumulator = 0;
            for (let column = 0; column < cols; column += 1) {
                const reference = activations[token * cols + column];
                const product = f32(reference * weights[row * cols + column]);
                accumulator = f32(accumulator + product);
            }
            out[token * rows + row] = accumulator;
        }
    }
    return out;
}

// Quantized matmul: the same dot product, with each weight decoded from the
// packed planes on the fly (`quant.dotRow` is the reference the WebGPU kernel is
// measured against, and it decodes per group rather than materializing a dense
// matrix).
export function matmul_quantized_reference(activations, tensor, format, tokens, rows, cols) {
    const groups_per_row = cols / GROUP_SIZE;
    const planes = {
        scales: tensor.bytes.subarray(0, rows * groups_per_row * GROUP_SCALE_BYTES),
        data: tensor.bytes.subarray(tensor.data_offset_bytes),
    };
    const out = new Float32Array(tokens * rows);
    for (let token = 0; token < tokens; token += 1) {
        for (let row = 0; row < rows; row += 1) {
            let accumulator = 0;
            for (let column = 0; column < cols; column += 1) {
                const weight = decode_weight(format, planes, groups_per_row, row, column);
                accumulator = f32(accumulator + f32(weight * activations[token * cols + column]));
            }
            out[token * rows + row] = accumulator;
        }
    }
    return out;
}

export function dequant_reference(planes, format, rows, cols) {
    const groups_per_row = cols / GROUP_SIZE;
    const out = new Float32Array(rows * cols);
    for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < cols; column += 1) {
            out[row * cols + column] = decode_weight(format, planes, groups_per_row, row, column);
        }
    }
    return out;
}

// Splits a packed tensor into the two plane views the dequantization reference
// reads, given the group count the shape implies.
export function plane_views(tensor, rows, cols) {
    const groups_per_row = cols / GROUP_SIZE;
    return {
        scales: tensor.bytes.subarray(0, rows * groups_per_row * GROUP_SCALE_BYTES),
        data: tensor.bytes.subarray(tensor.data_offset_bytes),
    };
}

// rmsNormInPlace from src/core/math.zig.
export function rmsnorm_reference(x, weight, rows, cols, eps) {
    const out = new Float32Array(rows * cols);
    for (let row = 0; row < rows; row += 1) {
        let sum_squares = 0;
        for (let column = 0; column < cols; column += 1) {
            const value = x[row * cols + column];
            sum_squares = f32(sum_squares + f32(value * value));
        }
        const mean_square = f32(sum_squares / f32(cols));
        const inverse_rms = f32(1 / f32(Math.sqrt(f32(mean_square + eps))));
        for (let column = 0; column < cols; column += 1) {
            const normalized = f32(x[row * cols + column] * inverse_rms);
            out[row * cols + column] = f32(normalized * weight[column]);
        }
    }
    return out;
}

// Rotary embedding, non-interleaved pairs, position = token index. `pow` and the
// angle are rounded to f32 before the trigonometry, so the reference makes the
// same f32 rounding choices the kernel does and the remaining difference is the
// adapter's own `pow`/`sin`/`cos`.
export function rope_reference(x, tokens, heads, head_dim, theta, position_base = 0) {
    const half = head_dim / 2;
    const out = new Float32Array(tokens * heads * head_dim);
    for (let token = 0; token < tokens; token += 1) {
        for (let head = 0; head < heads; head += 1) {
            const base = (token * heads + head) * head_dim;
            for (let pair = 0; pair < half; pair += 1) {
                const inverse_frequency = f32(Math.pow(theta, (-2 * pair) / head_dim));
                const angle = f32(f32(token + position_base) * inverse_frequency);
                const cos_angle = f32(Math.cos(angle));
                const sin_angle = f32(Math.sin(angle));
                const first = x[base + pair];
                const second = x[base + pair + half];
                out[base + pair] = f32(f32(first * cos_angle) - f32(second * sin_angle));
                out[base + pair + half] = f32(f32(second * cos_angle) + f32(first * sin_angle));
            }
        }
    }
    return out;
}

// Windowed attention with an optional causal mask and a stable softmax.
export function attention_reference(queries, keys, values, shape) {
    const { tokens, heads, keys: key_count, head_dim, scale, window, causal } = shape;
    const out = new Float32Array(tokens * heads * head_dim);
    const scores = new Float32Array(window);
    for (let token = 0; token < tokens; token += 1) {
        for (let head = 0; head < heads; head += 1) {
            const range = visible_range(token, key_count, window, causal);
            const query_base = (token * heads + head) * head_dim;
            let score_max = -Infinity;
            for (let r = range.start; r < range.end; r += 1) {
                const key_base = (r * heads + head) * head_dim;
                let dot_product = 0;
                for (let d = 0; d < head_dim; d += 1) {
                    const product = f32(queries[query_base + d] * keys[key_base + d]);
                    dot_product = f32(dot_product + product);
                }
                scores[r - range.start] = f32(dot_product * scale);
                score_max = Math.max(score_max, scores[r - range.start]);
            }
            let total = 0;
            for (let r = 0; r < range.end - range.start; r += 1) {
                scores[r] = f32(Math.exp(scores[r] - score_max));
                total = f32(total + scores[r]);
            }
            const inverse_total = f32(1 / total);
            for (let d = 0; d < head_dim; d += 1) {
                let accumulator = 0;
                for (let r = range.start; r < range.end; r += 1) {
                    const value_base = (r * heads + head) * head_dim;
                    const probability = f32(scores[r - range.start] * inverse_total);
                    accumulator = f32(accumulator + f32(probability * values[value_base + d]));
                }
                out[query_base + d] = accumulator;
            }
        }
    }
    return out;
}

// The key range a query may attend to: the last `window` keys ending at the
// visible end, which is the full sequence for a non-causal kernel.
export function visible_range(token, key_count, window, causal) {
    let end = key_count;
    if (causal) end = Math.min(token + 1, key_count);
    return { start: Math.max(0, end - window), end };
}

// SiLU(gate) * up, with `silu(x) = x / (1 + exp(-x))` from src/core/math.zig.
export function silu_mul_reference(gate, up, count) {
    const out = new Float32Array(count);
    for (let index = 0; index < count; index += 1) {
        const value = gate[index];
        out[index] = f32(f32(value / f32(1 + Math.exp(-value))) * up[index]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// The quantized-pattern hash qw_selftest reports
// ---------------------------------------------------------------------------

const FNV1A_64_OFFSET_BASIS = 0xcbf29ce484222325n;
const FNV1A_64_PRIME = 0x100000001b3n;
const MASK_64 = 0xffffffffffffffffn;

// The error function GELU needs, by the same `erfc` rational fit `src/core/math.zig` uses.
//
// Ported coefficient for coefficient rather than taken from the language's own `erf`: this reference
// exists to say what the *core* computes, so a more accurate function here would report the core's
// accurate approximation as an error.
export function erf_reference(value) {
    const sign = value < 0 ? -1 : 1;
    const magnitude = Math.abs(value);
    const t = 1 / (1 + 0.5 * magnitude);
    const tau = t * Math.exp(-magnitude * magnitude - 1.26551223 +
        t * (1.00002368 +
            t * (0.37409196 +
                t * (0.09678418 +
                    t * (-0.18628806 +
                        t * (0.27886807 +
                            t * (-1.13520398 +
                                t * (1.48851587 +
                                    t * (-0.82215223 + t * 0.17087277)))))))));
    return f32(sign * (1 - tau));
}

// `gelu` from src/core/math.zig: the exact (error-function) form, as torch's default computes it.
export function gelu_value(value) {
    return f32(f32(0.5 * value) * (1 + erf_reference(f32(value * 0.70710678))));
}

export function gelu_reference(x, count) {
    const out = new Float32Array(count);
    for (let index = 0; index < count; index += 1) out[index] = gelu_value(x[index]);
    return out;
}

// Weight `index` of a packed f16 plane: the low half of element `index / 2` when `index` is even,
// the high half otherwise, which is how the container stores a u16 tensor.
function packed_f16_bits(packed_weights, index) {
    const element = packed_weights[index >> 1];
    return (element >>> ((index & 1) * 16)) & 0xffff;
}

// `conv3x3Stride2Gelu` from src/core/qwen3_asr/kernels.zig.
//
// The bias first, then each input channel's nine taps in row-major kernel order, then one GELU over
// the finished sum -- not one per channel. Padding is one, stride is two, so the output plane is
// `(in_height - 1) / 2 + 1` by `(in_width - 1) / 2 + 1`.
export function conv3x3_stride2_gelu_reference(
    input, packed_weights, bias, out_channels, in_channels, in_height, in_width,
) {
    const out_height = Math.floor((in_height - 1) / 2) + 1;
    const out_width = Math.floor((in_width - 1) / 2) + 1;
    const in_plane = in_height * in_width;
    const out_plane = out_height * out_width;
    const out = new Float32Array(out_channels * out_plane);
    for (let channel = 0; channel < out_channels; channel += 1) {
        for (let out_row = 0; out_row < out_height; out_row += 1) {
            for (let out_column = 0; out_column < out_width; out_column += 1) {
                let sum = bias[channel];
                for (let in_channel = 0; in_channel < in_channels; in_channel += 1) {
                    const weight_base = (channel * in_channels + in_channel) * 9;
                    for (let kernel_row = 0; kernel_row < 3; kernel_row += 1) {
                        const in_row = out_row * 2 + kernel_row - 1;
                        if (in_row < 0 || in_row >= in_height) continue;
                        for (let kernel_column = 0; kernel_column < 3; kernel_column += 1) {
                            const in_column = out_column * 2 + kernel_column - 1;
                            if (in_column < 0 || in_column >= in_width) continue;
                            const sample =
                                input[in_channel * in_plane + in_row * in_width + in_column];
                            const bits = packed_f16_bits(
                                packed_weights, weight_base + kernel_row * 3 + kernel_column,
                            );
                            sum += from_f16_bits(bits) * sample;
                        }
                    }
                }
                out[channel * out_plane + out_row * out_width + out_column] = gelu_value(f32(sum));
            }
        }
    }
    return out;
}

// Packs f32 weights into the u16-pair layout `conv3x3_stride2_gelu_reference` and the shader read,
// through the same f16 rounding the container's writer uses.
export function pack_f16_pairs(weights) {
    const packed = new Uint32Array(Math.ceil(weights.length / 2));
    for (let index = 0; index < weights.length; index += 1) {
        packed[index >> 1] |= to_f16_bits(weights[index]) << ((index & 1) * 16);
    }
    return packed;
}

// `layerNormInPlace` from src/core/math.zig.
//
// Two passes on purpose: the mean is gathered first, and the deviations are measured from it. A
// single pass accumulating a sum and a sum of squares is algebraically equal and numerically
// different, and the kernel under test mirrors the core's two passes for that reason.
export function layernorm_reference(x, weight, bias, rows, cols, eps) {
    const out = new Float32Array(rows * cols);
    for (let row = 0; row < rows; row += 1) {
        const base = row * cols;
        let sum = 0;
        for (let column = 0; column < cols; column += 1) sum += x[base + column];
        const mean = f32(sum / cols);
        let squared = 0;
        for (let column = 0; column < cols; column += 1) {
            const centered = f32(x[base + column] - mean);
            squared += centered * centered;
        }
        const variance = f32(squared / cols);
        const inverse_std = f32(1 / Math.sqrt(variance + eps));
        for (let column = 0; column < cols; column += 1) {
            const centered = f32(x[base + column] - mean);
            out[base + column] = f32(f32(centered * inverse_std) * weight[column] + bias[column]);
        }
    }
    return out;
}

// `out[column][row] = input[row][column]`, the layout change between the tower's convolution stack
// and its downsample projection.
export function transpose_reference(input, rows, cols) {
    const out = new Float32Array(cols * rows);
    for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < cols; column += 1) {
            out[column * rows + row] = input[row * cols + column];
        }
    }
    return out;
}

// The q8 cache planes for one batch of rows, mirroring `quantizeRow` in `src/core/quant.zig`.
//
// The scale is stored as f16 and the codes are computed against that *stored* value, not the f32
// quotient, so a group can never decode above its true maximum. Rounding is half away from zero.
export function quantize_q8_rows_reference(values, rows, cols, data_offset_bytes) {
    const groups_per_row = cols / GROUP_SIZE;
    const group_count = rows * groups_per_row;
    const bytes = new Uint8Array(data_offset_bytes + group_count * GROUP_SIZE);
    for (let group = 0; group < group_count; group += 1) {
        let max_abs = 0;
        for (let index = 0; index < GROUP_SIZE; index += 1) {
            max_abs = Math.max(max_abs, Math.abs(values[group * GROUP_SIZE + index]));
        }
        const scale_bits = to_f16_bits(max_abs / 128);
        const scale = from_f16_bits(scale_bits);
        bytes[group * 2] = scale_bits & 0xff;
        bytes[group * 2 + 1] = (scale_bits >>> 8) & 0xff;
        for (let index = 0; index < GROUP_SIZE; index += 1) {
            const value = values[group * GROUP_SIZE + index];
            const scaled = scale === 0 ? 0 : value / scale;
            const rounded = scale === 0
                ? 0
                : (scaled >= 0 ? Math.floor(scaled + 0.5) : Math.ceil(scaled - 0.5));
            const clamped = Math.max(-128, Math.min(127, rounded));
            bytes[data_offset_bytes + group * GROUP_SIZE + index] = clamped + 128;
        }
    }
    return bytes;
}

// The decoded value at (position, column) of a q8 cache plane: `scale * (code - 128)`, with the
// scale read from the stored f16.
function decode_cache_at(planes, data_offset_bytes, groups_per_row, position, column) {
    const group = position * groups_per_row + Math.floor(column / GROUP_SIZE);
    const scale = from_f16_bits(
        planes[group * 2] | (planes[group * 2 + 1] << 8),
    );
    const code = planes[data_offset_bytes + position * groups_per_row * GROUP_SIZE + column];
    return scale * (code - 128);
}

// One query's attention over a q8 key/value cache, mirroring the shape `decode_attention_q8.wgsl`
// serves: grouped heads, a maximum-subtracted softmax, and a weighted sum over the cached positions.
export function decode_attention_q8_reference(query, key_planes, value_planes, shape) {
    const { heads, kv_heads, head_dim, positions, groups_per_row, data_offset_bytes } = shape;
    const scale = 1 / Math.sqrt(head_dim);
    const out = new Float32Array(heads * head_dim);
    const group_size = heads / kv_heads;
    for (let head = 0; head < heads; head += 1) {
        const kv_head = Math.floor(head / group_size);
        const column_base = kv_head * head_dim;
        const scores = new Float32Array(positions);
        for (let position = 0; position < positions; position += 1) {
            let accumulator = 0;
            for (let index = 0; index < head_dim; index += 1) {
                accumulator += query[head * head_dim + index] *
                    decode_cache_at(key_planes, data_offset_bytes, groups_per_row, position,
                        column_base + index);
            }
            scores[position] = accumulator * scale;
        }
        let maximum = -Infinity;
        for (const score of scores) maximum = Math.max(maximum, score);
        let total = 0;
        for (let position = 0; position < positions; position += 1) {
            scores[position] = Math.exp(scores[position] - maximum);
            total += scores[position];
        }
        for (let index = 0; index < head_dim; index += 1) {
            let accumulator = 0;
            for (let position = 0; position < positions; position += 1) {
                accumulator += (scores[position] / total) *
                    decode_cache_at(value_planes, data_offset_bytes, groups_per_row, position,
                        column_base + index);
            }
            out[head * head_dim + index] = accumulator;
        }
    }
    return out;
}

export function fnv1a_64(byte_arrays) {
    let hash = FNV1A_64_OFFSET_BASIS;
    for (const bytes of byte_arrays) {
        for (const byte of bytes) {
            hash = ((hash ^ BigInt(byte)) * FNV1A_64_PRIME) & MASK_64;
        }
    }
    return hash;
}

// The fixed pattern `selftest.checkQuantization` in src/core/selftest.zig
// quantizes and hashes: 128 values of `cos(i * 0.19) * 1.75`, packed as q4, then
// q5, then q8, hashing each format's scale plane followed by its code plane.
export function quant_pattern_values(cols = 128) {
    const values = new Float32Array(cols);
    for (let index = 0; index < cols; index += 1) {
        const position = f32(index);
        values[index] = f32(f32(Math.cos(f32(f32(position * f32(0.19))))) * f32(1.75));
    }
    return values;
}

export function quant_hash() {
    const values = quant_pattern_values();
    const planes = [];
    for (const format of ["q4", "q5", "q8"]) {
        const quantized = quantize_row(format, values);
        planes.push(quantized.scales, quantized.data);
    }
    return fnv1a_64(planes);
}
