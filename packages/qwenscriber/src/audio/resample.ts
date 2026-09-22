//! Sample-rate conversion, for audio that did not arrive at the core's 16 kHz.
//!
//! Method: band-limited windowed-sinc interpolation, evaluated per output sample from a table of
//! `RESAMPLE_PHASE_BINS` fractional phases with `RESAMPLE_TAPS` taps each, Blackman-windowed and
//! normalized per phase to unity DC gain. This is not linear interpolation: linear interpolation's
//! response falls off inside the band of interest and folds energy above the *target* Nyquist back
//! into it, which is audible on speech recorded at 48 kHz. The windowed sinc costs 32 multiply-adds
//! per output sample -- 512 kMAC for a second of audio -- and keeps aliasing below about -58 dB
//! (Blackman sidelobes), which is below the f16 storage of every quantized weight in the model.
//!
//! Two properties matter more than the numbers:
//!
//!   * Determinism. The kernel is a pure function of the rate ratio, and the loop reads only its
//!     arguments, so the same clip always produces the same samples on every run and every engine.
//!   * No clipping. A windowed-sinc interpolator has a small passband ripple and can overshoot on
//!     transients, and the core documents its input as f32 in `[-1, 1]`, so the result is clamped.

import { SDK_STATUS, QwenscriberError } from "../errors.ts";

/** Fractional positions per input sample. 512 keeps the phase quantization error under -70 dB. */
export const RESAMPLE_PHASE_BINS = 512;
/** Taps either side of the output position, so the kernel spans 32 input samples. */
export const RESAMPLE_HALF_TAPS = 16;
const RESAMPLE_TAPS = RESAMPLE_HALF_TAPS * 2;
/** Cut a little below Nyquist: the transition band has to fit inside the kernel, not outside it. */
const RESAMPLE_CUTOFF_GUARD = 0.98;
const RESAMPLE_RATE_HZ_MAX = 768000;

/** Blackman window over `[-1, 1]`. */
function blackmanWindow(position: number): number {
  return 0.42 + 0.5 * Math.cos(Math.PI * position) + 0.08 * Math.cos(2 * Math.PI * position);
}

/** Normalized sinc, `sin(pi x) / (pi x)`, defined as 1 at zero. */
function sinc(value: number): number {
  if (value === 0) return 1;
  const angle = Math.PI * value;
  return Math.sin(angle) / angle;
}

/** One-entry cache: a clip is converted at one ratio, and rebuilding 16k weights per call is waste. */
let cachedKernel: { readonly ratio: number; readonly weights: Float32Array } | undefined;

function kernelFor(ratio: number): Float32Array {
  if (cachedKernel !== undefined && cachedKernel.ratio === ratio) return cachedKernel.weights;
  const cutoff = Math.min(1, ratio) * RESAMPLE_CUTOFF_GUARD;
  const weights = new Float32Array(RESAMPLE_PHASE_BINS * RESAMPLE_TAPS);
  for (let phase = 0; phase < RESAMPLE_PHASE_BINS; phase += 1) {
    const fraction = phase / RESAMPLE_PHASE_BINS;
    let sum = 0;
    for (let tap = 0; tap < RESAMPLE_TAPS; tap += 1) {
      // Distance from the output position to this input sample, in input samples. The output
      // position sits `fraction` of a sample past the window's first sample.
      const offset = tap - (RESAMPLE_HALF_TAPS - 1) - fraction;
      const weight = cutoff * sinc(offset * cutoff) * blackmanWindow(offset / RESAMPLE_HALF_TAPS);
      weights[phase * RESAMPLE_TAPS + tap] = weight;
      sum += weight;
    }
    if (sum === 0) {
      throw new QwenscriberError(SDK_STATUS.internal, "resample", {
        message: `the interpolation kernel for ratio ${ratio} summed to zero`,
        context: { ratio, phase, cutoff },
      });
    }
    // Unity DC gain per phase: resampling must not change the level, and normalizing is also what
    // keeps the overshoot of an interpolating kernel small enough that the clamp below is a guard
    // rather than a distortion.
    for (let tap = 0; tap < RESAMPLE_TAPS; tap += 1) {
      weights[phase * RESAMPLE_TAPS + tap] = (weights[phase * RESAMPLE_TAPS + tap] ?? 0) / sum;
    }
  }
  cachedKernel = { ratio, weights };
  return weights;
}

function sampleOrZero(samples: Float32Array, index: number): number {
  if (index < 0) return 0;
  if (index >= samples.length) return 0;
  // The bounds check makes the fallback unreachable; `?? 0` is what `noUncheckedIndexedAccess`
  // wants to see before it will hand back a number.
  return samples[index] ?? 0;
}

/**
 * Resamples mono f32 audio. Output length is `floor(samples.length * target / source)`.
 *
 * The signal is treated as zero outside its own extent, which is the conventional choice and keeps
 * the result independent of whatever surrounds the clip. Audio outside the input range therefore
 * decays through the kernel instead of wrapping around to the other end.
 */
export function resample(
  samples: Float32Array,
  source_rate_hz: number,
  target_rate_hz: number,
): Float32Array {
  for (const [name, rate] of [
    ["source_rate_hz", source_rate_hz],
    ["target_rate_hz", target_rate_hz],
  ] as const) {
    if (!Number.isFinite(rate) || rate <= 0 || rate > RESAMPLE_RATE_HZ_MAX) {
      throw new QwenscriberError(SDK_STATUS.internal, "resample", {
        message: `${name} must be in (0, ${RESAMPLE_RATE_HZ_MAX}], got ${rate}`,
        context: { source_rate_hz, target_rate_hz },
      });
    }
  }
  if (source_rate_hz === target_rate_hz) return samples.slice();
  const ratio = target_rate_hz / source_rate_hz;
  const output_length = Math.floor(samples.length * ratio);
  if (output_length === 0) return new Float32Array(0);
  const weights = kernelFor(ratio);
  const output = new Float32Array(output_length);
  const last_tap = RESAMPLE_HALF_TAPS - 1;
  for (let index = 0; index < output_length; index += 1) {
    const position = index / ratio;
    const base = Math.floor(position);
    const fraction = position - base;
    const phase = Math.min(
      RESAMPLE_PHASE_BINS - 1,
      Math.floor(fraction * RESAMPLE_PHASE_BINS),
    );
    const row = phase * RESAMPLE_TAPS;
    let total = 0;
    for (let tap = 0; tap < RESAMPLE_TAPS; tap += 1) {
      total += (weights[row + tap] ?? 0) * sampleOrZero(samples, base + tap - last_tap);
    }
    // A non-finite sample can only come from a non-finite input or a non-finite weight, and both are
    // checked; asserting here keeps NaN from reaching the mel frontend as a silent poison.
    if (!Number.isFinite(total)) {
      throw new QwenscriberError(SDK_STATUS.internal, "resample", {
        message: `sample ${index} is not finite`,
        context: { index, source_rate_hz, target_rate_hz },
      });
    }
    output[index] = total < -1 ? -1 : total > 1 ? 1 : total;
  }
  return output;
}
