//! The quantized plane layout, as far as a caller has to know it.
//!
//! `src/core/quant.zig` owns the layout and `tests/gpu/layout_drift.mjs` compares every constant
//! here against it on each run, which is what makes a mirror in TypeScript safe to have at all. The
//! SDK is transport, not layout authority: it never re-derives the *packing* -- the shaders and the
//! converter own that -- but it does have to say where a tensor's code plane starts, and that is
//! `planeLayout`'s arithmetic, so it lives here once rather than in every wrapper that needs it.
//!
//! Layout, per tensor: groups of `QUANT_GROUP_SIZE` weights along a row, each group's scale in the
//! scale plane at offset 0, then the code plane at the next `TENSOR_ALIGNMENT_BYTES` boundary.

/** Weights per quantization group. Mirrors `quant.group_size`. */
export const QUANT_GROUP_SIZE = 64;
/** Code bytes per group, per format: q4 packs two codes a byte, q5 adds a bit plane, q8 a byte each. */
export const QUANT_CODE_BYTES_PER_GROUP = { q4: 32, q5: 40, q8: 64 } as const;
/** Scale bytes per group: one f16 for every quantized format. */
export const QUANT_SCALE_BYTES_PER_GROUP = 2;
/** Alignment of a quantized payload, hence of its data plane. Mirrors `quant.tensor_alignment_bytes`. */
export const TENSOR_ALIGNMENT_BYTES = 16;

/** `dtype.Format` values for the quantized formats, as the container writes them. */
export const FORMAT_Q4 = 16;
export const FORMAT_Q5 = 17;
export const FORMAT_Q8 = 18;

export type QuantFormat = keyof typeof QUANT_CODE_BYTES_PER_GROUP;

function alignForward(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

/**
 * Where this tensor's code plane starts, relative to the start of its payload.
 *
 * `rows` and `cols` are the tensor's shape as its descriptor reports it; `cols` must be a multiple
 * of the group size, which the container already enforced when it validated the tensor.
 *
 * The format does not appear: every quantized format scales one f16 per group, so the scale plane
 * is the same width whether the codes behind it are four, five, or eight bits wide.
 */
export function dataPlaneOffsetBytes(rows: number, cols: number): number {
  const groups_per_row = cols / QUANT_GROUP_SIZE;
  const scale_bytes = rows * groups_per_row * QUANT_SCALE_BYTES_PER_GROUP;
  // The payload's own start is 16-byte aligned in the container, which is why the alignment applies
  // to the scale plane's length alone.
  return alignForward(scale_bytes, TENSOR_ALIGNMENT_BYTES);
}

/** The format id of a quantized `QuantFormat`, matching what a tensor descriptor reports. */
export function formatId(format: QuantFormat): number {
  if (format === "q4") return FORMAT_Q4;
  if (format === "q5") return FORMAT_Q5;
  return FORMAT_Q8;
}

/** The quantized format a descriptor's `format` value names, or `undefined` for an element format. */
export function quantFormatOf(format: number): QuantFormat | undefined {
  if (format === FORMAT_Q4) return "q4";
  if (format === FORMAT_Q5) return "q5";
  if (format === FORMAT_Q8) return "q8";
  return undefined;
}
