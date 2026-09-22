//! The container's tensor kinds, mirroring `TensorKind` in `src/core/container.zig`.
//!
//! The ABI reports a kind as an integer (`TensorDescriptor.kind`), and this table is the only place
//! the numbers mean anything on this side of the boundary: a caller finds a weight by looking for
//! the kind it wants while walking `qw_model_tensor_descriptor`.
//!
//! The container's rule is "append new kinds, never renumber", which would make a stale mirror
//! silently wrong rather than loudly broken, so `tests/gpu/layout_drift.mjs` compares this table
//! against the Zig enum -- names and values both -- on every run. A kind added there fails that gate
//! until it is mirrored here, the same way a WGSL constant does.

/** Every tensor kind a converted model can hold, in the container's numbering. */
export const TENSOR_KIND = {
  audio_conv1_weight: 1,
  audio_conv1_bias: 2,
  audio_conv2_weight: 3,
  audio_conv2_bias: 4,
  audio_conv3_weight: 5,
  audio_conv3_bias: 6,
  audio_conv_out_weight: 7,
  audio_final_norm_weight: 8,
  audio_final_norm_bias: 9,
  audio_layer_attention_q_weight: 10,
  audio_layer_attention_q_bias: 11,
  audio_layer_attention_k_weight: 12,
  audio_layer_attention_k_bias: 13,
  audio_layer_attention_v_weight: 14,
  audio_layer_attention_v_bias: 15,
  audio_layer_attention_out_weight: 16,
  audio_layer_attention_out_bias: 17,
  audio_layer_attention_norm_weight: 18,
  audio_layer_attention_norm_bias: 19,
  audio_layer_ffn_in_weight: 20,
  audio_layer_ffn_in_bias: 21,
  audio_layer_ffn_out_weight: 22,
  audio_layer_ffn_out_bias: 23,
  audio_layer_final_norm_weight: 24,
  audio_layer_final_norm_bias: 25,
  projector_in_weight: 26,
  projector_in_bias: 27,
  projector_out_weight: 28,
  projector_out_bias: 29,
  decoder_embed_tokens_weight: 30,
  decoder_final_norm_weight: 31,
  decoder_output_weight: 32,
  decoder_layer_attention_q_weight: 33,
  decoder_layer_attention_k_weight: 34,
  decoder_layer_attention_v_weight: 35,
  decoder_layer_attention_out_weight: 36,
  decoder_layer_attention_q_norm_weight: 37,
  decoder_layer_attention_k_norm_weight: 38,
  decoder_layer_attention_norm_weight: 39,
  decoder_layer_ffn_norm_weight: 40,
  decoder_layer_ffn_gate_weight: 41,
  decoder_layer_ffn_up_weight: 42,
  decoder_layer_ffn_down_weight: 43,
} as const;

/**
 * Layer numbering, from `src/core/container.zig`: zero for tensors that belong to no layer,
 * `DECODER_LAYER_BASE..` for decoder layers, and `AUDIO_LAYER_BASE + i` for audio tower layers.
 *
 * The bases are tags rather than offsets -- 1024 keeps the tower's blocks clear of any decoder a
 * released checkpoint could have -- so a caller looking for the tower's layer `i` has to add the
 * base, and a caller that forgets finds nothing at all rather than the wrong weights.
 */
export const DECODER_LAYER_BASE = 1;
export const AUDIO_LAYER_BASE = 1024;

/** A kind's name, as the container spells it. */
export type TensorKindName = keyof typeof TENSOR_KIND;

/** The integer a tensor descriptor reports for a kind. */
export type TensorKind = (typeof TENSOR_KIND)[TensorKindName];
