# Model format

Format version 1. This page describes the layout the converter writes and the runtime parses today;
`src/core/container.zig`, `src/core/model_config.zig`, and `src/core/quant.zig` are the normative
definitions, and their offsets are asserted at compile time.

A converted Qwenscriber model is a directory:

    model/
      manifest.json     index for the TypeScript side: shards, sizes, hashes
      config.bin        architecture configuration, fixed 160-byte layout
      tokens.bin        tokenizer table: id -> token bytes
      shard-000.qw      weights, one or more, each self-describing
      shard-001.qw
      ...

The split is deliberate. `manifest.json` is read by the browser, which has a native JSON parser and
wants a fetch list with integrity hashes. `config.bin` and the shards are read by the WASM runtime,
which must not parse JSON, allocate unpredictably, or hold a string-keyed map of 600 tensor names.

Everything is little endian. Serialised integers are explicitly sized. Every structure that crosses a
language boundary is a fixed layout with its size asserted at compile time.

## `manifest.json`

Written by `qwenscriber-convert`, read by the SDK. Fields:

| Field | Meaning |
|---|---|
| `format_version` | Container format version, currently 1 |
| `architecture` | `"qwen3_asr"` |
| `model_id` | Free-form identifier, e.g. `qwen3-asr-0.6b-q4` |
| `quantization` | `q4`, `q5`, `q8`, or `none` |
| `config` | `{ name, bytes, sha256 }` for `config.bin` |
| `tokens` | `{ name, bytes, sha256, count }` for `tokens.bin` |
| `shards` | One entry per shard (see below) |
| `totals` | `{ tensors, payload_bytes, source_bytes, bits_per_weight }` |
| `output_is_tied` | True when the checkpoint has no separate output projection and the embedding matrix doubles as the unembedding |
| `tool_version` | The converter's version string |

Each `shards` entry:

| Field | Meaning |
|---|---|
| `name` | File name within the model directory |
| `bytes` | File length |
| `sha256` | Hex digest of the whole file, for cache integrity |
| `tensor_count` | Tensors in this shard |
| `first_layer`, `last_layer` | Layer range the shard covers, or null when it holds only layerness tensors |

Layer numbering (see below) makes a shard's `first_layer`/`last_layer` a useful loading hint: a
browser that is about to run layers 8 to 15 needs exactly the shards whose ranges overlap, and
nothing else.

## `config.bin`

A 160-byte `extern` structure, `model_config.Config`, validated after parsing. Its size and the
offsets of its fields are asserted at compile time, so adding a field is a format version bump rather
than an accident.

| Offset | Field | Notes |
|---|---|---|
| 0 | `magic` | `"QWCFG001"` |
| 8 | `format_version` | 1 |
| 12 | `architecture` | 1 = `qwen3_asr` |
| 16 | `audio_d_model` | Audio tower width (0.6B: 896, 1.7B: 1024) |
| 20 | `audio_layers` | 18 / 24 |
| 24 | `audio_attention_heads` | 14 / 16 |
| 28 | `audio_ffn_dim` | 3584 / 4096 |
| 32 | `audio_downsample_hidden_size` | 480 |
| 36 | `audio_n_window` | Mel frames per half chunk; the convolution stack consumes twice this |
| 40 | `audio_n_window_infer` | Mel frames per encoder attention window |
| 44 | `audio_max_position_steps` | Rows of the sinusoidal position table that are used |
| 48 | `audio_output_dim` | Projector output; equals the decoder's hidden size |
| 52 | `mel_bins` | 128 |
| 56 | `audio_layer_norm_eps` | 1e-5, torch's `LayerNorm` default |
| 60 | `text_hidden_size` | 1024 / 2048 |
| 64 | `text_layers` | 28 |
| 68 | `text_attention_heads` | 16 |
| 72 | `text_key_value_heads` | 8 |
| 76 | `text_head_dim` | 128 |
| 80 | `text_ffn_dim` | 3072 / 6144 |
| 84 | `vocab_size` | 151936 |
| 88 | `text_rms_norm_eps` | 1e-6 |
| 92 | `rope_theta` | 1e6 |
| 96 | `max_positions` | Key/value cache capacity; a hard limit on sequence length |
| 100 | `max_decode_tokens` | Default generation limit |
| 104 | `token_audio_start` | 151669 |
| 108 | `token_audio_end` | 151670 |
| 112 | `token_audio_pad` | 151676, the placeholder the audio features replace |
| 116 | `token_im_start` | 151644 |
| 120 | `token_im_end` | 151645 |
| 124 | `token_endoftext` | 151643 |
| 128 | `token_asr_text` | 151704 |
| 132 | `token_eos_primary` | 151643 |
| 136 | `token_eos_secondary` | 151645 |
| 140 | `token_pad` | 151643 |
| 144 | four reserved words | Must be zero; rejected otherwise |

Validation rejects a configuration whose head geometry, mel geometry, chunk geometry, epsilons, rope
base, or token ids cannot describe a real model. It also rejects a value that would make the
key/value cache unbounded.

Note what is *derived* rather than stored, because storing it twice is how the converter and the
runtime drift apart:

* the frequency bins after the convolution stack,
  `(((mel_bins + 1) / 2 + 1) / 2 + 1) / 2`;
* the width of the convolution output projection, `downsample_hidden_size * frequency_bins`;
* the post-convolution step count of a chunk, `(n - 1) / 2 + 1` applied three times;
* the packed step count for a clip, which counts whole chunks and the partial chunk separately. This
  one matters: applying the downsampling once to the concatenated frames loses a step at every chunk
  boundary, and yields 375 instead of 390 for a 30-second clip.

## `tokens.bin`

    offset 0                 u32 token count
    offset 4                 u32 offsets[count + 1]   (byte offsets into the pool)
    offset 4 + 4*(count+1)   the token byte pool

`count` is `max_id + 1` over the vocabulary and the added special tokens, so the table covers ids up
to 151704 including the added tokens. A gap is an error, not a filled-in blank. Token strings are
stored exactly as the vocabulary has them, in the GPT-2 byte alphabet (`Ġ` for a space), which is what
the detokenizer expects; `src/core/tokenizer.zig` undoes that alphabet.

The runtime reads this file directly as a `tokenizer.TokenTable`: no parsing, no allocation, no copy.

## `.qw` shards

    offset 0                          header (32 bytes)
    offset 32                         tensor index (tensor_count x 40 bytes)
    offset payload_offset_bytes       payload (aligned to 256)

### Header

| Offset | Field | Notes |
|---|---|---|
| 0 | `magic` | `"QWSHARD1"` |
| 8 | `format_version` | 1 |
| 12 | `index_len_bytes` | Must equal `tensor_count * 40` |
| 16 | `payload_offset_bytes` | Must be at least the header plus the index, and a multiple of 256 |
| 20 | `payload_len_bytes` | |
| 24 | `tensor_count` | |
| 28 | `payload_checksum` | Low 32 bits of the FNV-1a hash of the payload |

### Index entry

| Offset | Field | Notes |
|---|---|---|
| 0 | `kind` | `container.TensorKind`; permanent numbering |
| 2 | `layer` | See numbering below |
| 4 | `format` | `dtype.Format`; permanent numbering |
| 5 | `rank` | 1 to 4 |
| 6 | reserved | Must be zero |
| 8 | `dims` | Four `u32`s; the first `rank` are meaningful |
| 24 | `offset_bytes` | From the payload start; a multiple of 16 |
| 32 | `len_bytes` | Exact payload length, quantization padding included |

Entries are sorted by `(layer, kind)` and parsed defensively: the parser rejects a bad magic or
version, an index length that disagrees with the tensor count, a payload offset that overlaps the
index or is misaligned, entries out of order, overlapping tensor ranges, a range past the payload, a
format this build does not know, a shape that disagrees with its format, and a byte length that
disagrees with its shape. A shard that passes is one whose every tensor can be viewed without further
checks.

### Layer numbering

| Range | Meaning |
|---|---|
| 0 | Tensors that belong to no layer: the convolution stack, the projector, the embedding, the output projection, the final norm |
| 1 .. text_layers | Decoder layers |
| 1024 .. | Audio tower layers |

Layer-first ordering is why one shard can hold a contiguous range of layers with a correctly sorted
index.

### Tensor kinds

`container.TensorKind` enumerates every tensor the model needs. Per-layer kinds are paired with the
layer number in the index entry. The values are on disk, so they are permanent: new kinds are
appended, never renumbered.

### Quantized tensor payload

A quantized matrix of `rows x cols` is stored as two planes:

    scales plane:  group_count * 2 bytes   (f16 scale per group)
    padding to the tensor's 16-byte alignment
    data plane:    group_count * data_bytes_per_group

with `group_count = rows * cols / 64` and one group spanning 64 consecutive weights of a row. Codes
are unsigned with a midpoint bias, so a decoder computes `(code - bias) * scale` without a sign
extension:

| Format | Data bytes per group | Bias | Code range | Bits per weight |
|---|---|---|---|---|
| `q4` | 32 (two weights per byte, low nibble first) | 8 | -8 .. 7 | 4.25 |
| `q5` | 40 (32 nibble bytes plus 8 high-bit bytes) | 16 | -16 .. 15 | 5.25 |
| `q8` | 64 (one byte per weight) | 128 | -128 .. 127 | 8.25 |

Non-quantized tensors are plain `f32`, `f16`, or `bf16` arrays. `src/core/quant.zig` is the single
source of truth for all of this, `gpu/shaders/quant_layout.wgsl` mirrors the constants for the
shaders, and `tests/gpu/layout_drift.mjs` fails if the two ever disagree.

## Determinism

For the same input bytes, the converter produces byte-identical output: tensor order is fixed by the
inventory, the index is sorted, padding is zeroed, and the only inputs to the layout are the
configuration and the tensor shapes. That is what makes the checksums in the manifest meaningful as
cache keys.

## Relationship to the earlier sketch

This format stores tensor descriptors inside each shard rather than as a named tensor table inside
`manifest.json`. The reason is memory and parse cost in the browser: a named table of several hundred
tensors would be a large JSON document that has to be parsed and then re-associated with byte ranges,
whereas the in-shard index is already sorted, already fixed-width, and is mapped to the exact tensor
the runtime asks for by `(kind, layer)`. `reference/model-manifest.md` keeps the earlier field list as
the checklist this layout must satisfy.
