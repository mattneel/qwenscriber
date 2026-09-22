# Model manifest

This page is the requirements checklist the manifest must satisfy. The field set actually emitted at
format version 1 is documented in [model format](model-format.md); that page and the parser in
`src/host/manifest.zig` are authoritative, and this one records what any revision of the manifest is
required to carry.

## Required information

| Area | Fields | Where it lives at format version 1 |
| --- | --- | --- |
| Identity | Format version, architecture, model variant, upstream/source identity | `format_version`, `architecture`, `model_id`, `tool_version` |
| Compatibility | Minimum/maximum runtime or required feature bits | `format_version` plus the runtime's `qw_features()` bits; a version mismatch fails closed |
| Vocabulary | Tokenizer/vocabulary artifact identity and checksums | `tokens` (name, bytes, SHA-256, count) |
| Tensor | Canonical name, shape, logical dtype, packed dtype, quantization descriptor | Per-shard binary index; `(kind, layer)` replaces the name, see [model format](model-format.md) |
| Storage | Shard identity, byte offset, byte length, required alignment, checksum | `shards[]` (name, bytes, SHA-256, `tensor_count`, layer range) plus the in-shard index |
| Quantization | Scheme, block size, scale type/layout, zero-point semantics | `quantization` plus the fixed block layout in `src/core/quant.zig` |
| Integrity | Manifest and shard digest algorithm/value | `sha256` per shard and per auxiliary artifact |

Two deliberate departures from the earlier sketch on this page are worth recording, because they
change what a manifest is:

- **Tensors are not listed by name in the manifest.** Several hundred named entries in JSON would have
  to be parsed and then re-associated with byte ranges in the browser, which is exactly the work the
  shard index already does in fixed-width form. The manifest stays small and fetch-oriented; the shard
  index carries shape, dtype, offset, and length.
- **The auxiliary artifacts are addressed by field, not by convention.** `config` and `tokens` name
  their files and digests explicitly, so a revision can rename them without breaking consumers.

## Validation order

1. Bound total manifest bytes and nesting/counts before allocation.
2. Validate syntax, required fields, enum values, and version.
3. Validate integer conversions, shape products, block divisibility, offsets, lengths, alignment,
   and non-overlap without overflow.
4. Validate architecture-specific tensor inventory and relationships.
5. Validate shard declarations and integrity metadata.
6. Only then fetch/accept large shard ranges.

Unknown required fields/features fail closed. Unknown optional metadata may be ignored only when the
format version explicitly permits it. `src/host/manifest.zig` tests cover the malformed cases: missing
fields, negative and oversized numbers, unknown architecture, unknown quantization, shard digests that
are not hex, and duplicate shard names.
