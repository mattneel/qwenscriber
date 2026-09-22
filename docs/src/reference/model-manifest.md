# Model manifest

The manifest is fetched and validated before large weight artifacts. This page is provisional until
the first format version is implemented and frozen.

## Required information

| Area | Fields |
| --- | --- |
| Identity | Format version, architecture, model variant, upstream/source identity |
| Compatibility | Minimum/maximum runtime or required feature bits |
| Vocabulary | Tokenizer/vocabulary artifact identity and checksums |
| Tensor | Canonical name, shape, logical dtype, packed dtype, quantization descriptor |
| Storage | Shard identity, byte offset, byte length, required alignment, checksum |
| Quantization | Scheme, block size, scale type/layout, zero-point semantics |
| Integrity | Manifest and shard digest algorithm/value where applicable |

## Illustrative shape

```json
{
  "format": "qwenscriber-model",
  "format_version": 1,
  "architecture": "qwen3-asr",
  "variant": "0.6b",
  "quantization": {
    "scheme": "q4",
    "block_size": 32
  },
  "shards": [
    {
      "id": "audio-000",
      "path": "audio/shard-000.qw",
      "byte_length": 0,
      "checksum": "algorithm:value"
    }
  ],
  "tensors": [
    {
      "name": "example.weight",
      "shape": [0, 0],
      "dtype": "q4",
      "shard": "audio-000",
      "byte_offset": 0,
      "byte_length": 0,
      "alignment": 256
    }
  ]
}
```

Zero lengths/shapes and placeholder names above intentionally prevent this illustration from being
mistaken for a valid model.

## Validation order

1. Bound total manifest bytes and nesting/counts before allocation.
2. Validate syntax, required fields, enum values, and version.
3. Validate integer conversions, shape products, block divisibility, offsets, lengths, alignment,
   and non-overlap without overflow.
4. Validate architecture-specific tensor inventory and relationships.
5. Validate shard declarations and integrity metadata.
6. Only then fetch/accept large shard ranges.

Unknown required fields/features fail closed. Unknown optional metadata may be ignored only when the
format version explicitly permits it.

