# Model format

Qwenscriber's browser distribution format is purpose-built for bounded parsing, cacheable delivery,
sharded loading, and direct quantized execution. GGUF or official checkpoint formats may be accepted
by conversion tools, but they do not define the runtime architecture.

## Design requirements

- A small, versioned manifest that can be fetched first.
- Fixed-width, validated offsets and lengths.
- Multiple independently cacheable shards.
- Tensor lookup without scanning giant files.
- Explicit shape, dtype, quantization, block layout, and alignment.
- Checksums for complete artifacts and, where useful, ranges.
- Streaming/range-friendly delivery.
- Layer-oriented loading compatible with WebGPU buffer limits.
- Format definitions shared by converter and runtime.

## Quantization

Initial likely layouts are block-oriented Q4 and Q5 with packed values, a scale, an optional zero
point when the chosen scheme needs one, and an explicit block size. Layout decisions are made from
correctness and measured WebGPU behavior, not resemblance to another container.

## Conceptual package

A model may be represented as a manifest plus audio-encoder and decoder shards. This is conceptual,
not a frozen filesystem contract: a single indexed container or another layout may win if it
improves streaming, cacheability, integrity, and memory pressure without weakening validation.

## Compatibility

The manifest declares a format version and runtime compatibility range. Unknown required features,
unknown dtypes, impossible shapes, overlapping ranges, invalid alignment, and checksum failures are
ordinary parse errors. The runtime never guesses.

See the [model manifest reference](../reference/model-manifest.md) for the provisional schema.

