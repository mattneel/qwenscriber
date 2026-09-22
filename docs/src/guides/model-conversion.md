# Model conversion

Official Qwen3-ASR checkpoints must be converted into Qwenscriber's versioned, sharded distribution
format before browser use.

## Intended command

```sh
qwenscriber-convert \
  --input Qwen3-ASR-0.6B \
  --output model/ \
  --quant q4
```

The command and flags are planned until the converter is released. Production conversion should
trend toward a Zig tool that can ship as a self-contained native binary.

## Conversion stages

1. Read official configuration, tokenizer/vocabulary data, and tensor metadata.
2. Identify the exact supported architecture and reject unknown variants.
3. Map source tensor names to canonical runtime names.
4. Validate every shape before transposition or packing.
5. Transpose/repack tensors required by runtime kernels.
6. Quantize in deterministic blocks and record scheme parameters.
7. Partition tensors into bounded, cacheable shards.
8. Emit the versioned manifest and shard checksums.
9. Reopen the output with the runtime parser.
10. Optionally compare sampled/reconstructed tensor values with the source.

## Drift prevention

Converter and runtime share format constants, tensor-name rules, shape validators, dtype enums,
quantization descriptors, and checksum semantics wherever feasible. A converter-only understanding
of a layout is a latent corruption bug.

## Verification scripts

A temporary Python script is acceptable during bring-up when it materially accelerates comparison
with official tooling. Python must not become a browser/runtime dependency, and any reference script
used for release verification should be pinned and reproducible.

## Licensing and provenance

Conversion does not erase model terms. Record upstream repository/revision, model identifier,
license, tokenizer sources, conversion-tool revision, quantization settings, and output checksums.
Do not commit upstream checkpoints or generated shards to the source repository.

