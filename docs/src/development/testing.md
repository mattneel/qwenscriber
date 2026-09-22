# Testing

Qwenscriber uses layered tests so a full-model transcript is the integration proof, not the first
place a tensor bug becomes visible.

## Test layers

| Layer | Examples |
| --- | --- |
| Unit | Bounds/overflow helpers, handle tables, audio transforms, tokenizer rules, shape math |
| Format/ABI | Layout/version fixtures, malformed manifests, unknown features, invalid state, old
  compatibility fixtures |
| Reference math | Deterministic tensor operations and quantize/dequantize behavior |
| GPU conformance | WGSL output compared with Zig reference across edge shapes and tolerances |
| Component | Encoder/decoder blocks, preprocessing pipeline, model loading, cache transitions |
| Browser integration | Worker/WASM/WebGPU lifecycle, cancellation, device loss, capability paths |
| End to end | Legal small audio/model fixtures producing known decoded text |

## Determinism

Use fixed seeds and explicit tensor values. Test fixtures record shapes, dtypes, quantization
parameters, expected values, tolerance policy, and the source/tool revision that produced them.

## Negative space

For every successful parser or ABI path, test invalid versions, truncated inputs, overflow, boundary
alignment, impossible shape products, overlapping ranges, excessive counts, wrong-state handles, and
cleanup after partial initialization.

## GPU conformance

Each kernel starts with a straightforward CPU reference. Compare tiny and awkward dimensions before
large benchmarks. Include values around quantization boundaries and verify both absolute and relative
tolerance where floating-point order differs.

Never weaken a tolerance solely to make a new kernel pass. Explain the numerical model.

## Full models

Large model checkpoints do not belong in git. End-to-end jobs obtain identified artifacts through a
controlled cache/source, verify checksums, and separate artifact/network failures from runtime test
failures. PR checks should retain a fast deterministic subset.

## Fuzzing

Manifest/container parsing, ABI byte ranges, audio headers/adapters, tokenizer inputs, and handle
state transitions are high-value fuzz targets. Fuzzers complement assertions and code review; they
do not prove absence of bugs.

