# ADR-0003: Purpose-built model format

- **Status:** Accepted
- **Date:** 2026-09-22

## Context

Browser delivery needs cacheable/range-friendly shards, strict bounded parsing, compatibility with
WebGPU buffer limits, and packed layouts that fused kernels can consume directly. General checkpoint
or container formats need not optimize for those constraints.

## Decision

Define a versioned Qwenscriber distribution format with a small manifest and bounded shards. Record
architecture/model identity, tokenizer artifacts, tensor names/shapes, dtypes, quantization/block
layout, shard offsets/lengths/alignment, compatibility features, and checksums.

Q4 and Q5 are initial quantization candidates. Layouts are selected from correctness and measured
WebGPU behavior. GGUF and official checkpoints may be converter inputs but do not define the runtime
format.

## Consequences

Qwenscriber owns converter and compatibility work. Converter/runtime definitions must be shared to
prevent drift. The format can optimize direct upload and fused dequantization without materializing
full floating-point weights.

## Rejected alternatives

- Making GGUF compatibility the internal architecture.
- One unindexed multi-gigabyte file/buffer.
- Dequantizing entire tensors before every matmul.
- Letting each backend define its own model layout.

