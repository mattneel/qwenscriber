# Model conversion

Official Qwen3-ASR checkpoints are converted into Qwenscriber's versioned, sharded distribution
format by `qwenscriber-convert`, a native Zig tool that `zig build` installs next to the runtime:

```sh
zig build
zig-out/bin/qwenscriber-convert \
  --input models/Qwen3-ASR-0.6B \
  --output models/qwen3-asr-0.6b-q5 \
  --quant q5 \
  --verify
```

| Flag | Meaning |
| --- | --- |
| `--input <dir>` | Upstream checkpoint directory: `config.json`, the tokenizer files, and one or more `*.safetensors` |
| `--output <dir>` | Model directory to create or overwrite |
| `--quant q4\|q5\|q8\|none` | Weight format; `none` writes f16, which is what reference runs use |
| `--shard-bytes <n>` | Shard budget in bytes (default 192 MiB) |
| `--max-positions <n>` | Decoder positions to allocate the key/value cache for (default 8192) |
| `--model-id <name>` | Identifier recorded in the manifest (default: the input directory's name) |
| `--verify` | Re-read every shard and compare it against the checkpoint |

The converter walks the core's own tensor inventory (`src/core/qwen3_asr/layout.zig`), so a tensor the
runtime asks for but the checkpoint lacks, or one the checkpoint has that the configuration does not
describe, stops the conversion instead of producing a model that fails at load time.

## Quantization policy

Measured on `tests/fixtures/audio/asr_zh.wav`, which the reference pipeline transcribes as
「甚至出现交易几乎停滞的情况。」. Sizes are exact; the outcome column is the transcript of the same clip
through the same runtime:

| Format | Payload | Bits/weight | Reference transcript |
| --- | --- | --- | --- |
| q4 | 422.7 MB | 4.32 | **none** — the run produces no tokens |
| q5 | 520.0 MB | 5.32 | yes |
| q8 | 811.7 MB | 8.30 | yes |
| f16 | 1565.4 MB | 16.01 | yes |

The same clip and the same user prompt through 1.7B, converted the same way:

| Format | Payload | Bits/weight | Reference transcript |
| --- | --- | --- | --- |
| q5 | 1344.6 MB | 5.28 | yes |
| f16 | 4077.0 MB | 16.00 | yes |

"Reference transcript" means the sentence the released implementation produces for this clip, read
through the `-hf` checkpoint upstream publishes for each size. Compare against those repositories and
not the raw ones: the raw checkpoint is the converter's *input*, and its tensor names and nested
configuration are pre-conversion.

q4 does not fail by producing garbage. It fails by losing a near-tie: the first decision on this clip
separates two candidate tokens by 0.095 logits, while q4's mean first-step logit error is about 3.6.
One wrong first token is not a slightly worse transcript, it is no transcript, because generation
never recovers. That is why **q5 is the format the browser targets** and q4 is not offered until a
4-bit scheme with more headroom per group — a zero point, a smaller group, or selective precision on
the audio tower — has been measured rather than assumed.

Quantization is not a speed compromise here either. On this host, ReleaseFast, f16 measured *slower*
than both q5 and q8 on the same clip: the decoder is bandwidth-bound, so three times the weight bytes
costs more than the unpacking arithmetic saves. Per-run numbers are in the metrics records described
in [Performance](../development/performance.md).

The key/value cache is the other half of the memory budget, and it is quantized with the same
machinery as the weights: `--cache q8` stores one signed byte per element with an f16 scale per group
of 64, which is 462 MiB at 8192 positions instead of 1792 MiB for both 0.6B and 1.7B — their text
configurations share a layer count, head counts, and head width. Measured on the same clip through the
same binary, the f32 and q8 caches decode at the same speed (1.49 and 1.55 tokens per second) and
produce the same ten token ids, so the smaller cache is a capacity win rather than a trade. Through the ABI the
width is chosen with `qw_model_set_cache_format` before the load, which is what makes the choice
visible in `qw_model_requirements`; the browser-side SDK does not call it yet — see
[Project status](../status.md).

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

