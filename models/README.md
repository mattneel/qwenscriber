# Models

This directory holds model artifacts. Nothing here is committed: `.gitignore` excludes every model
subdirectory, and this file is exempt from those rules. A checkout clones the runtime, never the
weights.

## Layout

| Path | Contents |
| --- | --- |
| `Qwen3-ASR-0.6B/` | Official upstream checkpoint directory, as downloaded from the Qwen repository |
| `Qwen3-ASR-0.6B-hf/`, `Qwen3-ASR-1.7B-hf/` | Upstream's published `transformers` layout of each checkpoint, downloaded from `Qwen/Qwen3-ASR-<size>-hf`. `tools/reference/prepare_hf_layout.py` can rebuild this layout from something else, but the published repositories are the path to prefer: upstream's own `convert_qwen3_asr_to_hf.py` flattens the configuration *and* renames every tensor |
| `Qwen3-ASR-1.7B/` | The raw 1.7B checkpoint, which upstream publishes as two `model-0000N-of-00002.safetensors` files. It is the conversion *source*: the reader takes every `*.safetensors` in a directory, so sharding is not a special case, but `transformers` reads its `-hf` sibling |
| `<model-id>/` | A converted Qwenscriber model directory: `manifest.json`, `config.bin`, `tokens.bin`, and `.qw` shards |

The converter reads an upstream checkpoint and writes the converted directory; the runtime reads only
the converted directory. Keeping both under `models/` is a bring-up convenience, not an architecture:
a converted model is a self-contained artifact that can live on a CDN.

## Obtaining artifacts

Download checkpoints from the official Qwen distribution for the model you intend to run, and review
the model's license before use. Conversion does not change the terms that apply to the weights.

Verify a download before converting it: the converter checks shapes and dtypes, but it cannot detect a
truncated or substituted file whose tensors happen to parse. Compare the digest against the one the
upstream distribution publishes.

## Recording provenance

A converted model must be traceable. Record, next to the shards or in a release note:

- the upstream repository and revision;
- the model identifier and variant;
- the model license;
- the converter revision (`tool_version` in `manifest.json`);
- quantization scheme and block layout, as recorded in the manifest;
- the digest of each emitted shard.

`manifest.json` carries the tool version, the per-shard SHA-256 digests, and the quantization
descriptor, so a converted model that has been regenerated with different inputs will not present
itself as the same artifact.

## Disk pressure

The 0.6B checkpoint is roughly 2 GB in `f32`, the 1.7B checkpoint roughly 4 GB, and a converted Q4
model is roughly a quarter of that. Do not keep converted output on the same path as an in-progress
download if the filesystem is small; the converter writes shards atomically, so a partial output
directory can be deleted safely.
