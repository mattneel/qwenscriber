# Security and privacy

Local-first is a property to enforce, not marketing shorthand.

## Privacy model

Qwenscriber should not transmit audio, features, transcripts, or model-derived content by default.
Network activity in the browser path is limited to application-controlled code and fetching model or
runtime artifacts from configured sources. Telemetry, if ever added, must be separate, opt-in, and
incapable of silently including audio/transcript bytes.

Applications can still upload results themselves; Qwenscriber cannot make a hostile embedding page
private. The SDK should make its own network behavior inspectable and narrow.

## Untrusted inputs

Treat these as untrusted:

- audio bytes, metadata, sizes, sample rates, and channel counts;
- manifest/tokenizer/container bytes;
- shard contents, offsets, lengths, alignment, and checksums;
- cache entries and interrupted downloads;
- C/WASM pointers, offsets, lengths, handles, and enum values;
- TypeScript configuration and custom artifact resolvers; and
- GPU limits, device loss, validation errors, and shader-visible dimensions.

Parsing is bounded, overflow-checked, and complete before large allocation or dispatch.

## Supply chain

- CI builds canonical release artifacts from a reviewed revision and pinned toolchain.
- Releases carry checksums, provenance, and an SBOM.
- Model artifacts record upstream identity/license, conversion revision, format, quantization, and
  checksums.
- New dependencies require license/security/update ownership review.
- Third-party CI actions and package publication credentials receive least privilege.

## Memory and compute abuse

Bound audio duration, manifest/tensor counts, dimensions, shard concurrency, decode steps, queue
depth, output bytes, and scratch memory. Fail early when adapter/runtime limits are insufficient.
Cancellation must release staged network, WASM, Worker, and GPU resources.

## Browser boundary

WASM and Worker isolation reduce blast radius but do not replace validation. Cross-origin isolation
and `SharedArrayBuffer` are optional for basic non-streaming use where feasible. Content Security
Policy and model origin recommendations will be specified with the published package.

## Disclosure

Follow the repository
[security policy](https://github.com/mattneel/qwenscriber/blob/main/SECURITY.md) for private
vulnerability reports.
