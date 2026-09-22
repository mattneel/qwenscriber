<!--
  Release notes preamble.

  GitHub appends its generated notes (commits, contributors, merged pull
  requests) after this text, so keep it short and factual: it exists to say what
  the artifacts are, how to check them, and what is not finished. Update it when
  the artifact set changes, not when a release is cut.
-->

## What is in this release

| Artifact | Contents |
| --- | --- |
| `qwenscriber-<version>-<target>.tar.gz` / `.zip` | Native command line tools for that target: `qwenscriber-transcribe`, `qwenscriber-convert`, `qwenscriber-inspect`, `qwenscriber-selftest` |
| `qwenscriber-core-<version>.wasm` | The `wasm32-freestanding` core module: zero imports, exports its own memory, and runs the ABI self-test |
| `qwenscriber-qwenscriber-<version>.tgz` | The TypeScript package, containing the matching WASM module |
| `SHA256SUMS` | Digests of every artifact above |
| `provenance.json` | Source revision, toolchain versions, ABI and model-format versions, and per-artifact digests |

Native tools are pure Zig, so every target is cross-compiled from one runner with no per-platform
toolchain. Verify a download with `sha256sum -c SHA256SUMS`.

## Status

Qwenscriber is **pre-alpha** and this is not a supported release. The
[status page](https://mattneel.github.io/qwenscriber/status.html) records what actually works, with
the command that demonstrates it, and it is deliberately pessimistic. In particular:

- The WASM core runs preprocessing, tokenization, and detokenization today. Model loading and decode
  are not in ABI v1 yet, so the browser SDK stops short of a transcript with a typed error rather
  than producing a guess.
- The model format is version 1 and is not frozen. Converted models must be regenerated for a new
  format version; the manifest carries `format_version` so a mismatch fails closed.
- Nothing here has been benchmarked under the benchmark hooks the architecture pages describe, so no
  performance claim in these notes or in the documentation is a measured one yet.
- Model weights are not part of any artifact. Obtain them from the upstream Qwen distribution and
  convert them with `qwenscriber-convert`; review the model's own license before use.

## Compatibility

The ABI version is reported by `qw_abi_version()` and recorded in `provenance.json`. It is
independent of the release tag: a `0.1.x` release may still speak ABI v1.0, and a change that breaks
the ABI increments its major version rather than hiding inside a minor release.
