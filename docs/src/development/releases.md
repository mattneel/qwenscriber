# Release engineering

GitHub Actions CI is the release authority. Developer machines can build and test candidates, but
canonical artifacts are produced from a tagged, reviewed repository revision in CI.

## Release outputs

A release may include:

- native core static/shared libraries and `qwenscriber.h`;
- native CLI/conversion/inspection binaries per supported platform;
- the `wasm32-freestanding` module, and, when it exists, the thread-enabled `wasm32-emscripten`
  module, each identified in the artifact metadata (they are not byte-interchangeable even though the
  ABI matches: [ADR-0005](../project/decisions/0005-thread-enabled-wasm-build.md));
- the TypeScript package containing or resolving the matching WASM artifact;
- approved language integration packages;
- model converter artifacts, never third-party model weights by accident;
- checksums, provenance/attestations, and an SBOM; and
- the mdBook site deployed to <https://mattneel.github.io/qwenscriber>.

Every artifact embeds or accompanies enough version information to relate runtime version, ABI
version, model-format version, source revision, and toolchain.

## Pipeline order

1. Check formatting, generated-file drift, licenses, and repository hygiene.
2. Build/test Zig natively with the pinned Zig master revision.
3. Build/test deterministic `wasm32-freestanding` artifacts with intended SIMD features, and run the
   same ABI conformance fixtures against the thread-enabled build once it exists.
4. Run ABI/layout/format fixtures and CPU-versus-WGSL conformance.
5. Run TypeScript/browser/integration tests on the supported matrix.
6. Build native target archives and language packages from the same source revision.
7. Build the mdBook and reject broken navigation/links.
8. Generate checksums, SBOM, and provenance.
9. Publish immutable GitHub Release assets and package-registry artifacts.
10. Deploy the already-validated mdBook output/source revision to GitHub Pages.

The book is published by the `pages` job in `.github/workflows/ci.yml` from `main` only, and the job
deploys the artifact the `book` job already built rather than rebuilding it. This requires the
repository's Pages source to be set to "GitHub Actions" once; the deploy job fails loudly rather than
silently, which is the intended behavior for a misconfigured publication path.

Publishing begins only after validation. A failed target does not produce a partial release carrying
the same version.

## Cutting a release

```sh
git tag v0.1.0 && git push origin v0.1.0
```

`.github/workflows/release.yml` does the rest, and refuses to publish if any gate fails:

1. **Version gate.** The tag, `build.zig.zon`, and `packages/qwenscriber/package.json` must agree.
   Artifacts that disagree about their own version are worse than no release, so this runs before
   anything is built.
2. **Gates.** `zig build check` in Debug and in ReleaseFast, the shader/quantization drift check, and
   the TypeScript package's build, typecheck, and tests.
3. **Cross-compiled native tools.** The four command line tools are built for
   `x86_64-linux-gnu`, `aarch64-linux-gnu`, `x86_64-macos`, `aarch64-macos`, and `x86_64-windows`,
   all from one runner: they are pure Zig, so no per-platform toolchain has to be installed or kept
   in step.
4. **WASM module and TypeScript package**, with the module instantiated and self-tested before it is
   packaged.
5. **Checksums, provenance, attestation.** `SHA256SUMS` covers every artifact; `provenance.json`
   records the source revision, toolchain versions, ABI and model-format versions, and the digest of
   each file; `actions/attest-build-provenance` signs the artifacts.
6. **Publication.** `gh release create` with a preamble describing what the artifacts are and what is
   not finished (`.github/release-notes.md`), followed by GitHub's generated notes for commits,
   contributors, and merged pull requests.

`workflow_dispatch` re-runs the pipeline for an existing tag, which is the tool for a failed upload or
a flaky runner. Nothing is published until every job above succeeds, and a partial release is never
published under the same version.

## Reproducibility

- Pin the Zig revision and release tooling.
- Make generated inputs and feature flags explicit.
- Avoid timestamps or host paths in deterministic artifacts where practical.
- Record target triples and optimization modes.
- Compare rebuilt outputs where the toolchain supports reproducibility.

## Compatibility gates

A release cannot silently break the C/WASM ABI or model format. CI runs compatibility fixtures for
supported prior versions. Intentional breaks require a declared new version, migration notes, and a
support decision for old artifacts.

## Documentation publication

The root README links to the Pages site; the detailed source lives in `docs/`. CI builds that source
with a pinned mdBook version. Documentation describing an unreleased feature retains its planned or
experimental label even when deployed from the default branch.

