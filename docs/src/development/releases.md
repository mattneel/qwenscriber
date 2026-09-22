# Release engineering

GitHub Actions CI is the release authority. Developer machines can build and test candidates, but
canonical artifacts are produced from a tagged, reviewed repository revision in CI.

## Release outputs

A release may include:

- native core static/shared libraries and `qwenscriber.h`;
- native CLI/conversion/inspection binaries per supported platform;
- the `wasm32-freestanding` module;
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
3. Build/test deterministic `wasm32-freestanding` artifacts with intended SIMD features.
4. Run ABI/layout/format fixtures and CPU-versus-WGSL conformance.
5. Run TypeScript/browser/integration tests on the supported matrix.
6. Build native target archives and language packages from the same source revision.
7. Build the mdBook and reject broken navigation/links.
8. Generate checksums, SBOM, and provenance.
9. Publish immutable GitHub Release assets and package-registry artifacts.
10. Deploy the already-validated mdBook output/source revision to GitHub Pages.

Publishing begins only after validation. A failed target does not produce a partial release carrying
the same version.

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

