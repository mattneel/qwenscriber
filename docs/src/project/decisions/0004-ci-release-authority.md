# ADR-0004: CI is release authority

- **Status:** Accepted
- **Date:** 2026-09-22

## Context

Qwenscriber will distribute mutually compatible WASM, TypeScript, native, C-header, converter,
integration, and documentation artifacts. Building pieces manually risks toolchain drift, mismatched
ABIs, incomplete validation, and unreproducible releases.

## Decision

GitHub Actions CI builds, tests, packages, attests, and publishes every canonical release artifact
from one tagged/reviewed source revision. It also builds and deploys the mdBook site to
<https://mattneel.github.io/qwenscriber>. Release artifacts include checksums, provenance, and an
SBOM where applicable.

## Consequences

The Zig and documentation toolchains must be pinned. Cross-artifact compatibility becomes a CI
gate. Local builds remain useful for development but are not canonical releases. Publication
credentials and third-party actions require minimal permissions and explicit review.

## Rejected alternatives

- Uploading binaries from a maintainer workstation.
- Publishing language packages in separate, uncoordinated jobs/revisions.
- Deploying documentation independently of repository/release validation.

