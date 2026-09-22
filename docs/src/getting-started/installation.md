# Installation

Qwenscriber does not yet have a supported public release. Do not install an unrelated package that
happens to use the name.

## Release channels

Once releases begin, CI is the authority for every distributable artifact. A release is expected to
contain, as applicable:

- the TypeScript/WASM package;
- native static and shared libraries;
- the public C header;
- native command-line tools;
- model inspection/conversion tools;
- platform archives;
- checksums, provenance, and an SBOM; and
- the exact mdBook source/site revision for that release.

Install native command-line artifacts by extracting the platform archive and placing its executable
on `PATH`. The goal is a self-contained binary with as much compiled in statically as platform and
license constraints permit.

## From source

Use the exact Zig master revision pinned by the repository. The project tracks the `0.17.0-dev`
lineage; an arbitrary stable Zig release may not compile it.

The intended root commands are:

```sh
zig build
zig build test
zig build wasm
zig build test-wasm
```

These commands become authoritative only when their build steps exist on the checked-out revision.
See [setup and workflow](../development/setup.md) for the complete development environment.

## Models

Model weights are not stored in the source repository and should not be bundled implicitly with the
runtime. A model release must identify:

- the upstream checkpoint and license;
- the converter/runtime format version;
- quantization and block layout;
- shard checksums; and
- runtime compatibility requirements.

Use only model artifacts whose provenance and terms you have reviewed.

