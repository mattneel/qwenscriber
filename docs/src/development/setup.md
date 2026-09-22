# Setup and workflow

Qwenscriber tracks Zig master rather than a historical stable release. Use the exact compiler
revision pinned by the repository; the current architectural target is the `0.17.0-dev` lineage.

## Tools

Required for core work:

- the pinned Zig compiler;
- Git; and
- a browser/runtime suitable for the test being changed.

Additional areas may require:

- the repository-selected TypeScript package manager;
- mdBook for documentation previews;
- browser automation for WebGPU integration tests;
- the Emscripten SDK, only when working on the thread-enabled WASM build
  ([ADR-0005](../project/decisions/0005-thread-enabled-wasm-build.md)); and
- official Qwen3-ASR model/configuration sources for conversion work.

Do not introduce a package manager or framework merely to avoid a small bounded implementation.

## Standard commands

The build should converge on these stable entry points:

```sh
zig fmt --check .
zig build
zig build test
zig build wasm
zig build test-wasm
mdbook build docs
```

The root Zig build is authoritative for Zig outputs, including the deterministic WASM artifact used
by the TypeScript package. TypeScript tooling may wrap these commands but must not create a second,
semantically different build.

## Working loop

1. Confirm the pinned tool versions and a clean understanding of existing changes.
2. Write down the boundary, maximum sizes/work, ownership, error cases, and test oracle.
3. Resolve uncertain compiler/model/WebGPU behavior with a focused experiment.
4. Implement the smallest vertical slice.
5. Format and run the narrow test repeatedly.
6. Run all applicable builds and tests.
7. Update status, public/API docs, and an ADR if the decision is durable.
8. Inspect repository status for generated junk or weights.

## Generated and local data

Never commit Zig caches, package installations, downloaded checkpoints, converted shards, browser
caches, large benchmark output, or transient test artifacts. Keep small deterministic, legally
redistributable fixtures under the test fixture policy.

## Documentation preview

```sh
mdbook serve docs --open
```

The local site is a preview. CI publishes the canonical site from reviewed source.

