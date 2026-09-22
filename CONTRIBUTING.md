# Contributing to Qwenscriber

Qwenscriber is in pre-alpha bring-up. Small, verifiable vertical slices are more valuable than
broad speculative frameworks.

## Before you start

1. Read [`AGENTS.md`](AGENTS.md) and [`docs/TIGER_STYLE.md`](docs/TIGER_STYLE.md).
2. Check the [project status](docs/src/status.md) and [roadmap](docs/src/project/roadmap.md).
3. Open or find an issue before undertaking a large API, ABI, model-format, or backend change.
4. Prefer a tiny experiment when compiler, model, or WebGPU behavior is uncertain.

## Development loop

Use the Zig version pinned by the repository. It follows Zig master (`0.17.0-dev` lineage), so do
not assume examples written for older stable releases still compile.

As the corresponding build steps land, the standard checks are:

```sh
zig fmt --check .
zig build
zig build test
zig build wasm
zig build test-wasm
mdbook build docs
```

Run the narrowest relevant test while iterating, then the complete applicable suite before asking
for review.

## Change shape

A good contribution usually contains:

- one coherent behavior change;
- an explicit boundary and ownership model;
- tests for valid, boundary, and invalid inputs;
- a deterministic CPU reference for new GPU math;
- measurements for performance claims; and
- documentation updated in the same change.

Do not commit downloaded checkpoints, generated model shards, browser caches, large benchmark data,
Zig caches, or package-manager installation directories.

## Durable decisions

Add an architecture decision record under `docs/src/project/decisions/` when changing a decision
that affects multiple layers or future compatibility. State the context, decision, consequences,
and alternatives. Supersede old decisions rather than silently rewriting history.

## Review checklist

- [ ] Implemented/partial/planned status is accurate.
- [ ] Inputs, sizes, loops, queues, and resource use are bounded.
- [ ] Ownership and cleanup are explicit.
- [ ] ABI or serialized layouts use fixed-width fields and compatibility tests.
- [ ] Error paths are exercised; routine input errors do not trap.
- [ ] GPU behavior is compared against a deterministic reference.
- [ ] New dependencies are justified and their licenses recorded.
- [ ] Public docs and examples match the code.
- [ ] Formatting, tests, and builds pass with the pinned toolchain.

