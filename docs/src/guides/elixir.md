# Elixir with Zigler

The server-side Elixir integration uses Zigler to call the Zig core directly. This keeps shared
Qwenscriber semantics in Zig while allowing the binding to model BEAM scheduling, resources, and
binaries deliberately.

> The Elixir package is planned; module names and Mix configuration are not stable yet.

## Boundary design

- A loaded model/session is represented by a BEAM resource with one explicit native owner.
- Long inference operations run on an appropriate dirty scheduler or are divided into bounded work;
  they never block a normal scheduler for unbounded time.
- Inputs use binaries without unnecessary copies where lifetime rules permit.
- Native state is not retained through an unrooted raw pointer.
- Resource destructors are safe, bounded, and idempotent with explicit close semantics.
- Core error codes become structured Elixir errors; invariant violations remain defects.

## Concurrency

Document whether a runtime/session is single-owner, internally synchronized, or requires one process
to serialize calls. Do not infer safety from BEAM process isolation when multiple resources point to
the same native state.

Model weights may be shared only through a designed reference-counted or immutable ownership model.
Per-transcription decode/KV state should remain isolated.

## Build and release

The integration targets the repository-pinned Zig master toolchain. CI builds and tests supported
native targets, packages any approved precompiled artifact, and records the exact core/ABI version.
A local Mix build must not silently compile a semantically different core.

## Why not WASM on the server?

WASM remains the portable JavaScript/browser integration point. Elixir already has a direct native
bridge through Zigler, which avoids an extra runtime and lets the binding express BEAM-specific
scheduling and resource rules.

