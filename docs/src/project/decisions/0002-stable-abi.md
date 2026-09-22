# ADR-0002: Stable C-like ABI

- **Status:** Accepted
- **Date:** 2026-09-22

## Context

The Zig core must be callable from TypeScript/WASM, native languages with C FFI, tooling, and server
integrations. Zig compiler data layouts and error/slice representations are not stable foreign ABIs.
Zig master also changes quickly.

## Decision

Expose a small, explicit, versioned C-like ABI using fixed-width fields, pointer/offset-plus-length
byte ranges, opaque or integer handles, caller-visible ownership, stable result codes, and explicit
state transitions. Never export Zig slices, optionals, error unions, or accidental struct layouts.

The idiomatic Zig SDK remains the source foundation. The CLI and bindings depend on it; it does not
depend on them. Elixir may use Zigler while preserving the same core semantics.

## Consequences

ABI types and layouts require compile-time assertions and compatibility fixtures. Wrappers perform
some adaptation but avoid semantic reimplementation. Breaking changes require a new ABI version or
parallel compatibility surface.

## Rejected alternatives

- Exporting Zig-native function signatures directly.
- A broad JSON/message RPC interface for in-process calls.
- Separate semantics for WASM and native integrations.

