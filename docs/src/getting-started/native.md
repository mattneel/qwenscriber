# Native quickstart

> **Planned surface:** the native package and ABI are not a supported release yet.

The native foundation exposes two related interfaces:

- an idiomatic Zig API for Zig consumers; and
- a deliberately stable C ABI for languages with native FFI.

The Zig API is the source of truth. The C ABI is a narrow compatibility boundary with opaque
handles, fixed-width fields, explicit ownership, version negotiation, and machine-readable errors.

## Zig consumer

The eventual package should be added through the repository's pinned Zig package workflow and
integrated through `build.zig`. Exact import and module names will be documented when the package is
published; do not copy a speculative package hash from this page.

## C-compatible consumer

A released native bundle is expected to contain:

- `include/qwenscriber.h`;
- a static library where supported;
- a shared library where supported; and
- ABI/version metadata and checksums.

The lifecycle follows this pattern:

1. Query ABI/runtime compatibility.
2. Build an explicit configuration.
3. Create an opaque runtime handle.
4. Provide validated model and audio bytes.
5. Start and step transcription or call the bounded convenience operation.
6. Copy result bytes using the documented buffer contract.
7. Destroy the handle exactly once.

See the [native and C FFI guide](../guides/native-c.md) and [C ABI reference](../reference/c-abi.md).

