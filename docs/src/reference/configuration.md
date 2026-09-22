# Runtime configuration

This reference describes the planned configuration contract. The released TypeScript declarations
and C header become authoritative when they exist.

## Core options

| Option | Type | Required | Meaning |
| --- | --- | --- | --- |
| `model` | string/model descriptor | Yes | Model identity or explicit manifest source |
| `quantization` | `"q4" \| "q5"` | Yes initially | Required packed-weight layout |
| `backend` | `"auto" \| "webgpu" \| "wasm"` | No | Backend policy; intended default is `"auto"` |

The library should not mutate caller-owned configuration. It resolves options into an immutable,
fully explicit internal configuration before model loading.

## Model identity

A short model name is resolved through a package-defined catalog only when that catalog is part of
the current release. Advanced callers may eventually supply an explicit manifest URL/bytes and
artifact resolver. Custom resolvers still pass through the same schema, bounds, version, and
integrity validation.

## Backend behavior

- `auto` may fall back from WebGPU to WASM only when semantics remain supported.
- `webgpu` never silently falls back; it returns a capability error.
- `wasm` does not initialize WebGPU merely because it is available.

See [backends](../architecture/backends.md) for selection rules.

## Future options

Cache policy, progress, cancellation, language hints, decode limits, sampling, timestamps, and
streaming parameters require separate contracts. They are intentionally omitted here until their
ownership, defaults, bounds, and compatibility semantics are designed.

