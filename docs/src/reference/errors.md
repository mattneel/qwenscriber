# Errors and traps

Qwenscriber separates expected operating/input failures from violated implementation invariants.

## Expected errors

Expected errors include:

- unsupported ABI, model-format, architecture, quantization, or backend version;
- malformed or out-of-bounds manifest/tensor metadata;
- checksum mismatch or incomplete cache entry;
- unavailable WebGPU/WASM feature or inadequate adapter/runtime limit;
- invalid audio format, duration, channel count, or sample rate;
- insufficient caller-provided output memory;
- invalid, stale, wrong-type, or wrong-state handle;
- cancellation, device loss, download failure, and storage quota failure; and
- bounded decode/resource limits being reached.

They return stable machine-readable codes. Host wrappers attach operation context and preserve the
underlying cause where meaningful.

## Traps and assertions

A trap/assertion indicates a bug or invariant failure: internal overflow after validated bounds,
impossible state transition, corrupted ownership bookkeeping, mismatched compile-time layout, or
memory corruption. Applications are not expected to recover a corrupted instance in place.

Do not turn hostile input into an assertion by validating too late.

## Error stability

Callers may branch on documented error codes/classes, never on prose. New specific codes may be
added compatibly within a documented category; changing the meaning of an existing stable code is a
breaking change.

Diagnostics can include bounded detail for humans, but must not leak transcript/audio/model bytes or
unbounded attacker-controlled strings by default.

