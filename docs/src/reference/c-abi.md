# C ABI

The C ABI is a versioned integration boundary shared conceptually by native FFI and the raw WASM
exports. This page defines invariants and function families; the released `qwenscriber.h` defines
exact names, layouts, and calling conventions.

## Required properties

- Fixed-width integer and byte-buffer fields.
- Explicit ABI version negotiation.
- No Zig slices, optionals, error unions, sentinel assumptions, or accidental struct layout.
- Opaque/integer handles with type, generation, and lifecycle validation.
- Pointer plus length for caller memory; offset plus length in WASM linear memory.
- Explicit alignment and byte order for serialized layouts.
- Stable numeric result codes.
- Query-then-write or caller-buffer output contracts; no hidden cross-allocator free.

## Function families

| Family | Conceptual operations |
| --- | --- |
| Version | ABI query, runtime version, supported feature bits |
| Memory | WASM allocation/free where host-to-linear-memory transfer requires it |
| Runtime | Create/destroy, limits, diagnostics, scratch planning |
| Model | Parse/load/release, enumerate tensor descriptors, validate compatibility, report what a model keeps resident and the audio tower's geometry |
| Audio | Push/convert/resample/preprocess bounded PCM ranges |
| Decode | Begin/step/end a decode session; inspect bounded token output |
| Text | Resolve token bytes and assemble validated output |
| Error | Stable result code and bounded contextual detail |

Conceptual early exports include `qw_version`, `qw_init`, `qw_deinit`, `qw_alloc`, `qw_free`,
`qw_audio_push`, `qw_preprocess`, `qw_model_parse`, `qw_tensor_descriptor`, `qw_decode_begin`,
`qw_decode_step`, `qw_decode_end`, and `qw_token_to_text`. These names are not normative until the
header lands.

A family grows in place where it can. `qw_model_set_cache_format` was added to the model family after
the family was in use and needed no version bump: a caller that never calls it gets the behavior it
already had, and the width it chooses is reported back through `qw_model_requirements` rather than
assumed. It is refused once a model is loaded, because the load is what allocates the cache.

`qw_model_audio_config` joined the same family the same way, and for the same reason: a caller that
dispatches the audio tower itself needs its geometry, and a caller that does not dispatch the tower
never asks. It reports `AudioConfig` -- width, layer and head counts, the chunk's frame and step
counts, the bins left after the convolution stack, the projector width, and the LayerNorm epsilon --
with every derived value resolved by the core's own configuration helpers rather than left to the
caller to recompute, so a dispatch cannot drift from the path the reference transcript came from.

## Ownership

Each function documents who owns every input and output, how long borrowed bytes remain valid, and
which operation releases each handle. A buffer allocated by one allocator is never freed by another
without an explicit matching function.

## One ABI, two WASM builds

The ABI is a property of the core's source, not of the toolchain that compiled it. Both sanctioned
builds — the default `wasm32-freestanding` module and the thread-enabled `wasm32-emscripten` module
([ADR-0005](../project/decisions/0005-thread-enabled-wasm-build.md)) — export the same entry points
with the same semantics, and the ABI self-test runs against both.

What differs is instantiation: the freestanding module resolves zero imports and exports its own
memory, while the threaded module imports its host glue and requires a shared memory, which in a
browser means cross-origin isolation. A host loader must therefore supply an import object for the
threaded build and must not assume that a module with imports is a broken artifact.

## Errors versus traps

Malformed input, unsupported versions/features, insufficient caller buffers, resource limits,
invalid state transitions, and device/runtime failures return errors. Integer overflow, internal
memory corruption, or a proven invariant becoming false may trap/crash in assertion-enabled builds.

## Evolution

Append compatible fields only through size/version-aware structs. Reserve no mystery padding whose
meaning depends on compiler layout. Breaking semantics require a new ABI version, parallel entry
points or an explicit compatibility layer, and conformance fixtures for old callers.

