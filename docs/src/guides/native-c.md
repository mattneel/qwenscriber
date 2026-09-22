# Native and C FFI

The native SDK is implemented in Zig and exposes an idiomatic Zig API plus a deliberately small C
ABI. Languages that already support C FFI should bind that ABI instead of embedding WASM or
reimplementing model semantics.

## Integration model

- Zig callers use the native module and Zig types internally.
- C, C++, Rust, Go, Python extensions, and similar runtimes bind `qwenscriber.h`.
- Elixir uses the dedicated Zigler integration so BEAM scheduling and resources remain explicit.
- TypeScript uses WASM by default; Bun may eventually opt into `bun:ffi` when the native path is
  explicitly requested and packaged.

## ABI principles

- Opaque handles hide Zig layout and allocator choices.
- Every byte range is pointer/offset plus explicit length.
- Serialized fields use fixed-width integers and declared byte order.
- Creation and destruction have one clear owner.
- Functions return stable result/error codes rather than Zig error unions.
- Callers can query required output size before providing buffers.
- ABI version and runtime version are separate concepts.
- All entry points validate handle type/state and reject overflow before dereference.

## Illustrative lifecycle

```c
qw_abi_info abi = {0};
qw_result rc = qw_abi_query(QW_ABI_VERSION_1, &abi);

qw_runtime *runtime = NULL;
rc = qw_runtime_create(&config, &runtime);
rc = qw_model_load(runtime, model_bytes, model_bytes_len);
rc = qw_transcribe(runtime, pcm, frame_count, &request);
qw_runtime_destroy(runtime);
```

This is pseudocode. Names and signatures become normative only in the generated/released header.
See the [C ABI reference](../reference/c-abi.md) for function families and compatibility rules.

## Language wrapper rule

A wrapper may adapt errors, memory, strings, and async execution to its host language. It must not
fork tokenizer behavior, manifest parsing, quantization semantics, or decode state. Bindings should
include ABI compatibility tests against the exact native artifacts they distribute.

