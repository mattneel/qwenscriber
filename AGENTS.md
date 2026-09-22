# AGENTS.md

## Qwenscriber

Qwenscriber is a small, high-performance, local-first speech-to-text runtime for
Qwen3-ASR.

The bring-up target is Qwen3-ASR 0.6B.

The primary production target is Qwen3-ASR 1.7B running entirely client-side in
modern browsers using:

- Zig compiled to `wasm32-freestanding`.
- WASM SIMD for CPU-side work.
- WebGPU/WGSL for large tensor operations.
- TypeScript for browser orchestration and the public SDK.

The intended stack is deliberately small:

    TypeScript
        |
        +-- WebGPU -> WGSL
        |
        +-- WASM -> Zig -> WASM SIMD

No server-side inference is required.

Audio should not need to leave the user's machine.

---

## Read This First

Before modifying code, read:

    docs/TIGER_STYLE.md

TigerStyle is the coding and engineering style for this repository.

Do not skim it.

Its principles apply to Zig, TypeScript, WGSL, build tooling, tests, model
conversion code, and architecture.

Qwenscriber-specific constraints in this file supplement TigerStyle. If a
project-specific requirement here conflicts directly with a generic rule in
TigerStyle, follow this file, but preserve the intent of TigerStyle wherever
possible.

The priority order is:

1. Safety.
2. Performance.
3. Developer experience.

Simplicity serves those goals; it does not override them.

---

## Working Method

Understand before abstracting.

Measure before optimizing.

Prefer a small experiment over architectural speculation.

Work in vertical slices toward real transcription rather than accumulating
framework code.

The first meaningful milestone is:

    real Qwen3-ASR audio
        -> Qwenscriber preprocessing
        -> Qwenscriber inference
        -> decoded text

using Qwen3-ASR 0.6B.

After correctness, generalize and optimize toward 1.7B.

Do not declare success because infrastructure exists.

---

## Repository State

Assume the repository may be under active development.

Before changing anything:

1. Run `git status`.
2. Inspect the relevant files.
3. Read nearby tests.
4. Understand existing conventions.
5. Preserve user changes.

Do not overwrite, revert, or "clean up" unrelated work.

Do not create commits unless explicitly asked.

Run `git status` again before finishing.

---

## Zig Version

This project tracks current Zig master.

Do not assume APIs from a historical stable Zig release.

Before making build-system or standard-library assumptions, inspect:

    zig version

Prefer the compiler and documentation actually present in the development
environment over stale examples from the internet.

Run `zig fmt` on Zig sources.

---

## Zig Runtime

The browser core targets:

    wasm32-freestanding

It must not require:

- WASI.
- Emscripten.
- libc.
- Node.js.
- A JS runtime embedded into the WASM module.

Keep the WASM module genuinely freestanding.

Use Zig-native facilities wherever practical.

CPU hot paths should be designed so Zig can emit effective WASM SIMD.

Prefer idiomatic Zig `@Vector` implementations before reaching for handwritten
WASM intrinsics.

Inspect generated code and benchmark before introducing lower-level machinery.

---

## Runtime Responsibilities

Keep ownership boundaries sharp.

### TypeScript owns

- Public SDK.
- Browser feature detection.
- Workers.
- AudioWorklet integration.
- Network fetching.
- IndexedDB model caching.
- WebGPU adapter/device acquisition.
- GPU resource lifetime.
- GPUBuffer creation.
- Pipeline creation.
- Bind groups.
- Dispatch scheduling.
- Model shard loading.
- High-level inference orchestration.

### Zig/WASM owns

- Stable low-level ABI.
- Audio sample conversion.
- Resampling.
- Log-mel preprocessing.
- Normalization.
- Tokenization and detokenization where appropriate.
- Model metadata parsing.
- Quantization metadata.
- Decode state.
- KV-cache bookkeeping.
- Greedy/sampling decisions.
- CPU reference kernels.
- WASM SIMD fallback kernels.
- Correctness/reference implementations for GPU work.

### WGSL/WebGPU owns

- Large matrix operations.
- Fused dequantization + matmul.
- Attention.
- RMSNorm.
- RoPE.
- Activations.
- Convolution required by the audio encoder.
- Elementwise tensor operations.
- Encoder execution.
- Decoder tensor execution.

Do not route every tensor operation through the WASM/JS boundary.

---

## WebGPU

WebGPU is the primary high-performance inference backend.

Weights should generally flow:

    network/cache
        -> TypeScript
        -> GPUBuffer
        -> remain GPU-resident

Avoid repeated CPU/GPU movement.

Read back only compact results when practical.

Do not assume support for a single giant model buffer.

Inspect adapter/device limits, including:

    maxBufferSize
    maxStorageBufferBindingSize

Design weight storage around sharding from the beginning.

Prefer layer-oriented or otherwise naturally bounded GPU resources.

---

## WASM Backend

Support the conceptual backend choices:

    auto
    webgpu
    wasm

`auto` should select WebGPU when supported and suitable, otherwise use the WASM
fallback.

The WASM implementation serves three purposes:

1. Portable fallback.
2. Correctness reference.
3. Test oracle for WebGPU kernels.

Do not distort the WebGPU architecture merely to optimize 1.7B CPU inference.

The 0.6B WASM path should still receive serious SIMD optimization where useful.

---

## WASM ABI

Keep the JS/WASM ABI:

- Small.
- Explicit.
- Versioned.
- Stable.
- C-like.

Never expose accidental Zig ABI details.

Do not export:

- Zig slices.
- Zig error unions.
- Zig optionals.
- Layout-sensitive Zig structs.

Prefer:

- Fixed-width integers.
- Integer handles.
- Linear-memory offsets.
- Explicit lengths.
- Explicit enum values.
- Documented byte layouts.
- Machine-readable error codes.

The ABI should have an explicit version from the beginning.

Conceptually:

    qw_version()

    qw_init(...)
    qw_deinit(...)

    qw_alloc(...)
    qw_free(...)

    qw_audio_push(...)
    qw_preprocess(...)

    qw_model_parse(...)
    qw_tensor_descriptor(...)

    qw_decode_begin(...)
    qw_decode_step(...)
    qw_decode_end(...)

The exact API may evolve. Keep it minimal.

Routine invalid input must return a useful error rather than trap.

A WASM trap should indicate a programmer error or violated invariant.

---

## Memory

Browser memory is a first-class design constraint.

Avoid unnecessary copies.

In particular, resist pipelines like:

    network
        -> JS copy
        -> WASM copy
        -> JS copy
        -> GPU copy

Prefer direct movement where browser APIs permit it.

Separate:

- Long-lived weights.
- Persistent inference state.
- KV cache.
- Reusable activation buffers.
- Short-lived scratch memory.

Reuse fixed-capacity memory when practical.

Follow TigerStyle's preference for bounded and initialization-time allocation in
long-lived runtime paths.

Know the maximum size of queues, buffers, loops, tensors, token sequences,
audio windows, shards, and caches.

Put a limit on everything.

---

## Integer Types

Use explicitly sized integer types for persisted formats, ABI structures,
serialized metadata, tensor dimensions, indexes where bounds are known, and
cross-language interfaces.

Do not casually leak `usize` into persistent or external interfaces.

Distinguish semantically between:

- Index.
- Count.
- Byte size.
- Element size.
- Offset.

Names should make units obvious.

Examples:

    tensor_count
    tensor_index
    shard_size_bytes
    shard_offset_bytes
    latency_ms_max

---

## Assertions

Assertions are expected.

Assert:

- Preconditions.
- Postconditions.
- Bounds.
- Shape relationships.
- Tensor sizes.
- Alignment.
- Quantization block invariants.
- ABI invariants.
- Compile-time constants.
- State transitions.
- Impossible enum values.
- Relationships between serialized offsets and lengths.

Prefer separate assertions:

    assert(a);
    assert(b);

over:

    assert(a and b);

Test both valid and invalid spaces.

Expected operational failures must be handled as errors, not assertions.

---

## Control Flow

Use explicit, bounded control flow.

No recursion in runtime code.

Avoid clever control flow.

Loops must have understandable upper bounds.

Keep branching centralized.

Prefer:

    push `if`s up
    push `for`s down

Parent functions should own control flow and state changes.

Leaf helpers should preferably compute rather than mutate.

Functions have a hard maximum of 70 lines.

If a function grows beyond that, find the correct conceptual split rather than
mechanically slicing it.

---

## Naming

Use TigerStyle naming.

In particular:

- `snake_case` for Zig functions, variables, and files.
- Descriptive names over abbreviations.
- Proper capitalization for acronyms.
- Units and qualifiers at the end.
- Nouns for concepts where practical.
- Names that reveal ownership and lifetime.

Avoid vague names such as:

    data
    thing
    object
    tmp
    ctx

unless the scope makes their meaning genuinely obvious.

Choose terminology once and use it consistently across Zig, WGSL, TypeScript,
the model format, tests, and documentation.

---

## Dependencies

Dependency austerity is a core design constraint.

The browser runtime should preferably have zero production dependencies.

Do not add:

- ONNX Runtime.
- TensorFlow.
- PyTorch.
- llama.cpp as a runtime dependency.
- ggml as a runtime dependency.
- Emscripten.
- Large JavaScript ML frameworks.

Native dependencies must be one of:

1. Pure Zig.
2. A Zig-native wrapper around C/C++.
3. A C/C++ library consumed through Zig when justified.
4. Our own narrow Zig wrapper around such a library.

Before adding any dependency, explain why implementing the required subset
ourselves is worse.

Existing runtimes may be studied for:

- Algorithms.
- Tensor layouts.
- Quantization.
- Model semantics.
- Correctness comparison.

Do not turn them into hidden architectural dependencies.

---

## Qwen3-ASR

Implement the real Qwen3-ASR architecture.

Do not treat it as merely a generic Qwen text model accepting arbitrary audio
embeddings.

Inspect official model configuration and source artifacts.

The implementation must correctly account for:

- Audio frontend.
- Audio encoder.
- Projection/adapter layers.
- Decoder architecture.
- Tokenization.
- Generation semantics.
- Special tokens.
- Context handling.
- Timestamp/alignment behavior when implemented.

Bring up 0.6B first.

Do not bake 0.6B-specific tensor dimensions throughout the runtime.

1.7B compatibility is an architectural constraint from the beginning.

Model-specific dimensions belong in validated configuration/metadata.

---

## Model Format

Qwenscriber may use its own browser-oriented model format.

GGUF may be supported as an import/conversion source, but GGUF compatibility is
not an architectural requirement.

The distribution format should support:

- Architecture/version metadata.
- Model variant.
- Tokenizer metadata.
- Tensor names.
- Tensor shapes.
- Tensor data types.
- Quantization format.
- Quantization block size.
- Shard identifier.
- Byte offset.
- Byte length.
- Alignment.
- Integrity checks where useful.

Optimize for:

- CDN delivery.
- Range requests where useful.
- IndexedDB caching.
- Incremental loading.
- GPU upload.
- Browser memory pressure.

Serialized formats must be versioned.

Parsing must reject malformed or unsupported input cleanly.

---

## Quantization

Primary browser targets are expected to include Q4 and Q5-class formats.

The desired data path is:

    packed quantized weights
        -> WGSL unpack/dequant
        -> multiply
        -> accumulation

Avoid materializing complete FP16 copies of quantized weights merely to perform
matmul.

Quantization formats should be designed around measured WebGPU behavior.

Do not choose block sizes or layouts because they look elegant.

Benchmark them.

Conversion and runtime definitions must share format/layout definitions where
possible to prevent drift.

---

## TypeScript

Keep the public SDK small and unsurprising.

Target an API roughly like:

    import { Qwenscriber } from "@qwenscriber/qwenscriber";

    const asr = await Qwenscriber.create({
        model: "qwen3-asr-1.7b",
        quantization: "q4",
        backend: "auto",
    });

    const result = await asr.transcribe(audio);

    asr.dispose();

Eventually:

    for await (const segment of asr.stream(microphone)) {
        console.log(segment.text);
    }

Provide capability inspection for things such as:

- WebGPU.
- WASM SIMD.
- Worker support.
- SharedArrayBuffer.
- Adapter limits.
- Selected backend.

Do not expose internal GPU machinery through the normal SDK unless it is
required for an advanced low-level API.

Avoid production npm dependencies unless clearly justified.

---

## Browser Execution

Inference must not block the UI thread.

Prefer:

    main thread
        -> Worker
            -> WASM
            -> WebGPU

Use AudioWorklet for realtime capture when appropriate.

Do not require SharedArrayBuffer for basic offline transcription unless there is
a compelling technical reason.

Streaming may use SharedArrayBuffer when it materially improves the design.

Always provide explicit bounds on streaming queues and buffered audio.

---

## WGSL

Treat shaders as production source code.

Keep kernels:

- Small.
- Focused.
- Explicit.
- Testable independently.

Avoid shader metaprogramming complexity unless measurement justifies it.

Document memory layouts shared between Zig, TypeScript, and WGSL.

Changes to a shared layout must include corresponding tests.

GPU kernels must be checked against deterministic CPU reference implementations
before being trusted as part of full-model inference.

---

## Correctness Strategy

Do not debug the entire model as one black box.

Build upward.

For each important kernel:

1. Generate a small deterministic input.
2. Compute expected output using straightforward Zig reference code.
3. Execute the WGSL implementation.
4. Read the result back.
5. Compare using justified tolerances.

Apply this to at least:

- Matmul.
- Quantized matmul.
- Dequantization.
- RMSNorm.
- RoPE.
- Attention.
- Activations.
- Convolution.
- Encoder blocks.
- Decoder blocks.

Then compare progressively larger model fragments against a trusted reference.

---

## Tests

Tests are part of implementation, not cleanup.

Maintain tests for:

- ABI version/layout.
- Manifest parsing.
- Invalid manifests.
- Audio conversion.
- Resampling.
- Log-mel preprocessing.
- Quantization.
- Dequantization.
- Tokenization.
- Shape validation.
- Tensor indexing.
- Reference math.
- State transitions.
- Bounds.
- Error paths.

Browser/WebGPU tests should use small deterministic fixtures.

Do not commit full model checkpoints to the repository.

Test negative space, not just happy paths.

---

## Performance

Performance starts at design time.

Before implementing large subsystems, sketch expected costs in terms of:

- Network bandwidth/latency.
- Storage bandwidth/latency.
- Memory bandwidth/capacity.
- CPU.
- GPU.
- CPU/GPU synchronization.

Pay particular attention to model download size and memory bandwidth.

Batch work where possible.

Avoid repeated small GPU submissions when larger predictable batches are
possible.

Avoid unnecessary synchronization and readback.

Hot Zig loops should be isolated into simple functions with primitive arguments
when doing so improves optimization and inspectability.

Do not claim performance improvements without measurement.

---

## Benchmarks

Maintain hooks to measure:

- Model download/load time.
- Cached load time.
- Model size.
- GPU upload time.
- Audio preprocessing throughput.
- First-token latency.
- Decode tokens/second.
- Real-time factor.
- Individual GPU kernel timings.
- WASM memory usage.
- Approximate GPU memory usage.

Record enough environment information for numbers to mean something.

Performance regressions should be explainable.

---

## Tooling

Prefer Zig for repository tooling where practical.

A Zig tool is generally preferable to another shell/Python dependency for
permanent project infrastructure.

Temporary scripts used for reference validation are acceptable when they
materially accelerate bring-up.

Do not make Python a production runtime requirement.

Model conversion should trend toward a Zig executable, conceptually:

    qwenscriber-convert \
        --input <checkpoint> \
        --output <model> \
        --quant q4

The converter should eventually:

- Read official metadata.
- Validate architecture.
- Map tensor names.
- Transpose/repack tensors.
- Quantize.
- Shard.
- Generate manifests.
- Generate integrity metadata.
- Validate emitted tensors.

---

## Build

The Zig build is authoritative for Zig artifacts.

Expected commands should remain simple, ideally converging on:

    zig build
    zig build test
    zig build wasm
    zig build test-wasm

TypeScript/browser tooling may have its own commands where necessary.

Keep the toolchain small.

The WASM output must be deterministic enough to package reliably.

Explicitly configure the intended WASM feature set rather than depending on
accidental compiler defaults.

---

## Formatting

For Zig:

    zig fmt

Hard limit source lines to 100 columns.

Use 4 spaces where formatting is not automatically controlled.

Follow equivalent discipline in TypeScript and WGSL.

Do not disable formatting rules simply to accommodate awkward code. Improve the
code shape.

---

## Comments

Comments explain why and how.

Do not write comments that merely translate the next line into English.

Write full sentences.

Explain:

- Non-obvious invariants.
- Architectural constraints.
- Numeric constants.
- Memory layouts.
- Browser quirks.
- Quantization choices.
- Performance tradeoffs.
- Workarounds.

A surprising assertion can sometimes document an invariant better than a
comment.

---

## Documentation

Keep documentation truthful.

Clearly distinguish:

- Implemented.
- Experimental.
- Partial.
- Planned.

Do not advertise planned functionality as working.

Update documentation when changing:

- Public API.
- Model format.
- ABI.
- Browser requirements.
- Build steps.
- Supported models.
- Supported quantization.
- Backend behavior.

---

## Generated and Large Files

Do not commit:

- Downloaded checkpoints.
- Converted full model weights.
- Browser caches.
- `node_modules`.
- Zig cache artifacts.
- Large benchmark outputs.
- Temporary tensor dumps.
- Generated scratch data.

Maintain `.gitignore` accordingly.

Small deterministic fixtures needed for testing are acceptable.

---

## External Research

Prefer primary sources:

- Official Qwen/Qwen3-ASR repositories and model files.
- Zig documentation and source.
- WebGPU specification.
- WGSL specification.
- Browser vendor documentation.

When uncertain about the model architecture or an API, research it.

Do not invent details.

When two sources disagree, identify the disagreement and verify empirically
where practical.

---

## Finishing a Change

Before considering work complete:

1. Run formatting.
2. Run relevant unit tests.
3. Run the broader test suite when practical.
4. Build affected targets.
5. Check generated artifacts where relevant.
6. Check `git diff`.
7. Check `git status`.
8. Confirm no unrelated user changes were modified.
9. Confirm documentation still describes reality.

For performance-sensitive changes, run the relevant benchmark.

For GPU changes, compare against the CPU reference implementation.

For model-format or ABI changes, verify backward/version handling explicitly.

---

## Standard

Do the hard thinking early.

Prefer deleting an abstraction to explaining why it is necessary.

Prefer bounded data structures to open-ended ones.

Prefer an explicit state machine to implicit state.

Prefer known memory use to convenient allocation.

Prefer a boring ABI to a clever binding layer.

Prefer one well-understood kernel to three generic frameworks.

Prefer measured performance to assumed performance.

Prefer working end-to-end transcription to architectural pageantry.

Keep it small.
Keep it fast.
Keep it correct.
