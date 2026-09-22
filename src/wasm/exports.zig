//! The `qw_*` exports: the only surface the browser sees.
//!
//! Every export validates its arguments before touching memory, converts core
//! errors into `abi.Status` codes, and never lets a routine failure become a
//! trap. A trap from this module means JavaScript broke the ABI contract or a
//! core invariant was violated -- not that the user's audio was odd.
//!
//! Memory ownership is explicit and split:
//!
//!   * JavaScript allocates every buffer it passes in with `qw_alloc` and frees
//!     it with `qw_free`. The runtime retains no caller pointer except the
//!     vocabulary it decodes with, which stays valid until `qw_tokenizer_set`
//!     replaces it or `qw_destroy` runs.
//!   * The runtime allocates nothing per call. `qw_mel_compute` writes straight
//!     into the caller's output buffer.
//!
//! v1 exposes a single runtime instance, which is why the state below is flat
//! rather than a heap-allocated object. `qw_create` still returns a handle so
//! that multi-instance support would not require an ABI break.

const std = @import("std");
const qw = @import("qwenscriber");
const abi = @import("abi.zig");
const model_mod = @import("model.zig");

/// WebAssembly pages are 64 KiB.
const page_bytes: u64 = 65536;

const allocator = std.heap.wasm_allocator;
const mel = qw.mel;
const model_config = qw.model_config;
const tokenizer = qw.tokenizer;
const selftest = qw.selftest;

/// Handle of the single instance, or zero when it does not exist.
var created_handle: u32 = 0;

/// A view over memory JavaScript owns; never freed by the runtime.
var vocabulary: tokenizer.TokenTable = .{ .offsets = &.{}, .bytes = &.{}, .count = 0 };
var has_vocabulary: bool = false;

/// The model this instance is running, if any. Its shard bytes are JavaScript's
/// and stay resident in linear memory; the model only borrows them.
var model: model_mod.Model = .{};

// ---------------------------------------------------------------------------
// Version and lifecycle
// ---------------------------------------------------------------------------

export fn qw_abi_version() u32 {
    return abi.version;
}

export fn qw_core_version() u32 {
    return qw.core_version;
}

/// Creates the runtime instance and returns its handle, or zero if one already
/// exists.
export fn qw_create() u32 {
    if (created_handle != 0) return 0;
    created_handle = 1;
    return created_handle;
}

export fn qw_destroy(handle: u32) i32 {
    if (handle == 0 or handle != created_handle) return abi.Status.invalid_argument.code();
    created_handle = 0;
    has_vocabulary = false;
    vocabulary = .{ .offsets = &.{}, .bytes = &.{}, .count = 0 };
    // Releases a finished model and a partially built one alike: the caller may
    // destroy at any point in the model state machine.
    model.reset();
    return abi.Status.ok.code();
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/// Largest alignment `qw_alloc` will honour. Buffers the runtime touches are
/// never more aligned than a 64-byte cache line, and the ABI stays explicit
/// about the closed set it supports.
pub const alignment_max: u32 = 64;

/// Allocates `size` bytes aligned to `alignment` and returns a linear memory
/// offset, or zero on failure. Zero is never a valid allocation.
///
/// Zig's allocator interface requires a comptime alignment, so the supported
/// set is spelled out rather than passed through.
export fn qw_alloc(size: u32, alignment: u32) u32 {
    if (size == 0) return 0;
    const memory = switch (alignment) {
        1 => allocate(size, 1),
        2 => allocate(size, 2),
        4 => allocate(size, 4),
        8 => allocate(size, 8),
        16 => allocate(size, 16),
        32 => allocate(size, 32),
        64 => allocate(size, 64),
        else => null,
    } orelse return 0;
    const offset = @intFromPtr(memory.ptr);
    if (offset == 0 or offset > std.math.maxInt(u32)) return 0;
    return @intCast(offset);
}

fn allocate(size: u32, comptime alignment_bytes: usize) ?[]u8 {
    return allocator.alignedAlloc(u8, alignmentFor(alignment_bytes), size) catch null;
}

/// Releases a buffer previously returned by `qw_alloc`. `size` and `alignment`
/// must repeat the allocation exactly, mirroring `free`.
export fn qw_free(ptr: u32, size: u32, alignment: u32) void {
    if (ptr == 0 or size == 0) return;
    switch (alignment) {
        1 => release(ptr, size, 1),
        2 => release(ptr, size, 2),
        4 => release(ptr, size, 4),
        8 => release(ptr, size, 8),
        16 => release(ptr, size, 16),
        32 => release(ptr, size, 32),
        64 => release(ptr, size, 64),
        else => {},
    }
}

fn release(ptr: u32, size: u32, comptime alignment_bytes: usize) void {
    const bytes = regionAt(ptr, size, alignment_bytes) orelse return;
    const aligned: []align(alignment_bytes) u8 = @alignCast(bytes);
    allocator.free(aligned);
}

fn alignmentFor(comptime alignment_bytes: usize) std.mem.Alignment {
    return std.mem.Alignment.fromByteUnits(alignment_bytes);
}

/// Current linear memory size in bytes, so JavaScript can assert before writing.
export fn qw_memory_bytes() u32 {
    const bytes = memoryBytes();
    if (bytes > std.math.maxInt(u32)) return std.math.maxInt(u32);
    return @intCast(bytes);
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export fn qw_error_message_ptr(code: i32) u32 {
    return @intCast(@intFromPtr(abi.statusMessage(statusFromCode(code)).ptr));
}

export fn qw_error_message_len(code: i32) u32 {
    return @intCast(abi.statusMessage(statusFromCode(code)).len);
}

// ---------------------------------------------------------------------------
// Audio preprocessing
// ---------------------------------------------------------------------------

/// Log-mel frames the runtime produces for `sample_count` samples.
export fn qw_mel_frames_for_samples(sample_count: u32) u32 {
    return @intCast(mel.framesForSamples(sample_count));
}

/// Mono samples below which the reference pipeline zero-pads a clip.
export fn qw_mel_min_samples() u32 {
    return @intCast(mel.min_samples);
}

/// Whisper log-mel for one complete clip.
///
/// `samples` are f32 in [-1, 1] at 16 kHz. `out` must hold
/// `mel_bins * qw_mel_frames_for_samples(sample_count)` f32 values, written row
/// major as `[mel_bin][frame]`. `result` receives an `abi.MelResult`.
export fn qw_mel_compute(
    handle: u32,
    samples_ptr: u32,
    sample_count: u32,
    out_ptr: u32,
    out_bytes: u32,
    result_ptr: u32,
) i32 {
    if (requireInstance(handle)) |status| return status;
    if (sample_count == 0) return abi.Status.invalid_argument.code();
    if (out_bytes % 4 != 0) return abi.Status.invalid_argument.code();

    const samples = f32Region(samples_ptr, sample_count) orelse
        return abi.Status.invalid_argument.code();
    const out = f32Region(out_ptr, out_bytes / 4) orelse
        return abi.Status.invalid_argument.code();
    const result = structRegion(abi.MelResult, result_ptr) orelse
        return abi.Status.invalid_argument.code();

    const frames = mel.framesForSamples(samples.len);
    if (frames > mel.capacity_frames) return abi.Status.audio_too_long.code();
    const needed = @as(usize, mel.mel_bins) * frames;
    if (out.len < needed) return abi.Status.shape_mismatch.code();

    const global_max_log = mel.compute(samples, out[0..needed], frames) catch |err| {
        return abi.statusFromError(err).code();
    };
    result.frames_written = @intCast(frames);
    result.bytes_written = @intCast(needed * 4);
    result.global_max_log = global_max_log;
    result.reserved = 0;
    return abi.Status.ok.code();
}

/// Log-mel value the reference produces for zero-padded audio, so a partially
/// filled encoder chunk matches the reference's mel padding.
export fn qw_mel_padding_value(global_max_log: f32) f32 {
    return mel.paddingFrameValue(global_max_log);
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Points the runtime at a vocabulary. The caller keeps that memory alive.
export fn qw_tokenizer_set(handle: u32, descriptor_ptr: u32) i32 {
    if (requireInstance(handle)) |status| return status;
    const descriptor = structRegion(abi.TokenizerDescriptor, descriptor_ptr) orelse
        return abi.Status.invalid_argument.code();
    if (descriptor.reserved != 0) return abi.Status.invalid_argument.code();
    if (descriptor.token_count == 0) return abi.Status.invalid_argument.code();

    const offsets_len = std.math.mul(u64, descriptor.token_count + 1, 4) catch
        return abi.Status.limit_exceeded.code();
    if (descriptor.offsets_len != offsets_len) return abi.Status.shape_mismatch.code();

    const offsets_memory = regionAt(descriptor.offsets_ptr, descriptor.offsets_len, 4) orelse
        return abi.Status.invalid_argument.code();
    const bytes_memory = regionAt(descriptor.bytes_ptr, descriptor.bytes_len, 1) orelse
        return abi.Status.invalid_argument.code();
    if (bytes_memory.len == 0) return abi.Status.invalid_argument.code();

    const table = tokenizer.TokenTable{
        .offsets = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(offsets_memory))),
        .bytes = bytes_memory,
        .count = descriptor.token_count,
    };
    table.validate() catch |err| return abi.statusFromError(err).code();

    vocabulary = table;
    has_vocabulary = true;
    return abi.Status.ok.code();
}

/// Decodes token ids into UTF-8. `written_ptr` receives the byte count.
export fn qw_detokenize(
    handle: u32,
    ids_ptr: u32,
    id_count: u32,
    out_ptr: u32,
    out_bytes: u32,
    written_ptr: u32,
) i32 {
    if (requireInstance(handle)) |status| return status;
    if (!has_vocabulary) return abi.Status.invalid_state.code();

    const ids = u32Region(ids_ptr, id_count) orelse return abi.Status.invalid_argument.code();
    const out = regionAt(out_ptr, out_bytes, 1) orelse return abi.Status.invalid_argument.code();
    const written_out = structRegion(u32, written_ptr) orelse
        return abi.Status.invalid_argument.code();

    const written = tokenizer.detokenize(&vocabulary, ids, out) catch |err| {
        return abi.statusFromError(err).code();
    };
    written_out.* = @intCast(written);
    return abi.Status.ok.code();
}

// ---------------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------------

/// Runs the core self-test and writes an `abi.SelfTestResult`.
///
/// Returns success even when individual checks fail: inspect `failures` in the
/// result. A failure means this build of the module does not compute what the
/// host build computes, which invalidates its use as the WebGPU oracle.
export fn qw_selftest(result_ptr: u32) i32 {
    const result = structRegion(abi.SelfTestResult, result_ptr) orelse
        return abi.Status.invalid_argument.code();
    const report = selftest.run();
    result.failures = report.failures;
    result.loudest_band = report.loudest_band;
    result.silence_value = report.silence_value;
    result.reserved = 0;
    result.quant_hash = report.quant_hash;
    result.mel_hash = report.mel_hash;
    return abi.Status.ok.code();
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

/// The capability families this build compiles in, as `abi.Feature` bits.
///
/// JavaScript asks this before it downloads a model or calls into a family:
/// a module built from an older source of the same ABI version reports fewer
/// bits, and the SDK turns that into a typed `not_implemented` rather than
/// calling an export that is not there.
export fn qw_features() u32 {
    return abi.features;
}

/// Starts a model attempt, releasing whatever the handle holds.
///
/// The arena holds the runtime's allocations for one model: tensor bindings,
/// layer tables, prompt tokens, and every scratch buffer, including the
/// key/value cache. The weights are not copied into it -- they stay in the
/// caller's shard buffers.
///
/// This is also the unload path. Calling it again releases a finished model, or
/// discards a partial one, without destroying the instance: the log-mel
/// frontend, the tokenizer, and the self-test stay usable either way.
export fn qw_model_begin(handle: u32) i32 {
    if (requireInstance(handle)) |status| return status;
    model.begin();
    return abi.Status.ok.code();
}

/// Parses one `.qw` shard in place and keeps a view over it.
///
/// `shard_ptr` must be `shard_len` bytes of a complete shard, starting on a
/// 16-byte boundary -- what `qw_alloc(size, 16)` returns -- and must stay
/// resident and unmodified until `qw_destroy`: the weight tensors point into
/// it. Magic, version, index ordering, tensor shapes, byte lengths, and the
/// payload checksum are all validated here.
export fn qw_model_add_shard(handle: u32, shard_ptr: u32, shard_len: u32) i32 {
    if (requireInstance(handle)) |status| return status;
    const bytes = regionAt(shard_ptr, shard_len, 16) orelse
        return abi.Status.invalid_argument.code();
    const shard: []align(16) const u8 = @alignCast(bytes);
    model.addShard(shard) catch |err| return abi.statusFromError(err).code();
    return abi.Status.ok.code();
}

/// Parses `config.bin`, resolves every tensor, and prepares the decoder.
///
/// Returns once the model is usable. A failure releases the arena, so a model
/// that does not fit leaves no partial allocation behind and the call may be
/// Chooses the key/value cache width the next `qw_model_finish` loads.
///
/// The cache is allocated during the load, so this is refused once a model exists: a format chosen
/// afterwards would describe a model nobody built. `qw_model_requirements` reports what the chosen
/// width costs, which is how a caller budgets for an instance whose memory is capped — the cache is
/// a quarter of its f32 size at `q8` and decodes at the same speed.
export fn qw_model_set_cache_format(handle: u32, format: u32) i32 {
    if (requireInstance(handle)) |status| return status;
    const requested = std.enums.fromInt(abi.CacheFormat, format) orelse
        return abi.Status.invalid_argument.code();
    const chosen: qw.model_config.CacheFormat = switch (requested) {
        .full_precision => .f32,
        .q8 => .q8,
        else => return abi.Status.invalid_argument.code(),
    };
    model.setCacheFormat(chosen) catch return abi.Status.invalid_state.code();
    return abi.Status.ok.code();
}

/// retried after the caller removes a shard or supplies a smaller budget.
export fn qw_model_finish(handle: u32, config_ptr: u32, config_len: u32) i32 {
    if (requireInstance(handle)) |status| return status;
    if (model.state != .ready) return abi.Status.invalid_state.code();
    const config_bytes = regionAt(config_ptr, config_len, 4) orelse
        return abi.Status.invalid_argument.code();
    const config = model_config.parse(config_bytes) catch |err| {
        return abi.statusFromError(err).code();
    };
    model.finish(config) catch |err| return abi.statusFromError(err).code();
    return abi.Status.ok.code();
}

/// Writes an `abi.ModelRequirements` describing the loaded model.
///
/// This is the measurement a caller needs to answer "does this model fit here":
/// the weight bytes it must keep resident, the key/value cache, the scratch, the
/// total, and the limits the model decodes within.
export fn qw_model_requirements(handle: u32, out_ptr: u32) i32 {
    if (requireInstance(handle)) |status| return status;
    const out = structRegion(abi.ModelRequirements, out_ptr) orelse
        return abi.Status.invalid_argument.code();
    model.requirements(out) catch |err| return abi.statusFromError(err).code();
    return abi.Status.ok.code();
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Encodes one clip, resets the decoder, prefills the prompt, and holds the
/// first predicted token.
///
/// `features_ptr` is the `qw_mel_compute` output for this clip: `mel_bins x
/// frames` f32 values, row major, `features_len` bytes. The features are read
/// during this call and never retained. `max_tokens` bounds the generated
/// sequence for this utterance and may not exceed the configuration's
/// `max_decode_tokens`; the vocabulary must be installed, because the prompt is
/// built from it.
export fn qw_decode_begin(
    handle: u32,
    features_ptr: u32,
    features_len: u32,
    max_tokens: u32,
) i32 {
    if (requireInstance(handle)) |status| return status;
    // The prompt's ordinary words come from the vocabulary, so decoding without
    // one is a state error rather than a bad argument.
    if (!has_vocabulary) return abi.Status.invalid_state.code();
    if (features_len == 0) return abi.Status.invalid_argument.code();
    if (features_len % 4 != 0) return abi.Status.invalid_argument.code();
    const features = f32Region(features_ptr, features_len / 4) orelse
        return abi.Status.invalid_argument.code();
    model.decodeBegin(&vocabulary, features, max_tokens) catch |err| {
        return abi.statusFromError(err).code();
    };
    return abi.Status.ok.code();
}

/// Writes the next produced token id to `token_out_ptr` and returns 1, or
/// writes nothing and returns 0 when the sequence has finished -- an
/// end-of-sequence token, or the `max_tokens` budget. Any other return value is
/// a negative status code.
export fn qw_decode_step(handle: u32, token_out_ptr: u32) i32 {
    if (requireInstance(handle)) |status| return status;
    const token_out = structRegion(u32, token_out_ptr) orelse
        return abi.Status.invalid_argument.code();
    const step = model.decodeStep(token_out) catch |err| {
        return abi.statusFromError(err).code();
    };
    return switch (step) {
        .token => 1,
        .finished => 0,
    };
}

/// Copies the token ids produced so far and returns how many were written, so
/// the caller can detokenize them with `qw_detokenize`.
///
/// `out_capacity_tokens` must hold every id produced: a truncating copy would
/// silently shorten the transcript, so a buffer that is too small is reported
/// as `limit_exceeded`.
export fn qw_decode_tokens(handle: u32, out_ptr: u32, out_capacity_tokens: u32) i32 {
    if (requireInstance(handle)) |status| return status;
    const out = u32Region(out_ptr, out_capacity_tokens) orelse
        return abi.Status.invalid_argument.code();
    const count = model.decodeTokens(out) catch |err| return abi.statusFromError(err).code();
    return @intCast(count);
}

/// Ends the utterance: the key/value cache goes back to empty and the model
/// stays resident for the next call.
export fn qw_decode_end(handle: u32) i32 {
    if (requireInstance(handle)) |status| return status;
    model.decodeEnd() catch |err| return abi.statusFromError(err).code();
    return abi.Status.ok.code();
}

// ---------------------------------------------------------------------------
// Argument handling
// ---------------------------------------------------------------------------

/// Returns a status code when the handle is not usable, null when it is.
fn requireInstance(handle: u32) ?i32 {
    if (handle == 0 or handle != created_handle) return abi.Status.invalid_argument.code();
    return null;
}

fn statusFromCode(code: i32) abi.Status {
    inline for (@typeInfo(abi.Status).@"enum".field_names) |field_name| {
        const status: abi.Status = @field(abi.Status, field_name);
        if (status.code() == code) return status;
    }
    return .invalid_argument;
}

fn memoryBytes() u64 {
    return @as(u64, @wasmMemorySize(0)) * page_bytes;
}

fn alignmentIsSupported(alignment: u32) bool {
    if (alignment == 0) return false;
    if (!std.math.isPowerOfTwo(alignment)) return false;
    return alignment <= alignment_max;
}

/// Validates that `ptr .. ptr + len` lies inside linear memory at the requested
/// alignment, and returns it as a mutable slice.
fn regionAt(ptr: u32, len: u32, alignment: u32) ?[]u8 {
    if (!alignmentIsSupported(alignment)) return null;
    if (ptr % alignment != 0) return null;
    const end = @as(u64, ptr) + @as(u64, len);
    if (end > memoryBytes()) return null;
    if (end > std.math.maxInt(u32)) return null;
    if (len == 0) return &.{};
    const raw: [*]u8 = @ptrFromInt(ptr);
    return raw[0..len];
}

fn f32Region(ptr: u32, count: u32) ?[]f32 {
    return sliceRegion(f32, ptr, count);
}

fn u32Region(ptr: u32, count: u32) ?[]u32 {
    return sliceRegion(u32, ptr, count);
}

fn sliceRegion(comptime T: type, ptr: u32, count: u32) ?[]T {
    const len = std.math.mul(u64, count, @sizeOf(T)) catch return null;
    if (len > std.math.maxInt(u32)) return null;
    const bytes = regionAt(ptr, @intCast(len), @alignOf(T)) orelse return null;
    if (bytes.len == 0) return &.{};
    const aligned: []align(@alignOf(T)) u8 = @alignCast(bytes);
    return std.mem.bytesAsSlice(T, aligned);
}

fn structRegion(comptime T: type, ptr: u32) ?*T {
    const bytes = regionAt(ptr, @sizeOf(T), @alignOf(T)) orelse return null;
    if (bytes.len != @sizeOf(T)) return null;
    const aligned: []align(@alignOf(T)) u8 = @alignCast(bytes);
    return @ptrCast(aligned.ptr);
}
