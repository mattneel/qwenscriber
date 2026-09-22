//! The model and decode state machine behind the `qw_model_*` and
//! `qw_decode_*` exports.
//!
//! `exports.zig` owns argument validation: it turns linear-memory offsets into
//! slices and typed pointers, and maps this file's errors onto `abi.Status`.
//! This file owns what happens after that: which calls are legal in which
//! state, what the model owns, and when it is released.
//!
//! # The state machine
//!
//!     idle  --qw_model_begin-->  ready  --qw_model_add_shard-->  ready
//!     ready --qw_model_finish--> loaded --qw_decode_begin-->  decoding
//!     decoding --qw_decode_step / qw_decode_tokens--> decoding
//!     decoding --qw_decode_end--> loaded
//!     any   --qw_model_begin-->  ready       (starts over: release, then load again)
//!     any   --qw_destroy-->      (released)
//!
//! A call that does not fit the arrows returns `error.InvalidState`, which
//! `abi.statusFromError` reports as `invalid_state`. Nothing here traps for a
//! routine input problem, including a call in the wrong order.
//!
//! `qw_model_begin` is legal in every state because shards can be added but
//! never removed: starting over is the only way to retry a load that failed,
//! and it doubles as the unload path for a model that is no longer wanted.
//! Everything else is strict, because a silent state reset mid-utterance would
//! turn a caller's bug into a wrong transcript.
//!
//! # Ownership
//!
//! The shard bytes are the caller's. `addShard` parses the container in place
//! and keeps a *view*; the model's weight tensors point straight into those
//! bytes, so the caller must keep them resident and unmodified until
//! `qw_destroy`. Nothing is copied: loading a 426 MB model dir moves 426 MB
//! into linear memory once, and the runtime allocates only its own arithmetic
//! buffers on top.
//!
//! Everything the runtime does allocate -- tensor bindings, layer tables,
//! prompt tokens, the token budget, and every scratch buffer including the
//! key/value cache -- comes from one arena per handle. `decode_end` clears the
//! key/value cache in place and keeps the arena, because the next utterance
//! needs the same buffers; a failed load releases the arena, so a model that
//! does not fit does not leave its partial allocation behind.

const std = @import("std");
const qw = @import("qwenscriber");
const abi = @import("abi.zig");

const assert = std.debug.assert;
const container = qw.container;
const decoder_mod = qw.qwen3_asr.decoder;
const mel = qw.mel;
const model_config = qw.model_config;
const model_mod = qw.qwen3_asr.model;
const prompt = qw.qwen3_asr.prompt;
const tokenizer = qw.tokenizer;

/// Arena backing store. The freestanding allocator is the only one available,
/// and it is exactly right here: the runtime allocates on load and on the first
/// decode of an utterance, never inside a token loop.
const allocator = std.heap.wasm_allocator;

pub const Error =
    error{
        /// The call is legal, just not in the state this handle is in.
        InvalidState,
        /// The caller supplied more shards than one model may hold.
        LimitExceeded,
        /// The same shard was added twice. Ambiguous input is refused rather
        /// than resolved by order, which would make the loaded model depend on
        /// the caller's loop.
        DuplicateShard,
    } ||
    container.Error ||
    model_config.Error ||
    model_mod.Error ||
    decoder_mod.Error ||
    prompt.Error;

/// Shards one model directory may hold. The 0.6B conversion emits three and the
/// 1.7B target scales linearly, so this is generous while staying a bound.
pub const shard_count_max: u32 = 64;

pub const State = enum(u8) {
    /// `qw_model_begin` has not run: no arena exists.
    idle = 0,
    /// `qw_model_begin` ran. Shards may be added, and `finish` may be called.
    ready = 1,
    /// `qw_model_finish` ran: the model is resident and an utterance may start.
    loaded = 2,
    /// `qw_decode_begin` ran: one utterance is in flight.
    decoding = 3,
};

/// What `decodeStep` did, which the ABI spells as its two return values.
pub const Step = enum(u8) {
    /// A token was written to the caller's buffer.
    token,
    /// The sequence ended: end-of-sequence token, or the token budget.
    finished,
};

/// One handle's model, decode state, and arena.
///
/// The value lives in a global in `exports.zig`, so its address is stable and
/// pointers into it (`self.model`, `self.shards`) never move.
pub const Model = struct {
    state: State = .idle,
    /// Model-lifetime allocations. Valid once `state != .idle`.
    arena: std.heap.ArenaAllocator = undefined,
    /// Parsed shards. Kept as values, not arena allocations, so a failed load
    /// can release the whole arena and still let the caller retry `finish`.
    shards: [shard_count_max]container.File = undefined,
    shard_count: u32 = 0,

    model: ?*model_mod.Model = null,
    decoder: ?*decoder_mod.Decoder = null,
    /// Width the next `finish` loads the key/value cache at. Chosen through the ABI before the load,
    /// because the load is what allocates it.
    cache_format: model_config.CacheFormat = .f32,
    /// Prompt tokens for one utterance, `prompt.tokenCount(maxSteps)` long.
    prompt_tokens: []u32 = &.{},
    /// Token ids this utterance produced, `config.max_decode_tokens` long.
    generated: []u32 = &.{},
    generated_count: u32 = 0,
    /// The token the decoder predicted but has not produced yet.
    next_token: u32 = 0,
    max_tokens: u32 = 0,
    finished: bool = false,

    /// Starts a model attempt: releases whatever the handle holds -- including a finished model --
    /// and leaves the handle ready to take shards.
    ///
    /// This is the load entry point and the unload path in one, because shards can be added but
    /// never removed: starting over is the only way to retry a load that failed, and it is also how
    /// a caller releases a model's memory without destroying the instance.
    pub fn begin(self: *Model) void {
        if (self.state != .idle) self.arena.deinit();
        self.* = .{};
        self.arena = std.heap.ArenaAllocator.init(allocator);
        self.state = .ready;
    }

    /// Releases everything, including a model that was only partially built or
    /// never finished. Idempotent: `qw_destroy` may run on a fresh handle.
    pub fn reset(self: *Model) void {
        if (self.state != .idle) self.arena.deinit();
        self.* = .{};
    }

    /// Parses a shard in place and keeps the view.
    ///
    /// The parse is the whole validation: magic, version, index ordering,
    /// tensor shapes, byte lengths, and the payload checksum. A shard that
    /// fails any of them is rejected here rather than surfacing as a missing
    /// tensor during a forward pass.
    pub fn addShard(self: *Model, bytes: []align(16) const u8) Error!void {
        if (self.state != .ready) return Error.InvalidState;
        if (self.shard_count >= shard_count_max) return Error.LimitExceeded;

        const file = try container.File.parse(bytes);
        try file.verifyChecksum();

        var index: u32 = 0;
        while (index < self.shard_count) : (index += 1) {
            const existing = &self.shards[index];
            // The same buffer twice is the common accident (a loop that did not
            // advance), and it is worth an exact answer.
            if (existing.bytes.ptr == file.bytes.ptr) return Error.DuplicateShard;
            // The same payload at a different address is a real duplicate too,
            // which is what the container's payload checksum detects.
            if (existing.header.payload_checksum == file.header.payload_checksum) {
                return Error.DuplicateShard;
            }
        }
        self.shards[self.shard_count] = file;
        self.shard_count += 1;
        assert(self.shard_count <= shard_count_max);
    }

    /// Chooses the key/value cache width the next `finish` uses.
    ///
    /// Refused once a model is loaded: the cache is allocated during the load, so a format that
    /// arrived afterwards would describe a model nobody built.
    pub fn setCacheFormat(self: *Model, format: model_config.CacheFormat) Error!void {
        if (self.state != .ready) return Error.InvalidState;
        if (self.shard_count == 0) return Error.InvalidState;
        self.cache_format = format;
    }

    /// Resolves every tensor the architecture requires and prepares the decoder.
    ///
    /// This is where a model either becomes usable or fails cleanly: the
    /// configuration is validated against the runtime's capacity budgets, every
    /// tensor named by `layout.Iterator` is resolved, and the scratch buffers
    /// and key/value cache are allocated. The vocabulary is not consulted here;
    /// the prompt that needs it is built by `decodeBegin`.
    pub fn finish(self: *Model, config: *const model_config.Config) Error!void {
        self.finishInner(config) catch |err| {
            // A partial load holds most of the model's memory; releasing it here
            // is what keeps a failed attempt from costing a second attempt.
            self.releaseArena();
            return err;
        };
    }

    fn finishInner(self: *Model, config: *const model_config.Config) Error!void {
        if (self.state != .ready) return Error.InvalidState;
        if (self.shard_count == 0) return Error.InvalidState;

        const arena = self.arena.allocator();
        var files: [shard_count_max]*const container.File = undefined;
        for (self.shards[0..self.shard_count], 0..) |*file, index| files[index] = file;

        const loaded = try arena.create(model_mod.Model);
        loaded.* = try model_mod.Model.loadWithCache(arena, config.*, files[0..self.shard_count], self.cache_format);

        const decoder = try arena.create(decoder_mod.Decoder);
        decoder.* = decoder_mod.Decoder.init(loaded);

        self.prompt_tokens = try arena.alloc(u32, prompt.tokenCount(loaded.maxSteps()));
        self.generated = try arena.alloc(u32, config.max_decode_tokens);
        self.model = loaded;
        self.decoder = decoder;
        self.state = .loaded;
        // Both buffers are sized from the configuration, and `decodeBegin` relies
        // on that: the prompt cannot overflow, and a generated id always fits.
        assert(self.prompt_tokens.len == prompt.tokenCount(loaded.maxSteps()));
        assert(self.generated.len == config.max_decode_tokens);
    }

    /// What the loaded model keeps resident, and the limits it decodes within.
    ///
    /// The numbers are measured from the loaded model rather than predicted: a
    /// caller deciding whether a model fits has already paid for it here, and
    /// `ModelRequirements` documents where the numbers come from.
    pub fn requirements(self: *const Model, out: *abi.ModelRequirements) Error!void {
        // A model exists exactly in the loaded and decoding states, so this is
        // the state check as well.
        const loaded = self.model orelse return Error.InvalidState;
        assert(self.state == .loaded or self.state == .decoding);

        // The shard bytes as supplied, container padding included, because the
        // model borrows them and they must stay resident.
        var weight_bytes: u64 = 0;
        for (self.shards[0..self.shard_count]) |*file| weight_bytes += file.bytes.len;

        const cache_bytes = loaded.cacheBytes();
        const scratch_bytes = loaded.scratchBytes();
        out.* = .{
            .weight_bytes = weight_bytes,
            .cache_bytes = cache_bytes,
            .scratch_bytes = scratch_bytes,
            .total_bytes = weight_bytes + cache_bytes + scratch_bytes,
            .max_positions = loaded.config.max_positions,
            .max_audio_frames = @intCast(mel.capacity_frames),
            .max_decode_tokens = loaded.config.max_decode_tokens,
            .reserved = 0,
        };
        // The total is what a caller plans memory against, so it must be
        // exactly the sum of the parts it was handed.
        assert(out.total_bytes == weight_bytes + cache_bytes + scratch_bytes);
        assert(out.reserved == 0);
    }

    /// Runs the audio tower and projector over one clip's log-mel features,
    /// resets the decoder, prefills the prompt, and holds the first token.
    ///
    /// `features` is `mel.mel_bins x frames` row major, exactly what
    /// `qw_mel_compute` writes: `features.len / mel_bins` frames. `max_tokens`
    /// bounds this utterance's generated sequence and must not exceed the
    /// configuration's own budget.
    pub fn decodeBegin(
        self: *Model,
        vocabulary: *const tokenizer.TokenTable,
        features: []const f32,
        max_tokens: u32,
    ) Error!void {
        if (self.state != .loaded) return Error.InvalidState;
        const loaded = self.model orelse return Error.InvalidState;
        const decoder = self.decoder orelse return Error.InvalidState;

        if (max_tokens == 0) return Error.LimitExceeded;
        if (max_tokens > loaded.config.max_decode_tokens) return Error.LimitExceeded;
        if (features.len == 0) return Error.UnexpectedShape;
        if (features.len % loaded.config.mel_bins != 0) return Error.UnexpectedShape;
        const frames: u32 = @intCast(features.len / loaded.config.mel_bins);
        if (frames > mel.capacity_frames) return Error.AudioTooLong;

        // The template's ordinary words are resolved from the vocabulary by
        // byte comparison, so the prompt cannot be built without one, and a
        // vocabulary the template cannot be spelled from fails here rather than
        // producing a prompt the model was not trained on.
        const words = try prompt.Words.resolve(vocabulary);

        const steps = try loaded.encodeAudio(features, frames);
        try loaded.projectSteps(steps, loaded.scratch.projected);
        const length = try prompt.build(
            &loaded.config,
            words,
            self.prompt_tokens,
            steps,
            null,
        );

        decoder.reset();
        self.next_token = try decoder.prefill(
            self.prompt_tokens[0..length],
            loaded.scratch.projected,
            steps,
        );
        self.generated_count = 0;
        self.max_tokens = max_tokens;
        self.finished = false;
        self.state = .decoding;
        assert(self.next_token < loaded.config.vocab_size);
    }

    /// Advances the sequence by one token.
    ///
    /// Writes the produced token into `token_out` and returns `.token`, or
    /// writes nothing and returns `.finished` once the sequence has ended: an
    /// end-of-sequence token, or the caller's token budget. The token budget is
    /// checked before the next forward pass, so the last token of a full
    /// sequence costs no prediction.
    pub fn decodeStep(self: *Model, token_out: *u32) Error!Step {
        if (self.state != .decoding) return Error.InvalidState;
        const decoder = self.decoder orelse return Error.InvalidState;
        assert(self.finished or self.generated_count < self.max_tokens);
        if (self.finished) return .finished;

        if (endsSequence(&decoder.model.config, self.next_token)) {
            // The end-of-sequence token is not part of the transcript, which is
            // also what the native runner does.
            self.finished = true;
            return .finished;
        }
        if (self.generated_count >= self.max_tokens) {
            self.finished = true;
            return .finished;
        }

        token_out.* = self.next_token;
        self.generated[self.generated_count] = self.next_token;
        self.generated_count += 1;
        assert(self.generated_count <= self.max_tokens);

        if (self.generated_count >= self.max_tokens) {
            self.finished = true;
            return .token;
        }
        self.next_token = try decoder.step(self.next_token);
        return .token;
    }

    /// Copies the token ids produced so far. The buffer must hold all of them:
    /// a truncating copy would silently edit the transcript.
    pub fn decodeTokens(self: *Model, out: []u32) Error!u32 {
        if (self.state != .decoding) return Error.InvalidState;
        assert(self.generated_count <= self.max_tokens);
        if (out.len < self.generated_count) return Error.LimitExceeded;
        @memcpy(out[0..self.generated_count], self.generated[0..self.generated_count]);
        return self.generated_count;
    }

    /// Ends the utterance: the key/value cache goes back to empty, and the
    /// model stays resident for the next call.
    pub fn decodeEnd(self: *Model) Error!void {
        if (self.state != .decoding) return Error.InvalidState;
        const decoder = self.decoder orelse return Error.InvalidState;
        decoder.reset();
        self.generated_count = 0;
        self.next_token = 0;
        self.max_tokens = 0;
        self.finished = false;
        self.state = .loaded;
    }

    /// Releases the arena and forgets the model, leaving the shard views alone.
    fn releaseArena(self: *Model) void {
        assert(self.state != .idle);
        self.arena.deinit();
        self.arena = std.heap.ArenaAllocator.init(allocator);
        self.model = null;
        self.decoder = null;
        self.prompt_tokens = &.{};
        self.generated = &.{};
        self.generated_count = 0;
        self.next_token = 0;
        self.max_tokens = 0;
        self.finished = false;
        self.state = .ready;
    }
};

/// Whether `token` ends a sequence: either end-of-sequence token, or the
/// end-of-text token the reference also stops on.
fn endsSequence(config: *const model_config.Config, token: u32) bool {
    if (token == config.token_eos_primary) return true;
    if (token == config.token_eos_secondary) return true;
    return token == config.token_endoftext;
}
