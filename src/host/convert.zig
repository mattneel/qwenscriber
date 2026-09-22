//! Converts an official Qwen3-ASR checkpoint into a Qwenscriber model
//! directory.
//!
//! The conversion is a walk over the core's tensor inventory:
//!
//!     read config.json and the tokenizer files  -> config.bin, tokens.bin
//!     for each `layout.Iterator` entry:
//!         find the checkpoint tensor by its mapped name
//!         assert its shape is exactly the declared shape
//!         decode it to f32, quantize it, append it to the current shard
//!                                               -> shard-NNN.qw
//!     summarize the directory                   -> manifest.json
//!
//! Nothing is converted that the runtime did not ask for, and nothing the
//! runtime asks for may be missing: both directions are checked against
//! `layout.zig`, which is the contract the runtime loads with.
//!
//! # Sharding
//!
//! Tensors are assigned to shards in `(layer, kind)` order, which is the order a
//! shard's index must be sorted in. A shard ends when the next **layer** would
//! push it past the byte budget: a boundary inside a layer would make the
//! runtime fetch a shard it cannot yet use, and the point of sharding is that a
//! layer's weights arrive together. A single layer larger than the budget --
//! possible only with a budget far below any real layer's size -- overflows its
//! shard instead, because splitting a layer costs the runtime more than the
//! bytes it saves.
//!
//! # Memory
//!
//! The checkpoint is read into memory and never copied: every tensor is a slice
//! of the file image. One shard is built at a time and then written, so peak
//! memory is the checkpoint plus one shard, not the whole model directory.

const std = @import("std");
const qwenscriber = @import("qwenscriber");

const assert = std.debug.assert;
const container = qwenscriber.container;
const dtype = qwenscriber.dtype;
const half_float = qwenscriber.half_float;
const layout = qwenscriber.qwen3_asr.layout;
const model_config = qwenscriber.model_config;
const quant = qwenscriber.quant;
const tensor = qwenscriber.tensor;

const checkpoint_config = @import("checkpoint_config.zig");
const manifest = @import("manifest.zig");
const safetensors = @import("safetensors.zig");
const tensor_names = @import("tensor_names.zig");
const tokenizer_file = @import("tokenizer_file.zig");

/// Version of the conversion tooling, recorded in every manifest. Bumped when a
/// change alters the bytes a model directory contains.
pub const tool_version = "qwenscriber-convert 1";

/// Default shard size. Chosen so one shard fits a browser's fetch and a GPU
/// buffer on every adapter the runtime targets, and so a 0.6B conversion
/// produces a handful of shards rather than dozens.
pub const shard_bytes_default: u64 = 192 << 20;

/// Rows compared per quantized tensor when verification is on. The error bound
/// is per row, so sampling spread over every row catches a per-row mistake
/// without decoding the whole model a second time.
pub const verify_rows_max: u32 = 4096;

/// Elements decoded per pass when a tensor is stored as elements rather than
/// quantized. Bounds the scratch buffer at 16 KiB.
const element_chunk_max: u32 = 4096;

/// Largest shard count this converter produces. A longer shard list means the
/// budget is absurdly small for the model being converted.
pub const shards_max: u32 = 4096;

pub const Options = struct {
    /// Checkpoint directory, open.
    input_dir: std.Io.Dir,
    /// Model directory, which the caller has already created.
    output_dir: std.Io.Dir,
    /// Directory names, for the report and the summary table.
    input_name: []const u8 = "checkpoint",
    output_name: []const u8 = "model",
    /// Identifier recorded in the manifest. The CLI defaults it to the input
    /// directory's name.
    model_id: []const u8,
    quantization: dtype.Format = .q4,
    shard_bytes_max: u64 = shard_bytes_default,
    /// Re-read every shard and compare it against the checkpoint.
    verify: bool = false,
};

/// Names the checkpoint tensor a failure was about, so an operator can look at
/// the checkpoint rather than at this converter.
pub const Diagnostics = struct {
    /// Official checkpoint name of the tensor that failed, empty when the
    /// failure was not about one tensor.
    tensor_name: []const u8 = "",
    /// What was wrong with it.
    detail: []const u8 = "",
};

pub const ShardReport = struct {
    name: []const u8,
    bytes: u64,
    tensor_count: u32,
    first_layer: u16,
    last_layer: u16,
    payload_checksum: u32,
    sha256_hex: [64]u8,
};

pub const VerifyReport = struct {
    /// False when verification was not requested.
    checked: bool = false,
    shards_parsed: u32 = 0,
    tensors_compared: u32 = 0,
    rows_compared: u64 = 0,
    elements_compared: u64 = 0,
    /// Largest absolute difference seen, to compare against a format's step.
    error_max: f32 = 0,
};

pub const Report = struct {
    model_id: []const u8,
    quantization: dtype.Format,
    /// Tensors written: the inventory minus a tied output projection.
    tensors: u32,
    /// Elements written.
    parameters: u64,
    /// Bytes the checkpoint occupies.
    source_bytes: u64,
    /// Sum of the shards' payload sections.
    payload_bytes: u64,
    /// Every byte the model directory contains.
    output_bytes: u64,
    bits_per_weight: f64,
    token_count: u32,
    tokenizer_bytes: u64,
    merges_bytes: u64,
    output_is_tied: bool,
    /// Checkpoint tensors no rule recognized, which the model does not need.
    unused_tensors: u32,
    shards: []const ShardReport,
    verify: VerifyReport,
};

pub const Error = error{
    /// The checkpoint directory has no `config.json`.
    MissingConfigFile,
    /// The inventory requires a tensor the checkpoint does not have.
    MissingTensor,
    /// The checkpoint has a tensor for a layer or kind the configuration does
    /// not describe: the checkpoint and its configuration disagree.
    UnexpectedTensor,
    /// Two checkpoint tensors map to one inventory position.
    DuplicateTensor,
    /// A checkpoint tensor's shape is not the declared shape.
    ShapeMismatch,
    /// A checkpoint tensor is not a 16- or 32-bit float.
    UnsupportedSourceDtype,
    /// A tensor's row length is not a whole number of quantization groups, or
    /// its shape cannot be stored in the requested format.
    NotQuantizable,
    /// A tensor holds a NaN or an infinity, which would poison a group's scale.
    NonFiniteValue,
    /// The written output did not match the checkpoint within the format's
    /// error bound.
    VerificationFailed,
    /// A shard's payload does not fit the container's 32-bit length field.
    ShardTooLarge,
    /// The conversion needs more shards than `shards_max`.
    TooManyShards,
    /// The checkpoint declares that its output projection is not tied, but ships
    /// no output projection for the runtime to use.
    UntiedOutputMissing,
};

pub fn run(
    arena: std.mem.Allocator,
    io: std.Io,
    options: Options,
    diagnostics: *Diagnostics,
) !Report {
    var token_diagnostics: tokenizer_file.Diagnostics = .{};
    const tokens = try tokenizer_file.read(arena, io, options.input_dir, &token_diagnostics);
    const checkpoint = try readCheckpointConfig(arena, io, options.input_dir, tokens);
    const config = checkpoint.config;

    const source = try safetensors.Checkpoint.open(arena, io, options.input_dir);
    const names = try source.names(arena);
    const classification = try tensor_names.classifyNames(arena, &config, names);

    var driver = Driver{
        .arena = arena,
        .io = io,
        .options = options,
        .config = &config,
        .source = &source,
        .classification = &classification,
        .diagnostics = diagnostics,
        .config_output_is_tied = checkpoint.output_is_tied,
        .verify_arena = std.heap.ArenaAllocator.init(arena),
    };
    defer driver.verify_arena.deinit();
    try driver.init();

    try driver.requireCompleteInventory();
    try writeConfigFile(io, options.output_dir, &config);
    const tokenizer_bytes = try writeTokenizer(arena, io, options.output_dir, &tokens);
    const merges_bytes = try writeMerges(io, options.output_dir, &tokens);
    try driver.writeAll();

    const manifest_bytes = try driver.writeManifest(&tokens, tokenizer_bytes, merges_bytes);
    var report = try driver.report(&tokens, tokenizer_bytes, merges_bytes, manifest_bytes);
    report.source_bytes = source.sourceBytes();
    return report;
}

fn readCheckpointConfig(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    tokens: tokenizer_file.Data,
) !checkpoint_config.Parsed {
    const config_json = dir.readFileAlloc(io, "config.json", arena, .limited(1 << 20)) catch |err| {
        return switch (err) {
            error.FileNotFound => Error.MissingConfigFile,
            else => err,
        };
    };
    var diagnostics: checkpoint_config.Diagnostics = .{};
    return checkpoint_config.parse(arena, .{
        .config_json = config_json,
        .generation_json = readTextFile(arena, io, dir, "generation_config.json"),
        .tokenizer_config_json = readTextFile(arena, io, dir, "tokenizer_config.json"),
        .specials = tokens.specials,
    }, &diagnostics);
}

/// Reads a file a checkpoint may legitimately not have.
fn readTextFile(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    name: []const u8,
) ?[]u8 {
    return dir.readFileAlloc(io, name, arena, .limited(1 << 20)) catch null;
}

/// The conversion itself: a cursor into the inventory, one shard in memory, and
/// the bookkeeping the report and the manifest are made of.
const Driver = struct {
    arena: std.mem.Allocator,
    io: std.Io,
    options: Options,
    config: *const model_config.Config,
    source: *const safetensors.Checkpoint,
    classification: *const tensor_names.Classification,
    diagnostics: *Diagnostics,

    /// Row and element scratch for decoding checkpoint tensors: long enough for
    /// any matrix row, any element chunk, and any vector.
    scratch: []f32 = &.{},
    /// Second scratch buffer, for comparing two tensors element by element.
    other: []f32 = &.{},
    /// Payload of the shard being built, reused between shards.
    payload: std.ArrayList(u8) = .empty,
    /// Index entries of the shard being built, in `(layer, kind)` order.
    entries: std.ArrayList(container.Entry) = .empty,
    shards: std.ArrayList(ShardReport) = .empty,
    /// Arena for reading a shard back during verification, reset per shard so
    /// only one shard's bytes are held at a time.
    verify_arena: std.heap.ArenaAllocator,
    /// What the checkpoint's configuration claims about the output projection.
    config_output_is_tied: bool = false,
    /// True when the decoder's output projection is the token embedding and is
    /// therefore not stored separately.
    output_is_tied: bool = false,
    tensors: u32 = 0,
    parameters: u64 = 0,
    payload_bytes: u64 = 0,
    verify: VerifyReport = .{},

    fn init(self: *Driver) !void {
        const elements = scratchElements(self.config);
        self.scratch = try self.arena.alloc(f32, elements);
        self.other = try self.arena.alloc(f32, elements);
        self.output_is_tied = try outputIsTied(
            self.source,
            self.classification,
            self.config_output_is_tied,
            self.scratch,
            self.other,
        );
    }

    /// Every inventory position must be filled, and the checkpoint must not
    /// contain a tensor that contradicts the configuration.
    fn requireCompleteInventory(self: *Driver) !void {
        const classification = self.classification;
        if (classification.duplicate.len != 0) {
            self.diagnostics.tensor_name = classification.duplicate[0];
            self.diagnostics.detail = "two checkpoint tensors map to one tensor";
            return Error.DuplicateTensor;
        }
        if (classification.unexpected.len != 0) {
            self.diagnostics.tensor_name = classification.unexpected[0];
            self.diagnostics.detail = "not part of the configured model";
            return Error.UnexpectedTensor;
        }
        for (classification.slots) |slot| {
            if (slot.source_index != null) continue;
            if (self.output_is_tied and slot.required.kind == .decoder_output_weight) {
                continue;
            }
            return self.reportMissing(slot.required, "missing from the checkpoint");
        }
    }

    fn reportMissing(self: *Driver, required: layout.Required, detail: []const u8) Error {
        var buffer: [128]u8 = undefined;
        const name = tensor_names.officialName(required.kind, required.layer, &buffer) catch
            "unknown";
        self.diagnostics.tensor_name = self.arena.dupe(u8, name) catch name;
        self.diagnostics.detail = detail;
        return Error.MissingTensor;
    }

    fn writeAll(self: *Driver) !void {
        const count = layout.Iterator.count(self.config);
        var position: u32 = 0;
        while (position < count) : (position += 1) {
            const required = layout.at(self.config, position) orelse unreachable;
            if (self.output_is_tied and required.kind == .decoder_output_weight) continue;
            try self.appendTensor(required);
        }
        try self.finishShard();
    }

    fn sourceTensor(self: *Driver, required: layout.Required) !*const safetensors.Tensor {
        const slot = self.classification.slotFor(required.kind, required.layer) orelse {
            return Error.UnexpectedTensor;
        };
        const index = slot.source_index orelse {
            return self.reportMissing(required, "missing from the checkpoint");
        };
        return self.source.tensors[index];
    }

    /// Converts one inventory entry into the current shard's payload.
    fn appendTensor(self: *Driver, required: layout.Required) !void {
        const entry_source = try self.sourceTensor(required);
        if (!entry_source.shape.eql(&required.shape)) {
            self.diagnostics.tensor_name = entry_source.name;
            self.diagnostics.detail = "shape differs from the configured shape";
            return Error.ShapeMismatch;
        }
        if (!entry_source.dtype.isFloat()) {
            self.diagnostics.tensor_name = entry_source.name;
            self.diagnostics.detail = "not a 16- or 32-bit float";
            return Error.UnsupportedSourceDtype;
        }
        const format = required.format(self.options.quantization);
        const len_bytes = required.shape.byteLength(format) catch {
            self.diagnostics.tensor_name = entry_source.name;
            self.diagnostics.detail = "cannot be stored in the requested format";
            return Error.NotQuantizable;
        };
        try self.startShardIfNeeded(required.layer, len_bytes);
        try self.appendPayload(entry_source, required, format, len_bytes);
        self.tensors += 1;
        self.parameters += try required.shape.elementCount();
    }

    fn appendPayload(
        self: *Driver,
        entry_source: *const safetensors.Tensor,
        required: layout.Required,
        format: dtype.Format,
        len_bytes: u64,
    ) !void {
        // Every tensor starts on the container's alignment, because the
        // quantization planes' offsets are computed from an aligned base.
        const offset = std.mem.alignForward(
            u64,
            self.payload.items.len,
            quant.tensor_alignment_bytes,
        );
        try self.payload.appendNTimes(self.arena, 0, @intCast(offset - self.payload.items.len));
        const start = self.payload.items.len;
        try self.payload.appendNTimes(self.arena, 0, @intCast(len_bytes));
        const payload = self.payload.items[start..][0..@intCast(len_bytes)];
        try fillPayload(entry_source, required, format, payload, self.scratch, self.diagnostics);
        try self.entries.append(self.arena, .{
            .kind = @backingInt(required.kind),
            .layer = required.layer,
            .format = @backingInt(format),
            .rank = required.shape.rank,
            .dims = required.shape.dims,
            .offset_bytes = offset,
            .len_bytes = len_bytes,
        });
    }

    /// Ends the current shard and starts a new one when the next layer would
    /// push this one past the budget. See the module comment for why the
    /// boundary must fall between layers.
    fn startShardIfNeeded(self: *Driver, layer: u16, next_bytes: u64) !void {
        if (self.entries.items.len == 0) return;
        const previous = self.entries.items[self.entries.items.len - 1];
        if (previous.layer == layer) return;
        if (self.payload.items.len + next_bytes <= self.options.shard_bytes_max) return;
        try self.finishShard();
    }

    /// Writes the shard being built -- header, index, padding, payload -- and
    /// records what the manifest and the report need from it.
    fn finishShard(self: *Driver) !void {
        if (self.entries.items.len == 0) return;
        if (self.shards.items.len == shards_max) return Error.TooManyShards;

        var name_buffer: [24]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buffer, "shard-{d:0>3}.qw", .{
            self.shards.items.len,
        });
        const file_name = try self.arena.dupe(u8, name);
        const written = try self.writeShardFile(file_name);
        const first = self.entries.items[0];
        const last = self.entries.items[self.entries.items.len - 1];
        try self.shards.append(self.arena, .{
            .name = file_name,
            .bytes = written.bytes,
            .tensor_count = @intCast(self.entries.items.len),
            .first_layer = first.layer,
            .last_layer = last.layer,
            .payload_checksum = written.payload_checksum,
            .sha256_hex = written.sha256_hex,
        });
        self.payload_bytes += self.payload.items.len;
        if (self.options.verify) {
            self.verify.checked = true;
            try self.verifyShard(file_name);
        }

        self.payload.clearRetainingCapacity();
        self.entries.clearRetainingCapacity();
    }

    const WrittenShard = struct {
        bytes: u64,
        payload_checksum: u32,
        sha256_hex: [64]u8,
    };

    fn writeShardFile(self: *Driver, name: []const u8) !WrittenShard {
        const index_len_bytes: u32 = @intCast(self.entries.items.len * container.index_entry_bytes);
        const payload_len_bytes = std.math.cast(u32, self.payload.items.len) orelse
            return Error.ShardTooLarge;
        const payload_offset_bytes: u32 = @intCast(std.mem.alignForward(
            u64,
            container.header_bytes + index_len_bytes,
            container.payload_alignment,
        ));
        const header = container.Header{
            .magic = container.magic_bytes,
            .format_version = container.format_version,
            .index_len_bytes = index_len_bytes,
            .payload_offset_bytes = payload_offset_bytes,
            .payload_len_bytes = payload_len_bytes,
            .tensor_count = @intCast(self.entries.items.len),
            .payload_checksum = @truncate(std.hash.Fnv1a_64.hash(self.payload.items)),
        };
        // The container's own validation runs before the bytes reach a file: a
        // shard the runtime would reject must never be written.
        try header.validate();

        var file = try self.options.output_dir.createFile(self.io, name, .{ .truncate = true });
        defer file.close(self.io);
        var file_buffer: [64 * 1024]u8 = undefined;
        var file_writer: std.Io.File.Writer = .init(file, self.io, &file_buffer);
        const out = &file_writer.interface;

        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        const header_bytes = std.mem.asBytes(&header);
        const entries_bytes = std.mem.sliceAsBytes(self.entries.items);
        const padding: [container.payload_alignment]u8 = @splat(0);
        const padding_len = payload_offset_bytes - container.header_bytes - index_len_bytes;
        assert(padding_len < container.payload_alignment);

        try out.writeAll(header_bytes);
        try out.writeAll(entries_bytes);
        try out.writeAll(padding[0..padding_len]);
        try out.writeAll(self.payload.items);
        try out.flush();
        hasher.update(header_bytes);
        hasher.update(entries_bytes);
        hasher.update(padding[0..padding_len]);
        hasher.update(self.payload.items);

        const stat = try file.stat(self.io);
        const expected_bytes = header.fileLenBytes();
        if (stat.size != expected_bytes) return Error.ShardTooLarge;
        const digest = hasher.finalResult();
        return .{
            .bytes = expected_bytes,
            .payload_checksum = header.payload_checksum,
            .sha256_hex = manifest.hexOfDigest(&digest),
        };
    }

    /// Re-reads a written shard and compares every tensor in it against the
    /// checkpoint it came from.
    fn verifyShard(self: *Driver, name: []const u8) !void {
        _ = self.verify_arena.reset(.retain_capacity);
        const bytes = try self.options.output_dir.readFileAllocOptions(
            self.io,
            name,
            self.verify_arena.allocator(),
            .limited(safetensors.checkpoint_bytes_max),
            .@"16",
            null,
        );
        const file = try container.File.parse(bytes);
        try file.verifyChecksum();
        self.verify.shards_parsed += 1;
        for (file.index) |*index_entry| {
            const required = layout.find(
                self.config,
                @fromBackingInt(index_entry.kind),
                index_entry.layer,
            ) orelse return Error.VerificationFailed;
            const entry_source = try self.sourceTensor(required);
            try compareTensor(file, index_entry, entry_source, self.scratch, self.other, &self.verify);
        }
    }

    fn writeManifest(
        self: *Driver,
        tokens: *const tokenizer_file.Data,
        tokenizer_bytes: u64,
        merges_bytes: u64,
    ) !u64 {
        const shards = try self.arena.alloc(manifest.Shard, self.shards.items.len);
        for (self.shards.items, 0..) |shard, index| {
            shards[index] = .{
                .name = shard.name,
                .bytes = shard.bytes,
                .sha256_hex = try self.arena.dupe(u8, &shard.sha256_hex),
                .tensor_count = shard.tensor_count,
                .first_layer = shard.first_layer,
                .last_layer = shard.last_layer,
            };
        }
        const value = manifest.Value{
            .model_id = self.options.model_id,
            .quantization = self.options.quantization.name(),
            .config_bytes = model_config.size_bytes,
            .tokenizer_bytes = tokenizer_bytes,
            .token_count = tokens.count,
            .merges_file = if (tokens.merges != null) "merges.txt" else "",
            .merges_bytes = merges_bytes,
            .output_is_tied = self.output_is_tied,
            .shards = shards,
            .tensors = self.tensors,
            .parameters_approximate = self.parameters,
            .payload_bytes = self.payload_bytes,
            .bits_per_weight = self.bitsPerWeight(),
            .tool_version = tool_version,
        };
        var file = try self.options.output_dir.createFile(self.io, "manifest.json", .{
            .truncate = true,
        });
        defer file.close(self.io);
        var file_buffer: [16 * 1024]u8 = undefined;
        var file_writer: std.Io.File.Writer = .init(file, self.io, &file_buffer);
        try manifest.write(&file_writer.interface, &value);
        try file_writer.interface.flush();
        const stat = try file.stat(self.io);
        return stat.size;
    }

    fn bitsPerWeight(self: *const Driver) f64 {
        if (self.parameters == 0) return 0;
        const bits = @as(f64, @floatFromInt(self.payload_bytes)) * 8.0;
        return bits / @as(f64, @floatFromInt(self.parameters));
    }

    fn report(
        self: *Driver,
        tokens: *const tokenizer_file.Data,
        tokenizer_bytes: u64,
        merges_bytes: u64,
        manifest_bytes: u64,
    ) !Report {
        var output_bytes: u64 = model_config.size_bytes + manifest_bytes;
        output_bytes += tokenizer_bytes + merges_bytes;
        for (self.shards.items) |shard| output_bytes += shard.bytes;
        assert(self.payload_bytes > 0);
        assert(self.tensors > 0);
        return .{
            .model_id = self.options.model_id,
            .quantization = self.options.quantization,
            .tensors = self.tensors,
            .parameters = self.parameters,
            // Filled in by `run`, which knows the checkpoint's size.
            .source_bytes = 0,
            .payload_bytes = self.payload_bytes,
            .output_bytes = output_bytes,
            .bits_per_weight = self.bitsPerWeight(),
            .token_count = tokens.count,
            .tokenizer_bytes = tokenizer_bytes,
            .merges_bytes = merges_bytes,
            .output_is_tied = self.output_is_tied,
            .unused_tensors = @intCast(self.classification.unknown.len),
            .shards = self.shards.items,
            .verify = self.verify,
        };
    }
};

/// Scratch length: a whole matrix row, or an element chunk, whichever is larger.
fn scratchElements(config: *const model_config.Config) u32 {
    var elements: u32 = element_chunk_max;
    var iterator = layout.Iterator.init(config);
    while (iterator.next()) |required| {
        if (required.shape.rank != 2) continue;
        elements = @max(elements, required.shape.dims[1]);
    }
    return elements;
}

/// Decides whether the decoder's output projection is the token embedding, and
/// therefore whether it is stored at all.
///
/// Two checkpoints are tied in two different ways: the native export ships no
/// `lm_head` at all (weight-tied by construction), and the vLLM export ships one
/// whose values equal the embedding's. In both cases the embedding already holds
/// those weights, so the model directory stores one copy and the manifest
/// records the tiedness; a checkpoint whose head differs keeps its own tensor.
fn outputIsTied(
    source: *const safetensors.Checkpoint,
    classification: *const tensor_names.Classification,
    config_output_is_tied: bool,
    scratch: []f32,
    other: []f32,
) !bool {
    const output_slot = classification.slotFor(.decoder_output_weight, 0) orelse return false;
    const output_index = output_slot.source_index orelse {
        // No output projection at all: the embedding has to carry it, or the
        // checkpoint does not describe a model the runtime can run.
        if (!config_output_is_tied) return Error.UntiedOutputMissing;
        return true;
    };
    const embed_slot = classification.slotFor(.decoder_embed_tokens_weight, 0) orelse return false;
    const embed_index = embed_slot.source_index orelse return false;
    return tensorsEqual(
        source.tensors[output_index],
        source.tensors[embed_index],
        scratch,
        other,
    );
}

/// True when two checkpoint tensors hold the same values. Equal bytes are equal
/// values; otherwise both are decoded, because one may be stored at a different
/// width and the comparison is after rounding to f32.
fn tensorsEqual(
    left: *const safetensors.Tensor,
    right: *const safetensors.Tensor,
    scratch: []f32,
    other: []f32,
) !bool {
    if (left.data.len == right.data.len) {
        if (std.mem.eql(u8, left.data, right.data)) return true;
    }
    const count = try left.elementCount();
    if (try right.elementCount() != count) return false;
    var offset: u64 = 0;
    while (offset < count) {
        const chunk: u32 = @intCast(@min(scratch.len, count - offset));
        try left.decodeRangeF32(offset, scratch[0..chunk]);
        try right.decodeRangeF32(offset, other[0..chunk]);
        if (!std.mem.eql(f32, scratch[0..chunk], other[0..chunk])) return false;
        offset += chunk;
    }
    return true;
}

/// Converts one tensor into its container payload, in the format the storage
/// policy chose for it.
fn fillPayload(
    source: *const safetensors.Tensor,
    required: layout.Required,
    format: dtype.Format,
    payload: []u8,
    scratch: []f32,
    diagnostics: *Diagnostics,
) !void {
    switch (format) {
        .q4, .q5, .q8 => try quantizeMatrix(source, required, format, payload, scratch, diagnostics),
        .f16 => try writeElements(source, payload, 2, scratch, diagnostics),
        .f32 => try writeElements(source, payload, 4, scratch, diagnostics),
        else => unreachable,
    }
}

/// Decodes a matrix row by row and quantizes each row into the two planes.
fn quantizeMatrix(
    source: *const safetensors.Tensor,
    required: layout.Required,
    format: dtype.Format,
    payload: []u8,
    scratch: []f32,
    diagnostics: *Diagnostics,
) !void {
    const rows = try required.shape.rows();
    const cols = try required.shape.cols();
    assert(scratch.len >= cols);
    const plane = try quant.planeLayout(format, rows, cols, 0, quant.tensor_alignment_bytes);
    assert(plane.total_len_bytes == payload.len);
    assert(plane.scales_offset_bytes == 0);

    const scales_row_bytes = plane.groups_per_row * quant.q4_scale_bytes_per_group;
    const data_row_bytes = plane.groups_per_row * quant.dataBytesPerGroup(format);
    const values = scratch[0..cols];
    var row: u32 = 0;
    while (row < rows) : (row += 1) {
        try source.decodeRangeF32(@as(u64, row) * cols, values);
        try requireFinite(values, source.name, diagnostics);
        const scales = payload[plane.scales_offset_bytes + row * scales_row_bytes ..][0..scales_row_bytes];
        const codes = payload[plane.data_offset_bytes + row * data_row_bytes ..][0..data_row_bytes];
        quant.quantizeRow(format, values, scales, codes);
    }
}

/// Writes an element-format tensor: `element_bytes` bytes per value, decoded in
/// bounded chunks so a 4-D convolution and a 311 MiB embedding take one path.
fn writeElements(
    source: *const safetensors.Tensor,
    payload: []u8,
    comptime element_bytes: u32,
    scratch: []f32,
    diagnostics: *Diagnostics,
) !void {
    const count = payload.len / element_bytes;
    assert(count * element_bytes == payload.len);
    var written: u64 = 0;
    while (written < count) {
        const chunk: u32 = @intCast(@min(scratch.len, count - written));
        const values = scratch[0..chunk];
        try source.decodeRangeF32(written, values);
        try requireFinite(values, source.name, diagnostics);
        for (values, 0..) |value, index| {
            const at: usize = @intCast((written + index) * element_bytes);
            if (comptime element_bytes == 2) {
                std.mem.writeInt(u16, payload[at..][0..2], half_float.toF16(value), .little);
            } else {
                std.mem.writeInt(u32, payload[at..][0..4], @bitCast(value), .little);
            }
        }
        written += chunk;
    }
}

fn requireFinite(values: []const f32, name: []const u8, diagnostics: *Diagnostics) !void {
    if (quant.rowIsFinite(values)) return;
    diagnostics.tensor_name = name;
    diagnostics.detail = "holds a NaN or an infinity";
    return Error.NonFiniteValue;
}

/// Compares a container entry against the checkpoint tensor it came from.
///
/// Quantized tensors are compared row by row against the format's own step
/// bound: the largest value in a group decides its scale, so the largest error a
/// correctly quantized group can show is one step -- the most positive value has
/// no code of its own -- plus the stored f16 scale's rounding. Element tensors
/// are compared exactly, because f16 rounding is the only loss they suffer and
/// it is reproducible.
fn compareTensor(
    file: container.File,
    entry: *const container.Entry,
    source: *const safetensors.Tensor,
    scratch: []f32,
    other: []f32,
    report: *VerifyReport,
) !void {
    const format = try entry.storageFormat();
    const shape = try entry.shape();
    if (!shape.eql(&source.shape)) return Error.VerificationFailed;
    report.tensors_compared += 1;

    if (format.isQuantized()) {
        const planes = file.quantizedPlanes(entry.kind, entry.layer) orelse
            return Error.VerificationFailed;
        return compareQuantized(planes, source, scratch, other, format, report);
    }
    if (format == .f32) {
        return compareF32(file, entry, source, scratch, other, report);
    }
    return compareF16(file, entry, source, scratch, other, report);
}

fn compareQuantized(
    planes: container.QuantizedPlanes,
    source: *const safetensors.Tensor,
    scratch: []f32,
    other: []f32,
    format: dtype.Format,
    report: *VerifyReport,
) !void {
    const rows = planes.rows();
    const cols = planes.cols();
    assert(scratch.len >= cols);
    const scales_row_bytes = planes.scales.len / rows;
    const data_row_bytes = planes.data.len / rows;
    // Sampled rows are spread over the whole tensor: at most `verify_rows_max`
    // of them, evenly spaced, so a bug in one row's group cannot hide.
    const stride = @max((rows + verify_rows_max - 1) / verify_rows_max, 1);
    const values = scratch[0..cols];
    const expected = other[0..cols];
    var row: u32 = 0;
    while (row < rows) : (row += stride) {
        const scales = planes.scales[row * scales_row_bytes ..][0..scales_row_bytes];
        const codes = planes.data[row * data_row_bytes ..][0..data_row_bytes];
        quant.dequantizeRow(format, scales, codes, values);
        try source.decodeRangeF32(@as(u64, row) * cols, expected);
        try compareRow(expected, values, format, report);
        report.rows_compared += 1;
    }
}

fn compareF32(
    file: container.File,
    entry: *const container.Entry,
    source: *const safetensors.Tensor,
    scratch: []f32,
    other: []f32,
    report: *VerifyReport,
) !void {
    const stored = file.f32Values(entry.kind, entry.layer) orelse return Error.VerificationFailed;
    const count = try entry.elementCount();
    assert(count == stored.len);
    var offset: u64 = 0;
    while (offset < count) {
        const chunk: u32 = @intCast(@min(scratch.len, count - offset));
        try source.decodeRangeF32(offset, other[0..chunk]);
        for (stored[@intCast(offset)..][0..chunk], other[0..chunk]) |actual, expected| {
            if (actual != expected) return Error.VerificationFailed;
            report.elements_compared += 1;
        }
        offset += chunk;
    }
}

fn compareF16(
    file: container.File,
    entry: *const container.Entry,
    source: *const safetensors.Tensor,
    scratch: []f32,
    other: []f32,
    report: *VerifyReport,
) !void {
    const bits = file.f16Bits(entry.kind, entry.layer) orelse return Error.VerificationFailed;
    const count = try entry.elementCount();
    assert(count == bits.len);
    var offset: u64 = 0;
    while (offset < count) {
        const chunk: u32 = @intCast(@min(scratch.len, count - offset));
        for (scratch[0..chunk], bits[@intCast(offset)..][0..chunk]) |*value, raw| {
            value.* = half_float.fromF16(raw);
        }
        try source.decodeRangeF32(offset, other[0..chunk]);
        for (scratch[0..chunk], other[0..chunk]) |actual, expected| {
            if (actual != half_float.fromF16(half_float.toF16(expected))) {
                return Error.VerificationFailed;
            }
            report.elements_compared += 1;
        }
        offset += chunk;
    }
}

/// One row of a quantized tensor against its source, with the format's bound.
fn compareRow(
    expected: []const f32,
    actual: []const f32,
    format: dtype.Format,
    report: *VerifyReport,
) !void {
    assert(expected.len == actual.len);
    var max_abs: f32 = 0.0;
    for (expected) |value| max_abs = @max(max_abs, @abs(value));
    const step = max_abs / @as(f32, @floatFromInt(quant.bias(format)));
    // One step for the clamped positive extreme, plus the stored f16 scale's own
    // rounding, which moves every decoded value by up to `bias * 2^-11` steps.
    const allowance = step *
        (1.0 + @as(f32, @floatFromInt(quant.bias(format))) * 0.00048828125) + 1e-6;
    for (expected, actual) |original, restored| {
        const difference = @abs(original - restored);
        report.error_max = @max(report.error_max, difference);
        if (difference > allowance) return Error.VerificationFailed;
        report.elements_compared += 1;
    }
}

fn writeConfigFile(io: std.Io, dir: std.Io.Dir, config: *const model_config.Config) !void {
    try config.validate();
    try dir.writeFile(io, .{
        .sub_path = "config.bin",
        .data = std.mem.asBytes(config),
        .flags = .{ .truncate = true },
    });
}

fn writeTokenizer(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    tokens: *const tokenizer_file.Data,
) !u64 {
    var buffer: std.Io.Writer.Allocating = .init(arena);
    defer buffer.deinit();
    try tokenizer_file.writeTable(&buffer.writer, tokens);
    const bytes = buffer.written();
    try dir.writeFile(io, .{
        .sub_path = "tokens.bin",
        .data = bytes,
        .flags = .{ .truncate = true },
    });
    return bytes.len;
}

fn writeMerges(io: std.Io, dir: std.Io.Dir, tokens: *const tokenizer_file.Data) !u64 {
    const merges = tokens.merges orelse return 0;
    try dir.writeFile(io, .{
        .sub_path = "merges.txt",
        .data = merges,
        .flags = .{ .truncate = true },
    });
    return merges.len;
}

// --- the synthetic checkpoint the conversion tests are built on -------------
//
// A real checkpoint is 1.2 GiB and cannot be committed, so the contract these
// tests defend is exercised on the smallest model the inventory admits:
// 64-wide attention, one audio layer, one decoder layer, and a 128-entry
// vocabulary. The shapes are real shapes, produced by the same `layout.Iterator`
// the runtime loads with, so a conversion that passes here has done every step a
// full conversion does -- only with fewer repetitions.

/// Dimensionality the tests shrink to. Everything that must stay fixed does:
/// `mel_bins` is 128, `n_window` is 50 (and therefore a chunk is 100 frames and
/// 13 post-convolution steps), and every matrix row is a whole number of
/// quantization groups.
const tiny_audio_d_model: u32 = 64;
const tiny_audio_heads: u32 = 4;
const tiny_audio_ffn: u32 = 64;
const tiny_audio_downsample: u32 = 64;
const tiny_text_hidden: u32 = 64;
const tiny_text_heads: u32 = 2;
const tiny_text_kv_heads: u32 = 1;
const tiny_text_head_dim: u32 = 32;
const tiny_text_ffn: u32 = 64;
const tiny_vocab: u32 = 128;
const tiny_token_audio_start: u32 = 120;
const tiny_token_audio_end: u32 = 121;
const tiny_token_audio_pad: u32 = 122;
const tiny_token_im_start: u32 = 123;
const tiny_token_im_end: u32 = 124;
const tiny_token_endoftext: u32 = 125;
const tiny_token_asr_text: u32 = 126;

/// The runtime configuration the tiny `config.json` must produce.
fn tinyConfig() model_config.Config {
    return .{
        .magic = model_config.magic_bytes,
        .format_version = model_config.format_version,
        .architecture = @backingInt(model_config.Architecture.qwen3_asr),
        .audio_d_model = tiny_audio_d_model,
        .audio_layers = 1,
        .audio_attention_heads = tiny_audio_heads,
        .audio_ffn_dim = tiny_audio_ffn,
        .audio_downsample_hidden_size = tiny_audio_downsample,
        .audio_n_window = 50,
        .audio_n_window_infer = 800,
        .audio_max_position_steps = 13,
        .audio_output_dim = tiny_text_hidden,
        .mel_bins = 128,
        .audio_layer_norm_eps = checkpoint_config.audio_layer_norm_eps_default,
        .text_hidden_size = tiny_text_hidden,
        .text_layers = 1,
        .text_attention_heads = tiny_text_heads,
        .text_key_value_heads = tiny_text_kv_heads,
        .text_head_dim = tiny_text_head_dim,
        .text_ffn_dim = tiny_text_ffn,
        .vocab_size = tiny_vocab,
        .text_rms_norm_eps = 1e-6,
        .rope_theta = 1000000.0,
        .max_positions = checkpoint_config.max_positions_default,
        .max_decode_tokens = checkpoint_config.max_decode_tokens_default,
        .token_audio_start = tiny_token_audio_start,
        .token_audio_end = tiny_token_audio_end,
        .token_audio_pad = tiny_token_audio_pad,
        .token_im_start = tiny_token_im_start,
        .token_im_end = tiny_token_im_end,
        .token_endoftext = tiny_token_endoftext,
        .token_asr_text = tiny_token_asr_text,
        .token_eos_primary = tiny_token_endoftext,
        .token_eos_secondary = tiny_token_im_end,
        .token_pad = tiny_token_endoftext,
    };
}

/// `config.json` in the vLLM export's shape: the model's own configuration
/// wrapped in `thinker_config`, with the audio marker ids beside it and the
/// prompt markers left to the tokenizer files.
const tiny_config_json =
    \\{"architectures":["Qwen3ASRForConditionalGeneration"],"model_type":"qwen3_asr",
    \\ "thinker_config":{"model_type":"qwen3_asr",
    \\  "audio_config":{"d_model":64,"encoder_layers":1,"encoder_attention_heads":4,
    \\    "encoder_ffn_dim":64,"downsample_hidden_size":64,"n_window":50,
    \\    "n_window_infer":800,"output_dim":64,"num_mel_bins":128,
    \\    "max_source_positions":1500},
    \\  "text_config":{"hidden_size":64,"num_hidden_layers":1,"num_attention_heads":2,
    \\    "num_key_value_heads":1,"head_dim":32,"intermediate_size":64,
    \\    "vocab_size":128,"rms_norm_eps":1e-06,"rope_theta":1000000,
    \\    "max_position_embeddings":65536,"tie_word_embeddings":TIE},
    \\  "audio_start_token_id":120,"audio_end_token_id":121,"audio_token_id":122,
    \\  "dtype":"bfloat16"}}
;

/// The tiny `config.json`, with `TIE` replaced by the tiedness the case needs.
fn tinyConfigJson(arena: std.mem.Allocator, tied: bool) ![]const u8 {
    return std.mem.replaceOwned(u8, arena, tiny_config_json, "TIE", if (tied) "true" else "false");
}

const TinyImageOptions = struct {
    /// What the checkpoint says about the decoder's output projection.
    output: TinyOutput = .absent,
    /// What `tie_word_embeddings` claims, which must agree with `output`: a
    /// checkpoint that declares untied weights must ship a head of its own.
    tied_flag: bool = true,
    /// Leave this inventory position out of the checkpoint.
    omit: ?container.TensorKind = null,
    /// Store one tensor with a second extent one larger than the shape the
    /// inventory declares.
    break_shape_of: ?container.TensorKind = null,
    /// Add a tensor for a layer the configuration does not describe.
    extra_layer: bool = false,
};

/// Writes a complete checkpoint into `dir`: configuration, tokenizer, and an
/// image holding every tensor the inventory requires.
fn writeTinyCheckpoint(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    options: TinyImageOptions,
) !void {
    try dir.writeFile(io, .{
        .sub_path = "config.json",
        .data = try tinyConfigJson(arena, options.tied_flag),
    });
    try dir.writeFile(io, .{
        .sub_path = "generation_config.json",
        .data =
        \\{"eos_token_id":[125,124],"pad_token_id":125,"do_sample":false}
        ,
    });
    try dir.writeFile(io, .{
        .sub_path = "tokenizer_config.json",
        .data =
        \\{"added_tokens_decoder":{
        \\  "120":{"content":"<|audio_start|>"},"121":{"content":"<|audio_end|>"},
        \\  "122":{"content":"<|audio_pad|>"},"123":{"content":"<|im_start|>"},
        \\  "124":{"content":"<|im_end|>"},"125":{"content":"<|endoftext|>"},
        \\  "126":{"content":"<asr_text>"}}}
        ,
    });
    try dir.writeFile(io, .{ .sub_path = "merges.txt", .data = "#version: 0.2\nt0 t1\nt1 t2\n" });
    try writeTinyVocab(arena, io, dir);
    try writeTinyImage(arena, io, dir, options);
}

/// `vocab.json` with ids 0..119 and no gaps. The added tokens in
/// `tokenizer_config.json` cover 120..126, so the union is contiguous.
fn writeTinyVocab(arena: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !void {
    var buffer: std.Io.Writer.Allocating = .init(arena);
    defer buffer.deinit();
    const out = &buffer.writer;
    try out.writeByte('{');
    var id: u32 = 0;
    while (id < tiny_token_audio_start) : (id += 1) {
        if (id != 0) try out.writeByte(',');
        try out.print("\"t{d}\":{d}", .{ id, id });
    }
    try out.writeByte('}');
    try dir.writeFile(io, .{ .sub_path = "vocab.json", .data = buffer.written() });
}

/// Values are spread over the range a real weight occupies, so quantization has
/// to choose a real scale and so a stored row is visibly not a zero row.
fn tinyValues(arena: std.mem.Allocator, count: u64, seed: f32) ![]u8 {
    const bytes = try arena.alloc(u8, @intCast(count * 2));
    var index: u64 = 0;
    while (index < count) : (index += 1) {
        const position: f32 = @floatFromInt(index);
        const value = @sin(position * 0.017 + seed) * 0.25 + @cos(position * 0.0031) * 0.05;
        std.mem.writeInt(u16, bytes[@intCast(index * 2)..][0..2], half_float.toBf16(value), .little);
    }
    return bytes;
}

/// Builds `model.safetensors` from the inventory, so the image and the layout
/// cannot disagree by construction -- except where a test asks them to.
fn writeTinyImage(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    options: TinyImageOptions,
) !void {
    const config = tinyConfig();
    var tensors: std.ArrayList(safetensors.StoredTensor) = .empty;
    var iterator = layout.Iterator.init(&config);
    var seed: f32 = 0;
    while (iterator.next()) |required| {
        // The output projection is the checkpoint's choice, not the
        // inventory's: `appendOutputTensor` adds it when the checkpoint ships
        // one, because a tied checkpoint has none.
        if (required.kind == .decoder_output_weight) continue;
        const omit = options.omit != null and options.omit.? == required.kind;
        const break_shape = options.break_shape_of != null and
            options.break_shape_of.? == required.kind;
        var dims = required.shape.dims;
        if (break_shape) dims[1] += 1;
        const count = productOf(dims[0..required.shape.rank]);
        const payload_count = if (break_shape) productOf(dims[0..required.shape.rank]) else count;
        const entry = safetensors.StoredTensor{
            .name = try tinyName(arena, required),
            .dtype_name = "BF16",
            .dims = try arena.dupe(u32, dims[0..required.shape.rank]),
            .payload = try tinyValues(arena, payload_count, seed),
        };
        seed += 0.37;
        if (!omit) try tensors.append(arena, entry);
    }
    if (options.extra_layer) {
        var buffer: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(
            &buffer,
            "thinker.model.layers.{d}.mlp.up_proj.weight",
            .{config.text_layers},
        );
        try tensors.append(arena, .{
            .name = try arena.dupe(u8, name),
            .dtype_name = "BF16",
            .dims = try arena.dupe(u32, &.{ tiny_text_ffn, tiny_text_hidden }),
            .payload = try tinyValues(
                arena,
                @as(u64, tiny_text_ffn) * tiny_text_hidden,
                seed,
            ),
        });
    }
    try appendOutputTensor(arena, &tensors, options.output);
    std.mem.sort(safetensors.StoredTensor, tensors.items, {}, storedTensorLessThan);

    const storage = try arena.alloc(u8, @intCast(safetensors.imageLength(tensors.items, 0)));
    const used = try safetensors.writeImage(storage, tensors.items, 0);
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = storage[0..used] });
}

/// What the checkpoint says about the decoder's output projection.
const TinyOutput = enum { absent, identical, distinct };

/// Adds the checkpoint's `lm_head`, which the native export does not ship and
/// the vLLM export ships with the embedding's own weights. `identical` copies
/// the embedding's payload, which is what the released vLLM checkpoint does;
/// `distinct` gives it values of its own.
fn appendOutputTensor(
    arena: std.mem.Allocator,
    tensors: *std.ArrayList(safetensors.StoredTensor),
    output: TinyOutput,
) !void {
    if (output == .absent) return;
    const embed_name = "thinker.model.embed_tokens.weight";
    var embedding: ?safetensors.StoredTensor = null;
    for (tensors.items) |entry| {
        if (std.mem.eql(u8, entry.name, embed_name)) embedding = entry;
    }
    const source = embedding orelse return error.TestFixtureIncomplete;
    assert(source.dims.len == 2);
    const payload = switch (output) {
        .identical => source.payload,
        .distinct => try tinyValues(arena, productOf(source.dims), 17.25),
        .absent => unreachable,
    };
    try tensors.append(arena, .{
        .name = "thinker.lm_head.weight",
        .dtype_name = "BF16",
        .dims = source.dims,
        .payload = payload,
    });
}

fn productOf(dims: []const u32) u64 {
    var product: u64 = 1;
    for (dims) |dim| product *= dim;
    return product;
}

fn tinyName(arena: std.mem.Allocator, required: layout.Required) ![]const u8 {
    var buffer: [128]u8 = undefined;
    const name = try tensor_names.officialName(required.kind, required.layer, &buffer);
    return arena.dupe(u8, name);
}

fn storedTensorLessThan(context: void, lhs: safetensors.StoredTensor, rhs: safetensors.StoredTensor) bool {
    _ = context;
    return std.mem.lessThan(u8, lhs.name, rhs.name);
}

// --- the conversion contract ------------------------------------------------

/// Reads a file from the model directory the conversion produced.
fn readModelFile(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    name: []const u8,
) ![]u8 {
    return dir.readFileAlloc(io, name, arena, .limited(safetensors.checkpoint_bytes_max));
}

/// What a test expects of the converted directory beyond the report's totals.
const ExpectedModel = struct {
    /// True when the directory holds no `decoder.output.weight`.
    output_is_tied: bool,
    /// Shards, and the layer range each one covers. A shard boundary must fall
    /// between layers, never inside one.
    layer_ranges: []const [2]u16,
};

/// The contract test: parse every produced file the way the runtime does, and
/// check that every tensor the inventory requires is present exactly once, with
/// the shape and format the storage policy chose.
fn expectModelDirectory(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    config: *const model_config.Config,
    report: Report,
    expected: ExpectedModel,
) !void {
    try expectConfigFile(arena, io, dir, config);
    try expectTokenFile(arena, io, dir, report.token_count);

    const manifest_bytes = try readModelFile(arena, io, dir, "manifest.json");
    const value = try manifest.read(arena, manifest_bytes);
    try std.testing.expectEqualStrings(report.model_id, value.model_id);
    try std.testing.expectEqualStrings(report.quantization.name(), value.quantization);
    try std.testing.expectEqual(report.tensors, value.tensors);
    try std.testing.expectEqual(report.parameters, value.parameters_approximate);
    try std.testing.expectEqual(report.payload_bytes, value.payload_bytes);
    try std.testing.expectEqual(report.token_count, value.token_count);
    try std.testing.expectEqual(report.output_is_tied, value.output_is_tied);
    try std.testing.expectEqual(expected.output_is_tied, value.output_is_tied);
    try std.testing.expectEqual(report.shards.len, value.shards.len);
    try std.testing.expectEqual(@as(u64, 160), value.config_bytes);
    try std.testing.expectEqualStrings("merges.txt", value.merges_file);

    // Every inventory position must be covered exactly once, except the output
    // projection when the model is tied.
    const positions = layout.Iterator.count(config);
    const seen = try arena.alloc(bool, positions);
    @memset(seen, false);
    _ = &seen;
    var covered: u32 = 0;
    for (report.shards, 0..) |shard, index| {
        covered += try expectShard(
            arena,
            io,
            dir,
            config,
            report.quantization,
            shard,
            seen,
        );
        const range = expected.layer_ranges[index];
        try std.testing.expectEqual(range[0], shard.first_layer);
        try std.testing.expectEqual(range[1], shard.last_layer);
        // The manifest's digest must be the file's digest.
        const bytes = try readModelFile(arena, io, dir, shard.name);
        const hex = manifest.sha256Hex(bytes);
        try std.testing.expectEqualStrings(&shard.sha256_hex, &hex);
        try std.testing.expectEqual(bytes.len, shard.bytes);
    }
    try std.testing.expectEqual(report.tensors, covered);

    var position: u32 = 0;
    while (position < positions) : (position += 1) {
        const required = layout.at(config, position).?;
        const tied_output = expected.output_is_tied and
            required.kind == .decoder_output_weight;
        try std.testing.expectEqual(!tied_output, seen[position]);
    }
}

/// Checks one shard: parses it, verifies its checksum, and checks every index
/// entry against the inventory. Returns the number of tensors it held.
fn expectShard(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    config: *const model_config.Config,
    quantization: dtype.Format,
    shard: ShardReport,
    seen: []bool,
) !u32 {
    const bytes = try dir.readFileAllocOptions(
        io,
        shard.name,
        arena,
        .limited(safetensors.checkpoint_bytes_max),
        .@"16",
        null,
    );
    const file = try container.File.parse(bytes);
    try file.verifyChecksum();
    try std.testing.expectEqual(shard.tensor_count, file.header.tensor_count);
    try std.testing.expectEqual(shard.payload_checksum, file.header.payload_checksum);
    try std.testing.expectEqual(shard.bytes, bytes.len);

    var previous_layer: u16 = 0;
    var first = true;
    for (file.index) |*entry| {
        const required = layout.find(
            config,
            @fromBackingInt(entry.kind),
            entry.layer,
        ) orelse return error.TestUnexpectedResult;
        try std.testing.expect(required.shape.eql(&(try entry.shape())));
        try std.testing.expectEqual(
            @backingInt(required.format(quantization)),
            entry.format,
        );
        if (!first) try std.testing.expect(entry.layer >= previous_layer);
        previous_layer = entry.layer;
        first = false;
        const position = tensor_names.positionOf(config, required.kind, required.layer).?;
        try std.testing.expect(!seen[position]);
        seen[position] = true;
    }
    return file.header.tensor_count;
}

fn expectConfigFile(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    config: *const model_config.Config,
) !void {
    const bytes = try dir.readFileAllocOptions(
        io,
        "config.bin",
        arena,
        .limited(1 << 20),
        .@"4",
        null,
    );
    const parsed = try model_config.parse(bytes);
    try std.testing.expectEqualSlices(
        u8,
        std.mem.asBytes(config),
        std.mem.asBytes(parsed),
    );
}

/// The offset of a token in the table, read the way the runtime reads it.
fn tokenOffset(bytes: []const u8, id: u32) u32 {
    return std.mem.readInt(u32, bytes[@as(usize, id) * 4 ..][0..4], .little);
}

/// The token table is `count + 1` offsets followed by the concatenated strings,
/// so the added tokens must land at the ids the tokenizer files gave them.
fn expectTokenFile(
    arena: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    token_count: u32,
) !void {
    const bytes = try readModelFile(arena, io, dir, "tokens.bin");
    const offsets_bytes = @as(usize, token_count + 1) * 4;
    try std.testing.expect(bytes.len > offsets_bytes);
    const text = bytes[offsets_bytes..];
    try std.testing.expectEqual(@as(u32, 0), tokenOffset(bytes, 0));
    var id: u32 = 0;
    while (id < token_count) : (id += 1) {
        try std.testing.expect(tokenOffset(bytes, id) <= tokenOffset(bytes, id + 1));
    }
    try std.testing.expectEqual(@as(usize, text.len), @as(usize, tokenOffset(bytes, token_count)));
    try std.testing.expectEqualStrings(
        "t0",
        text[tokenOffset(bytes, 0)..tokenOffset(bytes, 1)],
    );
    try std.testing.expectEqualStrings(
        "<|audio_start|>",
        text[tokenOffset(bytes, tiny_token_audio_start)..][0..15],
    );
    try std.testing.expectEqualStrings(
        "<asr_text>",
        text[tokenOffset(bytes, tiny_token_asr_text)..][0..10],
    );
}

/// Converts the tiny checkpoint and returns the report, with the directories
/// left open for the caller's checks.
fn convertTiny(
    arena: std.mem.Allocator,
    io: std.Io,
    input: std.Io.Dir,
    output: std.Io.Dir,
    shard_bytes_max: u64,
    verify: bool,
    diagnostics: *Diagnostics,
) !Report {
    return run(arena, io, .{
        .input_dir = input,
        .output_dir = output,
        .input_name = "checkpoint",
        .output_name = "model",
        .model_id = "tiny-qwen3-asr",
        .quantization = .q4,
        .shard_bytes_max = shard_bytes_max,
        .verify = verify,
    }, diagnostics);
}

/// The three shards the tiny model produces at a 8 KiB budget: the layer-zero
/// tensors (the embeddings and the convolution stack, which do not fit and are
/// not split), then the single decoder layer, then the single audio layer.
const tiny_layer_ranges = [_][2]u16{
    .{ 0, 0 },
    .{ 1, 1 },
    .{ container.audio_layer_base, container.audio_layer_base },
};

test "a synthetic checkpoint converts into a model directory the runtime can load" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var input = try tmp.dir.createDirPathOpen(io, "checkpoint", .{
        .open_options = .{ .iterate = true },
    });
    defer input.close(io);
    var output = try tmp.dir.createDirPathOpen(io, "model", .{});
    defer output.close(io);

    // The native export's shape: no `lm_head`, because the weights are tied.
    try writeTinyCheckpoint(arena, io, input, .{});

    const config = tinyConfig();
    var diagnostics: Diagnostics = .{};
    const report = try convertTiny(arena, io, input, output, 8192, true, &diagnostics);
    try std.testing.expectEqualStrings("", diagnostics.tensor_name);

    // Nine fixed, four projector, three decoder fixed, eleven decoder layer
    // tensors, sixteen audio layer tensors: 43 in the inventory, minus the tied
    // output projection.
    try std.testing.expectEqual(@as(u32, 43 - 1), report.tensors);
    try std.testing.expect(report.output_is_tied);
    try std.testing.expectEqual(@as(u32, 127), report.token_count);
    try std.testing.expectEqual(@as(usize, 3), report.shards.len);
    try std.testing.expect(report.merges_bytes > 0);
    // q4 weights plus f32 norms and f16 convolutions land between 4 and 10 bits
    // per weight for a model this small.
    try std.testing.expect(report.bits_per_weight > 4.0);
    try std.testing.expect(report.bits_per_weight < 10.0);
    try std.testing.expect(report.output_bytes > report.payload_bytes);
    try std.testing.expectEqual(@as(u32, 0), report.unused_tensors);

    // Verification parsed every shard and compared every tensor in it.
    try std.testing.expect(report.verify.checked);
    try std.testing.expectEqual(@as(u32, 3), report.verify.shards_parsed);
    try std.testing.expectEqual(report.tensors, report.verify.tensors_compared);
    try std.testing.expect(report.verify.elements_compared > 0);

    try expectModelDirectory(arena, io, output, &config, report, .{
        .output_is_tied = true,
        .layer_ranges = &tiny_layer_ranges,
    });
    try expectF32TensorRoundTrips(arena, io, input, output, &config);
}

/// An f32 tensor's payload is the decoded checkpoint values, bit for bit, so
/// comparing the two is the cheapest proof that the element path reads the
/// checkpoint and writes the container correctly.
fn expectF32TensorRoundTrips(
    arena: std.mem.Allocator,
    io: std.Io,
    input: std.Io.Dir,
    output: std.Io.Dir,
    config: *const model_config.Config,
) !void {
    const image = try readModelFile(arena, io, input, "model.safetensors");
    const source = try safetensors.File.parse(arena, image, .{});
    const required = layout.find(config, .audio_conv1_bias, 0).?;
    var expected: [tiny_audio_downsample]f32 = undefined;
    try source.find("thinker.audio_tower.conv2d1.bias").?.decodeF32(&expected);

    const manifest_bytes = try readModelFile(arena, io, output, "manifest.json");
    const value = try manifest.read(arena, manifest_bytes);
    for (value.shards) |shard| {
        const bytes = try output.readFileAllocOptions(
            io,
            shard.name,
            arena,
            .limited(safetensors.checkpoint_bytes_max),
            .@"16",
            null,
        );
        const file = try container.File.parse(bytes);
        const stored = file.f32Values(@backingInt(required.kind), required.layer) orelse continue;
        try std.testing.expectEqualSlices(f32, &expected, stored);
        return;
    }
    return error.TestUnexpectedResult;
}

test "an output projection identical to the embedding is stored once" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var input = try tmp.dir.createDirPathOpen(io, "checkpoint", .{
        .open_options = .{ .iterate = true },
    });
    defer input.close(io);
    var output = try tmp.dir.createDirPathOpen(io, "model", .{});
    defer output.close(io);

    // The vLLM export's shape: `lm_head` present, byte-identical to the
    // embedding on disk.
    try writeTinyCheckpoint(arena, io, input, .{ .output = .identical });

    const config = tinyConfig();
    var diagnostics: Diagnostics = .{};
    const report = try convertTiny(arena, io, input, output, 8192, false, &diagnostics);
    try std.testing.expect(report.output_is_tied);
    try std.testing.expectEqual(@as(u32, 43 - 1), report.tensors);
    try expectModelDirectory(arena, io, output, &config, report, .{
        .output_is_tied = true,
        .layer_ranges = &tiny_layer_ranges,
    });
}

test "an output projection that differs from the embedding is stored and untied" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var input = try tmp.dir.createDirPathOpen(io, "checkpoint", .{
        .open_options = .{ .iterate = true },
    });
    defer input.close(io);
    var output = try tmp.dir.createDirPathOpen(io, "model", .{});
    defer output.close(io);

    try writeTinyCheckpoint(arena, io, input, .{
        .output = .distinct,
        .tied_flag = false,
    });

    const config = tinyConfig();
    var diagnostics: Diagnostics = .{};
    const report = try convertTiny(arena, io, input, output, 8192, true, &diagnostics);
    try std.testing.expect(!report.output_is_tied);
    try std.testing.expectEqual(@as(u32, 43), report.tensors);
    try expectModelDirectory(arena, io, output, &config, report, .{
        .output_is_tied = false,
        .layer_ranges = &tiny_layer_ranges,
    });
}

test "a checkpoint missing a tensor the runtime needs is rejected by name" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var input = try tmp.dir.createDirPathOpen(io, "checkpoint", .{
        .open_options = .{ .iterate = true },
    });
    defer input.close(io);
    var output = try tmp.dir.createDirPathOpen(io, "model", .{});
    defer output.close(io);

    try writeTinyCheckpoint(arena, io, input, .{ .omit = .decoder_layer_ffn_up_weight });

    var diagnostics: Diagnostics = .{};
    try std.testing.expectError(
        Error.MissingTensor,
        convertTiny(arena, io, input, output, 8192, false, &diagnostics),
    );
    // The diagnostic names the tensor in the checkpoint's own spelling, which is
    // what an operator can search the file for.
    try std.testing.expectEqualStrings(
        "thinker.model.layers.0.mlp.up_proj.weight",
        diagnostics.tensor_name,
    );
    try std.testing.expectEqualStrings("missing from the checkpoint", diagnostics.detail);
}

test "a checkpoint whose weights claim to be untied must ship an output projection" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var input = try tmp.dir.createDirPathOpen(io, "checkpoint", .{
        .open_options = .{ .iterate = true },
    });
    defer input.close(io);
    var output = try tmp.dir.createDirPathOpen(io, "model", .{});
    defer output.close(io);

    try writeTinyCheckpoint(arena, io, input, .{ .tied_flag = false });

    var diagnostics: Diagnostics = .{};
    try std.testing.expectError(
        Error.UntiedOutputMissing,
        convertTiny(arena, io, input, output, 8192, false, &diagnostics),
    );
}

test "a tensor stored at a shape the inventory did not declare is rejected" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var input = try tmp.dir.createDirPathOpen(io, "checkpoint", .{
        .open_options = .{ .iterate = true },
    });
    defer input.close(io);
    var output = try tmp.dir.createDirPathOpen(io, "model", .{});
    defer output.close(io);

    try writeTinyCheckpoint(arena, io, input, .{ .break_shape_of = .projector_in_weight });

    var diagnostics: Diagnostics = .{};
    try std.testing.expectError(
        Error.ShapeMismatch,
        convertTiny(arena, io, input, output, 8192, false, &diagnostics),
    );
    try std.testing.expectEqualStrings(
        "thinker.audio_tower.proj1.weight",
        diagnostics.tensor_name,
    );
}

test "a checkpoint with more layers than its configuration describes is rejected" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var input = try tmp.dir.createDirPathOpen(io, "checkpoint", .{
        .open_options = .{ .iterate = true },
    });
    defer input.close(io);
    var output = try tmp.dir.createDirPathOpen(io, "model", .{});
    defer output.close(io);

    try writeTinyCheckpoint(arena, io, input, .{ .extra_layer = true });

    var diagnostics: Diagnostics = .{};
    try std.testing.expectError(
        Error.UnexpectedTensor,
        convertTiny(arena, io, input, output, 8192, false, &diagnostics),
    );
    try std.testing.expectEqualStrings(
        "thinker.model.layers.1.mlp.up_proj.weight",
        diagnostics.tensor_name,
    );
}

test "a directory that is not a checkpoint is rejected before anything is written" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var output = try tmp.dir.createDirPathOpen(io, "model", .{});
    defer output.close(io);
    var diagnostics: Diagnostics = .{};

    // Nothing at all: the tokenizer is read first, and its absence is the first
    // thing that stops a conversion.
    var empty = try tmp.dir.createDirPathOpen(io, "empty", .{
        .open_options = .{ .iterate = true },
    });
    defer empty.close(io);
    try std.testing.expectError(
        tokenizer_file.Error.MissingTokenizerFiles,
        convertTiny(arena, io, empty, output, 8192, false, &diagnostics),
    );

    // A tokenizer but no configuration.
    var no_config = try tmp.dir.createDirPathOpen(io, "no-config", .{
        .open_options = .{ .iterate = true },
    });
    defer no_config.close(io);
    try writeTinyVocab(arena, io, no_config);
    try std.testing.expectError(
        Error.MissingConfigFile,
        convertTiny(arena, io, no_config, output, 8192, false, &diagnostics),
    );

    // A configuration but no weights.
    var no_weights = try tmp.dir.createDirPathOpen(io, "no-weights", .{
        .open_options = .{ .iterate = true },
    });
    defer no_weights.close(io);
    try writeTinyCheckpoint(arena, io, no_weights, .{});
    try no_weights.deleteFile(io, "model.safetensors");
    try std.testing.expectError(
        safetensors.Error.NoSafetensorsFile,
        convertTiny(arena, io, no_weights, output, 8192, false, &diagnostics),
    );
}
