//! The `.qw` shard container.
//!
//! A model is a directory: a JSON manifest for the JavaScript side, a binary
//! configuration, a tokenizer table, and one or more `.qw` shards. This file
//! defines the shard.
//!
//! # Layer numbering
//!
//! Entries sort by `(layer, kind)`, and a tensor's `layer` field uses one
//! numbering for the whole model:
//!
//!     0                      tensors that belong to no layer: the convolution
//!                            stack, the projector, the token embedding, the
//!                            output projection, the final norm
//!     1..=text_layers        decoder layers
//!     audio_layer_base + i   audio tower layers
//!
//! Layer-first ordering is what lets one shard hold a contiguous range of
//! layers with a correctly sorted index, which in turn lets the browser fetch
//! the weights for the layers it is about to run.
//!
//! # Why an index of kinds rather than names
//!
//! A browser runtime should never build a string-keyed hash map of 600 tensor
//! names to find its weights: the names would have to be shipped, hashed, and
//! matched, and a typo in either the converter or the runtime would only surface
//! as a missing tensor at load time. Instead every tensor is identified by a
//! `(TensorKind, layer)` pair, both fixed-width integers. The runtime asks for
//! exactly what the architecture module knows it needs, the converter asserts
//! that everything required was emitted, and a lookup is a bounded scan over a
//! sorted index.
//!
//! # Layout
//!
//!     offset 0                        header (32 bytes)
//!     offset header_bytes             tensor index (tensor_count * 40 bytes)
//!     offset payload_offset_bytes     payload (aligned, tensor data in index order)
//!
//! All integers are little endian. Every tensor's payload range is relative to
//! `payload_offset_bytes`, so a shard can be uploaded to a GPU or copied into
//! linear memory in one piece and every tensor addressed without further work.
//!
//! # Alignment
//!
//! The payload starts on a 256-byte boundary and each quantized tensor's scale
//! and data planes are aligned to 16 bytes within it (see `quant.planeLayout`).
//! That keeps 16-byte vector loads legal for both WASM SIMD and WGSL.

const std = @import("std");
const dtype = @import("dtype.zig");
const quant = @import("quant.zig");
const tensor = @import("tensor.zig");

pub const magic = "QWSHARD1";
pub const magic_bytes: [8]u8 = magic[0..8].*;
pub const format_version: u32 = 1;
pub const header_bytes: u32 = 32;
pub const index_entry_bytes: u32 = 40;
/// Payload start alignment. 256 bytes is the largest alignment any GPU upload
/// path here needs, and it keeps every plane inside a tensor aligned too.
pub const payload_alignment: u64 = 256;

pub const Error = error{
    BadMagic,
    BadVersion,
    Truncated,
    ShapeMismatch,
    ByteLengthMismatch,
    ChecksumMismatch,
    IndexNotSorted,
    TensorRangesOverlap,
    UnsupportedFormat,
    NotFound,
    CountOverflow,
    /// The shard buffer was not 16-byte aligned, which the zero-copy readers
    /// require. JavaScript allocates shard buffers with `qw_alloc(size, 16)`.
    Misaligned,
};

/// Header of a shard file. `extern` so its layout is exactly the 32 bytes on
/// disk, asserted below.
pub const Header = extern struct {
    /// `magic` followed by a NUL, so the field is exactly eight bytes.
    magic: [8]u8,
    format_version: u32,
    /// Bytes of tensor index that follow the header.
    index_len_bytes: u32,
    /// Offset of the payload from the start of the file.
    payload_offset_bytes: u32,
    payload_len_bytes: u32,
    tensor_count: u32,
    /// Low 32 bits of the FNV-1a hash of the payload.
    payload_checksum: u32,

    pub fn validate(self: *const Header) Error!void {
        if (!std.mem.eql(u8, &self.magic, &magic_bytes)) return Error.BadMagic;
        if (self.format_version != format_version) return Error.BadVersion;
        if (self.index_len_bytes != self.tensor_count * index_entry_bytes) {
            return Error.Truncated;
        }
        if (self.payload_offset_bytes < header_bytes + self.index_len_bytes) {
            return Error.Truncated;
        }
        if (self.payload_offset_bytes % payload_alignment != 0) return Error.Truncated;
    }

    /// Total file size the header implies.
    pub fn fileLenBytes(self: *const Header) u64 {
        return @as(u64, self.payload_offset_bytes) + self.payload_len_bytes;
    }
};

/// One tensor's location and shape.
pub const Entry = extern struct {
    /// `TensorKind` value.
    kind: u16,
    /// Layer index for per-layer kinds; zero otherwise.
    layer: u16,
    /// `dtype.Format` value.
    format: u8,
    /// Shape rank, 1..4.
    rank: u8,
    reserved: u16 = 0,
    dims: [tensor.rank_max]u32,
    /// Offset of this tensor's payload from the payload section start.
    offset_bytes: u64,
    /// Exact payload length, quantization padding included.
    len_bytes: u64,

    pub fn shape(self: *const Entry) Error!tensor.Shape {
        return tensor.Shape.fromDims(self.dims[0..self.rank]) catch return Error.ShapeMismatch;
    }

    pub fn storageFormat(self: *const Entry) Error!dtype.Format {
        const format: dtype.Format = @fromBackingInt(@intCast(self.format));
        if (!format.isKnown()) return Error.UnsupportedFormat;
        return format;
    }

    /// Validates the entry in isolation: known format, shape consistent with the
    /// format, and a recorded byte length that matches the shape exactly.
    pub fn validate(self: *const Entry) Error!void {
        const format = try self.storageFormat();
        const tensor_shape = try self.shape();
        tensor_shape.validateForFormat(format) catch return Error.ShapeMismatch;
        const expected = tensor_shape.byteLength(format) catch return Error.ShapeMismatch;
        if (expected != self.len_bytes) return Error.ByteLengthMismatch;
        if (format.isQuantized()) {
            if (tensor_shape.rank != 2) return Error.ShapeMismatch;
            const layout = quant.planeLayout(
                format,
                self.dims[0],
                self.dims[1],
                self.offset_bytes,
                quant.tensor_alignment_bytes,
            ) catch return Error.ShapeMismatch;
            if (layout.total_len_bytes != self.len_bytes) return Error.ByteLengthMismatch;
            if (layout.scales_offset_bytes != 0) return Error.ByteLengthMismatch;
        }
    }

    pub fn elementCount(self: *const Entry) Error!u64 {
        const tensor_shape = try self.shape();
        return tensor_shape.elementCount() catch Error.CountOverflow;
    }
};

/// A read-only view of a shard's bytes. The bytes stay owned by the caller:
/// `File` never copies and never allocates, so the same code path serves a
/// browser `ArrayBuffer`, a WASM linear-memory region, and a memory-mapped file.
pub const File = struct {
    bytes: []align(16) const u8,
    header: Header,
    index: []const Entry,

    /// Parses and fully validates a shard. `bytes` must be the entire shard and
    /// must start at a 16-byte boundary, because every tensor view handed out
    /// below aliases this buffer directly rather than copying.
    pub fn parse(bytes: []align(16) const u8) Error!File {
        if (bytes.len < header_bytes) return Error.Truncated;
        const header: *const Header = @ptrCast(bytes[0..header_bytes]);
        try header.validate();
        const index_start = header_bytes;
        const index_end = index_start + header.index_len_bytes;
        if (bytes.len < index_end) return Error.Truncated;
        if (bytes.len < header.fileLenBytes()) return Error.Truncated;

        const index: []const Entry = @alignCast(std.mem.bytesAsSlice(
            Entry,
            bytes[index_start..index_end],
        ));
        var file = File{ .bytes = bytes, .header = header.*, .index = index };
        try file.validate();
        return file;
    }

    fn validate(self: *const File) Error!void {
        var previous_end: u64 = 0;
        var previous_key: u32 = 0;
        for (self.index, 0..) |*entry, position| {
            try entry.validate();

            // The index is sorted by (layer, kind) so a lookup can stop early,
            // so layers cluster, and so a file is byte-deterministic for a given
            // input.
            const key = sortKey(entry.layer, entry.kind);
            if (position > 0 and key <= previous_key) return Error.IndexNotSorted;
            previous_key = key;

            // Tensors must tile the payload: no gaps are required, but no
            // overlaps and no out-of-range ranges are allowed.
            if (entry.offset_bytes < previous_end) return Error.TensorRangesOverlap;
            const end = entry.offset_bytes + entry.len_bytes;
            if (end > self.header.payload_len_bytes) return Error.Truncated;
            previous_end = end;
            if (entry.offset_bytes % quant.tensor_alignment_bytes != 0) {
                return Error.TensorRangesOverlap;
            }
        }
    }

    pub fn payload(self: *const File) []const u8 {
        const start = self.header.payload_offset_bytes;
        return self.bytes[start..][0..self.header.payload_len_bytes];
    }

    /// Finds a tensor by layer and kind. The index is sorted, so stop as soon as
    /// the search key is passed.
    pub fn find(self: *const File, kind: u16, layer: u16) ?*const Entry {
        const key = sortKey(layer, kind);
        for (self.index) |*entry| {
            const entry_key = sortKey(entry.layer, entry.kind);
            if (entry_key == key) return entry;
            if (entry_key > key) return null;
        }
        return null;
    }

    /// The bytes of one tensor, or null when it is absent.
    pub fn tensorBytes(self: *const File, kind: u16, layer: u16) ?[]const u8 {
        const entry = self.find(kind, layer) orelse return null;
        const start: usize = @intCast(entry.offset_bytes);
        const end: usize = @intCast(entry.offset_bytes + entry.len_bytes);
        return self.payload()[start..end];
    }

    /// Two planes of a quantized tensor: the f16 scales and the packed codes.
    pub fn quantizedPlanes(self: *const File, kind: u16, layer: u16) ?QuantizedPlanes {
        const entry = self.find(kind, layer) orelse return null;
        const format = entry.storageFormat() catch return null;
        if (!format.isQuantized()) return null;
        const layout = quant.planeLayout(
            format,
            entry.dims[0],
            entry.dims[1],
            entry.offset_bytes,
            quant.tensor_alignment_bytes,
        ) catch return null;
        const entry_bytes = self.tensorBytes(kind, layer) orelse return null;
        return .{
            .format = format,
            .entry = entry,
            .scales = entry_bytes[@intCast(layout.scales_offset_bytes)..][0..@intCast(layout.scales_len_bytes)],
            .data = entry_bytes[@intCast(layout.data_offset_bytes)..][0..@intCast(layout.data_len_bytes)],
        };
    }

    /// Element-format tensor as f32 values, when the format is `f32`.
    /// The slice aliases the shard buffer; the entry's 16-byte payload
    /// alignment makes the cast sound.
    pub fn f32Values(self: *const File, kind: u16, layer: u16) ?[]const f32 {
        const entry = self.find(kind, layer) orelse return null;
        const format = entry.storageFormat() catch return null;
        if (format != .f32) return null;
        const entry_bytes = self.tensorBytes(kind, layer) orelse return null;
        if (entry_bytes.len % 4 != 0) return null;
        return @alignCast(std.mem.bytesAsSlice(f32, entry_bytes));
    }

    /// Element-format tensor as f16 bit patterns.
    pub fn f16Bits(self: *const File, kind: u16, layer: u16) ?[]const u16 {
        const entry = self.find(kind, layer) orelse return null;
        const format = entry.storageFormat() catch return null;
        if (format != .f16) return null;
        const entry_bytes = self.tensorBytes(kind, layer) orelse return null;
        if (entry_bytes.len % 2 != 0) return null;
        return @alignCast(std.mem.bytesAsSlice(u16, entry_bytes));
    }

    pub fn checksum(self: *const File) u32 {
        return @truncate(std.hash.Fnv1a_64.hash(self.payload()));
    }

    /// Verifies the payload hash recorded in the header.
    pub fn verifyChecksum(self: *const File) Error!void {
        if (self.checksum() != self.header.payload_checksum) return Error.ChecksumMismatch;
    }
};

/// Sort key of an index entry: layer first, then kind.
pub fn sortKey(layer: u16, kind: u16) u32 {
    return (@as(u32, layer) << 16) | kind;
}

/// Layer number of the first audio tower layer. The gap leaves room for a
/// decoder with more layers than any released checkpoint without renumbering.
pub const audio_layer_base: u16 = 1024;

pub const QuantizedPlanes = struct {
    format: dtype.Format,
    entry: *const Entry,
    scales: []const u8,
    data: []const u8,

    pub fn rows(self: QuantizedPlanes) u32 {
        return self.entry.dims[0];
    }

    pub fn cols(self: QuantizedPlanes) u32 {
        return self.entry.dims[1];
    }
};

comptime {
    std.debug.assert(@sizeOf(Header) == header_bytes);
    std.debug.assert(@sizeOf(Entry) == index_entry_bytes);
    std.debug.assert(payload_alignment % quant.tensor_alignment_bytes == 0);
}

/// Tensor identity. Values are written into shard files, so they are permanent:
/// append new kinds, never renumber. Every per-layer kind is paired with the
/// layer index in the index entry.
pub const TensorKind = enum(u16) {
    audio_conv1_weight = 1,
    audio_conv1_bias,
    audio_conv2_weight,
    audio_conv2_bias,
    audio_conv3_weight,
    audio_conv3_bias,
    audio_conv_out_weight,
    audio_final_norm_weight,
    audio_final_norm_bias,
    audio_layer_attention_q_weight,
    audio_layer_attention_q_bias,
    audio_layer_attention_k_weight,
    audio_layer_attention_k_bias,
    audio_layer_attention_v_weight,
    audio_layer_attention_v_bias,
    audio_layer_attention_out_weight,
    audio_layer_attention_out_bias,
    audio_layer_attention_norm_weight,
    audio_layer_attention_norm_bias,
    audio_layer_ffn_in_weight,
    audio_layer_ffn_in_bias,
    audio_layer_ffn_out_weight,
    audio_layer_ffn_out_bias,
    audio_layer_final_norm_weight,
    audio_layer_final_norm_bias,
    projector_in_weight,
    projector_in_bias,
    projector_out_weight,
    projector_out_bias,
    decoder_embed_tokens_weight,
    decoder_final_norm_weight,
    decoder_output_weight,
    decoder_layer_attention_q_weight,
    decoder_layer_attention_k_weight,
    decoder_layer_attention_v_weight,
    decoder_layer_attention_out_weight,
    decoder_layer_attention_q_norm_weight,
    decoder_layer_attention_k_norm_weight,
    decoder_layer_attention_norm_weight,
    decoder_layer_ffn_norm_weight,
    decoder_layer_ffn_gate_weight,
    decoder_layer_ffn_up_weight,
    decoder_layer_ffn_down_weight,
    _,

    pub fn name(self: TensorKind) []const u8 {
        return switch (self) {
            .audio_conv1_weight => "audio.conv1.weight",
            .audio_conv1_bias => "audio.conv1.bias",
            .audio_conv2_weight => "audio.conv2.weight",
            .audio_conv2_bias => "audio.conv2.bias",
            .audio_conv3_weight => "audio.conv3.weight",
            .audio_conv3_bias => "audio.conv3.bias",
            .audio_conv_out_weight => "audio.conv_out.weight",
            .audio_final_norm_weight => "audio.final_norm.weight",
            .audio_final_norm_bias => "audio.final_norm.bias",
            .audio_layer_attention_q_weight => "audio.layer.attention.q.weight",
            .audio_layer_attention_q_bias => "audio.layer.attention.q.bias",
            .audio_layer_attention_k_weight => "audio.layer.attention.k.weight",
            .audio_layer_attention_k_bias => "audio.layer.attention.k.bias",
            .audio_layer_attention_v_weight => "audio.layer.attention.v.weight",
            .audio_layer_attention_v_bias => "audio.layer.attention.v.bias",
            .audio_layer_attention_out_weight => "audio.layer.attention.out.weight",
            .audio_layer_attention_out_bias => "audio.layer.attention.out.bias",
            .audio_layer_attention_norm_weight => "audio.layer.attention.norm.weight",
            .audio_layer_attention_norm_bias => "audio.layer.attention.norm.bias",
            .audio_layer_ffn_in_weight => "audio.layer.ffn.in.weight",
            .audio_layer_ffn_in_bias => "audio.layer.ffn.in.bias",
            .audio_layer_ffn_out_weight => "audio.layer.ffn.out.weight",
            .audio_layer_ffn_out_bias => "audio.layer.ffn.out.bias",
            .audio_layer_final_norm_weight => "audio.layer.final_norm.weight",
            .audio_layer_final_norm_bias => "audio.layer.final_norm.bias",
            .projector_in_weight => "projector.in.weight",
            .projector_in_bias => "projector.in.bias",
            .projector_out_weight => "projector.out.weight",
            .projector_out_bias => "projector.out.bias",
            .decoder_embed_tokens_weight => "decoder.embed_tokens.weight",
            .decoder_final_norm_weight => "decoder.final_norm.weight",
            .decoder_output_weight => "decoder.output.weight",
            .decoder_layer_attention_q_weight => "decoder.layer.attention.q.weight",
            .decoder_layer_attention_k_weight => "decoder.layer.attention.k.weight",
            .decoder_layer_attention_v_weight => "decoder.layer.attention.v.weight",
            .decoder_layer_attention_out_weight => "decoder.layer.attention.out.weight",
            .decoder_layer_attention_q_norm_weight => "decoder.layer.attention.q_norm.weight",
            .decoder_layer_attention_k_norm_weight => "decoder.layer.attention.k_norm.weight",
            .decoder_layer_attention_norm_weight => "decoder.layer.attention.norm.weight",
            .decoder_layer_ffn_norm_weight => "decoder.layer.ffn.norm.weight",
            .decoder_layer_ffn_gate_weight => "decoder.layer.ffn.gate.weight",
            .decoder_layer_ffn_up_weight => "decoder.layer.ffn.up.weight",
            .decoder_layer_ffn_down_weight => "decoder.layer.ffn.down.weight",
            else => "unknown",
        };
    }

    pub fn key(self: TensorKind, layer: u16) u32 {
        return (@as(u32, @backingInt(self)) << 16) | layer;
    }
};

test "header and entry are exactly the documented bytes" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Header));
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(Entry));
    try std.testing.expectEqual(@as(u32, 8), @as(u32, magic.len));
}

/// Layout of a 64x64 q4 tensor, derived at compile time from the layout code so
/// the test cannot drift from the format it is testing. Note that 64 rows of 64
/// columns is 64 groups, not one: a group spans a row, not a matrix.
const q4_64_layout = quant.planeLayout(
    .q4,
    64,
    64,
    0,
    quant.tensor_alignment_bytes,
) catch unreachable;

test "a minimal shard parses and exposes its tensor" {
    var payload = std.mem.zeroes([q4_64_layout.total_len_bytes]u8);
    // One real value, in the data plane, so the round trip below is not
    // trivially zero and would catch the two planes being swapped.
    payload[q4_64_layout.data_offset_bytes] = 0x0F;
    payload[q4_64_layout.scales_offset_bytes] = 0x3C;

    const storage = try testShard(&payload, .q4, 64, 64);
    defer std.testing.allocator.free(storage);
    const file = try File.parse(storage);
    try std.testing.expectEqual(@as(u32, 1), file.header.tensor_count);

    const planes = file.quantizedPlanes(@backingInt(TensorKind.audio_conv1_weight), 0).?;
    try std.testing.expectEqual(.q4, planes.format);
    try std.testing.expectEqual(@as(u32, 64), planes.rows());
    try std.testing.expectEqual(@as(u32, 64), planes.cols());
    try std.testing.expectEqual(@as(u8, 0x0F), planes.data[0]);
    try std.testing.expectEqual(@as(u8, 0x3C), planes.scales[0]);
    try file.verifyChecksum();

    var corrupted = try std.testing.allocator.dupe(u8, storage);
    defer std.testing.allocator.free(corrupted);
    corrupted[corrupted.len - 1] ^= 0xFF;
    const corrupted_file = try File.parse(@alignCast(corrupted));
    try std.testing.expectError(Error.ChecksumMismatch, corrupted_file.verifyChecksum());
}

test "lookup is by kind and layer, and missing tensors are absent not guessed" {
    // A q8 tensor of 64 x 64 has a payload of 64 * (2 + 64) = 4224 bytes, but
    // the builder only needs *a* payload, so this test uses a matching one.
    const zero_payload = std.mem.zeroes([4224]u8);
    const storage = try testShard(&zero_payload, .q8, 64, 64);
    defer std.testing.allocator.free(storage);
    const file = try File.parse(storage);
    try std.testing.expect(file.find(@backingInt(TensorKind.audio_conv1_weight), 0) != null);
    try std.testing.expect(file.find(@backingInt(TensorKind.audio_conv1_weight), 1) == null);
    try std.testing.expect(file.find(@backingInt(TensorKind.audio_conv2_weight), 0) == null);
    try std.testing.expect(file.tensorBytes(@backingInt(TensorKind.audio_conv1_weight), 0) != null);
    try std.testing.expect(file.f32Values(@backingInt(TensorKind.audio_conv1_weight), 0) == null);
}

test "corrupt headers and indexes are rejected" {
    const zero_payload = std.mem.zeroes([4224]u8);
    const good = try testShard(&zero_payload, .q8, 64, 64);
    defer std.testing.allocator.free(good);

    const cases = [_]struct { offset: usize, value: u32, expected: Error }{
        .{ .offset = 0, .value = 'X', .expected = Error.BadMagic },
        .{ .offset = 8, .value = 99, .expected = Error.BadVersion },
        .{ .offset = 24, .value = 3, .expected = Error.Truncated },
        .{ .offset = 12, .value = 0, .expected = Error.Truncated },
    };
    for (cases) |case| {
        const corrupt = try std.testing.allocator.dupe(u8, good);
        defer std.testing.allocator.free(corrupt);
        std.mem.writeInt(u32, corrupt[case.offset..][0..4], case.value, .little);
        try std.testing.expectError(case.expected, File.parse(@alignCast(corrupt)));
    }

    try std.testing.expectError(Error.Truncated, File.parse(good[0..16]));
    try std.testing.expectError(Error.Truncated, File.parse(good[0 .. good.len - 1]));
}

test "a tensor whose recorded length disagrees with its shape is rejected" {
    const zero_payload = std.mem.zeroes([4224]u8);
    const good = try testShard(&zero_payload, .q8, 64, 64);
    defer std.testing.allocator.free(good);
    const storage = try std.testing.allocator.dupe(u8, good);
    defer std.testing.allocator.free(storage);
    // The entry's len_bytes lives at offset 32 of the first index entry.
    std.mem.writeInt(u64, storage[header_bytes + 32 ..][0..8], 8, .little);
    try std.testing.expectError(Error.ByteLengthMismatch, File.parse(@alignCast(storage)));
}

test "tensor kinds are stable and uniquely named" {
    // These integers are on disk; renumbering them would silently reinterpret
    // existing model files.
    try std.testing.expectEqual(@as(u16, 1), @backingInt(TensorKind.audio_conv1_weight));
    try std.testing.expectEqual(@as(u16, 30), @backingInt(TensorKind.decoder_embed_tokens_weight));

    const kinds = [_]TensorKind{
        .audio_conv1_weight,
        .audio_layer_attention_q_weight,
        .projector_in_weight,
        .decoder_output_weight,
    };
    for (kinds, 0..) |kind, index| {
        for (kinds[0..index]) |other| {
            try std.testing.expect(!std.mem.eql(u8, kind.name(), other.name()));
        }
    }
    // Sorting is layer first, so a shard's index stays valid when it is
    // limited to a contiguous range of layers.
    try std.testing.expect(sortKey(1, 30) < sortKey(2, 1));
    try std.testing.expect(sortKey(0, 99) < sortKey(1, 1));
    try std.testing.expect(sortKey(audio_layer_base, 10) > sortKey(27, 43));
}

/// Builds a one-tensor shard for tests: header, index, padding, payload.
/// The caller owns the returned buffer.
fn testShard(payload: []const u8, format: dtype.Format, rows: u32, cols: u32) ![]align(16) u8 {
    const layout = try quant.planeLayout(format, rows, cols, 0, quant.tensor_alignment_bytes);
    if (payload.len != layout.total_len_bytes) return Error.ByteLengthMismatch;

    const index_len = index_entry_bytes;
    const payload_offset = std.mem.alignForward(u64, header_bytes + index_len, payload_alignment);
    const total = payload_offset + payload.len;
    const storage = try std.testing.allocator.alignedAlloc(u8, .@"16", @intCast(total));
    @memset(storage, 0);

    const header: *Header = @ptrCast(storage[0..header_bytes]);
    header.* = .{
        .magic = magic_bytes,
        .format_version = format_version,
        .index_len_bytes = index_len,
        .payload_offset_bytes = @intCast(payload_offset),
        .payload_len_bytes = @intCast(payload.len),
        .tensor_count = 1,
        .payload_checksum = @truncate(std.hash.Fnv1a_64.hash(payload)),
    };
    const entry: *Entry = @ptrCast(@alignCast(storage[header_bytes..][0..index_entry_bytes]));
    entry.* = .{
        .kind = @backingInt(TensorKind.audio_conv1_weight),
        .layer = 0,
        .format = @backingInt(format),
        .rank = 2,
        .dims = .{ rows, cols, 1, 1 },
        .offset_bytes = 0,
        .len_bytes = payload.len,
    };
    @memcpy(storage[@intCast(payload_offset)..][0..payload.len], payload);
    return storage;
}
