//! Zero-copy reader for `safetensors` checkpoint files.
//!
//! A safetensors file is three regions back to back:
//!
//!     offset 0                       header length (u64, little endian)
//!     offset 8                       JSON header, `header_length` bytes
//!     offset 8 + header_length       tensor payloads, packed in header order
//!
//! The JSON header maps a tensor name to its dtype, shape, and byte range
//! inside the data section. This reader validates all of that once, at parse
//! time, and then hands out slices that alias the caller's buffer: a 1.8 GiB
//! checkpoint must not be copied, and the converter must not hold two copies of
//! it while writing its output.
//!
//! The reader is stricter than the reference implementation, because a
//! checkpoint is an untrusted input and a mis-parse here would surface much
//! later as a silently wrong model:
//!
//!   * the header length must land inside the buffer and under a fixed bound
//!   * the header must be an object, and every entry must carry exactly the
//!     three documented keys with the documented JSON types
//!   * names must be unique, and every tensor's byte range must be inside the
//!     data section, shape-consistent, and non-overlapping with the others
//!   * only the dtypes this build can decode are accepted unless the caller
//!     asks for a lenient parse (the inspector does; the converter does not)
//!
//! Elements are decoded into the caller's f32 buffer rather than read in
//! place, because a checkpoint's payload is byte-aligned to the file layout,
//! not to an f32 or u16, and because the converter wants rows of f32 anyway.
//! The decoders take an aligned bulk path when the host is little-endian and
//! the payload happens to be aligned, and a scalar path otherwise; a test
//! compares the two paths on the same values.

const std = @import("std");
const builtin = @import("builtin");
const qwenscriber = @import("qwenscriber");

const assert = std.debug.assert;
const half_float = qwenscriber.half_float;
const tensor = qwenscriber.tensor;

/// Bytes of the little-endian header length that precedes the JSON header.
pub const header_length_bytes: usize = 8;

/// Largest JSON header this reader accepts. The released 1.7B checkpoint's
/// header is 78 KiB; the bound keeps a corrupt length field from asking for a
/// multi-gigabyte allocation before any validation has run.
pub const header_bytes_max: u64 = 64 << 20;

/// Element types this reader understands. `other` marks a dtype the checkpoint
/// declares but this build cannot decode; it is only reachable from a lenient
/// parse, and decoding it is an error rather than a guess.
pub const DType = enum {
    f32,
    f16,
    bf16,
    other,

    /// Bytes per element, or null for a dtype this build cannot decode.
    pub fn sizeBytes(self: DType) ?u32 {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .other => null,
        };
    }

    /// True when the dtype is a 16- or 32-bit float, which is what a conversion
    /// source has to be.
    pub fn isFloat(self: DType) bool {
        return switch (self) {
            .f32, .f16, .bf16 => true,
            .other => false,
        };
    }

    /// The spelling used in the safetensors header.
    pub fn providerName(self: DType) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .f16 => "F16",
            .bf16 => "BF16",
            .other => "unknown",
        };
    }
};

/// Maps a safetensors dtype string to a `DType`. Unknown strings become
/// `.other` rather than an error, so a lenient parse can report what the
/// checkpoint actually holds.
pub fn dtypeFromName(name: []const u8) DType {
    if (std.mem.eql(u8, name, "F32")) return .f32;
    if (std.mem.eql(u8, name, "F16")) return .f16;
    if (std.mem.eql(u8, name, "BF16")) return .bf16;
    return .other;
}

pub const Error = error{
    /// Fewer bytes than the 8-byte header length, or a header that runs past
    /// the end of the buffer.
    Truncated,
    /// The header length is zero or above `header_bytes_max`.
    HeaderLengthInvalid,
    /// The header is not JSON, or a tensor entry is not exactly
    /// `{dtype, shape, data_offsets}` with the documented JSON types.
    MalformedHeader,
    /// Two tensor entries share a name.
    DuplicateTensorName,
    /// A shape is empty, has more than `tensor.rank_max` extents, a zero
    /// extent, or an extent above `tensor.dims_max`.
    InvalidShape,
    /// The dtype value is not a JSON string.
    InvalidDTypeField,
    /// The dtype names a type this build cannot decode.
    UnsupportedDType,
    /// A tensor's recorded byte range does not match its shape.
    ByteLengthMismatch,
    /// A tensor's byte range falls outside the data section.
    DataOutOfRange,
    /// Two tensors' byte ranges overlap.
    TensorRangesOverlap,
    /// A decode range extends past the end of the tensor.
    DecodeOutOfRange,
    /// The caller's output buffer is not exactly as long as the requested range.
    DecodeLengthMismatch,
    /// The checkpoint directory holds no `*.safetensors` file.
    NoSafetensorsFile,
    /// The checkpoint has more `*.safetensors` files than this reader accepts.
    TooManySafetensorsFiles,
} || std.mem.Allocator.Error;

pub const ParseOptions = struct {
    /// Accept, but do not decode, dtypes this build does not know. The
    /// inspector sets this so it can report what a checkpoint holds; the
    /// converter leaves it false so an unsupported checkpoint fails at parse
    /// time rather than halfway through a conversion.
    allow_unknown_dtypes: bool = false,
};

/// One tensor's header entry plus a slice of its payload.
pub const Tensor = struct {
    /// Name as written in the header. Aliases the checkpoint buffer unless the
    /// JSON escaped it.
    name: []const u8,
    /// The dtype string exactly as written, kept for error messages about
    /// unsupported types.
    dtype_name: []const u8,
    dtype: DType,
    shape: tensor.Shape,
    /// Offset of the payload from the start of the data section.
    offset_bytes: u64,
    /// Payload, aliasing the checkpoint buffer.
    data: []const u8,

    pub fn dtypeOf(self: *const Tensor) DType {
        return self.dtype;
    }

    pub fn shapeOf(self: *const Tensor) tensor.Shape {
        return self.shape;
    }

    pub fn bytesOf(self: *const Tensor) []const u8 {
        return self.data;
    }

    /// Product of the shape's extents.
    pub fn elementCount(self: *const Tensor) Error!u64 {
        return self.shape.elementCount() catch Error.InvalidShape;
    }

    /// Decodes the whole tensor into `out`, which must be exactly as long as
    /// the element count.
    pub fn decodeF32(self: *const Tensor, out: []f32) Error!void {
        const count = try self.elementCount();
        if (out.len != count) return Error.DecodeLengthMismatch;
        try self.decodeRangeF32(0, out);
    }

    /// Decodes `out.len` elements starting at `element_offset`.
    pub fn decodeRangeF32(
        self: *const Tensor,
        element_offset: u64,
        out: []f32,
    ) Error!void {
        const count = try self.elementCount();
        if (element_offset > count) return Error.DecodeOutOfRange;
        if (out.len > count - element_offset) return Error.DecodeOutOfRange;
        assert(element_offset + out.len <= count);

        switch (self.dtype) {
            .f32 => widenF32(self.data, element_offset, out),
            .f16 => widenF16(self.data, element_offset, out),
            .bf16 => widenBf16(self.data, element_offset, out),
            .other => return Error.UnsupportedDType,
        }
    }
};

/// A parsed checkpoint file. `tensors` is arena-allocated and the payload
/// slices alias `bytes`, which the caller keeps alive.
pub const File = struct {
    bytes: []const u8,
    tensors: []const Tensor,

    pub fn parse(
        arena: std.mem.Allocator,
        bytes: []const u8,
        options: ParseOptions,
    ) Error!File {
        if (bytes.len < header_length_bytes) return Error.Truncated;
        const header_length = std.mem.readInt(u64, bytes[0..8], .little);
        if (header_length == 0) return Error.HeaderLengthInvalid;
        if (header_length > header_bytes_max) return Error.HeaderLengthInvalid;
        if (header_length > bytes.len - header_length_bytes) return Error.Truncated;
        const header_bytes = bytes[header_length_bytes..][0..@intCast(header_length)];
        const data = bytes[header_length_bytes + @as(usize, @intCast(header_length)) ..];
        assert(header_bytes.len == header_length);

        const root = std.json.parseFromSliceLeaky(
            std.json.Value,
            arena,
            header_bytes,
            .{},
        ) catch |err| {
            return switch (err) {
                error.OutOfMemory => Error.OutOfMemory,
                // The default duplicate-field behavior is `error`, which is
                // exactly the guarantee this reader needs: a name may not be
                // defined twice, silently keeping one definition.
                error.DuplicateField => Error.DuplicateTensorName,
                else => Error.MalformedHeader,
            };
        };
        const object = switch (root) {
            .object => |object| object,
            else => return Error.MalformedHeader,
        };

        var tensor_count: u32 = 0;
        var entries = object.iterator();
        while (entries.next()) |entry| {
            if (isMetadataKey(entry.key_ptr.*)) continue;
            tensor_count += 1;
        }

        const tensors = try arena.alloc(Tensor, tensor_count);
        var position: u32 = 0;
        entries = object.iterator();
        while (entries.next()) |entry| {
            if (isMetadataKey(entry.key_ptr.*)) continue;
            assert(position < tensor_count);
            tensors[position] = try tensorFromValue(
                entry.key_ptr.*,
                entry.value_ptr.*,
                data,
                options,
            );
            position += 1;
        }
        assert(position == tensor_count);

        assert(position == tensor_count);
        try validateRanges(arena, tensors);
        assert(tensors.len <= object.count());
        return .{ .bytes = bytes, .tensors = tensors };
    }

    pub fn count(self: *const File) u32 {
        return @intCast(self.tensors.len);
    }

    pub fn at(self: *const File, index: u32) Error!*const Tensor {
        if (index >= self.tensors.len) return Error.DecodeOutOfRange;
        return &self.tensors[index];
    }

    /// The tensor with this exact name, or null. A linear scan: the largest
    /// released checkpoint has 708 tensors, and the converter looks each one up
    /// once, so a hash map would cost more to build than it saves.
    pub fn find(self: *const File, name: []const u8) ?*const Tensor {
        for (self.tensors) |*entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry;
        }
        return null;
    }

    /// Index of the tensor with this name, or null.
    pub fn indexOf(self: *const File, name: []const u8) ?u32 {
        for (self.tensors, 0..) |*entry, index| {
            if (std.mem.eql(u8, entry.name, name)) return @intCast(index);
        }
        return null;
    }

    /// Total payload bytes the file accounts for, gaps included.
    pub fn payloadBytes(self: *const File) u64 {
        var total: u64 = 0;
        for (self.tensors) |entry| {
            total = @max(total, entry.offset_bytes + entry.data.len);
        }
        return total;
    }
};

/// Largest checkpoint this reader will load. The 1.7B checkpoint is 3.5 GiB in
/// two files; the bound keeps a wrong path from reading a disk image.
pub const checkpoint_bytes_max: u64 = 8 << 30;

/// Largest number of `*.safetensors` files one checkpoint may consist of. The
/// released 1.7B checkpoint is split in two.
pub const checkpoint_files_max: u32 = 64;

/// A checkpoint directory's `*.safetensors` files, in name order.
///
/// A released checkpoint is one file (0.6B) or several (1.7B), and the
/// converter neither knows nor cares which: it looks tensors up by name across
/// all of them.
pub const Checkpoint = struct {
    files: []const File,
    /// File names in the same order as `files`, for reporting.
    file_names: []const []const u8,
    /// Every tensor of every file, in file order. Indexing this list is what a
    /// name classification refers to.
    tensors: []const *const Tensor,

    /// `dir` must have been opened with iteration capability
    /// (`OpenOptions.iterate = true`), because which files exist is the
    /// directory's answer, not the caller's.
    pub fn open(arena: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !Checkpoint {
        var file_names: std.ArrayList([]const u8) = .empty;
        var iterator = dir.iterate();
        while (try iterator.next(io)) |entry| {
            // Some filesystems report `.unknown` for a regular file; the name
            // test below is what decides, and a directory named
            // `x.safetensors` fails when it is read.
            if (entry.kind != .file and entry.kind != .unknown) continue;
            if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
            try file_names.append(arena, try arena.dupe(u8, entry.name));
        }
        if (file_names.items.len == 0) return Error.NoSafetensorsFile;
        if (file_names.items.len > checkpoint_files_max) return Error.TooManySafetensorsFiles;
        assert(file_names.items.len <= checkpoint_files_max);
        // A directory's entry order is the filesystem's, not the format's, so
        // the files are sorted: two conversions of one checkpoint must produce
        // the same report and the same shards.
        std.mem.sort([]const u8, file_names.items, {}, lessThanFileName);

        const files = try arena.alloc(File, file_names.items.len);
        for (file_names.items, 0..) |name, index| {
            const file_bytes = try dir.readFileAlloc(
                io,
                name,
                arena,
                .limited(checkpoint_bytes_max),
            );
            files[index] = try File.parse(arena, file_bytes, .{});
        }

        var tensor_count: u32 = 0;
        for (files) |file| tensor_count += file.count();
        const tensors = try arena.alloc(*const Tensor, tensor_count);
        var position: u32 = 0;
        for (files) |*file| {
            for (file.tensors) |*entry| {
                tensors[position] = entry;
                position += 1;
            }
        }
        assert(position == tensor_count);
        return .{
            .files = files,
            .file_names = file_names.items,
            .tensors = tensors,
        };
    }

    pub fn count(self: *const Checkpoint) u32 {
        return @intCast(self.tensors.len);
    }

    /// Bytes across every file, which is what the source costs to read.
    pub fn sourceBytes(self: *const Checkpoint) u64 {
        var total: u64 = 0;
        for (self.files) |file| total += file.bytes.len;
        return total;
    }

    /// The tensor with this exact name, or null.
    pub fn find(self: *const Checkpoint, name: []const u8) ?*const Tensor {
        for (self.tensors) |entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry;
        }
        return null;
    }

    /// Every tensor name, in file order, as a name classification consumes it.
    pub fn names(self: *const Checkpoint, arena: std.mem.Allocator) ![]const []const u8 {
        const all_names = try arena.alloc([]const u8, self.tensors.len);
        for (self.tensors, 0..) |entry, index| all_names[index] = entry.name;
        return all_names;
    }
};

fn lessThanFileName(context: void, lhs: []const u8, rhs: []const u8) bool {
    _ = context;
    return std.mem.lessThan(u8, lhs, rhs);
}

/// One entry of the header's top-level object is metadata rather than a tensor;
/// the reference implementation writes exactly this key.
fn isMetadataKey(key: []const u8) bool {
    return std.mem.eql(u8, key, "__metadata__");
}

fn tensorFromValue(
    name: []const u8,
    value: std.json.Value,
    data: []const u8,
    options: ParseOptions,
) Error!Tensor {
    const fields = switch (value) {
        .object => |object| object,
        else => return Error.MalformedHeader,
    };

    var dtype_name: ?[]const u8 = null;
    var dims: ?[]const std.json.Value = null;
    var offsets: ?[]const std.json.Value = null;

    var entries = fields.iterator();
    while (entries.next()) |entry| {
        const key = entry.key_ptr.*;
        if (std.mem.eql(u8, key, "dtype")) {
            dtype_name = switch (entry.value_ptr.*) {
                .string => |text| text,
                else => return Error.InvalidDTypeField,
            };
        } else if (std.mem.eql(u8, key, "shape")) {
            dims = switch (entry.value_ptr.*) {
                .array => |array| array.items,
                else => return Error.MalformedHeader,
            };
        } else if (std.mem.eql(u8, key, "data_offsets")) {
            offsets = switch (entry.value_ptr.*) {
                .array => |array| array.items,
                else => return Error.MalformedHeader,
            };
        } else {
            // An unrecognized key means the header is not the format this
            // reader validates; guessing which key to ignore is how a
            // version skew becomes a silently wrong tensor.
            return Error.MalformedHeader;
        }
    }

    const dtype_text = dtype_name orelse return Error.MalformedHeader;
    const extents = dims orelse return Error.MalformedHeader;
    const limits = offsets orelse return Error.MalformedHeader;
    if (limits.len != 2) return Error.MalformedHeader;

    const start = switch (limits[0]) {
        .integer => |number| number,
        else => return Error.MalformedHeader,
    };
    const end = switch (limits[1]) {
        .integer => |number| number,
        else => return Error.MalformedHeader,
    };
    if (start < 0) return Error.DataOutOfRange;
    if (end < start) return Error.DataOutOfRange;
    if (end > data.len) return Error.DataOutOfRange;

    const dtype = dtypeFromName(dtype_text);
    if (dtype == .other and !options.allow_unknown_dtypes) {
        return Error.UnsupportedDType;
    }

    var shape: tensor.Shape = .{};
    for (extents) |item| {
        const extent = switch (item) {
            .integer => |number| number,
            else => return Error.MalformedHeader,
        };
        if (shape.rank == tensor.rank_max) return Error.InvalidShape;
        if (extent <= 0) return Error.InvalidShape;
        if (extent > tensor.dims_max) return Error.InvalidShape;
        shape.dims[shape.rank] = @intCast(extent);
        shape.rank += 1;
    }
    if (shape.rank == 0) return Error.InvalidShape;
    assert(shape.rank <= tensor.rank_max);

    const payload = data[@intCast(start)..@intCast(end)];
    var entry = Tensor{
        .name = name,
        .dtype_name = dtype_text,
        .dtype = dtype,
        .shape = shape,
        .offset_bytes = @intCast(start),
        .data = payload,
    };
    if (dtype.sizeBytes()) |element_bytes| {
        const elements = entry.elementCount() catch return Error.InvalidShape;
        if (elements * element_bytes != payload.len) return Error.ByteLengthMismatch;
    }
    return entry;
}

/// Every tensor's range must sit inside the data section and no two ranges may
/// overlap. Overlap detection sorts a index array by offset, so it costs
/// O(n log n) rather than O(n^2) for a 708-tensor checkpoint.
fn validateRanges(arena: std.mem.Allocator, tensors: []const Tensor) Error!void {
    const order = try arena.alloc(u32, tensors.len);
    for (order, 0..) |*slot, index| slot.* = @intCast(index);
    std.mem.sort(u32, order, RangeOrder{ .tensors = tensors }, RangeOrder.lessThan);

    var previous_end: u64 = 0;
    for (order) |index| {
        const entry = tensors[index];
        assert(entry.data.len > 0); // A shape with a zero extent is rejected.
        if (entry.offset_bytes < previous_end) return Error.TensorRangesOverlap;
        previous_end = entry.offset_bytes + entry.data.len;
    }
}

const RangeOrder = struct {
    tensors: []const Tensor,

    fn lessThan(context: RangeOrder, lhs: u32, rhs: u32) bool {
        return context.tensors[lhs].offset_bytes < context.tensors[rhs].offset_bytes;
    }
};

/// True when a bulk reinterpretation of the payload is both endian-correct and
/// alignment-legal on this host.
fn canReinterpret(data: []const u8, comptime Element: type) bool {
    if (comptime builtin.cpu.arch.endian() != .little) return false;
    return std.mem.isAligned(@intFromPtr(data.ptr), @alignOf(Element));
}

fn widenF32(data: []const u8, element_offset: u64, out: []f32) void {
    const bytes = data[@intCast(element_offset * 4)..][0 .. out.len * 4];
    assert(bytes.len == out.len * 4);
    if (canReinterpret(bytes, f32)) {
        const source: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, bytes));
        @memcpy(out, source);
        return;
    }
    for (out, 0..) |*value, index| {
        const bits = std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little);
        value.* = @bitCast(bits);
    }
}

fn widenF16(data: []const u8, element_offset: u64, out: []f32) void {
    const bytes = data[@intCast(element_offset * 2)..][0 .. out.len * 2];
    assert(bytes.len == out.len * 2);
    if (canReinterpret(bytes, u16)) {
        const source: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, bytes));
        for (source, out) |bits, *value| value.* = half_float.fromF16(bits);
        return;
    }
    for (out, 0..) |*value, index| {
        const bits = std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little);
        value.* = half_float.fromF16(bits);
    }
}

fn widenBf16(data: []const u8, element_offset: u64, out: []f32) void {
    const bytes = data[@intCast(element_offset * 2)..][0 .. out.len * 2];
    assert(bytes.len == out.len * 2);
    if (canReinterpret(bytes, u16)) {
        const source: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, bytes));
        for (source, out) |bits, *value| value.* = half_float.fromBf16(bits);
        return;
    }
    for (out, 0..) |*value, index| {
        const bits = std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little);
        value.* = half_float.fromBf16(bits);
    }
}

/// One tensor of an image being built. Payloads are laid out in the order the
/// entries appear, which is what `safetensors` itself does.
pub const StoredTensor = struct {
    name: []const u8,
    dtype_name: []const u8,
    dims: []const u32,
    payload: []const u8,
};

/// Bytes the image of `tensors` occupies, so a caller can size its buffer
/// exactly. `header_pad` is available because trailing spaces in the JSON header
/// are legal and shift every payload, which is how a test forces the scalar
/// decode path.
pub fn imageLength(tensors: []const StoredTensor, header_pad: usize) u64 {
    var total: u64 = header_length_bytes + header_pad;
    for (tensors) |entry| total += entry.payload.len;
    // A rough allowance for the JSON itself: names, keys, and numbers. Callers
    // use `writeImage`, which reports the exact length.
    for (tensors) |entry| total += entry.name.len + entry.dims.len * 3 + 64;
    return total;
}

/// Writes a safetensors image into `storage`, returning how many bytes it used.
/// `storage` must be at least `imageLength` bytes.
pub fn writeImage(
    storage: []u8,
    tensors: []const StoredTensor,
    header_pad: usize,
) !usize {
    // The header has to be written before the payload offsets are known, and the
    // offsets have to be known before the header is written. Both are computed
    // from the payload lengths, which are known up front, so the JSON is emitted
    // into the caller's buffer first and the payload copied in after.
    var json = std.Io.Writer.fixed(storage[header_length_bytes..]);
    try writeJsonHeader(&json, tensors, header_pad);
    const header_len = json.end;
    const payload_start = header_length_bytes + header_len;

    var offset_end: u64 = 0;
    for (tensors) |entry| offset_end += entry.payload.len;
    if (payload_start + offset_end > storage.len) return Error.Truncated;

    std.mem.writeInt(u64, storage[0..8], header_len, .little);
    var write_offset: usize = payload_start;
    for (tensors) |entry| {
        @memcpy(storage[write_offset..][0..entry.payload.len], entry.payload);
        write_offset += entry.payload.len;
    }
    return write_offset;
}

fn writeJsonHeader(
    out: *std.Io.Writer,
    tensors: []const StoredTensor,
    header_pad: usize,
) !void {
    try out.writeByte('{');
    var offset_end: u64 = 0;
    for (tensors, 0..) |entry, index| {
        if (index != 0) try out.writeByte(',');
        const start = offset_end;
        offset_end += entry.payload.len;
        try out.print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{
            entry.name,
            entry.dtype_name,
        });
        for (entry.dims, 0..) |dim, dim_index| {
            if (dim_index != 0) try out.writeByte(',');
            try out.print("{d}", .{dim});
        }
        try out.print("],\"data_offsets\":[{d},{d}]}}", .{ start, offset_end });
    }
    try out.writeByte('}');
    for (0..header_pad) |_| try out.writeByte(' ');
}

/// Builds a well-formed safetensors image in a freshly allocated buffer.
fn buildImage(
    allocator: std.mem.Allocator,
    tensors: []const StoredTensor,
    header_pad: usize,
) ![]u8 {
    const storage = try allocator.alloc(u8, @intCast(imageLength(tensors, header_pad)));
    assert(storage.len >= header_length_bytes);
    const used = try writeImage(storage, tensors, header_pad);
    return storage[0..used];
}

/// Builds an image from a hand-written header, for the malformed cases.
fn buildRawImage(
    allocator: std.mem.Allocator,
    header: []const u8,
    payload: []const u8,
) ![]u8 {
    const image = try allocator.alloc(u8, header_length_bytes + header.len + payload.len);
    std.mem.writeInt(u64, image[0..8], header.len, .little);
    @memcpy(image[header_length_bytes..][0..header.len], header);
    @memcpy(image[header_length_bytes + header.len ..][0..payload.len], payload);
    return image;
}

fn littleEndianF32(values: []const f32, allocator: std.mem.Allocator) ![]u8 {
    const bytes = try allocator.alloc(u8, values.len * 4);
    for (values, 0..) |value, index| {
        std.mem.writeInt(u32, bytes[index * 4 ..][0..4], @bitCast(value), .little);
    }
    return bytes;
}

fn littleEndianF16(values: []const f32, allocator: std.mem.Allocator) ![]u8 {
    const bytes = try allocator.alloc(u8, values.len * 2);
    for (values, 0..) |value, index| {
        std.mem.writeInt(u16, bytes[index * 2 ..][0..2], half_float.toF16(value), .little);
    }
    return bytes;
}

fn littleEndianBf16(values: []const f32, allocator: std.mem.Allocator) ![]u8 {
    const bytes = try allocator.alloc(u8, values.len * 2);
    for (values, 0..) |value, index| {
        std.mem.writeInt(u16, bytes[index * 2 ..][0..2], half_float.toBf16(value), .little);
    }
    return bytes;
}

test "a well-formed image exposes every tensor with its dtype, shape and bytes" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const f32_values = [_]f32{ 1.0, -2.0, 0.5, 8.25 };
    const f16_values = [_]f32{ 0.25, -0.5, 3.0, 6.0, 12.0, -24.0 };
    const bf16_values = [_]f32{ 16.0, -32.0, 64.0, -128.0 };

    const f32_bytes = try littleEndianF32(&f32_values, arena);
    const f16_bytes = try littleEndianF16(&f16_values, arena);
    const bf16_bytes = try littleEndianBf16(&bf16_values, arena);

    const entries = [_]StoredTensor{
        .{ .name = "w.f32", .dtype_name = "F32", .dims = &.{ 2, 2 }, .payload = f32_bytes },
        .{ .name = "w.f16", .dtype_name = "F16", .dims = &.{ 2, 3 }, .payload = f16_bytes },
        .{ .name = "w.bf16", .dtype_name = "BF16", .dims = &.{4}, .payload = bf16_bytes },
    };
    const image = try buildImage(arena, &entries, 0);

    const file = try File.parse(arena, image, .{});
    try std.testing.expectEqual(@as(u32, 3), file.count());

    const f32_tensor = file.find("w.f32").?;
    try std.testing.expectEqual(DType.f32, f32_tensor.dtypeOf());
    try std.testing.expectEqual(@as(u8, 2), f32_tensor.shapeOf().rank);
    try std.testing.expectEqual(@as(u32, 2), try f32_tensor.shapeOf().rows());
    try std.testing.expectEqual(@as(u32, 2), try f32_tensor.shapeOf().cols());
    try std.testing.expectEqual(@as(usize, 16), f32_tensor.bytesOf().len);
    // The reader must not copy: the payload is inside the caller's buffer.
    try std.testing.expect(@intFromPtr(f32_tensor.bytesOf().ptr) >= @intFromPtr(image.ptr));
    const image_end = @intFromPtr(image.ptr) + image.len;
    try std.testing.expect(@intFromPtr(f32_tensor.bytesOf().ptr) < image_end);

    var decoded: [4]f32 = undefined;
    try f32_tensor.decodeF32(&decoded);
    try std.testing.expectEqualSlices(f32, &f32_values, &decoded);

    var decoded_f16: [6]f32 = undefined;
    try file.find("w.f16").?.decodeF32(&decoded_f16);
    try std.testing.expectEqualSlices(f32, &f16_values, &decoded_f16);

    var decoded_bf16: [4]f32 = undefined;
    try file.find("w.bf16").?.decodeF32(&decoded_bf16);
    try std.testing.expectEqualSlices(f32, &bf16_values, &decoded_bf16);

    // A range read decodes exactly the requested elements.
    var middle: [2]f32 = undefined;
    try file.find("w.f16").?.decodeRangeF32(2, &middle);
    try std.testing.expectEqualSlices(f32, f16_values[2..4], &middle);

    try std.testing.expect(file.find("absent") == null);
    try std.testing.expect(file.indexOf("w.bf16").? == 2);
    try std.testing.expectEqual(@as(u64, 16 + 12 + 8), file.payloadBytes());
}

test "the aligned and scalar decode paths agree on the same values" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var f32_values: [64]f32 = undefined;
    var f16_values: [64]f32 = undefined;
    var bf16_values: [64]f32 = undefined;
    for (0..64) |index| {
        const position: f32 = @floatFromInt(index);
        f32_values[index] = @sin(position * 0.31) * 4.0;
        f16_values[index] = @cos(position * 0.17) * 2.0;
        bf16_values[index] = @sin(position * 0.07) * 8.0;
    }
    const f32_bytes = try littleEndianF32(&f32_values, arena);
    const f16_bytes = try littleEndianF16(&f16_values, arena);
    const bf16_bytes = try littleEndianBf16(&bf16_values, arena);

    const entries = [_]StoredTensor{
        .{ .name = "f32", .dtype_name = "F32", .dims = &.{ 8, 8 }, .payload = f32_bytes },
        .{ .name = "f16", .dtype_name = "F16", .dims = &.{ 8, 8 }, .payload = f16_bytes },
        .{ .name = "bf16", .dtype_name = "BF16", .dims = &.{ 8, 8 }, .payload = bf16_bytes },
    };
    // Three trailing header bytes make every payload odd-addressed, so the
    // scalar path runs; the aligned image runs the bulk path.
    const aligned = try buildImage(arena, &entries, 0);
    const shifted = try buildImage(arena, &entries, 3);
    const aligned_file = try File.parse(arena, aligned, .{});
    const shifted_file = try File.parse(arena, shifted, .{});

    for ([_][]const u8{ "f32", "f16", "bf16" }) |name| {
        var bulk: [64]f32 = undefined;
        var scalar: [64]f32 = undefined;
        try aligned_file.find(name).?.decodeF32(&bulk);
        try shifted_file.find(name).?.decodeF32(&scalar);
        try std.testing.expectEqualSlices(f32, &bulk, &scalar);
    }
}

test "the header length is validated before anything is parsed" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var short: [4]u8 = .{ '{', '}', 0, 0 };
    try std.testing.expectError(Error.Truncated, File.parse(arena, &short, .{}));

    const payload = [_]u8{ 0, 0, 0, 0 };
    const good = try buildRawImage(arena, "{}", &payload);
    defer arena.free(good);
    try std.testing.expectEqual(@as(u32, 0), (try File.parse(arena, good, .{})).count());

    const beyond = try std.testing.allocator.dupe(u8, good);
    defer std.testing.allocator.free(beyond);
    std.mem.writeInt(u64, beyond[0..8], beyond.len + 1, .little);
    try std.testing.expectError(Error.Truncated, File.parse(arena, beyond, .{}));

    const huge = try std.testing.allocator.dupe(u8, good);
    defer std.testing.allocator.free(huge);
    std.mem.writeInt(u64, huge[0..8], header_bytes_max + 1, .little);
    try std.testing.expectError(Error.HeaderLengthInvalid, File.parse(arena, huge, .{}));

    const empty = try std.testing.allocator.dupe(u8, good);
    defer std.testing.allocator.free(empty);
    std.mem.writeInt(u64, empty[0..8], 0, .little);
    try std.testing.expectError(Error.HeaderLengthInvalid, File.parse(arena, empty, .{}));
}

test "malformed and unfamiliar headers are rejected rather than guessed" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const payload: [8]u8 = @splat(0);

    const cases = [_]struct { header: []const u8, expected: Error }{
        .{ .header = "not json", .expected = Error.MalformedHeader },
        .{ .header = "[1, 2]", .expected = Error.MalformedHeader },
        .{ .header = "{\"a\": 7}", .expected = Error.MalformedHeader },
        .{ .header = "{\"a\": {}}", .expected = Error.MalformedHeader },
        .{
            .header = "{\"a\": {\"dtype\": 7, \"shape\": [1], \"data_offsets\": [0, 4]}}",
            .expected = Error.InvalidDTypeField,
        },
        .{
            .header = "{\"a\": {\"dtype\": \"F32\", \"shape\": \"nope\"," ++
                " \"data_offsets\": [0, 4]}}",
            .expected = Error.MalformedHeader,
        },
        .{
            .header = "{\"a\": {\"dtype\": \"F32\", \"shape\": [1.5], \"data_offsets\": [0, 4]}}",
            .expected = Error.MalformedHeader,
        },
        .{
            .header = "{\"a\": {\"dtype\": \"F32\", \"shape\": [1], \"data_offsets\": [0, 4, 8]}}",
            .expected = Error.MalformedHeader,
        },
        .{
            .header = "{\"a\": {\"dtype\": \"F32\", \"shape\": [1], \"data_offsets\": [0, 4]," ++
                " \"extra\": 1}}",
            .expected = Error.MalformedHeader,
        },
        .{
            .header = "{\"a\": {\"dtype\": \"F32\", \"shape\": [1, 2, 3, 4, 5]," ++
                " \"data_offsets\": [0, 4]}}",
            .expected = Error.InvalidShape,
        },
        .{
            .header = "{\"a\": {\"dtype\": \"F32\", \"shape\": [], \"data_offsets\": [0, 4]}}",
            .expected = Error.InvalidShape,
        },
        .{
            .header = "{\"a\": {\"dtype\": \"F32\", \"shape\": [0], \"data_offsets\": [0, 4]}}",
            .expected = Error.InvalidShape,
        },
        .{
            // Two entries with the same name: the second definition would
            // silently replace the first.
            .header = "{\"a\": {\"dtype\": \"F32\", \"shape\": [1], \"data_offsets\": [0, 4]}," ++
                "\"a\": {\"dtype\": \"F32\", \"shape\": [2], \"data_offsets\": [0, 4]}}",
            .expected = Error.DuplicateTensorName,
        },
    };

    for (cases) |case| {
        const image = try buildRawImage(arena, case.header, &payload);
        try std.testing.expectError(case.expected, File.parse(arena, image, .{}));
    }

    // A named metadata entry is skipped rather than treated as a tensor.
    const with_metadata = "{\"__metadata__\": {\"format\": \"pt\"}," ++
        "\"a\": {\"dtype\": \"F32\", \"shape\": [2], \"data_offsets\": [0, 8]}}";
    const metadata_image = try buildRawImage(arena, with_metadata, &payload);
    const metadata_file = try File.parse(arena, metadata_image, .{});
    try std.testing.expectEqual(@as(u32, 1), metadata_file.count());
    try std.testing.expect(metadata_file.find("__metadata__") == null);
    try std.testing.expect(metadata_file.find("a") != null);
}

test "byte ranges must match the shape, stay inside the payload and not overlap" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const payload: [32]u8 = @splat(0);

    const cases = [_]struct { header: []const u8, expected: Error }{
        .{
            // 4 f32 elements need 16 bytes, not 8.
            .header = "{\"a\": {\"dtype\": \"F32\", \"shape\": [4], \"data_offsets\": [0, 8]}}",
            .expected = Error.ByteLengthMismatch,
        },
        .{
            // The data section is 32 bytes.
            .header = "{\"a\": {\"dtype\": \"F32\", \"shape\": [1], \"data_offsets\": [30, 34]}}",
            .expected = Error.DataOutOfRange,
        },
        .{
            .header = "{\"a\": {\"dtype\": \"F32\", \"shape\": [1], \"data_offsets\": [8, 0]}}",
            .expected = Error.DataOutOfRange,
        },
        .{
            .header = "{\"a\": {\"dtype\": \"F16\", \"shape\": [8], \"data_offsets\": [0, 16]}," ++
                "\"b\": {\"dtype\": \"F16\", \"shape\": [8], \"data_offsets\": [8, 24]}}",
            .expected = Error.TensorRangesOverlap,
        },
        .{
            .header = "{\"a\": {\"dtype\": \"F16\", \"shape\": [8], \"data_offsets\": [0, 16]}," ++
                "\"b\": {\"dtype\": \"F16\", \"shape\": [8], \"data_offsets\": [0, 16]}}",
            .expected = Error.TensorRangesOverlap,
        },
    };
    for (cases) |case| {
        const image = try buildRawImage(arena, case.header, &payload);
        try std.testing.expectError(case.expected, File.parse(arena, image, .{}));
    }

    // Unaligned, gap-carrying ranges are legal: offsets need only lie inside
    // the data section and stay disjoint, because a checkpoint's own alignment
    // is a property of the tool that wrote it, not of this format.
    const first = "{\"a\": {\"dtype\": \"F16\", \"shape\": [3], \"data_offsets\": [0, 6]},";
    const second = "\"b\": {\"dtype\": \"F16\", \"shape\": [3], \"data_offsets\": [8, 14]}}";
    const sparse_image = try buildRawImage(arena, first ++ second, &payload);
    const sparse_file = try File.parse(arena, sparse_image, .{});
    try std.testing.expectEqual(@as(u32, 2), sparse_file.count());
    try std.testing.expectEqual(@as(u64, 0), sparse_file.find("a").?.offset_bytes);
    try std.testing.expectEqual(@as(u64, 8), sparse_file.find("b").?.offset_bytes);
    try std.testing.expectEqual(@as(u64, 14), sparse_file.payloadBytes());
}

test "an unknown dtype is reported instead of decoded" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const payload: [8]u8 = @splat(0);
    const header = "{\"a\": {\"dtype\": \"F8_E4M3\", \"shape\": [8], \"data_offsets\": [0, 8]}}";
    const image = try buildRawImage(arena, header, &payload);

    try std.testing.expectError(Error.UnsupportedDType, File.parse(arena, image, .{}));

    const lenient = try File.parse(arena, image, .{ .allow_unknown_dtypes = true });
    const entry = lenient.find("a").?;
    try std.testing.expectEqual(DType.other, entry.dtypeOf());
    try std.testing.expectEqualStrings("F8_E4M3", entry.dtype_name);
    try std.testing.expect(!entry.dtypeOf().isFloat());
    try std.testing.expect(entry.dtypeOf().sizeBytes() == null);
    var decoded: [8]f32 = undefined;
    try std.testing.expectError(Error.UnsupportedDType, entry.decodeF32(&decoded));
    try std.testing.expectEqualStrings("unknown", DType.other.providerName());
}

test "the element reader is bounds checked" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const values: [6]f32 = @splat(1.0);
    const bytes = try littleEndianF32(&values, arena);
    const entries = [_]StoredTensor{
        .{ .name = "a", .dtype_name = "F32", .dims = &.{ 2, 3 }, .payload = bytes },
    };
    const image = try buildImage(arena, &entries, 0);
    const file = try File.parse(arena, image, .{});
    const entry = file.find("a").?;

    var whole: [6]f32 = undefined;
    try entry.decodeF32(&whole);
    var too_small: [5]f32 = undefined;
    try std.testing.expectError(Error.DecodeLengthMismatch, entry.decodeF32(&too_small));
    var too_large: [7]f32 = undefined;
    try std.testing.expectError(Error.DecodeLengthMismatch, entry.decodeF32(&too_large));

    var boundary: [3]f32 = undefined;
    try entry.decodeRangeF32(3, &boundary);
    var past_end: [3]f32 = undefined;
    try std.testing.expectError(Error.DecodeOutOfRange, entry.decodeRangeF32(4, &past_end));
    var past_start: [1]f32 = undefined;
    try std.testing.expectError(Error.DecodeOutOfRange, entry.decodeRangeF32(7, &past_start));

    var at_end: [0]f32 = .{};
    try entry.decodeRangeF32(6, &at_end);
    try std.testing.expectError(Error.DecodeOutOfRange, entry.decodeRangeF32(7, &at_end));

    try std.testing.expectError(Error.DecodeOutOfRange, file.at(1));
    try std.testing.expectEqual(@as(u32, 4), entry.dtypeOf().sizeBytes().?);
}
