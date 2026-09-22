#!/usr/bin/env node
// Anti-drift gate for the quantized weight layout.
//
// `src/core/quant.zig` writes the layout, `gpu/shaders/quant_layout.wgsl`
// documents it, and every kernel in `gpu/shaders/` repeats the handful of
// numbers it needs (WGSL has no `#include`). Three copies of one truth is a bug
// waiting to happen, so this script extracts the constants from all three and
// fails on any disagreement, any constant that disappeared, and any kernel
// constant that this script has not been taught about.
//
// It also checks the *structure* the numbers describe: the nibble packing, the
// q5 high bit plane, and the alignment call in `planeLayout` are matched by
// regex in the Zig source, so a rewrite that keeps the constants but changes the
// packing fails too.
//
// One more table lives outside Zig entirely: the container numbers its tensor
// kinds and the ABI reports the number, so `src/core/container.zig`'s
// `TensorKind` and the JavaScript mirror in
// `packages/qwenscriber/src/gpu/tensor_kind.ts` are compared here, names and
// values both. Its rule is "append new kinds, never renumber", which is exactly
// the rule a hand-copied mirror can break without anything else noticing.
//
// Usage: node tests/gpu/layout_drift.mjs
// Exit code 0 when everything agrees, 1 otherwise.
//
// No dependencies: a gate that needs an install is a gate that stops running.

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const root = fileURLToPath(new URL("../../", import.meta.url));
const zig_quant_path = "src/core/quant.zig";
const zig_dtype_path = "src/core/dtype.zig";
const zig_container_path = "src/core/container.zig";
const tensor_kind_path = "packages/qwenscriber/src/gpu/tensor_kind.ts";
const abi_zig_path = "src/wasm/abi.zig";
const abi_ts_path = "packages/qwenscriber/src/wasm/abi.ts";
const quant_ts_path = "packages/qwenscriber/src/gpu/quant_layout.ts";
const layout_path = "gpu/shaders/quant_layout.wgsl";
const shader_dir = "gpu/shaders";

// Every constant the layout file declares, and how to find it in the Zig
// sources. `zig` is a regex with the number in group 1; `slice` narrows the
// match to one function body so that `bias` cannot pick up a value from
// `dataBytesForGroups`.
const LAYOUT = [
    {
        name: "QW_GROUP_SIZE",
        source: zig_quant_path,
        pattern: /pub const group_size: u32 = (\d+);/,
        reason: "weights per quantization group",
    },
    {
        name: "QW_Q4_SCALE_BYTES_PER_GROUP",
        source: zig_quant_path,
        pattern: /pub const q4_scale_bytes_per_group: u32 = (\d+);/,
        reason: "f16 scale bytes per q4 group",
    },
    {
        name: "QW_Q4_DATA_BYTES_PER_GROUP",
        source: zig_quant_path,
        pattern: /pub const q4_data_bytes_per_group: u32 = (\d+);/,
        reason: "packed code bytes per q4 group",
    },
    {
        name: "QW_Q5_SCALE_BYTES_PER_GROUP",
        source: zig_quant_path,
        pattern: /pub const q5_scale_bytes_per_group: u32 = (\d+);/,
        reason: "f16 scale bytes per q5 group",
    },
    {
        name: "QW_Q5_DATA_BYTES_PER_GROUP",
        source: zig_quant_path,
        pattern: /pub const q5_data_bytes_per_group: u32 = (\d+);/,
        reason: "packed code bytes per q5 group",
    },
    {
        name: "QW_Q8_SCALE_BYTES_PER_GROUP",
        source: zig_quant_path,
        pattern: /pub const q8_scale_bytes_per_group: u32 = (\d+);/,
        reason: "f16 scale bytes per q8 group",
    },
    {
        name: "QW_Q8_DATA_BYTES_PER_GROUP",
        source: zig_quant_path,
        pattern: /pub const q8_data_bytes_per_group: u32 = (\d+);/,
        reason: "packed code bytes per q8 group",
    },
    {
        name: "QW_TENSOR_ALIGNMENT_BYTES",
        source: zig_quant_path,
        pattern: /pub const tensor_alignment_bytes: u64 = (\d+);/,
        reason: "alignment of a quantized payload, hence of the data plane",
    },
    {
        name: "QW_Q4_BIAS",
        source: zig_quant_path,
        slice: "pub fn bias(format: dtype.Format) i32 {",
        pattern: /\.q4 => (\d+),/,
        reason: "q4 code midpoint",
    },
    {
        name: "QW_Q5_BIAS",
        source: zig_quant_path,
        slice: "pub fn bias(format: dtype.Format) i32 {",
        pattern: /\.q5 => (\d+),/,
        reason: "q5 code midpoint",
    },
    {
        name: "QW_Q8_BIAS",
        source: zig_quant_path,
        slice: "pub fn bias(format: dtype.Format) i32 {",
        pattern: /\.q8 => (\d+),/,
        reason: "q8 code midpoint",
    },
    {
        name: "QW_FORMAT_Q4",
        source: zig_dtype_path,
        slice: "pub const Format = enum(u8) {",
        pattern: /^\s*q4 = (\d+),/m,
        reason: "format id passed to dequant_reference.wgsl",
    },
    {
        name: "QW_FORMAT_Q5",
        source: zig_dtype_path,
        slice: "pub const Format = enum(u8) {",
        pattern: /^\s*q5 = (\d+),/m,
        reason: "format id passed to dequant_reference.wgsl",
    },
    {
        name: "QW_FORMAT_Q8",
        source: zig_dtype_path,
        slice: "pub const Format = enum(u8) {",
        pattern: /^\s*q8 = (\d+),/m,
        reason: "format id passed to dequant_reference.wgsl",
    },
];

// Constants the layout file states that have no named counterpart in Zig: the
// packing code spells the same fact out inline, so `check_structure` verifies
// them against that code instead of against an extracted value.
const DERIVED = new Set([
    "QW_NIBBLE_LOW_SHIFT",
    "QW_NIBBLE_HIGH_SHIFT",
    "QW_Q5_HIGH_BIT_PLANE_OFFSET_BYTES",
    "QW_Q5_HIGH_BIT_BYTES_PER_GROUP",
]);

const sources = new Map();

function source_text(path) {
    if (!sources.has(path)) sources.set(path, readFileSync(join(root, path), "utf8"));
    return sources.get(path);
}

// The body of `header`, up to the closing brace of the declaration.
function slice_body(text, header) {
    const start = text.indexOf(header);
    if (start < 0) return null;
    const end = text.indexOf("\n}", start);
    if (end < 0) return null;
    return text.slice(start, end);
}

// `const QW_X: u32 = 64u;` / `const QW_X: i32 = 8;` anywhere in a WGSL file.
function wgsl_constants(text) {
    const found = new Map();
    const pattern = new RegExp(
        "^\\s*const\\s+(QW_[A-Z0-9_]+)\\s*:\\s*(?:u32|i32|f32)\\s*=\\s*" +
            "(-?\\d+(?:\\.\\d+)?)[uif]?\\s*;",
        "gm",
    );
    for (const match of text.matchAll(pattern)) {
        found.set(match[1], Number(match[2]));
    }
    return found;
}

function zig_value(entry) {
    const text = source_text(entry.source);
    const scope = entry.slice ? slice_body(text, entry.slice) : text;
    if (scope === null) {
        return { error: `${entry.source} no longer contains "${entry.slice}"` };
    }
    const match = scope.match(entry.pattern);
    if (match === null) {
        return { error: `${entry.source} no longer matches ${entry.pattern}` };
    }
    return { value: Number(match[1]) };
}

function compare_layout() {
    const layout = wgsl_constants(source_text(layout_path));
    const rows = [];
    const problems = [];
    for (const entry of LAYOUT) {
        const zig = zig_value(entry);
        const wgsl = layout.get(entry.name);
        if (zig.error !== undefined) {
            problems.push(`${entry.name}: ${zig.error}`);
            continue;
        }
        if (wgsl === undefined) {
            problems.push(
                `${entry.name}: missing from ${layout_path} (${entry.source} has ${zig.value})`,
            );
            continue;
        }
        rows.push({ name: entry.name, zig: zig.value, wgsl, reason: entry.reason });
        if (zig.value !== wgsl) {
            problems.push(
                `${entry.name}: ${entry.source} = ${zig.value}, ${layout_path} = ${wgsl}`,
            );
        }
    }
    for (const name of layout.keys()) {
        if (LAYOUT.some((entry) => entry.name === name)) continue;
        if (DERIVED.has(name)) continue;
        problems.push(`${name}: declared in ${layout_path} but unknown to this script`);
    }
    return { rows, problems };
}

// Structural facts about the packing itself, which the constants above only
// describe. Each is a regex that must still match the Zig source.
const STRUCTURE = [
    {
        name: "q4 low nibble holds weight 2i (no shift)",
        pattern: /codes\[index \/ 2\] \|= unsigned;/,
        describe: (match) => `shift ${0}`,
    },
    {
        name: "q4 high nibble holds weight 2i+1 (shift)",
        pattern: /codes\[index \/ 2\] \|= unsigned << (\d+);/,
        describe: (match) => `shift ${match[1]}`,
    },
    {
        name: "q5 high bit plane follows the q4 nibble plane",
        pattern: /codes\[(q4_data_bytes_per_group) \+ index \/ 8\]/,
        describe: (match) => `offset ${match[1]}`,
    },
    {
        name: "q5 fifth bit is bit j%8 of byte j/8",
        pattern: /<< @intCast\(index % (\d+)\)/,
        describe: (match) => `bit index modulo ${match[1]}`,
    },
    {
        name: "q8 stores one byte per weight",
        pattern: /codes\[index\] = @intCast\(code \+ shift\);/,
        describe: () => "one byte per weight",
    },
    {
        name: "planeLayout aligns the data plane to the tensor alignment",
        pattern: /std\.mem\.alignForward\(u64, data_start, alignment\)/,
        describe: () => "alignForward(data_start, alignment)",
    },
    {
        name: "the scale plane's low byte is written first",
        pattern: /bytes\[group_index \* 2\] = @truncate\(bits\);/,
        describe: () => "low byte at offset 0 of the group",
    },
    {
        name: "the scale plane's high byte follows",
        pattern: /bytes\[group_index \* 2 \+ 1\] = @truncate\(bits >> 8\);/,
        describe: () => "high byte at offset 1 of the group",
    },
];

function check_structure(values) {
    const text = source_text(zig_quant_path);
    const rows = [];
    const problems = [];
    for (const fact of STRUCTURE) {
        const match = text.match(fact.pattern);
        if (match === null) {
            problems.push(`structure: ${zig_quant_path} no longer matches ${fact.pattern}`);
            continue;
        }
        rows.push({ name: fact.name, detail: fact.describe(match) });
    }

    // Relations the layout file states between constants, which the packing code
    // above fixes: they cannot be read out of Zig as named values, so they are
    // checked against the number the packing uses.
    for (const name of DERIVED) {
        if (values.get(name) === undefined) {
            problems.push(`${name}: missing from ${layout_path}`);
        }
    }
    const shift_match = text.match(/codes\[index \/ 2\] \|= unsigned << (\d+);/);
    const high_shift = values.get("QW_NIBBLE_HIGH_SHIFT");
    if (shift_match !== null && high_shift !== Number(shift_match[1])) {
        problems.push(
            `QW_NIBBLE_HIGH_SHIFT: packing shifts by ${shift_match[1]}, ` +
                `${layout_path} declares ${high_shift}`,
        );
    }
    if (values.get("QW_NIBBLE_LOW_SHIFT") !== 0) {
        problems.push(`QW_NIBBLE_LOW_SHIFT: must be 0, ${layout_path} declares ` +
            `${values.get("QW_NIBBLE_LOW_SHIFT")}`);
    }
    const high_bit_offset = values.get("QW_Q5_HIGH_BIT_PLANE_OFFSET_BYTES");
    const q4_data_bytes = values.get("QW_Q4_DATA_BYTES_PER_GROUP");
    if (high_bit_offset !== q4_data_bytes) {
        problems.push(
            `QW_Q5_HIGH_BIT_PLANE_OFFSET_BYTES: must equal QW_Q4_DATA_BYTES_PER_GROUP ` +
                `(${q4_data_bytes}), ${layout_path} declares ${high_bit_offset}`,
        );
    }
    const high_bit_bytes = values.get("QW_Q5_HIGH_BIT_BYTES_PER_GROUP");
    const expected_high_bit_bytes = values.get("QW_GROUP_SIZE") / 8;
    if (high_bit_bytes !== expected_high_bit_bytes) {
        problems.push(
            `QW_Q5_HIGH_BIT_BYTES_PER_GROUP: must be QW_GROUP_SIZE / 8 ` +
                `(${expected_high_bit_bytes}), ${layout_path} declares ${high_bit_bytes}`,
        );
    }
    return { rows, problems };
}

function check_kernel_mirrors(layout) {
    const files = readdirSync(join(root, shader_dir)).filter((name) => name.endsWith(".wgsl"));
    files.sort();
    const mirrors = [];
    const problems = [];
    for (const file of files) {
        if (`${shader_dir}/${file}` === layout_path) continue;
        const text = source_text(`${shader_dir}/${file}`);
        for (const [name, value] of wgsl_constants(text)) {
            const declared = layout.get(name);
            if (declared === undefined) {
                problems.push(
                    `${shader_dir}/${file}: declares ${name}, which ${layout_path} does not`,
                );
                continue;
            }
            mirrors.push({ file, name, value });
            if (declared !== value) {
                problems.push(
                    `${name}: ${shader_dir}/${file} = ${value}, ${layout_path} = ${declared}`,
                );
            }
        }
    }
    return { mirrors, problems };
}

// The container's tensor kinds, in declaration order with implicit values resolved: `name = 1,`
// starts a count, `name,` continues it, and an explicit value restarts it. `_` ends the exhaustive
// list and is not a kind.
function zig_tensor_kinds(text) {
    const header = "pub const TensorKind = enum(u16) {";
    if (!text.includes(header)) return null;
    const kinds = [];
    let next = 0;
    for (const line of slice_body(text, header).split("\n")) {
        const match = line.match(/^\s*([a-z][a-z0-9_]*)\s*(?:=\s*(\d+))?,?\s*$/);
        if (match === null) continue;
        next = match[2] === undefined ? next : Number(match[2]);
        kinds.push({ name: match[1], value: next });
        next += 1;
    }
    return kinds;
}

// `audio_conv1_weight: 1,` inside the exported table, at the indentation the file uses.
function ts_tensor_kinds(text) {
    const kinds = [];
    for (const line of text.split("\n")) {
        const match = line.match(/^\s{2}([a-z][a-z0-9_]*):\s*(\d+),\s*$/);
        if (match !== null) kinds.push({ name: match[1], value: Number(match[2]) });
    }
    return kinds;
}

// A kind the container numbers but the mirror does not know is a caller looking for a tensor it
// cannot name; a kind the mirror invents is a caller looking for a tensor that does not exist.
function compare_tensor_kinds() {
    const rows = [];
    const problems = [];
    const zig = zig_tensor_kinds(source_text(zig_container_path));
    if (zig === null || zig.length === 0) {
        problems.push(`tensor kinds: ${zig_container_path} no longer declares TensorKind`);
        return { rows, problems };
    }
    const mirrored = ts_tensor_kinds(source_text(tensor_kind_path));
    const by_name = new Map(mirrored.map((kind) => [kind.name, kind.value]));
    for (const kind of zig) {
        const value = by_name.get(kind.name);
        if (value === undefined) {
            problems.push(
                `tensor kinds: ${kind.name} = ${kind.value} is not mirrored in ${tensor_kind_path}`,
            );
        } else if (value !== kind.value) {
            problems.push(
                `tensor kinds: ${kind.name} is ${kind.value} in ${zig_container_path} and ` +
                    `${value} in ${tensor_kind_path}`,
            );
        }
    }
    const declared = new Set(zig.map((kind) => kind.name));
    for (const kind of mirrored) {
        if (!declared.has(kind.name)) {
            problems.push(
                `tensor kinds: ${tensor_kind_path} mirrors ${kind.name}, which ` +
                    `${zig_container_path} does not declare`,
            );
        }
    }
    const first = zig[0];
    const last = zig[zig.length - 1];
    rows.push({
        name: "TensorKind",
        detail: `${zig.length} kinds, ${first.name} = ${first.value} .. ${last.name} = ${last.value}`,
    });
    return { rows, problems };
}

// The ABI structs are the other place two languages have to agree on numbers. Zig pins its own
// layout with comptime asserts; this compares those asserts against the DataView offsets the SDK
// reads with, so a field reordered on either side fails here rather than reading the neighbouring
// field in a browser.
// The layer bases are tags in the container's index, and the SDK adds them when it looks a tower
// layer up, so a silent renumbering would send every lookup to the wrong block.
const LAYER_BASES = [
    { ts: "DECODER_LAYER_BASE = (\\d+)", zig: "/pub const decoder_layer_base: u16 = (\\d+);/", source: "src/core/qwen3_asr/layout.zig" },
    { ts: "AUDIO_LAYER_BASE = (\\d+)", zig: "/pub const audio_layer_base: u16 = (\\d+);/", source: "src/core/container.zig" },
];

function compare_layer_bases() {
    const problems = [];
    const ts = source_text(tensor_kind_path);
    let compared = 0;
    for (const base of LAYER_BASES) {
        const mirrored = ts_constant(ts, base.ts);
        const zig = zig_constant(source_text(base.source), base.zig);
        if (mirrored === undefined || zig === undefined) {
            problems.push(`layer bases: ${base.ts} is not pinned in both files`);
            continue;
        }
        compared += 1;
        if (mirrored !== zig) {
            problems.push(
                `layer bases: ${base.ts.match(/^(\w+)/)[1]} is ${mirrored} in ${tensor_kind_path} ` +
                    `and ${zig} in ${base.source}`,
            );
        }
    }
    return {
        rows: [{ name: "layer bases", detail: `${compared} bases agree with the container` }],
        problems,
    };
}

const ABI_STRUCTS = [
    {
        name: "TensorDescriptor",
        size: "TENSOR_DESCRIPTOR_BYTES",
        offsets: "TENSOR_DESCRIPTOR_OFFSET",
    },
    {
        name: "ModelRequirements",
        size: "MODEL_REQUIREMENTS_BYTES",
        offsets: "MODEL_REQUIREMENTS_OFFSET",
    },
    {
        name: "AudioConfig",
        size: "AUDIO_CONFIG_BYTES",
        offsets: "AUDIO_CONFIG_OFFSET",
    },
];

// One offset table's body. Field names repeat across tables -- `reserved` is in several -- so the
// search has to stay inside the table named, or it reads a neighbouring struct's number.
function ts_offset_block(text, table) {
    const start = text.indexOf(`${table} = {`);
    if (start < 0) return null;
    const end = text.indexOf("}", start);
    return end < 0 ? null : text.slice(start, end);
}

// Every field Zig asserts an offset for, in the order the asserts appear.
function zig_abi_fields(text, struct) {
    const pattern = new RegExp(`@offsetOf\\(${struct}, "(\\w+)"\\) == (\\d+)`, "g");
    return [...text.matchAll(pattern)].map((match) => ({ name: match[1], offset: match[2] }));
}

function compare_abi_layout() {
    const rows = [];
    const problems = [];
    const zig = source_text(abi_zig_path);
    const ts = source_text(abi_ts_path);

    for (const struct of ABI_STRUCTS) {
        const fields = zig_abi_fields(zig, struct.name);
        if (fields.length === 0) {
            problems.push(`abi layout: ${abi_zig_path} no longer asserts offsets for ${struct.name}`);
            continue;
        }
        const zig_size = zig.match(new RegExp(`@sizeOf\\(${struct.name}\\) == (\\d+)`));
        const ts_size = ts.match(new RegExp(`${struct.size} = (\\d+)`));
        if (zig_size === null || ts_size === null) {
            problems.push(
                `abi layout: ${struct.name} size is pinned in ` +
                    `${zig_size === null ? abi_zig_path : abi_ts_path} only`,
            );
        } else if (zig_size[1] !== ts_size[1]) {
            problems.push(
                `abi layout: ${struct.name} is ${zig_size[1]} bytes in ${abi_zig_path} and ` +
                    `${ts_size[1]} in ${abi_ts_path}`,
            );
        }

        const table = ts_offset_block(ts, struct.offsets);
        if (table === null) {
            problems.push(`abi layout: ${abi_ts_path} no longer declares ${struct.offsets}`);
            continue;
        }

        let compared = 0;
        for (const field of fields) {
            const in_ts = table.match(new RegExp(`^\\s*${field.name}: (\\d+),`, "m"));
            if (in_ts === null) {
                problems.push(
                    `abi layout: ${struct.name}.${field.name} is at ${field.offset} in ` +
                        `${abi_zig_path} and is not pinned in ${struct.offsets}`,
                );
                continue;
            }
            compared += 1;
            if (in_ts[1] !== field.offset) {
                problems.push(
                    `abi layout: ${struct.name}.${field.name} is at ${field.offset} in ` +
                        `${abi_zig_path} and ${in_ts[1]} in ${abi_ts_path}`,
                );
            }
        }
        rows.push({ name: struct.name, detail: `${compared} field offsets agree` });
    }
    return { rows, problems };
}

// The TypeScript side of the layout: `quant_layout.ts` mirrors the same constants so a caller can say
// where a tensor's code plane starts. That is a second copy of one truth, for the browser what the
// WGSL mirrors are for the shaders, so it is compared here too -- each mirror against the Zig source
// it came from, with the Zig pattern written out rather than borrowed from `LAYOUT`, whose entries
// resolve to values in more than one shape.
const QUANT_TS_CONSTANTS = [
    {
        name: "QUANT_GROUP_SIZE",
        ts: "QUANT_GROUP_SIZE = (\\d+)",
        zig: "/pub const group_size: u32 = (\\d+);/",
        reason: "weights per quantization group",
    },
    {
        name: "QUANT_SCALE_BYTES_PER_GROUP",
        ts: "QUANT_SCALE_BYTES_PER_GROUP = (\\d+)",
        zig: "/pub const q4_scale_bytes_per_group: u32 = (\\d+);/",
        reason: "f16 scale bytes per group",
    },
    {
        name: "QUANT_CODE_BYTES_PER_GROUP.q4",
        ts: "q4: (\\d+),",
        zig: "/pub const q4_data_bytes_per_group: u32 = (\\d+);/",
        reason: "packed q4 code bytes per group",
    },
    {
        name: "QUANT_CODE_BYTES_PER_GROUP.q5",
        ts: "q5: (\\d+),",
        zig: "/pub const q5_data_bytes_per_group: u32 = (\\d+);/",
        reason: "packed q5 code bytes per group",
    },
    {
        name: "QUANT_CODE_BYTES_PER_GROUP.q8",
        // No trailing comma: q8 is the last entry of the table.
        ts: "q8: (\\d+)",
        zig: "/pub const q8_data_bytes_per_group: u32 = (\\d+);/",
        reason: "packed q8 code bytes per group",
    },
    {
        name: "TENSOR_ALIGNMENT_BYTES",
        ts: "TENSOR_ALIGNMENT_BYTES = (\\d+)",
        zig: "/pub const tensor_alignment_bytes: u64 = (\\d+);/",
        reason: "alignment of a quantized payload",
    },
    {
        name: "FORMAT_Q4",
        ts: "FORMAT_Q4 = (\\d+)",
        zig: "/^\\s*q4 = (\\d+),/m",
        dtype: true,
        reason: "q4 format id",
    },
    {
        name: "FORMAT_Q5",
        ts: "FORMAT_Q5 = (\\d+)",
        zig: "/^\\s*q5 = (\\d+),/m",
        dtype: true,
        reason: "q5 format id",
    },
    {
        name: "FORMAT_Q8",
        ts: "FORMAT_Q8 = (\\d+)",
        zig: "/^\\s*q8 = (\\d+),/m",
        dtype: true,
        reason: "q8 format id",
    },
];

// `const NAME = 64` / `q4: 32,` in the TypeScript file.
function ts_constant(text, pattern) {
    const match = text.match(new RegExp(`(?:const )?${pattern}`));
    return match === null ? undefined : Number(match[1]);
}

// `/regex/flags` as written in the table above.
function zig_constant(text, specification) {
    const match = specification.match(/^\/(.*)\/([a-z]*)$/s);
    if (match === null) return undefined;
    const found = text.match(new RegExp(match[1], match[2]));
    return found === null ? undefined : Number(found[1]);
}

function compare_quant_ts() {
    const problems = [];
    const quant = source_text(zig_quant_path);
    const types = source_text(zig_dtype_path);
    const ts = source_text(quant_ts_path);

    let compared = 0;
    for (const constant of QUANT_TS_CONSTANTS) {
        const mirrored = ts_constant(ts, constant.ts);
        const source = constant.dtype === true ? types : quant;
        const zig = zig_constant(source, constant.zig);
        if (mirrored === undefined || zig === undefined) {
            problems.push(
                `quant ts: ${constant.name} is not pinned in both files ` +
                    `(${quant_ts_path}: ${mirrored}, Zig: ${zig})`,
            );
            continue;
        }
        compared += 1;
        if (mirrored !== zig) {
            problems.push(
                `quant ts: ${constant.name} is ${mirrored} in ${quant_ts_path} and ${zig} in Zig ` +
                    `(${constant.reason})`,
            );
        }
    }
    return {
        rows: [{ name: "quant_layout.ts", detail: `${compared} constants agree with Zig` }],
        problems,
    };
}

const layout = wgsl_constants(source_text(layout_path));
const compared = compare_layout();
const structure = check_structure(layout);
const mirrors = check_kernel_mirrors(layout);
const kinds = compare_tensor_kinds();
const abi_layout = compare_abi_layout();
const quant_ts = compare_quant_ts();
const layer_bases = compare_layer_bases();

const problems = [
    ...compared.problems,
    ...structure.problems,
    ...mirrors.problems,
    ...kinds.problems,
    ...abi_layout.problems,
    ...quant_ts.problems,
    ...layer_bases.problems,
];
const shader_count = new Set(mirrors.mirrors.map((mirror) => mirror.file)).size;

console.log(
    `layout_drift: ${compared.rows.length} constants compared between ` +
        `${zig_quant_path} (+ ${zig_dtype_path}) and ${layout_path}`,
);
for (const row of compared.rows) {
    console.log(`  ${row.name} = ${row.zig}  (${row.reason})`);
}
console.log(`layout_drift: ${structure.rows.length} structural facts matched in ${zig_quant_path}`);
for (const row of structure.rows) {
    console.log(`  ${row.name}: ${row.detail}`);
}
console.log(
    `layout_drift: ${mirrors.mirrors.length} constant mirrors agree across ${shader_count} kernels`,
);
console.log(
    `layout_drift: tensor kinds compared between ${zig_container_path} and ${tensor_kind_path}`,
);
for (const row of kinds.rows) {
    console.log(`  ${row.name}: ${row.detail}`);
}
console.log(`layout_drift: constants compared between ${layout_path} and ${quant_ts_path}`);
for (const row of quant_ts.rows) {
    console.log(`  ${row.name}: ${row.detail}`);
}
for (const row of layer_bases.rows) {
    console.log(`  ${row.name}: ${row.detail}`);
}
console.log(`layout_drift: ABI struct layout compared between ${abi_zig_path} and ${abi_ts_path}`);
for (const row of abi_layout.rows) {
    console.log(`  ${row.name}: ${row.detail}`);
}

if (problems.length > 0) {
    console.error("");
    console.error(`layout_drift: FAIL, ${problems.length} problem(s)`);
    for (const problem of problems) console.error(`  - ${problem}`);
    process.exit(1);
}

console.log("");
console.log("layout_drift: ok, no drift");
