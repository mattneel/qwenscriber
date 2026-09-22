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

const layout = wgsl_constants(source_text(layout_path));
const compared = compare_layout();
const structure = check_structure(layout);
const mirrors = check_kernel_mirrors(layout);

const problems = [...compared.problems, ...structure.problems, ...mirrors.problems];
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

if (problems.length > 0) {
    console.error("");
    console.error(`layout_drift: FAIL, ${problems.length} problem(s)`);
    for (const problem of problems) console.error(`  - ${problem}`);
    process.exit(1);
}

console.log("");
console.log("layout_drift: ok, no drift");
