//! The kernel catalogue against the shaders it names.
//!
//! `gpu/kernels.ts` is a transcription of eight shader headers, and a transcription is exactly the
//! kind of thing that drifts: rename an entry point, add a binding, change a workgroup size, and the
//! page that runs one kernel would still pass while the other seven were unreachable.
//!
//! So the shaders are read back here and checked against the catalogue: the entry point exists, the
//! binding numbers are the ones the shader declares, and the workgroup size is what
//! `@workgroup_size(...)` says once the shader's own constants are resolved. No browser, no device,
//! no WebGPU -- this is a text check on purpose, and the thing that actually *executes* every kernel
//! is `node tests/gpu/harness.mjs`.

import { readFileSync } from "node:fs";
import { test } from "node:test";
import assert from "node:assert/strict";

import { WEBGPU_KERNELS } from "../src/gpu/kernels.ts";

function shaderSource(file: string): string {
  const url = new URL(`../../../gpu/shaders/${file}`, import.meta.url);
  return new TextDecoder().decode(readFileSync(url));
}

/** `const NAME: u32 = 256u;` -> 256, so `@workgroup_size(RMSNORM_LANES, 1, 1)` can be resolved. */
function shaderConstants(source: string): Readonly<Record<string, number>> {
  const constants: Record<string, number> = {};
  const pattern = /^const\s+([A-Z0-9_]+)\s*:\s*u32\s*=\s*(\d+)u?\s*;/gm;
  for (const match of source.matchAll(pattern)) {
    const name = match[1];
    const value = match[2];
    if (name !== undefined && value !== undefined) constants[name] = Number(value);
  }
  return constants;
}

/** The three numbers in the shader's `@workgroup_size(...)`, with its constants resolved. */
function workgroupSizeOf(source: string): readonly number[] | undefined {
  const match = source.match(/@workgroup_size\(([^)]*)\)/);
  if (match === null || match[1] === undefined) return undefined;
  const constants = shaderConstants(source);
  return match[1].split(",").map((term) => {
    const text = term.trim();
    if (/^\d+$/.test(text)) return Number(text);
    const resolved = constants[text];
    assert.ok(resolved !== undefined, `@workgroup_size uses unresolved ${text}`);
    return resolved;
  });
}

test("every catalogue entry names an entry point its shader declares", () => {
  for (const kernel of Object.values(WEBGPU_KERNELS)) {
    const source = shaderSource(kernel.file);
    assert.match(
      source,
      new RegExp(`fn\\s+${kernel.entryPoint}\\s*\\(`),
      `${kernel.file} does not declare ${kernel.entryPoint}`,
    );
    assert.equal(kernel.name, kernel.file.replace(/\.wgsl$/, ""));
  }
});

test("every catalogue binding is a binding the shader declares, in the same order", () => {
  for (const kernel of Object.values(WEBGPU_KERNELS)) {
    const source = shaderSource(kernel.file);
    const declared = [...source.matchAll(/@binding\((\d+)\)/g)].map((match) => Number(match[1]));
    assert.deepEqual(
      kernel.bindings.map((binding) => binding.binding),
      declared,
      `${kernel.file} declares bindings ${declared.join(",")}`,
    );
  }
});

test("every catalogue workgroup size is what @workgroup_size says", () => {
  for (const kernel of Object.values(WEBGPU_KERNELS)) {
    const declared = workgroupSizeOf(shaderSource(kernel.file));
    assert.ok(declared !== undefined, `${kernel.file} has no @workgroup_size`);
    assert.deepEqual(
      [...kernel.workgroupSize],
      declared,
      `${kernel.file} declares a ${declared.join("x")} workgroup`,
    );
  }
});

test("a kernel that declares workgroup memory is described as using some", () => {
  for (const kernel of Object.values(WEBGPU_KERNELS)) {
    const source = shaderSource(kernel.file);
    const declares_storage = /var<workgroup>/.test(source);
    assert.equal(
      kernel.workgroupStorageBytes > 0,
      declares_storage,
      `${kernel.file} ${declares_storage ? "declares" : "does not declare"} var<workgroup>`,
    );
  }
});
