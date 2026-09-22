#!/usr/bin/env node
// Stages the browser example into the book, so the documentation ships a page
// that actually runs instead of a screenshot of one.
//
// There is exactly one demo driver: `examples/browser/main.js`. Copying it (and
// the handful of files it imports) into `docs/src/demo/` keeps that true, and the
// rewrites below fail loudly if the strings they look for are gone. A silent
// no-op here would produce a page that looks fine in review and 404s in a browser,
// which is the failure mode this script exists to prevent.
//
// Usage:
//
//     node tools/book/stage-demo.mjs
//     mdbook build docs

import { access, cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const staged = join(root, "docs/src/demo");

/// Path rewrites that make the example work from inside the book's directory
/// layout. Each replacement has to fire, or the staged page would import files
/// that are not there.
const REWRITES = [
  ['"../../packages/qwenscriber/dist/index.js"', '"./sdk/index.js"'],
  ['new URL("../../gpu/shaders/", import.meta.url)', 'new URL("./shaders/", import.meta.url)'],
  ['import("../../tests/gpu/reference.mjs")', 'import("./tests/reference.mjs")'],
  ['import("../../tests/gpu/harness.mjs")', 'import("./tests/harness.mjs")'],
];

async function stageDriver() {
  const source = await readFile(join(root, "examples/browser/main.js"), "utf8");
  let staged_source = source;
  for (const [from, to] of REWRITES) {
    if (!staged_source.includes(from)) {
      throw new Error(
        `tools/book/stage-demo.mjs: examples/browser/main.js no longer contains ${from}. ` +
          "Update the rewrite list rather than shipping a demo that cannot load its imports.",
      );
    }
    staged_source = staged_source.replaceAll(from, to);
  }
  await writeFile(join(staged, "main.js"), staged_source);
}

async function main() {
  // Rebuilt from scratch: a stale copy of the SDK is how a demo ends up running
  // last week's module against this week's page.
  await rm(staged, { recursive: true, force: true });
  await mkdir(staged, { recursive: true });

  await cp(join(root, "examples/browser/index.html"), join(staged, "index.html"));
  await stageDriver();
  await cp(join(root, "packages/qwenscriber/dist"), join(staged, "sdk"), { recursive: true });
  await cp(join(root, "gpu/shaders"), join(staged, "shaders"), { recursive: true });
  // Only the two modules the page imports: `harness.mjs` (which imports
  // `reference.mjs`), and nothing else from the harness directory. Listed
  // explicitly because `cp`'s filter is applied to directories too, and rejecting
  // one skips its whole subtree — which is how an earlier version of this script
  // silently produced a demo whose imports 404'd.
  await mkdir(join(staged, "tests"), { recursive: true });
  await cp(join(root, "tests/gpu/reference.mjs"), join(staged, "tests/reference.mjs"));
  await cp(join(root, "tests/gpu/harness.mjs"), join(staged, "tests/harness.mjs"));

  const module = await readFile(join(staged, "sdk/index.js"), "utf8");
  if (!module.includes("SDK_VERSION")) {
    throw new Error(
      "tools/book/stage-demo.mjs: packages/qwenscriber/dist/index.js looks unbuilt. " +
        "Run `npm run build` in packages/qwenscriber first.",
    );
  }

  // Everything the page imports, checked as a set. A staging step that copies
  // nothing is indistinguishable from one that copies everything until a reader
  // opens the page, so the check is here rather than in a browser console.
  const required = [
    "index.html",
    "main.js",
    "sdk/index.js",
    "sdk/qwenscriber_core.wasm",
    "shaders/quant_layout.wgsl",
    "shaders/matmul_q4.wgsl",
    "tests/reference.mjs",
    "tests/harness.mjs",
  ];
  const missing = [];
  for (const relative of required) {
    try {
      await access(join(staged, relative));
    } catch {
      missing.push(relative);
    }
  }
  if (missing.length > 0) {
    throw new Error(`tools/book/stage-demo.mjs: staged demo is incomplete: ${missing.join(", ")}`);
  }
  console.log(`staged the browser demo into ${staged} (${required.length} required files present)`);
}

await main();
