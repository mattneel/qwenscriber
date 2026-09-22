#!/usr/bin/env node
// Runs the tower check in headless Chrome.
//
// The page loads a converted model and a reference fixture from the repository itself, so this
// starts a static server and then hands the URL to the kernel harness's driver -- same Chrome, same
// WebGPU flags, same `window.__result` contract, no second copy of that machinery.
//
// Usage: node tests/gpu/tower.mjs [driver flags]
//
// Prerequisites, both local because neither belongs in the repository:
//
//   1. a converted q5 model at models/qwen3-asr-0.6b-q5
//   2. its stage dump, which is the comparison target:
//        qwenscriber-transcribe --model models/qwen3-asr-0.6b-q5 \
//            --audio tests/fixtures/audio/asr_zh.wav --dump models/qwen3-asr-0.6b-q5/dump
//
// `tower_page.mjs` explains why the oracle is our own runtime's dump rather than the released
// implementation's fixture: the browser can only hold a quantized build, and the two agree stage by
// stage anyway (`tools/reference/compare_fixtures.py`).

import { readFile, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

import { run_driver } from "./driver.mjs";

const CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
    ".f32": "application/octet-stream",
    ".bin": "application/octet-stream",
    ".safetensors": "application/octet-stream",
};

async function serve(root, request, response) {
    const path = normalize(decodeURIComponent(new URL(request.url, "http://x").pathname));
    const full = join(root, path);
    if (!full.startsWith(root.replace(/\/$/, ""))) {
        response.writeHead(403).end();
        return;
    }
    try {
        const info = await stat(full);
        if (info.isDirectory()) {
            response.writeHead(404).end();
            return;
        }
        response.writeHead(200, {
            "content-type": CONTENT_TYPES[extname(full)] ?? "application/octet-stream",
            "content-length": String(info.size),
        });
        response.end(await readFile(full));
    } catch {
        response.writeHead(404).end();
    }
}

const root = fileURLToPath(new URL("../../", import.meta.url));
const server = createServer((request, response) => {
    serve(root, request, response);
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const url = `http://127.0.0.1:${server.address().port}/tests/gpu/tower.html`;
console.log(`serving ${root} at ${url}`);
try {
    process.exit(await run_driver(["--url", url, ...process.argv.slice(2)]));
} finally {
    server.close();
}
