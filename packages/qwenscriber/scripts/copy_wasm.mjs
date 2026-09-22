// Copies the built WASM core next to the compiled SDK, so `dist/` is self-contained.
//
// `dist/qwenscriber_core.wasm` is what `new URL("./qwenscriber_core.wasm", import.meta.url)`
// resolves to at runtime, which is how the SDK finds the module with no configuration and no
// bundler-specific asset rules. The Zig build owns the artifact; this step only places it.

import { copyFile, mkdir, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const source = new URL("../../../zig-out/bin/qwenscriber_core.wasm", import.meta.url);
const destination = new URL("../dist/qwenscriber_core.wasm", import.meta.url);

try {
  await stat(source);
} catch {
  console.error(`missing ${fileURLToPath(source)}; run \`zig build wasm\` first`);
  process.exit(1);
}

await mkdir(new URL("../dist/", import.meta.url), { recursive: true });
await copyFile(source, destination);

const { size } = await stat(destination);
console.log(`copied qwenscriber_core.wasm (${size} bytes) -> ${fileURLToPath(destination)}`);
