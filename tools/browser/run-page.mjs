// Drives a page in a real browser and reports what happened.
//
// Qwenscriber's WebGPU path has to be verified against a real adapter, and on a
// WSL2 host the GPU belongs to Windows, not to the Linux side. WebGPU is
// therefore exercised through Chrome on Windows:
//
//   1. Serve the repository over http from inside WSL:
//
//          python3 -m http.server 8766 --bind 127.0.0.1
//
//      Windows forwards `localhost` to WSL listeners, so the served pages and
//      `zig-out/bin/qwenscriber_core.wasm` are reachable from Chrome.
//
//   2. Install the driver's one dependency once, on the Windows side:
//
//          cd /mnt/c/Users/<you>/omp-browser
//          cmd.exe /c "npm.cmd install playwright@1.56.0"
//
//      Playwright drives the *installed* Chrome (`channel: "chrome"`), so no
//      browser download is needed and the run uses the machine's real GPU.
//
//   3. Run a page:
//
//          cmd.exe /c 'cd /d C:\Users\<you>\omp-browser && node.exe run-page.mjs \
//              http://127.0.0.1:8766/tests/gpu/harness.html --settle-ms 5000'
//
// The page under test sets `window.__result` to any JSON-serialisable value when
// its work is finished. This driver prints that value together with every
// console message and page error, so a harness reports structured numbers
// instead of a screenshot. Exit code is 0 only when `__result` exists and its
// `ok` field is not false.
//
// Fallback without a GPU: run under the WSL-side Chromium with SwiftShader using
// `--swiftshader`. It is correct but far slower, so prefer the Windows path for
// anything timing-related.

import { readFileSync } from "node:fs";
import { chromium } from "playwright";

function parseArgs(argv) {
  const args = {
    url: argv[0],
    settleMs: 3000,
    swiftshader: false,
    evalFile: null,
    headless: true,
    channel: "chrome",
  };
  for (let index = 1; index < argv.length; index += 1) {
    const flag = argv[index];
    if (flag === "--settle-ms") args.settleMs = Number(argv[++index]);
    else if (flag === "--swiftshader") args.swiftshader = true;
    else if (flag === "--eval-file") args.evalFile = argv[++index];
    else if (flag === "--headed") args.headless = false;
    else if (flag === "--channel") args.channel = argv[++index];
    else throw new Error(`unknown flag ${flag}`);
  }
  if (!args.url) throw new Error("usage: node run-page.mjs <url> [flags]");
  return args;
}

const args = parseArgs(process.argv.slice(2));

const flags = ["--no-sandbox", "--disable-dev-shm-usage", "--enable-unsafe-webgpu"];
if (args.swiftshader) {
  flags.push("--enable-unsafe-swiftshader", "--use-webgpu-adapter=swiftshader");
}

const launchOptions = { headless: args.headless, args: flags };
if (args.channel) launchOptions.channel = args.channel;

const browser = await chromium.launch(launchOptions);
const context = await browser.newContext();
const page = await context.newPage();

const logs = [];
const errors = [];
page.on("console", (message) => logs.push(`${message.type()}: ${message.text()}`));
page.on("pageerror", (error) => errors.push(String(error)));
page.on("requestfailed", (request) =>
  errors.push(`request failed: ${request.url()} ${request.failure()?.errorText ?? ""}`),
);

let status = 1;
try {
  await page.goto(args.url, { waitUntil: "load", timeout: 60000 });
  if (args.evalFile) {
    const evaluated = await page.evaluate(readFileSync(args.evalFile, "utf8"));
    console.log("eval result:", JSON.stringify(evaluated, null, 2));
  }
  await page.waitForTimeout(args.settleMs);

  const result = await page.evaluate(() =>
    typeof window.__result === "undefined" ? null : window.__result,
  );
  console.log("--- console ---");
  for (const line of logs) console.log(line);
  if (errors.length > 0) {
    console.log("--- page errors ---");
    for (const line of errors) console.log(line);
  }
  console.log("--- result ---");
  console.log(result === null ? "<window.__result was never set>" : JSON.stringify(result, null, 2));
  status = result !== null && result.ok !== false ? 0 : 1;
} catch (error) {
  console.log("--- driver error ---");
  console.log(String(error));
  for (const line of logs) console.log(line);
  for (const line of errors) console.log(line);
} finally {
  await browser.close();
}

process.exit(status);
