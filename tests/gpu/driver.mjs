// Node side of the GPU harness: serve the repository, launch headless Chrome
// with WebGPU enabled, wait for the page to publish `window.__result`, print the
// table, and exit nonzero when a kernel is outside its tolerance.
//
// Chrome is driven over the DevTools protocol with Node's built-in `WebSocket`,
// so neither side of this harness has a dependency to install.
//
// Usage: node tests/gpu/harness.mjs [flags]
//
//   --url <url>        drive an already served page instead of starting a server
//   --chrome <path>    browser binary (default: $QWENSCRIBER_CHROME, the
//                      puppeteer cache, or chrome/chromium on PATH)
//   --timeout <s>      how long to wait for `window.__result` (default 240)
//   --no-swiftshader   keep the GPU flags off, for a machine with a real adapter
//   --headed           show the window instead of running headless
//   --keep-open        leave Chrome and the server running for a look

import { createServer } from "node:http";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".wasm": "application/wasm",
    ".wgsl": "text/plain; charset=utf-8",
    ".json": "application/json; charset=utf-8",
};

export function parse_args(argv) {
    const options = { timeout_seconds: 240, swiftshader: true, headed: false, keep_open: false };
    for (let index = 0; index < argv.length; index += 1) {
        const flag = argv[index];
        if (flag === "--url") options.url = argv[++index];
        else if (flag === "--chrome") options.chrome = argv[++index];
        else if (flag === "--timeout") options.timeout_seconds = Number(argv[++index]);
        else if (flag === "--no-swiftshader") options.swiftshader = false;
        else if (flag === "--headed") options.headed = true;
        else if (flag === "--keep-open") options.keep_open = true;
        else throw new Error(`unknown flag ${flag}`);
    }
    return options;
}

function delay(milliseconds) {
    return new Promise((done) => setTimeout(done, milliseconds));
}

// ---------------------------------------------------------------------------
// Static server
// ---------------------------------------------------------------------------

async function serve_file(root, request, response) {
    const url = new URL(request.url, "http://127.0.0.1");
    const target = resolve(root, `.${decodeURIComponent(url.pathname)}`);
    if (!target.startsWith(root)) {
        response.writeHead(403).end("outside the served root");
        return;
    }
    try {
        const bytes = await readFile(target);
        const content_type = CONTENT_TYPES[extname(target)] ?? "application/octet-stream";
        response.writeHead(200, { "content-type": content_type });
        response.end(bytes);
    } catch (error) {
        response.writeHead(404).end(String(error));
    }
}

function start_server(root) {
    const server = createServer((request, response) => {
        void serve_file(root, request, response);
    });
    return new Promise((done, failed) => {
        server.on("error", failed);
        server.listen(0, "127.0.0.1", () => done({ server, port: server.address().port }));
    });
}

// ---------------------------------------------------------------------------
// Chrome
// ---------------------------------------------------------------------------

export async function find_chrome(explicit) {
    if (explicit !== undefined) return explicit;
    if (process.env.QWENSCRIBER_CHROME !== undefined) return process.env.QWENSCRIBER_CHROME;
    const cache = join(process.env.HOME ?? "/root", ".omp", "puppeteer", "chrome");
    const found = [];
    try {
        for (const version of await readdir(cache, { withFileTypes: true })) {
            if (!version.isDirectory()) continue;
            const candidate = join(cache, version.name, "chrome-linux64", "chrome");
            if (existsSync(candidate)) found.push(candidate);
        }
    } catch (error) {
        if (error.code !== "ENOENT") throw error;
    }
    found.sort();
    if (found.length > 0) return found[found.length - 1];
    const system_candidates = [
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ];
    for (const candidate of system_candidates) {
        if (existsSync(candidate)) return candidate;
    }
    throw new Error("no browser found: pass --chrome or set QWENSCRIBER_CHROME");
}

function launch_args(options, profile, url) {
    const args = [
        options.headed ? "--window-size=1280,900" : "--headless=new",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--enable-unsafe-webgpu",
        "--remote-debugging-port=0",
        // The DevTools endpoint rejects WebSocket handshakes that carry an
        // Origin header it does not know; Node's WebSocket client may send one.
        "--remote-allow-origins=*",
    ];
    if (options.swiftshader) {
        args.push(
            "--enable-unsafe-swiftshader",
            "--use-webgpu-adapter=swiftshader",
        );
    }
    args.push(
        `--user-data-dir=${profile}`,
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-sync",
        "--disable-background-networking",
        "--mute-audio",
        url,
    );
    return args;
}

async function launch_chrome(options, url) {
    const chrome = await find_chrome(options.chrome);
    const profile = await mkdtemp(join(tmpdir(), "qwenscriber-gpu-"));
    const child = spawn(chrome, launch_args(options, profile, url), {
        stdio: ["ignore", "pipe", "pipe"],
    });
    let stderr = "";
    child.stderr.on("data", (chunk) => {
        stderr += String(chunk);
    });
    let exited = false;
    child.on("exit", () => {
        exited = true;
    });

    const port_file = join(profile, "DevToolsActivePort");
    for (let waited_ms = 0; waited_ms < 20000; waited_ms += 100) {
        if (exited) throw new Error(`chrome exited before DevTools was ready:\n${stderr}`);
        if (existsSync(port_file)) {
            const port = Number(String(await readFile(port_file, "utf8")).split("\n")[0]);
            if (Number.isFinite(port) && port > 0) {
                return { child, profile, port };
            }
        }
        await delay(100);
    }
    child.kill("SIGKILL");
    throw new Error(`chrome did not open a DevTools port:\n${stderr}`);
}

// ---------------------------------------------------------------------------
// DevTools protocol
// ---------------------------------------------------------------------------

async function connect_page(port) {
    let target = null;
    for (let attempt = 0; attempt < 100; attempt += 1) {
        const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
        target = list.find((entry) => entry.type === "page" && entry.url.startsWith("http"));
        if (target !== undefined) break;
        await delay(100);
    }
    if (target === null || target === undefined) throw new Error("no page target appeared");

    const socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((done, failed) => {
        socket.addEventListener("open", done);
        socket.addEventListener("error", () => failed(new Error("DevTools socket failed")));
    });

    const client = { socket, next_id: 1, pending: new Map(), logs: [], errors: [] };
    socket.addEventListener("message", (event) => {
        const message = JSON.parse(String(event.data));
        if (message.id !== undefined) {
            const entry = client.pending.get(message.id);
            if (entry !== undefined) {
                client.pending.delete(message.id);
                entry(message);
            }
            return;
        }
        if (message.method === "Runtime.consoleAPICalled") {
            client.logs.push(
                `${message.params.type}: ` +
                    message.params.args.map((arg) => arg.value ?? arg.description ?? "").join(" "),
            );
        }
        if (message.method === "Runtime.exceptionThrown") {
            client.errors.push(message.params.exceptionDetails.text);
        }
    });
    await cdp_send(client, "Runtime.enable", {});
    return client;
}

function cdp_send(client, method, params) {
    const id = client.next_id;
    client.next_id += 1;
    return new Promise((done, failed) => {
        const timer = setTimeout(() => failed(new Error(`${method} timed out`)), 60000);
        client.pending.set(id, (message) => {
            clearTimeout(timer);
            if (message.error !== undefined) {
                failed(new Error(`${method}: ${JSON.stringify(message.error)}`));
            }
            else done(message.result);
        });
        client.socket.send(JSON.stringify({ id, method, params }));
    });
}

async function wait_for_result(client, timeout_seconds) {
    const expression = "window.__result === undefined ? null : window.__result";
    const deadline = Date.now() + timeout_seconds * 1000;
    while (Date.now() < deadline) {
        const evaluated = await cdp_send(client, "Runtime.evaluate", {
            expression,
            returnByValue: true,
        });
        if (evaluated.result !== undefined && evaluated.result.value !== null) {
            return evaluated.result.value;
        }
        await delay(500);
    }
    return null;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

export async function run_driver(argv) {
    const options = parse_args(argv);
    const root = fileURLToPath(new URL("../../", import.meta.url));
    let server = null;
    let browser = null;
    try {
        let url = options.url;
        if (url === undefined) {
            server = await start_server(root);
            url = `http://127.0.0.1:${server.port}/tests/gpu/harness.html`;
            console.log(`serving ${root} at http://127.0.0.1:${server.port}`);
        }
        console.log(`loading ${url}`);
        browser = await launch_chrome(options, url);
        const client = await connect_page(browser.port);
        const result = await wait_for_result(client, options.timeout_seconds);

        if (result === null) {
            console.error(`no window.__result after ${options.timeout_seconds}s`);
            for (const line of client.logs.slice(-40)) console.error(`  console ${line}`);
            for (const line of client.errors) console.error(`  page error ${line}`);
            const body = await cdp_send(client, "Runtime.evaluate", {
                expression: "document.body === null ? '' : document.body.innerText.slice(0, 4000)",
                returnByValue: true,
            });
            console.error(`  page text: ${String(body.result?.value ?? "")}`);
            return 1;
        }

        console.log(result.table_text ?? JSON.stringify(result, null, 2));
        for (const line of client.errors) console.error(`page error: ${line}`);
        return result.ok === false ? 1 : 0;
    } catch (error) {
        console.error(`driver error: ${String(error)}`);
        return 2;
    } finally {
        if (options.keep_open) {
            console.log("--keep-open: leaving the browser and server running");
        } else {
            if (browser !== null) {
                browser.child.kill("SIGKILL");
                // Chrome keeps writing to its profile for a moment after the
                // kill. Deleting it is best effort and must never mask the exit
                // code the run produced.
                await delay(400);
                const remove_options = { recursive: true, force: true, maxRetries: 5 };
                await rm(browser.profile, remove_options).catch(() => {});
            }
            if (server !== null) server.server.close();
        }
    }
}
