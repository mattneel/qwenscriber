//! Where the WGSL sources come from.
//!
//! `WebGpuRuntime` takes a `shaderSource` function instead of a directory because only the caller
//! knows how its own files are laid out: this package is served from `dist/` in one deployment and
//! bundled in another, so a default path would be correct in exactly one of them. What every http
//! caller does need is the same twenty lines of fetch-and-check, which is what this module is -- and
//! the one place a missing shader turns into a typed error instead of a compile failure that blames
//! the wrong file.

import { QwenscriberError } from "../errors.ts";
import { STATUS } from "../wasm/abi.ts";
import type { ShaderSource } from "./runtime.ts";

/**
 * A `ShaderSource` that reads `file` from `baseUrl`.
 *
 * `baseUrl` must be absolute -- `new URL("../../gpu/shaders/", import.meta.url)` from a module is
 * the intended spelling -- because a relative URL has nothing to resolve against inside a library
 * that is imported from anywhere. A shader that is not there raises `not_found` with the URL and the
 * status, so a 404 never arrives disguised as a WGSL syntax error.
 */
export function shaderSourceFromBaseUrl(baseUrl: URL): ShaderSource {
  return async (file: string): Promise<string> => {
    const url = new URL(file, baseUrl);
    let response: Response;
    try {
      response = await fetch(url);
    } catch (error) {
      throw new QwenscriberError(STATUS.truncated, "gpu.shaderSource", {
        message: `gpu.shaderSource: ${url.pathname} could not be fetched`,
        context: { file, url: url.href },
        cause: error,
      });
    }
    if (!response.ok) {
      throw new QwenscriberError(STATUS.not_found, "gpu.shaderSource", {
        message: `gpu.shaderSource: ${url.pathname} answered ${response.status}`,
        context: { file, url: url.href, status: response.status },
      });
    }
    return response.text();
  };
}
