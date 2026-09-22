//! Minimal ambient declarations for the Node built-ins the tests use.
//!
//! The package has exactly one devDependency, TypeScript, and `@types/node` is a dependency, so
//! these declarations cover the small surface the tests touch -- `node:test`, `node:assert/strict`,
//! `node:fs`, `node:child_process`, `node:url` -- and nothing else. They exist so the tests are
//! still typechecked under `strict` instead of being excluded from it. Application code that needs
//! the real Node API should install `@types/node`; this file is deliberately not a substitute.

declare module "node:test" {
  interface TestContext {
    readonly name: string;
    diagnostic(message: string): void;
  }
  type TestBody = (context: TestContext) => void | Promise<void>;
  interface TestOptions {
    readonly timeout?: number;
    readonly skip?: boolean | string;
    readonly concurrency?: number;
  }
  interface TestApi {
    (name: string, body: TestBody): Promise<void>;
    (name: string, options: TestOptions, body: TestBody): Promise<void>;
  }
  export const test: TestApi;
  export function describe(name: string, body: () => void | Promise<void>): void;
  export function before(body: () => void | Promise<void>): void;
  export function after(body: () => void | Promise<void>): void;
}

declare module "node:assert/strict" {
  type ErrorCheck =
    | ((error: unknown) => boolean)
    | RegExp
    | Error
    | (new (...args: never[]) => Error);
  interface StrictAssert {
    (value: unknown, message?: string | Error): asserts value;
    ok(value: unknown, message?: string | Error): asserts value;
    equal(actual: unknown, expected: unknown, message?: string | Error): void;
    notEqual(actual: unknown, expected: unknown, message?: string | Error): void;
    deepEqual(actual: unknown, expected: unknown, message?: string | Error): void;
    match(value: string, pattern: RegExp, message?: string | Error): void;
    throws(body: () => unknown, check?: ErrorCheck, message?: string | Error): void;
    rejects(
      value: Promise<unknown> | (() => Promise<unknown>),
      check?: ErrorCheck,
      message?: string | Error,
    ): Promise<void>;
  }
  const strict: StrictAssert;
  export default strict;
}

declare module "node:fs" {
  export function readFileSync(path: string | URL): Uint8Array;
  export function existsSync(path: string | URL): boolean;
}

declare module "node:child_process" {
  export function execFileSync(
    file: string,
    args?: readonly string[],
    options?: { readonly cwd?: string; readonly stdio?: "pipe" | "inherit" | "ignore" },
  ): Uint8Array;
}

declare module "node:url" {
  export function fileURLToPath(url: string | URL): string;
}

declare const process: {
  readonly env: Record<string, string | undefined>;
  exitCode: number | undefined;
  cwd(): string;
};
