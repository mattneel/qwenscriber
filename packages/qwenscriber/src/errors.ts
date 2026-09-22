//! Typed errors for the whole SDK.
//!
//! Every failure that crosses the SDK boundary is a `QwenscriberError` carrying:
//!
//!   * `code`      the ABI status integer (negative, stable, machine-readable), or an SDK-only code
//!                 below `-1000` for failures the ABI has no number for;
//!   * `status`    the name of that code, as a string, so logs read without a lookup table;
//!   * `operation` the ABI export or SDK method that failed;
//!   * `context`   the arguments that made it fail, so a bug report is reproducible.
//!
//! Nothing in the SDK throws a bare `Error`: a caller can always branch on `code` without parsing
//! text, and can always read `status`/`operation` without consulting documentation.

import { STATUS, statusFallbackMessage, statusName } from "./wasm/abi.ts";

/**
 * Status codes the SDK defines itself.
 *
 * They start at `-1000`, far below the ABI's `-1 .. -14`, so a future ABI code can never collide
 * with them. `-1000` itself is deliberately unused so that an off-by-one in either table is loud.
 */
export const SDK_STATUS = {
  /** The module's ABI major version is not the one this SDK was compiled against. */
  abi_mismatch: -1000,
  /** The ABI does not expose this capability yet. */
  not_implemented: -1001,
  /** The SDK broke its own contract: a malformed message, a missing export, a dead channel. */
  protocol: -1002,
  /** A failure with no code to report, wrapped so callers still get a typed error. */
  internal: -1003,
} as const;
export type SdkStatusName = keyof typeof SDK_STATUS;

const SDK_STATUS_BY_CODE: Readonly<Record<number, SdkStatusName>> = {
  [SDK_STATUS.abi_mismatch]: "abi_mismatch",
  [SDK_STATUS.not_implemented]: "not_implemented",
  [SDK_STATUS.protocol]: "protocol",
  [SDK_STATUS.internal]: "internal",
};

const SDK_FALLBACK_MESSAGES: Readonly<Record<SdkStatusName, string>> = {
  abi_mismatch: "the module's ABI major version is not the one this SDK speaks",
  not_implemented: "not implemented by this build of the core",
  protocol: "the SDK's internal contract was violated",
  internal: "unclassified failure",
};

/** `-1` -> `"invalid_argument"`, `-1000` -> `"abi_mismatch"`, `7` -> `"unknown"`. */
export function statusLabel(code: number): string {
  const sdk = SDK_STATUS_BY_CODE[code];
  if (sdk !== undefined) return sdk;
  return statusName(code);
}

/** Human text for a code, used only when the module cannot supply its own. */
export function fallbackStatusMessage(code: number): string {
  const sdk = SDK_STATUS_BY_CODE[code];
  if (sdk !== undefined) return SDK_FALLBACK_MESSAGES[sdk];
  return statusFallbackMessage(code);
}

export interface QwenscriberErrorOptions {
  /** Overrides the generated message. The `why` of a failure, not the `what`. */
  readonly message?: string | undefined;
  /** The arguments that produced the failure. Kept as-is, never cloned. */
  readonly context?: Readonly<Record<string, unknown>> | undefined;
  /** The underlying failure, when there is one. */
  readonly cause?: unknown;
}

const NO_CONTEXT: Readonly<Record<string, unknown>> = Object.freeze({});

/**
 * The SDK's only error type.
 *
 * Built from a status code rather than a message, because the code is the part programs need: the
 * ABI guarantees `invalid_argument` means the same thing at every call site, while text may change.
 */
export class QwenscriberError extends Error {
  /** The ABI status integer, or an `SDK_STATUS` value. Negative or zero. */
  readonly code: number;
  /** The code's name, for logs: `"invalid_argument"`, `"audio_too_long"`, `"not_implemented"`. */
  readonly status: string;
  /** The export or method that failed, spelled as the ABI spells it: `"qw_mel_compute"`. */
  readonly operation: string;
  /** The arguments that produced the failure. */
  readonly context: Readonly<Record<string, unknown>>;

  constructor(code: number, operation: string, options: QwenscriberErrorOptions = {}) {
    if (!Number.isInteger(code)) {
      throw new TypeError(`QwenscriberError needs an integer status code, got ${String(code)}`);
    }
    if (operation.length === 0) {
      throw new TypeError("QwenscriberError needs the operation that failed");
    }
    const message = options.message ?? `${operation}: ${fallbackStatusMessage(code)}`;
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = "QwenscriberError";
    this.code = code;
    this.status = statusLabel(code);
    this.operation = operation;
    this.context = options.context ?? NO_CONTEXT;
  }
}

/** True when `value` is a `QwenscriberError` built by this realm. */
export function isQwenscriberError(value: unknown): value is QwenscriberError {
  return value instanceof QwenscriberError;
}

/**
 * The module reports an ABI major version this SDK cannot speak.
 *
 * Major versions change struct layouts and semantics, so this is never recoverable by a fallback:
 * the caller must load a module built for this SDK.
 */
export class AbiMismatchError extends QwenscriberError {
  readonly expectedVersion: number;
  readonly actualVersion: number;

  constructor(actualVersion: number, expectedVersion: number, operation = "qw_abi_version") {
    super(SDK_STATUS.abi_mismatch, operation, {
      message:
        `ABI mismatch: the module reports 0x${actualVersion.toString(16)} but this SDK speaks ` +
        `0x${expectedVersion.toString(16)}. Rebuild the module with \`zig build wasm\`, or use a ` +
        `matching SDK version.`,
      context: { actual_version: actualVersion, expected_version: expectedVersion },
    });
    this.name = "AbiMismatchError";
    this.expectedVersion = expectedVersion;
    this.actualVersion = actualVersion;
  }
}

/**
 * The requested stage is not implemented.
 *
 * ABI v1 preprocesses audio and detokenizes ids; it cannot load a model or decode tokens into
 * text, so calls that need those stages fail here instead of returning a fabricated transcript.
 */
export class NotImplementedError extends QwenscriberError {
  /** The missing stage: `"model_load"`, `"decode"`. */
  readonly feature: string;

  constructor(feature: string, operation: string, options: QwenscriberErrorOptions = {}) {
    super(SDK_STATUS.not_implemented, operation, {
      message:
        options.message ??
        `${operation} needs "${feature}", which ABI v1 does not expose yet. ` +
          `Call preprocess() for the stage this build does implement.`,
      // `feature` travels in the context as well as in the field, because the context is what
      // survives a worker boundary: a serialized error can be rebuilt as this class from it.
      context: { ...options.context, feature },
      cause: options.cause,
    });
    this.name = "NotImplementedError";
    this.feature = feature;
  }
}

export interface ThrowForStatusOptions {
  /** The arguments that produced the status, for the error's `context`. */
  readonly context?: Readonly<Record<string, unknown>> | undefined;
  /** Reads the module's own error text. Omit before a module exists to read from. */
  readonly readMessage?: ((code: number) => string | undefined) | undefined;
}

/**
 * Turns a nonzero ABI status into a `QwenscriberError`, and does nothing on success.
 *
 * Every call into the module goes through here, so no export can return a failure that the SDK
 * silently ignores. The message prefers the module's own text (`qw_error_message_ptr`) over the
 * SDK's copy of the table, because the module is the authority on what it just refused.
 */
export function throwForStatus(
  status: number,
  operation: string,
  options: ThrowForStatusOptions = {},
): void {
  if (status === STATUS.ok) return;
  const text = options.readMessage?.(status);
  const detail = text === undefined || text.length === 0 ? fallbackStatusMessage(status) : text;
  throw new QwenscriberError(status, operation, {
    message: `${operation}: ${detail}`,
    context: options.context,
  });
}
