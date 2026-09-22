# Security policy

Qwenscriber handles untrusted audio, model metadata, tokenizer data, cache entries, and application
inputs. Treat malformed lengths, offsets, tensor shapes, manifests, and handles as ordinary hostile
input—not as impossible states.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for the repository when available. Do not open a
public issue for an unpatched vulnerability. If private reporting is unavailable, contact the
maintainers through an existing private channel before disclosing details publicly.

Include, where possible:

- affected revision or release;
- affected runtime and platform;
- a minimal reproduction;
- impact and required preconditions; and
- whether model or audio data must be attacker-controlled.

The maintainers will acknowledge the report, reproduce and triage it, coordinate a fix and release,
and credit the reporter if requested and appropriate. No fixed response SLA is promised before the
project's first supported release.

## Security boundaries

- Local-first means Qwenscriber should not transmit audio or transcripts by default.
- Model artifacts must be versioned and integrity-checked before use.
- Routine malformed input returns explicit errors across the ABI; traps indicate implementation
  defects or broken invariants.
- WASM isolation is defense in depth, not permission to skip bounds checks.
- WebGPU buffer sizes, offsets, bindings, and dispatch dimensions must be validated.
- Cache keys must include format/model identity and must not permit path or origin confusion.

See the [security and privacy design](docs/src/project/security-and-privacy.md) for the broader
threat model.

