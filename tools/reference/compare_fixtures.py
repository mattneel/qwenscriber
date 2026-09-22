#!/usr/bin/env python3
"""Compare Qwenscriber's intermediates against the reference pipeline.

Qwenscriber's bring-up runner writes each stage in the `QWFIX001` container:

    qwenscriber-transcribe --model <dir> --audio clip.wav --dump ours/

`tools/reference/transcribe_reference.py` writes the same stages for the same
clip from the released `transformers` implementation:

    tools/reference/transcribe_reference.py --model <hf checkpoint> \\
        --audio clip.wav --output reference/

This script compares the two stage by stage. A mismatch tells you *where* the
implementations diverge, which is the whole point: debugging a transcription as
one black box means guessing, and guessing about a 600-tensor model is slow.

Usage:
    compare_fixtures.py --ours ours --reference reference [--verbose]
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

# Stage name -> (file name, tolerance kind, justification).
STAGES = [
    (
        "input_features",
        "input_features.f32",
        ("abs", 1e-4),
        "log-mel: measured 8.6e-5 worst case, from a different FFT summation "
        "order and from computing re^2+im^2 instead of abs()**2",
    ),
    (
        "audio_encoded",
        "audio_encoded.f32",
        ("rel", 2e-2),
        "audio tower: 18 pre-norm blocks of f32 accumulation over f16/bf16 "
        "weights, so agreement is limited by the weight precision, not the code",
    ),
    (
        "audio_projected",
        "audio_projected.f32",
        ("rel", 2e-2),
        "projector output, inherits the audio tower's error and adds two more "
        "linear layers",
    ),
    (
        "decoder_input_ids",
        "decoder_input_ids.f32",
        ("exact", 0),
        "the prompt must be token for token identical; any difference here is a "
        "prompt bug, not a numeric one",
    ),
    (
        "logits_step0",
        "logits_step0.f32",
        ("abs", 5e-2),
        "first-step logits after 28 decoder layers",
    ),
    (
        "generated_ids",
        "generated_ids.f32",
        ("exact", 0),
        "the token sequence is the deliverable",
    ),
]


def load(path: pathlib.Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 28:
        raise SystemExit(f"{path}: shorter than the container header")
    if data[:8] != b"QWFIX001":
        raise SystemExit(f"{path}: not a QWFIX001 container")
    rank = struct.unpack_from("<I", data, 8)[0]
    dims = struct.unpack_from("<4I", data, 12)[:rank]
    count = int(np.prod(dims)) if dims else 0
    expected = 28 + count * 4
    if len(data) != expected:
        raise SystemExit(f"{path}: expected {expected} bytes for {dims}, found {len(data)}")
    return np.frombuffer(data[28:], dtype="<f4").reshape(dims)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    failures = 0
    missing: list[str] = []

    print(f"{'stage':22} {'shape':>16} {'max|d|':>12} {'mean|d|':>12} {'rel':>10}  verdict")
    print("-" * 92)

    for name, file_name, (kind, tolerance), justification in STAGES:
        ours_path = args.ours / file_name
        reference_path = args.reference / file_name
        if not ours_path.exists() or not reference_path.exists():
            missing.append(f"{name} ({'ours' if not ours_path.exists() else 'reference'} missing)")
            continue

        ours = load(ours_path)
        reference = load(reference_path)
        if ours.shape != reference.shape:
            print(f"{name:22} SHAPE MISMATCH ours={ours.shape} reference={reference.shape}")
            failures += 1
            continue

        difference = np.abs(ours - reference)
        scale = np.maximum(np.abs(reference), 1e-6)
        relative = float(np.max(difference / scale))
        worst = float(np.max(difference)) if difference.size else 0.0
        mean = float(np.mean(difference)) if difference.size else 0.0

        if kind == "exact":
            ok = bool(np.array_equal(ours, reference))
            verdict = "exact" if ok else "MISMATCH"
        elif kind == "abs":
            ok = worst <= tolerance
            verdict = f"ok (abs <= {tolerance:g})" if ok else f"FAIL (> {tolerance:g})"
        else:
            ok = relative <= tolerance
            verdict = f"ok (rel <= {tolerance:g})" if ok else f"FAIL (> {tolerance:g})"

        shape_text = "x".join(str(dim) for dim in ours.shape)
        print(f"{name:22} {shape_text:>16} {worst:12.3e} {mean:12.3e} {relative:10.3e}  {verdict}")
        if args.verbose:
            print(f"{'':22} tolerance: {justification}")
        if not ok:
            failures += 1
            if kind == "exact":
                mismatch = np.flatnonzero(ours != reference)
                print(f"{'':22} first mismatches at {mismatch[:8].tolist()}")
                print(f"{'':22} ours      {ours.flat[mismatch[:8]].tolist()}")
                print(f"{'':22} reference {reference.flat[mismatch[:8]].tolist()}")
            else:
                worst_index = int(np.argmax(difference))
                flat = np.unravel_index(worst_index, ours.shape)
                print(f"{'':22} worst at {flat}: ours {ours[flat]:.6f} reference {reference[flat]:.6f}")

    if missing:
        print()
        for entry in missing:
            print(f"skipped: {entry}")

    print()
    if failures == 0:
        print("all compared stages agree")
    else:
        print(f"{failures} stage(s) disagree")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
