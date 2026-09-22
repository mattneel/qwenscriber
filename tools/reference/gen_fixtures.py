#!/usr/bin/env python3
"""Generate deterministic numeric fixtures for Qwenscriber's test suite.

This is a bring-up aid, not part of the runtime. It uses the reference
implementation that ships with `transformers` to produce golden values for the
audio frontend, so the Zig implementation can be compared against the exact
numbers a real Qwen3-ASR pipeline produces.

Every fixture is written in the tiny `QWFIX001` container described in
docs/MODEL_FORMAT.md:

    offset 0   magic         8 bytes  "QWFIX001"
    offset 8   rank          u32
    offset 12  dims          4 x u32  (unused entries are 1)
    offset 28  payload       f32 x product(dims), little endian, row major

Usage:
    tools/reference/gen_fixtures.py --output tests/fixtures
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np


def write_fixture(path: pathlib.Path, array: np.ndarray) -> None:
    """Writes an array in the QWFIX001 container."""
    array = np.ascontiguousarray(array, dtype="<f4")
    if array.ndim > 4:
        raise SystemExit(f"{path}: rank {array.ndim} exceeds the container limit")
    dims = [int(d) for d in array.shape] + [1] * (4 - array.ndim)
    with path.open("wb") as handle:
        handle.write(b"QWFIX001")
        handle.write(struct.pack("<I", array.ndim))
        handle.write(struct.pack("<4I", *dims))
        handle.write(array.tobytes(order="C"))
    print(f"wrote {path} dims={dims[: array.ndim]} bytes={path.stat().st_size}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from transformers.audio_utils import mel_filter_bank
    from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import (
        Qwen3ASRFeatureExtractor,
    )

    extractor = Qwen3ASRFeatureExtractor()

    # The mel filterbank itself: 201 frequency bins (n_fft 400) x 128 filters.
    write_fixture(
        args.output / "mel_filters.f32",
        np.asarray(extractor.mel_filters, dtype=np.float32),
    )

    # A deterministic waveform: 12345 samples is deliberately not a multiple of
    # the 160-sample hop, so the final frame is a partial one.
    sample_count = 12345
    index = np.arange(sample_count, dtype=np.float64)
    waveform = (
        0.60 * np.sin(2.0 * np.pi * 220.0 * index / 16000.0)
        + 0.25 * np.sin(2.0 * np.pi * 1337.0 * index / 16000.0)
        + 0.15 * np.sin(2.0 * np.pi * 3999.0 * index / 16000.0)
    )
    # A slow amplitude envelope keeps quiet frames quiet, which exercises the
    # `max(log_spec, global_max - 8)` clamp instead of leaving every frame loud.
    waveform = waveform * (0.1 + 0.9 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.7 * index / 16000.0)))
    waveform = np.asarray(waveform, dtype=np.float32)
    write_fixture(args.output / "mel_waveform.f32", waveform)

    log_mel = extractor._torch_extract_fbank_features(waveform)
    write_fixture(args.output / "mel_expected_unpadded.f32", np.asarray(log_mel, dtype=np.float32))
    print(f"unpadded log-mel shape {log_mel.shape}")
    # The processor drops a final partial hop (floor(samples / hop)), and the
    # processor is what the model is fed, so the fixture the runtime is checked
    # against has to drop it too. The extractor's own call above keeps it, which
    # is why this count and the one below differ by one for this waveform.

    # The authoritative pipeline path: the extractor zero-pads the clip to a
    # whole 30 seconds, computes the spectrogram over that buffer, and marks a
    # frame valid while its first sample is a real one. The encoder only ever
    # sees those frames, so this is what the runtime must reproduce.
    batch = extractor(waveform, sampling_rate=16000, return_tensors="np", return_attention_mask=True)
    padded = np.asarray(batch["input_features"], dtype=np.float32)
    mask = np.asarray(batch["attention_mask"])
    valid_frames = int(mask.sum())
    hop_length = 160
    processed_frames = len(waveform) // hop_length
    if processed_frames != valid_frames:
        print(f"processor marks {valid_frames} frames valid; the runtime uses {processed_frames}")
    valid_frames = min(valid_frames, processed_frames)
    expected = padded[0, :, :valid_frames]
    write_fixture(args.output / "mel_expected.f32", expected)
    print(
        f"padded log-mel shape {padded.shape}, mask sum {valid_frames}, "
        f"expected {expected.shape}, min {expected.min():.6f}, max {expected.max():.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
