#!/usr/bin/env python3
"""Produce ground-truth transcriptions and per-stage activations for bring-up.

This is the reference side of Qwenscriber's correctness strategy: the released
`transformers` implementation runs the real checkpoint, and every intermediate
tensor Qwenscriber has to reproduce is dumped in the `QWFIX001` container so the
Zig runtime can be checked stage by stage instead of as one black box.

Stage order matches `Qwen3ASRForConditionalGeneration`:

    input_features      log-mel, exactly as the encoder receives it
    conv_out            audio tower output after the convolution stack and the
                        sinusoidal positional embedding, before packing
    audio_encoded       audio tower output after `ln_post` and packing
    audio_projected     after the multi-modal projector
    decoder_input_ids   the prompt the decoder is given
    decoder_logits      first-step logits, and the top-k of every step
    transcription       the final decoded text

Usage:
    tools/reference/transcribe_reference.py \
        --model models/Qwen3-ASR-0.6B \
        --audio tests/fixtures/audio/asr_zh.wav \
        --output tests/fixtures/reference
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration


def write_fixture(path: pathlib.Path, array: np.ndarray, dtype: str = "f4") -> None:
    """Writes an array in the QWFIX001 container (f32 payload)."""
    array = np.ascontiguousarray(array, dtype="<" + dtype)
    if array.ndim > 4:
        raise SystemExit(f"{path}: rank {array.ndim} exceeds the container limit")
    dims = [int(d) for d in array.shape] + [1] * (4 - array.ndim)
    with path.open("wb") as handle:
        handle.write(b"QWFIX001")
        handle.write(struct.pack("<I", array.ndim))
        handle.write(struct.pack("<4I", *dims))
        handle.write(array.tobytes(order="C"))
    print(f"wrote {path.name} dims={dims[: array.ndim]} dtype={dtype} bytes={path.stat().st_size}")


def load_audio(path: pathlib.Path) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sample_rate != 16000:
        raise SystemExit(f"{path}: expected 16 kHz audio, got {sample_rate} Hz")
    return np.asarray(samples, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--audio", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(str(args.model))
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(args.model), dtype=torch.float32
    )
    model.eval()

    waveform = load_audio(args.audio)
    print(f"audio: {waveform.shape[0]} samples ({waveform.shape[0] / 16000.0:.2f} s)")

    inputs = processor.apply_transcription_request(
        audio=waveform, language=args.language, prompt=None, return_tensors="pt"
    )
    input_ids = inputs["input_ids"]
    input_features = inputs["input_features"]
    input_features_mask = inputs["input_features_mask"]
    print(f"input_ids: {tuple(input_ids.shape)}")
    print(f"input_features: {tuple(input_features.shape)}, mask sum {int(input_features_mask.sum())}")
    print("prompt prefix:", processor.tokenizer.decode(input_ids[0, :24]))
    print("prompt suffix:", processor.tokenizer.decode(input_ids[0, -24:]))

    # Token ids travel as f32: the vocabulary is far below 2^24, so they round
    # trip exactly, and the container stays a single dtype.
    write_fixture(args.output / "decoder_input_ids.f32", input_ids[0].numpy().astype(np.float32))
    valid_frames = int(input_features_mask.sum())
    write_fixture(
        args.output / "input_features.f32",
        input_features[0, :, :valid_frames].numpy(),
    )

    # Audio tower, instrumented so each stage can be compared separately.
    audio_tower = model.model.audio_tower
    captured: dict[str, torch.Tensor] = {}

    def capture_conv_out(module, module_inputs, module_outputs):
        captured["conv_out"] = module_outputs.detach().clone()

    handle = audio_tower.register_forward_hook(
        lambda module, module_inputs, module_outputs: captured.__setitem__(
            "encoded", module_outputs.last_hidden_state.detach().clone()
        )
    )
    conv_handle = audio_tower.conv_out.register_forward_hook(capture_conv_out)

    audio_output = model.get_audio_features(
        input_features=input_features, input_features_mask=input_features_mask
    )
    conv_handle.remove()
    handle.remove()

    write_fixture(args.output / "audio_conv_out.f32", captured["conv_out"][0].numpy())
    write_fixture(args.output / "audio_encoded.f32", captured["encoded"].numpy())
    write_fixture(args.output / "audio_projected.f32", audio_output.pooler_output.numpy())
    packed_frames = int(captured["encoded"].shape[0])
    print(f"audio tower: packed frames {packed_frames}")

    # First-step logits, plus the top-k of each step, so a decoding bug can be
    # localized to a step without shipping a full vocab-sized fixture per step.
    first_step_logits = None
    step_records: list[tuple[int, np.ndarray, np.ndarray]] = []
    generated = None

    def capture_step(step: int, logits: torch.Tensor) -> None:
        row = logits[0].float()
        top = torch.topk(row, args.top_k)
        step_records.append((int(row.argmax()), top.indices.numpy(), top.values.numpy()))

    def on_logits(step: int):
        def hook(module, module_inputs, module_outputs):
            nonlocal first_step_logits
            logits = module_outputs if torch.is_tensor(module_outputs) else module_outputs[0]
            if step == 0:
                first_step_logits = logits[0].detach().clone()
            capture_step(step, logits.detach())
        return hook

    hooks = []
    for step in range(args.steps):
        hooks.append(model.lm_head.register_forward_hook(on_logits(step)))

    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    for hook in hooks:
        hook.remove()

    if first_step_logits is not None:
        write_fixture(args.output / "logits_step0.f32", first_step_logits.numpy())
    for step, (argmax_token, indices, values) in enumerate(step_records):
        write_fixture(
            args.output / f"logits_step{step}_topk.f32",
            np.asarray(indices, dtype=np.float32),
        )
        write_fixture(
            args.output / f"logits_step{step}_topk_values.f32",
            np.asarray(values, dtype=np.float32),
        )
        print(f"step {step}: argmax {argmax_token}")

    write_fixture(
        args.output / "generated_ids.f32",
        generated[0].numpy().astype(np.float32),
    )

    raw_text = processor.decode(generated, return_format="raw")
    parsed = processor.parse_output(raw_text)
    # `parse_output` returns a list of segments in this transformers release and a
    # single mapping in others, so both shapes are accepted rather than pinning
    # this script to one version's return type.
    record = parsed[0] if isinstance(parsed, list) else parsed
    if not isinstance(record, dict) or "transcription" not in record:
        raise SystemExit(f"unexpected parse_output shape: {type(parsed).__name__}")
    print("raw output:", repr(raw_text))
    print("parsed:", parsed)
    # Written verbatim, marker and all: this file is the comparison target, so
    # post-processing it here would hide a difference in the runtime.
    (args.output / "transcription.txt").write_text(record["transcription"], encoding="utf-8")
    print(f"transcription: {record['transcription']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
