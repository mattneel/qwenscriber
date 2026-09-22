"""Prepare a `transformers`-layout copy of a checkpoint, as `Qwen3-ASR-0.6B-hf` is.

The Python reference loads a processor, and a processor needs files the raw
checkpoint does not ship: `processor_config.json`, `chat_template.jinja`, and a
`tokenizer.json`. `save_pretrained` writes them. Weights are symlinked rather
than copied, so preparing the 1.7B layout costs a few kilobytes instead of
4.7 GB.

# The template trap

The released checkpoints carry an older `chat_template.json`, and its template
has no assistant branch. `apply_transcription_request` renders the assistant
turn as a prefilled final message and passes `continue_final_message=True`, so
that template makes transformers refuse the request outright:

    ValueError: continue_final_message is set but the final message does not
    appear in the chat after applying the chat template

Prepared layouts prepared by an older toolchain carry a template that does have
the branch. If the raw checkpoint's template fails that way, copy the working
template and `processor_config.json` from the other prepared checkpoint of the
same family: the processor files are model-agnostic, while the tokenizer and the
weights must come from the checkpoint being measured.

# Use the published `-hf` repositories

Upstream publishes each checkpoint twice: the conversion source (`Qwen3-ASR-1.7B`) and the conversion
output (`Qwen3-ASR-1.7B-hf`). `transformers` reads the output. The conversion is upstream's own
`convert_qwen3_asr_to_hf.py`, which flattens `thinker_config` **and renames every tensor**
(`thinker.model.` to `model.language_model.`, `thinker.lm_head.` to `lm_head.`), so a raw checkpoint
presented to the library in its own layout is not a slightly different version of the right artifact.

Loading a raw checkpoint produces two errors that look like library gaps and are not:

    ValueError: Audio features and audio tokens do not match, tokens: 55, features: 55

whose two numbers are equal because the check compares element counts, and — once the configuration is
flattened by hand to get past that —

    KeyError: 'qwen3_asr_audio_encoder'

whose name the library maps to nothing. Passing that second error by aliasing the name to the older
`qwen3_asr_encoder` yields a model that loads, runs, and is wrong: the audio tower receives weights
that were not renamed for it, its convolution output comes back in the wrong place, `logits_step0`
differs from the runtime by about 40, and generation produces a different sentence. That was measured,
not assumed, and it is why this script does not flatten unless told to.

Point `transcribe_reference.py` at the published `-hf` repository and stop. This script is for the case
where those files must be prepared from something else — a mirror, a local conversion, a raw release
with no `-hf` sibling — where it prepares the processor side beside an already-converted layout.
"""

import json
import pathlib
import shutil
import sys

from transformers import AutoProcessor

# Fields the flattened form carries that the nested one does not. They are
# family-wide: the released checkpoints share one tokenizer and one set of
# special tokens, so another prepared checkpoint of the family can supply them.
FAMILY_FIELDS = (
    "eos_token_id",
    "pad_token_id",
    "timestamp_token_id",
    "tie_word_embeddings",
    "token_classification_bias",
)

# Upstream renamed the audio encoder's `model_type` between the 0.6B and 1.7B
# releases, while the released transformers only maps the older name. The
# architecture is the same — the runtime reads both — so the prepared copy says
# what the library can look up.
CONFIG_MODEL_TYPE_ALIASES = {
    "qwen3_asr_audio_encoder": "qwen3_asr_encoder",
}


def flatten_config(source: pathlib.Path, target: pathlib.Path, family: pathlib.Path | None) -> bool:
    """Hoist `thinker_config` to the top level, which is the shape transformers reads.

    The released checkpoints nest their whole configuration under
    `thinker_config`, and the released model and processor classes take their
    attributes from the top level. A nested config is therefore not read at all:
    `audio_config.output_dim` falls back to a class default, the audio projector
    emits embeddings of the wrong width, and the forward pass dies with

        ValueError: Audio features and audio tokens do not match, tokens: 55, features: 55

    a message whose two numbers are equal because the real comparison is by
    element count. Returns whether the configuration was rewritten.
    """
    raw = json.loads((source / "config.json").read_text())
    thinker = raw.pop("thinker_config", None)
    if thinker is None:
        return False

    flat = dict(raw)
    flat.update(thinker)
    # A sub-config's `architectures` describes the parent, not the sub-config.
    for section in ("audio_config", "text_config"):
        if isinstance(flat.get(section), dict):
            flat[section].pop("architectures", None)
            flat[section].pop("_name_or_path", None)
            model_type = flat[section].get("model_type")
            if model_type in CONFIG_MODEL_TYPE_ALIASES:
                flat[section]["model_type"] = CONFIG_MODEL_TYPE_ALIASES[model_type]

    generation_path = source / "generation_config.json"
    if generation_path.exists():
        generation = json.loads(generation_path.read_text())
        for field in ("eos_token_id", "pad_token_id"):
            if field not in flat and field in generation:
                flat[field] = generation[field]

    if family is not None:
        known = json.loads((family / "config.json").read_text())
        for field in FAMILY_FIELDS:
            if field not in flat and field in known:
                flat[field] = known[field]

    (target / "config.json").write_text(json.dumps(flat, indent=2) + "\n")
    return True


def main() -> int:
    arguments = [argument for argument in sys.argv[1:] if argument != "--flatten"]
    flatten = len(arguments) != len(sys.argv[1:])
    if len(arguments) not in (2, 3):
        print(f"usage: {sys.argv[0]} <checkpoint dir> <prepared dir> [family prepared dir] [--flatten]")
        print("  the family directory supplies processor files the checkpoint lacks")
        print("  --flatten hoists a raw checkpoint's `thinker_config`; read the module docstring first")
        return 2
    source = pathlib.Path(arguments[0])
    target = pathlib.Path(arguments[1])
    family = pathlib.Path(arguments[2]) if len(arguments) == 3 else None
    if not source.is_dir():
        print(f"{source}: not a directory")
        return 1
    if family is not None and not (family / "chat_template.jinja").exists():
        print(f"{family}: has no chat_template.jinja to take processor files from")
        return 1
    if "thinker_config" in json.loads((source / "config.json").read_text()) and not flatten:
        print(f"{source}/config.json nests its configuration under `thinker_config`.")
        print("That is the conversion source, not the artifact transformers reads. Use the published")
        print(f"  https://huggingface.co/Qwen/Qwen3-ASR-<size>-hf")
        print("or pass --flatten to build a model that loads and is wrong. See the module docstring.")
        return 1
    target.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(str(source))
    processor.save_pretrained(str(target))

    # The model itself stays in the source directory. `from_pretrained` follows
    # symlinks, and a second 4.7 GB copy would only be paid for with disk.
    linked = 0
    for weight in sorted(source.glob("*.safetensors")) + sorted(source.glob("*.index.json")):
        destination = target / weight.name
        if destination.exists() or destination.is_symlink():
            continue
        destination.symlink_to(weight.resolve())
        linked += 1

    flattened = flatten_config(source, target, family) if flatten else False

    # The checkpoint's own template has no assistant branch, which
    # `continue_final_message` requires; the family's has it. Both files describe
    # the processor, not the weights, so taking them from a prepared checkpoint of
    # the same family is what makes the reference runnable at all.
    from_family = 0
    if family is not None:
        for name in ("chat_template.jinja", "processor_config.json"):
            origin = family / name
            if origin.exists():
                shutil.copyfile(origin, target / name)
                from_family += 1

    print(f"prepared {target} from {source}: {linked} weights linked")
    print(f"  configuration     {'flattened from thinker_config' if flattened else 'already flat'}")
    if from_family:
        print(f"  processor files   {from_family} taken from {family}")
    for written in sorted(target.iterdir()):
        print(f"  {written.name}{' -> ' + str(written.resolve()) if written.is_symlink() else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
