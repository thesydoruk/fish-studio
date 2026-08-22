"""Convert a merged fish-speech checkpoint to the HF layout served by vLLM-Omni.

The merged checkpoint (``./run.sh train merge``) stores weights as a
fish-native ``model.pth`` state dict. vLLM-Omni loads the HuggingFace layout
of ``fishaudio/s2-pro`` (safetensors with ``text_model.*`` /
``audio_decoder.*`` prefixes). The tensors are identical — only the key names
differ — so conversion is a mechanical rename plus copying the static files
(config, tokenizer, DAC codec) from the stock checkpoint.

Output goes to ``{data_root}/training/vllm/``; point ``fish_speech.model``
at it (``training/vllm``) and restart the vLLM server.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from fish_studio.config import load_config
from fish_studio.training.layout import resolve_base_checkpoint, resolve_merged_checkpoint

# fish-native key prefix → HF key prefix (order matters: fast_* before bare).
_RENAME_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^fast_embeddings\."), "audio_decoder.embeddings."),
    (re.compile(r"^fast_layers\."), "audio_decoder.layers."),
    (re.compile(r"^fast_norm\."), "audio_decoder.norm."),
    (re.compile(r"^fast_output\."), "audio_decoder.output."),
    (re.compile(r"^codebook_embeddings\."), "audio_decoder.codebook_embeddings."),
    (re.compile(r"^embeddings\."), "text_model.model.embeddings."),
    (re.compile(r"^layers\."), "text_model.model.layers."),
    (re.compile(r"^norm\."), "text_model.model.norm."),
]

# Static files taken from the stock HF checkpoint.
_STOCK_FILES = (
    "config.json",
    "chat_template.jinja",
    "codec.pth",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
)


def _rename_key(key: str) -> str:
    for pattern, replacement in _RENAME_RULES:
        if pattern.match(key):
            return pattern.sub(replacement, key)
    raise ValueError(f"unexpected key in merged checkpoint: {key}")


def convert(merged_dir: Path, stock_dir: Path, output_dir: Path) -> None:
    import torch
    from safetensors.torch import save_file

    weights_path = merged_dir / "model.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"merged checkpoint not found: {weights_path}. Run ./run.sh train merge first."
        )
    for name in _STOCK_FILES:
        if not (stock_dir / name).is_file():
            raise FileNotFoundError(f"stock checkpoint file missing: {stock_dir / name}")

    print(f"[export-vllm] loading {weights_path}")
    state = torch.load(weights_path, map_location="cpu", mmap=True, weights_only=True)

    renamed = {}
    for key, tensor in state.items():
        renamed[_rename_key(key)] = tensor.contiguous()

    stock_index = stock_dir / "model.safetensors.index.json"
    expected = None
    if stock_index.is_file():
        import json

        expected = set(json.loads(stock_index.read_text())["weight_map"])
        got = set(renamed)
        if expected != got:
            missing = sorted(expected - got)[:5]
            extra = sorted(got - expected)[:5]
            raise ValueError(
                f"converted keys do not match stock layout (missing={missing}, extra={extra})"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_weights = output_dir / "model.safetensors"
    print(f"[export-vllm] saving {out_weights}")
    save_file(renamed, str(out_weights))

    for name in _STOCK_FILES:
        shutil.copy2(stock_dir / name, output_dir / name)
    # Remove a stale shard index if a previous export copied one.
    (output_dir / "model.safetensors.index.json").unlink(missing_ok=True)

    print(f"[export-vllm] done: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=".env", help="Path to .env")
    parser.add_argument(
        "--output",
        default=None,
        help="Output dir (default: {data_root}/training/vllm)",
    )
    args = parser.parse_args()

    project = load_config(args.config)
    ws = project.workspace()
    merged_dir = resolve_merged_checkpoint(project, ws)
    stock_dir = resolve_base_checkpoint(project, ws)
    output_dir = Path(args.output) if args.output else ws.training_dir / "fish" / "vllm"

    try:
        convert(merged_dir, stock_dir, output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    try:
        rel = output_dir.relative_to(ws.data_root)
    except ValueError:
        rel = output_dir
    print("[export-vllm] to serve it, set in .env:")
    print(f"[export-vllm]   FISH_SPEECH_MODEL={rel}")
    print("[export-vllm] then: ./run.sh vllm restart")


if __name__ == "__main__":
    main()
