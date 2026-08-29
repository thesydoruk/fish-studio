"""Blend a merged Fish checkpoint toward stock: W = stock + scale * (ft − stock).

``./run.sh train merge`` applies ``TRAINING_MERGE_SCALE`` (default 0.5) after
the upstream fold. A raw attention+mlp merge at 1.0 kills in-context clone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

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


def rename_key(key: str) -> str:
    for pattern, replacement in _RENAME_RULES:
        if pattern.match(key):
            return pattern.sub(replacement, key)
    raise ValueError(f"unexpected fish-native key: {key}")


def load_stock_index(stock_dir: Path) -> dict[str, str]:
    index_path = stock_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        return {k: str(stock_dir / v) for k, v in weight_map.items()}
    single = stock_dir / "model.safetensors"
    if single.is_file():
        from safetensors.torch import load_file

        return {k: str(single) for k in load_file(str(single)).keys()}
    raise FileNotFoundError(f"no stock weights in {stock_dir}")


def interpolate(
    stock_dir: Path,
    merged_path: Path,
    scale: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= scale <= 1.0:
        raise ValueError(f"scale must be in [0, 1], got {scale}")
    key_to_shard = load_stock_index(stock_dir)
    state = torch.load(merged_path, map_location="cpu", mmap=True, weights_only=True)

    shard_handles: dict[str, safe_open] = {}

    def stock_tensor(hf_key: str) -> torch.Tensor:
        shard = key_to_shard[hf_key]
        if shard not in shard_handles:
            shard_handles[shard] = safe_open(shard, framework="pt", device="cpu")
        return shard_handles[shard].get_tensor(hf_key)

    out: dict[str, torch.Tensor] = {}
    blended = 0
    unchanged = 0
    skipped = 0
    max_rel = 0.0
    for key, ft in state.items():
        if not torch.is_floating_point(ft):
            out[key] = ft
            skipped += 1
            continue
        try:
            hf_key = rename_key(key)
        except ValueError:
            out[key] = ft.contiguous()
            skipped += 1
            continue
        if hf_key not in key_to_shard:
            out[key] = ft.contiguous()
            skipped += 1
            continue
        base = stock_tensor(hf_key)
        if base.shape != ft.shape:
            print(f"[warn] shape mismatch {key}: {tuple(base.shape)} vs {tuple(ft.shape)}")
            out[key] = ft.contiguous()
            skipped += 1
            continue
        delta = ft.float() - base.float()
        rel = float(delta.norm() / (base.float().norm() + 1e-12))
        if rel < 1e-8:
            out[key] = base
            unchanged += 1
            continue
        mixed = (base.float() + scale * delta).to(dtype=ft.dtype)
        out[key] = mixed.contiguous()
        blended += 1
        max_rel = max(max_rel, rel * scale)

    for handle in shard_handles.values():
        handle.__exit__(None, None, None)

    print(f"[interpolate] scale={scale}")
    print(f"[interpolate] blended={blended} unchanged={unchanged} skipped={skipped}")
    print(f"[interpolate] max remaining rel L2 ≈ {max_rel * 100:.4f}%")
    return out


def write_interpolated(state: dict[str, torch.Tensor], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(output)
    print(f"[interpolate] wrote {output}")


def apply_merge_scale(merged_dir: Path, stock_dir: Path, scale: float) -> None:
    """Blend ``merged_dir/model.pth`` toward stock. ``scale=1`` is a no-op."""
    if scale == 1.0:
        print("[merge] scale=1.0 — keeping the full LoRA fold")
        return
    model_path = merged_dir / "model.pth" if merged_dir.is_dir() else merged_dir
    if not model_path.is_file():
        raise FileNotFoundError(f"merged checkpoint not found: {model_path}")
    state = interpolate(stock_dir, model_path, scale)
    write_interpolated(state, model_path)
    print(f"[merge] applied scale={scale} toward {stock_dir}")
