#!/usr/bin/env python3
"""Compare stock Fish Speech weights against merged / vLLM-exported fine-tune."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
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


def module_group(key: str) -> str:
    if key.startswith("text_model."):
        return "text_model (slow AR)"
    if key.startswith("audio_decoder.layers."):
        return "audio_decoder.layers (fast AR)"
    if key.startswith("audio_decoder.embeddings."):
        return "audio_decoder.embeddings"
    if key.startswith("audio_decoder.output."):
        return "audio_decoder.output"
    if key.startswith("audio_decoder.norm."):
        return "audio_decoder.norm"
    if key.startswith("audio_decoder.codebook_embeddings."):
        return "audio_decoder.codebook"
    return "other"


@dataclass
class Accum:
    l2_sq: float = 0.0
    base_norm_sq: float = 0.0
    ft_norm_sq: float = 0.0
    max_abs: float = 0.0
    params: int = 0
    changed: int = 0

    def add(self, base: torch.Tensor, ft: torch.Tensor, *, eps: float) -> None:
        delta = ft.float() - base.float()
        l2 = float(delta.norm())
        base_norm = float(base.float().norm())
        rel = l2 / (base_norm + 1e-12)
        self.l2_sq += l2 * l2
        self.base_norm_sq += base_norm * base_norm
        self.ft_norm_sq += float(ft.float().norm()) ** 2
        self.max_abs = max(self.max_abs, float(delta.abs().max()))
        self.params += base.numel()
        if rel > eps:
            self.changed += base.numel()

    @property
    def rel_l2(self) -> float:
        return (self.l2_sq**0.5) / (self.base_norm_sq**0.5 + 1e-12)


def load_stock_index(stock_dir: Path) -> dict[str, str]:
    index_path = stock_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        single = stock_dir / "model.safetensors"
        if single.is_file():
            from safetensors.torch import load_file

            return {k: str(single) for k in load_file(str(single)).keys()}
        raise FileNotFoundError(f"no stock weights in {stock_dir}")
    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
    return {k: str(stock_dir / v) for k, v in weight_map.items()}


def load_ft_from_merged(merged_path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(merged_path, map_location="cpu", mmap=True, weights_only=True)
    return {rename_key(k): v for k, v in state.items()}


def load_ft_from_vllm(vllm_path: Path) -> safe_open:
    return safe_open(str(vllm_path), framework="pt", device="cpu")


def get_ft_tensor(ft_source: object, key: str) -> torch.Tensor:
    if isinstance(ft_source, safe_open):
        return ft_source.get_tensor(key)
    assert isinstance(ft_source, dict)
    return ft_source[key]


def compare(
    stock_dir: Path,
    *,
    merged: Path | None,
    vllm: Path | None,
    eps: float,
    top_n: int,
) -> int:
    key_to_shard = load_stock_index(stock_dir)
    stock_keys = set(key_to_shard)

    if vllm is not None:
        ft_handle = load_ft_from_vllm(vllm)
        ft_keys = set(ft_handle.keys())
        ft_source: object = ft_handle
    elif merged is not None:
        ft_source = load_ft_from_merged(merged)
        ft_keys = set(ft_source)
    else:
        raise ValueError("pass --merged or --vllm")

    stock_keys = set(key_to_shard)

    common = sorted(stock_keys & ft_keys)
    missing_stock = sorted(ft_keys - stock_keys)
    missing_ft = sorted(stock_keys - ft_keys)

    print(f"stock tensors: {len(stock_keys)}")
    print(f"fine-tuned tensors: {len(ft_keys)}")
    print(f"common: {len(common)}")
    if missing_stock:
        print(f"only in fine-tuned: {len(missing_stock)}")
    if missing_ft:
        print(f"only in stock: {len(missing_ft)}")

    shard_handles: dict[str, safe_open] = {}
    groups: dict[str, Accum] = defaultdict(Accum)
    total = Accum()
    top: list[tuple[float, float, str]] = []

    def get_stock(key: str) -> torch.Tensor:
        shard = key_to_shard[key]
        if shard not in shard_handles:
            shard_handles[shard] = safe_open(shard, framework="pt", device="cpu")
        return shard_handles[shard].get_tensor(key)

    for key in common:
        base = get_stock(key)
        ft = get_ft_tensor(ft_source, key)
        if base.shape != ft.shape:
            print(f"[warn] shape mismatch {key}: {tuple(base.shape)} vs {tuple(ft.shape)}")
            continue
        if not torch.is_floating_point(base):
            continue

        group = module_group(key)
        for acc in (total, groups[group]):
            acc.add(base, ft, eps=eps)

        delta = ft.float() - base.float()
        l2 = float(delta.norm())
        base_norm = float(base.float().norm())
        rel = l2 / (base_norm + 1e-12)
        if rel > eps:
            top.append((rel, l2, key))

    for handle in shard_handles.values():
        handle.__exit__(None, None, None)
    if isinstance(ft_source, safe_open):
        ft_source.__exit__(None, None, None)

    top.sort(reverse=True)

    print("\n=== global (all float tensors) ===")
    print(f"parameters compared: {total.params:,}")
    print(f"relative L2:         {total.rel_l2:.6e}  ({total.rel_l2 * 100:.4f}%)")
    print(f"max |delta|:         {total.max_abs:.6e}")
    print(
        f"params with rel > {eps:g}: {total.changed:,} "
        f"({100 * total.changed / max(total.params, 1):.2f}%)"
    )

    print("\n=== by module group ===")
    rows = sorted(groups.items(), key=lambda item: item[1].rel_l2, reverse=True)
    for name, acc in rows:
        pct_changed = 100 * acc.changed / max(acc.params, 1)
        print(
            f"{name:40} rel_L2={acc.rel_l2:10.6e} ({acc.rel_l2 * 100:7.4f}%) "
            f"max|d|={acc.max_abs:.3e} changed={pct_changed:5.2f}%"
        )

    print(f"\n=== top {top_n} changed tensors (rel L2) ===")
    for rel, l2, key in top[:top_n]:
        print(f"  rel={rel:.4e}  l2={l2:.4e}  {key}")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stock", type=Path, required=True, help="Stock s2-pro checkpoint dir")
    p.add_argument("--merged", type=Path, help="Merged model.pth directory")
    p.add_argument("--vllm", type=Path, help="vLLM export model.safetensors file")
    p.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Relative L2 threshold to count a tensor as changed",
    )
    p.add_argument("--top", type=int, default=15, help="Number of top tensors to print")
    args = p.parse_args()

    merged = args.merged
    if merged is not None:
        merged = merged / "model.pth" if merged.is_dir() else merged

    return compare(
        args.stock,
        merged=merged,
        vllm=args.vllm,
        eps=args.eps,
        top_n=args.top,
    )


if __name__ == "__main__":
    sys.exit(main())
