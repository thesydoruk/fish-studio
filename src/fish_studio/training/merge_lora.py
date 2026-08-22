#!/usr/bin/env python3
"""Merge Fish Speech LoRA weights into a standalone s2-pro LLAMA checkpoint."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from fish_studio.config import TrainingConfig
from fish_studio.project_context import try_load_project
from fish_studio.training.layout import (
    latest_lora_checkpoint,
    project_run_dir,
    resolve_base_checkpoint,
    resolve_merged_checkpoint,
)
from fish_studio.training.upstream import fish_site_packages, run_fish_merge

_TOKENIZER_FILES = (
    "tokenizer.tiktoken",
    "tokenizer.model",
    "tokenizer.json",
    "special_tokens_map.json",
)


def write_lora_config(training: TrainingConfig, config_name: str) -> str:
    """Mirror the training LoRA setup into a hydra config merge_lora can compose.

    Upstream composes ``fish_speech/configs/lora/<name>.yaml`` and then loads the
    checkpoint with ``strict=True``, so the target modules must match the ones
    used during training or the merge fails on missing LoRA keys.
    """
    name = f"fish-studio-{config_name}"
    configs_dir = fish_site_packages() / "fish_speech" / "configs" / "lora"
    configs_dir.mkdir(parents=True, exist_ok=True)

    targets = "\n".join(f"  - {module}" for module in training.lora_target_modules)
    (configs_dir / f"{name}.yaml").write_text(
        "_target_: fish_studio.training.lora_patch.LoraConfig\n"
        f"r: {training.lora_r}\n"
        f"lora_alpha: {training.lora_alpha}\n"
        f"lora_dropout: {training.lora_dropout}\n"
        f"target_modules:\n{targets}\n",
        encoding="utf-8",
    )
    return name


def _copy_tokenizer_files(base_checkpoint: Path, output_dir: Path) -> None:
    for name in _TOKENIZER_FILES:
        src = base_checkpoint / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", default=".env")
    pre_args, _ = pre.parse_known_args()
    project = try_load_project(pre_args.config)

    defaults: dict[str, object] = {
        "project_name": "fish-uk",
        "lora_config": "r_8_alpha_16",
        "lora_weight": "",
        "base_checkpoint": None,
        "output_dir": None,
    }
    if project is not None:
        ws = project.workspace()
        ft = project.training
        defaults.update(
            {
                "project_name": ft.project_name,
                "lora_config": ft.lora_config,
                "base_checkpoint": resolve_base_checkpoint(project, ws),
                "output_dir": resolve_merged_checkpoint(project, ws),
            }
        )
        latest = latest_lora_checkpoint(project_run_dir(ws, ft.project_name) / "checkpoints")
        if latest is not None:
            defaults["lora_weight"] = str(latest)

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--project-name", default=defaults["project_name"])
    parser.add_argument("--lora-config", default=defaults["lora_config"])
    parser.add_argument("--lora-weight", default=defaults["lora_weight"])
    parser.add_argument("--base-checkpoint", type=Path, default=defaults["base_checkpoint"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = try_load_project(args.config)
    if project is None:
        print("[error] .env not found", file=sys.stderr)
        sys.exit(1)

    ws = project.workspace()
    if not args.lora_weight:
        latest = latest_lora_checkpoint(project_run_dir(ws, args.project_name) / "checkpoints")
        if latest is None:
            print("[error] no LoRA checkpoint found; run train step first", file=sys.stderr)
            sys.exit(1)
        args.lora_weight = str(latest)

    # Upstream merge runs from the fish-speech package dir, so a relative path
    # supplied on the command line would not resolve there.
    lora_path = Path(args.lora_weight).resolve()
    if not lora_path.is_file():
        print(f"[error] LoRA checkpoint not found: {lora_path}", file=sys.stderr)
        sys.exit(1)

    base = (args.base_checkpoint or resolve_base_checkpoint(project, ws)).resolve()
    if not base.is_dir():
        print(f"[error] base checkpoint not found: {base}", file=sys.stderr)
        sys.exit(1)

    output = (args.output_dir or resolve_merged_checkpoint(project, ws)).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    lora_config = write_lora_config(project.training, args.lora_config)
    run_fish_merge(
        [
            "--lora-config",
            lora_config,
            "--base-weight",
            str(base),
            "--lora-weight",
            str(lora_path),
            "--output",
            str(output),
        ],
    )
    _copy_tokenizer_files(base, output)
    print(f"[done] merged checkpoint: {output}")


if __name__ == "__main__":
    main()
