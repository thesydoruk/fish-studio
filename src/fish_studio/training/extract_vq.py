#!/usr/bin/env python3
"""Extract Fish Speech semantic tokens (VQ) for a prepared raw dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fish_studio.project_context import try_load_project
from fish_studio.training.layout import ensure_training_dirs, resolve_base_checkpoint
from fish_studio.training.upstream import run_fish_command


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", default=".env")
    pre_args, _ = pre.parse_known_args()
    project = try_load_project(pre_args.config)

    defaults_input: Path | None = None
    defaults_codec: Path | None = None
    defaults_config_name = "modded_dac_vq"
    defaults_batch_size = 16
    defaults_workers = 1
    if project is not None:
        ws = project.workspace()
        ft = project.training
        defaults_input = ensure_training_dirs(ws)["raw"]
        base = resolve_base_checkpoint(project, ws)
        defaults_codec = base / "codec.pth"
        defaults_config_name = project.fish_speech.decoder_config_name
        defaults_batch_size = ft.vq_batch_size
        defaults_workers = ft.vq_num_workers

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=defaults_input,
        help="Fish raw dataset root with speaker subfolders",
    )
    parser.add_argument("--codec", type=Path, default=defaults_codec)
    parser.add_argument("--config-name", default=defaults_config_name)
    parser.add_argument("--batch-size", type=int, default=defaults_batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults_workers)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_dir is None or not args.input_dir.is_dir():
        print("[error] input dir not found; run export step first", file=sys.stderr)
        sys.exit(1)
    if args.codec is None or not args.codec.is_file():
        print(f"[error] codec checkpoint not found: {args.codec}", file=sys.stderr)
        sys.exit(1)

    run_fish_command(
        "tools/vqgan/extract_vq.py",
        [
            str(args.input_dir),
            "--num-workers",
            str(args.num_workers),
            "--batch-size",
            str(args.batch_size),
            "--config-name",
            args.config_name,
            "--checkpoint-path",
            str(args.codec),
        ],
    )
    print(f"[done] semantic tokens written next to wav files under {args.input_dir}")


if __name__ == "__main__":
    main()
