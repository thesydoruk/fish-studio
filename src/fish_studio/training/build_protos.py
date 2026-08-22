#!/usr/bin/env python3
"""Pack Fish Speech semantic tokens into protobuf shards for training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fish_studio.project_context import try_load_project
from fish_studio.training.layout import ensure_training_dirs
from fish_studio.training.upstream import run_fish_command


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", default=".env")
    pre_args, _ = pre.parse_known_args()
    project = try_load_project(pre_args.config)

    defaults_input: Path | None = None
    defaults_output: Path | None = None
    defaults_workers = 8
    defaults_shard_mb = 10
    if project is not None:
        ws = project.workspace()
        layout = ensure_training_dirs(ws)
        defaults_input = layout["raw"]
        defaults_output = layout["protos"]
        defaults_workers = project.training.proto_num_workers
        defaults_shard_mb = project.training.proto_shard_size_mb

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--input-dir", type=Path, default=defaults_input)
    parser.add_argument("--output-dir", type=Path, default=defaults_output)
    parser.add_argument("--num-workers", type=int, default=defaults_workers)
    parser.add_argument("--shard-size-mb", type=int, default=defaults_shard_mb)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_dir is None or not args.input_dir.is_dir():
        print("[error] input dir not found; run export + vq first", file=sys.stderr)
        sys.exit(1)
    if args.output_dir is None:
        print("[error] output dir is required", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_fish_command(
        "tools/llama/build_dataset.py",
        [
            "--input",
            str(args.input_dir),
            "--output",
            str(args.output_dir),
            "--text-extension",
            ".lab",
            "--num-workers",
            str(args.num_workers),
            "--shard-size",
            str(args.shard_size_mb),
        ],
    )
    print(f"[done] protobuf shards written to {args.output_dir}")


if __name__ == "__main__":
    main()
