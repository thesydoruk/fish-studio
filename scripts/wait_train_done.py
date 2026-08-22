#!/usr/bin/env python3
"""Wait until fish-uk LoRA training finishes (process gone) or hits target step."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def latest_step(ckpt_dir: Path) -> int:
    pat = re.compile(r"step_(\d+)(?:-v\d+)?\.ckpt$")
    steps: list[int] = []
    if ckpt_dir.is_dir():
        for path in ckpt_dir.glob("step_*.ckpt"):
            match = pat.search(path.name)
            if match:
                steps.append(int(match.group(1)))
    return max(steps) if steps else 0


def train_alive() -> bool:
    return (
        subprocess.run(["pgrep", "-f", "fish_speech.train"], capture_output=True).returncode
        == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=36000)
    parser.add_argument("--poll-sec", type=int, default=120)
    args = parser.parse_args()

    while True:
        step = latest_step(args.ckpt_dir)
        alive = train_alive()
        print(
            time.strftime("%H:%M:%S"),
            f"latest_step={step}",
            f"alive={int(alive)}",
            flush=True,
        )
        if step >= args.target and not alive:
            print(f"DONE step={step}", flush=True)
            return 0
        if not alive:
            # Process ended before/at target — still report.
            print(f"TRAIN_ENDED step={step}", flush=True)
            return 0 if step >= args.target else 2
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    sys.exit(main())
