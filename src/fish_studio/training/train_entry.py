#!/usr/bin/env python3
"""Entry point that patches Dual-AR LoRA before delegating to fish_speech.train."""

from __future__ import annotations

import sys

from fish_studio.training.lora_patch import apply_dual_ar_lora_patch
from fish_studio.training.upstream import ensure_fish_speech_root


def main() -> None:
    # Patch Dual-AR LoRA *before* importing fish_speech.train so Hydra picks up our setup_lora.
    apply_dual_ar_lora_patch()
    ensure_fish_speech_root()
    sys.argv = ["fish_speech.train", *sys.argv[1:]]
    from fish_speech.train import main as fish_train_main

    fish_train_main()


if __name__ == "__main__":
    main()
