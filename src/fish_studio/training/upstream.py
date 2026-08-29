"""Helpers for invoking the installed fish-speech package."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def fish_site_packages() -> Path:
    spec = importlib.util.find_spec("fish_speech")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "fish_speech is not installed. Run: ./run.sh install server  (or install all)"
        )
    return Path(spec.submodule_search_locations[0]).resolve().parent


def ensure_fish_speech_root() -> Path:
    """fish-speech uses pyrootutils with a `.project-root` marker in site-packages."""
    root = fish_site_packages()
    (root / ".project-root").touch()
    return root


def fish_tool_script(relative: str) -> Path:
    path = fish_site_packages() / relative
    if not path.is_file():
        raise FileNotFoundError(f"fish-speech tool not found: {path}")
    return path


def run_fish_command(
    script_relative: str,
    args: list[str],
    *,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    ensure_fish_speech_root()
    script = fish_tool_script(script_relative)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, str(script), *args]
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd or fish_site_packages()), env=env, check=True)


def run_fish_merge(script_args: list[str]) -> None:
    """Merge LoRA after applying Dual-AR patches in the same interpreter as fish-speech."""
    ensure_fish_speech_root()
    env = os.environ.copy()
    env["FISH_MERGE_ARGS"] = json.dumps(["merge_lora.py", *script_args])
    bootstrap = (
        "import json, os, sys; "
        "from fish_studio.training.lora_patch import apply_dual_ar_lora_patch; "
        "apply_dual_ar_lora_patch(); "
        "sys.argv = json.loads(os.environ['FISH_MERGE_ARGS']); "
        "from tools.llama.merge_lora import merge; "
        "merge()"
    )
    cmd = [sys.executable, "-c", bootstrap]
    print(f"[cmd] python -c <fish merge bootstrap> {' '.join(script_args)}", flush=True)
    subprocess.run(cmd, cwd=str(fish_site_packages()), env=env, check=True)


def run_fish_train(hydra_overrides: list[str]) -> None:
    ensure_fish_speech_root()
    env = os.environ.copy()
    env["FISH_TRAIN_OVERRIDES"] = json.dumps(hydra_overrides)
    # LoRA on the slow layers leaves a few GB stranded in reserved-but-unused
    # blocks, which is enough to fail an allocation on a 32 GB card.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    bootstrap = (
        "import json, os, runpy, sys; "
        "from fish_studio.training.lora_patch import apply_dual_ar_lora_patch; "
        "apply_dual_ar_lora_patch(); "
        "sys.argv = ['fish_speech.train', *json.loads(os.environ['FISH_TRAIN_OVERRIDES'])]; "
        "runpy.run_module('fish_speech.train', run_name='__main__')"
    )
    cmd = [sys.executable, "-c", bootstrap]
    print(f"[cmd] python -c <fish train bootstrap> {' '.join(hydra_overrides)}", flush=True)
    subprocess.run(cmd, cwd=str(fish_site_packages()), env=env, check=True)
