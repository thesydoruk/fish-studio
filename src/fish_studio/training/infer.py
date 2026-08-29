#!/usr/bin/env python3
"""Test synthesis with a fine-tuned Fish Speech s2-pro checkpoint."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from fish_studio.config import load_config
from fish_studio.runtime.fish_config import FishSpeechRuntimeSettings, resolve_speaker_reference
from fish_studio.training.native_engine import NativeFishSpeechEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default=".env")
    p.add_argument("--text", default=None, help="Text to synthesize")
    p.add_argument("--speaker-wav", default=None, help="Reference speaker WAV (~6s)")
    p.add_argument("--speaker-text", default=None, help="Reference transcript (recommended)")
    p.add_argument("--llama-checkpoint", default=None, help="Override merged LLAMA checkpoint")
    p.add_argument("--out", default="fish_synthesized.wav")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project = load_config(args.config)
    settings = FishSpeechRuntimeSettings.from_project(project, prefer_finetuned=True)
    if args.llama_checkpoint:
        settings = replace(settings, llama_checkpoint=Path(args.llama_checkpoint))
    if args.cpu:
        settings = replace(settings, device="cpu")

    text = (args.text or project.training.test_text).strip()
    if not text:
        print("[error] --text is required", file=sys.stderr)
        sys.exit(1)

    speaker_wav = Path(args.speaker_wav) if args.speaker_wav else resolve_speaker_reference(project)
    if not speaker_wav.is_file():
        print(f"[error] speaker wav not found: {speaker_wav}", file=sys.stderr)
        sys.exit(1)

    speaker_text = (args.speaker_text or settings.default_reference_text).strip()
    out = Path(args.out)

    engine = NativeFishSpeechEngine(settings)
    try:
        engine.load()
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] device={settings.device} llama={settings.llama_checkpoint}")
    result = engine.synthesize(
        text=text,
        language=settings.default_language,
        speaker_path=speaker_wav,
        speaker_text=speaker_text or None,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(result.wav_bytes)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
