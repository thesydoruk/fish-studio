#!/usr/bin/env python3
"""When a clone prompt scores near zero, what voice did the model actually use?

A cosine of 0.08 against a 0.30 gate says the take is not the prompt's speaker.
It does not say what went wrong, and the repairs differ: if the take matches a
training speaker, the model fell back to a voice it knows and the fix is data;
if it matches nothing, the take is degenerate and the fix is decoding; if the
prompt clip cannot even match itself, the clip is the problem and neither.

So every take is scored against three things: its own prompt, every other
prompt, and one reference clip per training speaker. The prompts are also
checked against themselves -- first half versus second half -- because a clip
holding music, noise or two speakers gives an unstable embedding, and that
would explain the low score without any model fault.

Takes are the WAVs an ``uk-eval`` run wrote with ``--out-dir``, named
``<voice-id>__<probe-id>.wav``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import soundfile as sf

from fish_studio.project_context import workspace_or_default
from fish_studio.server.voiceprint import VoiceEncoder, cosine_similarity


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    return np.asarray(samples, dtype=np.float32), int(rate)


def self_consistency(encoder: VoiceEncoder, path: Path) -> float | None:
    """Cosine between the clip's two halves. Low means the clip is not one voice."""
    samples, rate = read_audio(path)
    middle = samples.size // 2
    if middle < rate // 2:
        return None
    left = encoder.embed(samples[:middle], rate)
    right = encoder.embed(samples[middle:], rate)
    if not left or not right:
        return None
    return cosine_similarity(left, right)


def speaker_references(probe_path: Path, data_root: Path) -> dict[str, Path]:
    """One reference clip per training speaker, as the probe set froze them."""
    references: dict[str, Path] = {}
    with probe_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            references.setdefault(row["speaker"], data_root / row["ref_wav"])
    return references


def prompt_clips(index_path: Path, data_root: Path) -> dict[str, Path]:
    clips: dict[str, Path] = {}
    with index_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            path = data_root / row["wav"]
            if path.is_file():
                clips[row["id"]] = path
    return clips


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=".env")
    parser.add_argument("--takes", type=Path, required=True, help="uk-eval --out-dir")
    parser.add_argument("--probe", type=Path, default=Path("configs/uk_probe.tsv"))
    parser.add_argument("--voices", type=Path, default=None)
    parser.add_argument("--top", type=int, default=3, help="nearest other voices to print")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = workspace_or_default(args.config).data_root
    index_path = args.voices or (data_root / "eval" / "voices" / "index.tsv")

    encoder = VoiceEncoder()
    encoder.warmup()
    if not encoder.available:
        raise SystemExit("ECAPA encoder unavailable; nothing to compare")

    prompts = prompt_clips(index_path, data_root)
    speakers = speaker_references(args.probe, data_root)
    if not prompts:
        raise SystemExit(f"no usable prompt clips in {index_path}")

    print("prompt clips -- half against half (low = the clip is not one steady voice)")
    prompt_embeddings: dict[str, list[float]] = {}
    for voice_id, path in prompts.items():
        samples, rate = read_audio(path)
        embedding = encoder.embed(samples, rate)
        if embedding:
            prompt_embeddings[voice_id] = embedding
        halves = self_consistency(encoder, path)
        shown = f"{halves:.3f}" if halves is not None else "   -"
        print(f"  {voice_id:<10} self={shown}  {path.name}")

    speaker_embeddings: dict[str, list[float]] = {}
    for speaker, path in speakers.items():
        if not path.is_file():
            continue
        samples, rate = read_audio(path)
        embedding = encoder.embed(samples, rate)
        if embedding:
            speaker_embeddings[speaker] = embedding

    takes: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(args.takes.glob("*__*.wav")):
        takes[path.name.split("__", 1)[0]].append(path)
    if not takes:
        raise SystemExit(f"no <voice>__<probe>.wav takes under {args.takes}")

    print("\ntakes -- who does the model sound like?")
    for voice_id, paths in takes.items():
        own = prompt_embeddings.get(voice_id)
        own_scores: list[float] = []
        speaker_scores: dict[str, list[float]] = defaultdict(list)
        other_scores: dict[str, list[float]] = defaultdict(list)
        for path in paths:
            samples, rate = read_audio(path)
            embedding = encoder.embed(samples, rate)
            if not embedding:
                continue
            if own:
                own_scores.append(cosine_similarity(embedding, own))
            for speaker, reference in speaker_embeddings.items():
                speaker_scores[speaker].append(cosine_similarity(embedding, reference))
            for other, reference in prompt_embeddings.items():
                if other != voice_id:
                    other_scores[other].append(cosine_similarity(embedding, reference))

        mean_own = float(np.mean(own_scores)) if own_scores else float("nan")
        ranked = sorted(
            ((name, float(np.mean(values))) for name, values in speaker_scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        nearest = "  ".join(f"{name}:{score:.3f}" for name, score in ranked[: args.top])
        ranked_other = sorted(
            ((name, float(np.mean(values))) for name, values in other_scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        nearest_other = "  ".join(f"{name}:{score:.3f}" for name, score in ranked_other[:1])
        print(
            f"  {voice_id:<10} own={mean_own:>6.3f} n={len(paths):<3} "
            f"| nearest training: {nearest} | nearest other prompt: {nearest_other}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
