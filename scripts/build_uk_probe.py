#!/usr/bin/env python3
"""Freeze the Ukrainian probe set used to grade checkpoints.

Probe lines come from ``metadata_eval.csv`` -- the held-out split, with no file
in common with ``metadata_train.csv`` and nothing exported into
``training/raw``. Two properties make that pool worth more than hand-written
lines: the sentences are real Ukrainian rather than invented, and every line
ships with the human recording of itself, which gives the stress estimator a
ceiling measured on the same words instead of an assumed one.

The truth marks are frozen into the file on purpose. Re-deriving them at scoring
time would let a lexicon edit move the target between runs, and a probe set that
drifts cannot compare two checkpoints.

Reference clips for the clone prompt are picked per speaker from
``training/raw``: prompts are inputs, so their being in the training data costs
nothing, and a stable prompt per speaker keeps ECAPA comparable across runs.
Which clip becomes that prompt matters more than it looks -- see
``pick_reference``.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
import wave
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import soundfile as sf

from fish_studio.config import StressConfig
from fish_studio.project_context import try_load_project, workspace_or_default
from fish_studio.server.voiceprint import VoiceEncoder, cosine_similarity
from fish_studio.stress import COMBINING_ACUTE, stressify, strip_stress_marks

_VOWELS = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ'’ʼ́-]+", re.UNICODE)
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"\d")

HEADER = ("id", "speaker", "text", "truth", "human_wav", "ref_wav", "ref_text")


def clip_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / handle.getframerate()
    except Exception:
        return 0.0


def marked_word_count(marked: str) -> int:
    """Multi-vowel words the dictionary was sure about -- the scorable ones."""
    count = 0
    for match in _WORD_RE.finditer(marked):
        word = match.group(0)
        if COMBINING_ACUTE not in word:
            continue
        if sum(1 for ch in strip_stress_marks(word) if ch in _VOWELS) >= 2:
            count += 1
    return count


def read_eval_rows(dataset_dir: Path) -> list[tuple[str, str, str]]:
    path = dataset_dir / "metadata_eval.csv"
    if not path.is_file():
        raise SystemExit(f"eval metadata not found: {path}")
    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"eval metadata is empty: {path}")
        for row in reader:
            if len(row) < 3:
                continue
            rows.append((row[0].strip(), row[1].strip(), row[2].strip()))
    return rows


def _candidate_clips(
    raw_dir: Path, speaker: str, rng: random.Random, wanted: int
) -> list[tuple[Path, str]]:
    folder = raw_dir / speaker
    if not folder.is_dir():
        return []
    labs = sorted(folder.glob("*.lab"))
    if not labs:
        return []
    rng.shuffle(labs)
    found: list[tuple[Path, str]] = []
    for lab in labs[:400]:
        wav = lab.with_suffix(".wav")
        if not wav.is_file():
            continue
        if not 4.0 <= clip_seconds(wav) <= 12.0:
            continue
        text = lab.read_text(encoding="utf-8").strip()
        if not text or _LATIN.search(text):
            continue
        found.append((wav, text))
        if len(found) >= wanted:
            break
    return found


def pick_reference(
    raw_dir: Path,
    speaker: str,
    rng: random.Random,
    encoder: VoiceEncoder | None = None,
    *,
    candidates: int = 12,
    floor: float = 0.4,
) -> tuple[Path, str, float] | None:
    """The clip that best represents the speaker, not the first one that fits.

    Taking the first acceptable clip once froze a four-second Russian fragment
    as the reference for ``rostyslav_s2``: its own two halves scored 0.202
    against each other, so every clone measured against it was measuring the
    clip. Here each candidate is scored by its median cosine to the speaker's
    other candidates, and the most typical one wins. A speaker whose best clip
    still sits under ``floor`` is reported, because that is a fact about the
    corpus rather than about any checkpoint.
    """
    found = _candidate_clips(raw_dir, speaker, rng, candidates)
    if not found:
        return None
    if encoder is None or not encoder.available or len(found) < 3:
        wav, text = found[0]
        return wav, text, float("nan")

    embeddings: list[tuple[Path, str, list[float]]] = []
    for wav, text in found:
        samples, rate = sf.read(str(wav), dtype="float32", always_2d=False)
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        embedding = encoder.embed(np.asarray(samples, dtype=np.float32), int(rate))
        if embedding:
            embeddings.append((wav, text, embedding))
    if len(embeddings) < 3:
        wav, text = found[0]
        return wav, text, float("nan")

    best: tuple[Path, str, float] | None = None
    for index, (wav, text, embedding) in enumerate(embeddings):
        others = [
            cosine_similarity(embedding, other)
            for position, (_, _, other) in enumerate(embeddings)
            if position != index
        ]
        score = float(np.median(others))
        if best is None or score > best[2]:
            best = (wav, text, score)
    if best is not None and best[2] < floor:
        print(f"[warn] {speaker}: best reference only scores {best[2]:.3f} against its own clips")
    return best


def build(
    *,
    dataset_dir: Path,
    raw_dir: Path,
    stress: StressConfig,
    lines_per_speaker: int,
    min_marked: int,
    min_sec: float,
    max_sec: float,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    truth_settings = replace(stress, enabled=True, acoustic_fallback=False)
    encoder = VoiceEncoder()
    encoder.warmup()
    if not encoder.available:
        print("[warn] ECAPA unavailable: references fall back to the first clip that fits")

    by_speaker: dict[str, list[tuple[str, str]]] = {}
    for audio_file, text, speaker in read_eval_rows(dataset_dir):
        by_speaker.setdefault(speaker, []).append((audio_file, text))

    probes: list[dict[str, str]] = []
    for speaker in sorted(by_speaker):
        reference = pick_reference(raw_dir, speaker, rng, encoder)
        if reference is None:
            print(f"[skip] {speaker}: no usable reference clip in {raw_dir}")
            continue
        ref_wav, ref_text, ref_score = reference

        candidates = by_speaker[speaker]
        rng.shuffle(candidates)
        taken = 0
        for audio_file, raw_text in candidates:
            if taken >= lines_per_speaker:
                break
            # Exported text is already marked, and some of those marks come from
            # the energy estimator. Strip first, then re-derive from dictionary.
            plain = strip_stress_marks(raw_text).strip()
            if not plain or _LATIN.search(plain) or _DIGIT.search(plain):
                continue
            wav = dataset_dir / audio_file
            if not wav.is_file():
                continue
            seconds = clip_seconds(wav)
            if not min_sec <= seconds <= max_sec:
                continue
            truth = stressify(plain, truth_settings)
            if marked_word_count(truth) < min_marked:
                continue
            probes.append(
                {
                    "id": f"{speaker}_{Path(audio_file).stem}",
                    "speaker": speaker,
                    "text": plain,
                    "truth": truth,
                    "human_wav": str(wav),
                    "ref_wav": str(ref_wav),
                    "ref_text": ref_text,
                }
            )
            taken += 1
        typical = "n/a" if math.isnan(ref_score) else f"{ref_score:.3f}"
        print(f"[ok] {speaker}: {taken} lines, reference typicality {typical}")
    return probes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=".env")
    parser.add_argument("--dataset-id", default="combined")
    parser.add_argument("--out", type=Path, default=Path("configs/uk_probe.tsv"))
    parser.add_argument("--lines-per-speaker", type=int, default=10)
    parser.add_argument("--min-marked", type=int, default=3, help="scorable words per line")
    parser.add_argument("--min-sec", type=float, default=2.0)
    parser.add_argument("--max-sec", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project = try_load_project(args.config)
    stress = project.stress if project is not None else StressConfig()
    workspace = workspace_or_default(args.config)
    root = workspace.data_root

    probes = build(
        dataset_dir=workspace.datasets_root / args.dataset_id,
        raw_dir=workspace.training_dir / "raw",
        stress=stress,
        lines_per_speaker=args.lines_per_speaker,
        min_marked=args.min_marked,
        min_sec=args.min_sec,
        max_sec=args.max_sec,
        seed=args.seed,
    )
    if not probes:
        raise SystemExit("no probe lines selected")

    # Paths are written relative to data_root so the set resolves on any host.
    for probe in probes:
        for key in ("human_wav", "ref_wav"):
            probe[key] = str(Path(probe[key]).resolve().relative_to(root.resolve())).replace(
                "\\", "/"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HEADER), delimiter="\t")
        writer.writeheader()
        writer.writerows(probes)

    scorable = sum(marked_word_count(probe["truth"]) for probe in probes)
    print(f"wrote {args.out}: {len(probes)} lines, {scorable} scorable words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
