#!/usr/bin/env python3
"""Is each speaker folder actually one speaker?

Clips are grouped by diarization and then merged across videos on an embedding
cosine (``SPEAKER_CLUSTER_THRESHOLD``). When that merge is too loose the folder
holds several people, and the model is then trained on examples where the same
speaker id sounds different every time -- which teaches it that the prompt does
not determine the voice. That costs in-context cloning far more directly than
any merge-scale setting.

Coherence here is the mean cosine between random clips of the same folder. A
tight speaker sits well above the number two different speakers would score, so
the between-speaker figure is printed alongside as the floor to read against.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import soundfile as sf

from fish_studio.project_context import workspace_or_default
from fish_studio.server.voiceprint import VoiceEncoder, cosine_similarity


def embed_clip(encoder: VoiceEncoder, path: Path) -> list[float] | None:
    try:
        samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return None
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    return encoder.embed(np.asarray(samples, dtype=np.float32), int(rate))


def sample_embeddings(
    encoder: VoiceEncoder, folder: Path, count: int, rng: random.Random
) -> list[list[float]]:
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        return []
    rng.shuffle(wavs)
    embeddings: list[list[float]] = []
    for wav in wavs:
        embedding = embed_clip(encoder, wav)
        if embedding:
            embeddings.append(embedding)
        if len(embeddings) >= count:
            break
    return embeddings


def mean_pairwise(embeddings: list[list[float]]) -> float:
    pairs = [cosine_similarity(a, b) for a, b in itertools.combinations(embeddings, 2)]
    return float(np.mean(pairs)) if pairs else float("nan")


def best_two_way_split(embeddings: list[list[float]], rounds: int = 12) -> tuple[float, int]:
    """Split the folder in two and report how far apart the halves sit.

    A folder holding one person with a variable delivery still splits, but the
    two halves keep resembling each other. A folder holding two people splits
    into halves that score like strangers. The number returned is the mean
    cosine between the halves, plus the size of the smaller one.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.shape[0] < 6:
        return float("nan"), 0
    centres = matrix[[0, int(np.argmin(matrix @ matrix[0]))]]
    labels = np.zeros(matrix.shape[0], dtype=int)
    for _ in range(rounds):
        labels = np.argmax(matrix @ centres.T, axis=1)
        for group in (0, 1):
            members = matrix[labels == group]
            if members.size:
                centres[group] = members.mean(axis=0)
    left = matrix[labels == 0]
    right = matrix[labels == 1]
    if left.shape[0] < 2 or right.shape[0] < 2:
        return float("nan"), 0
    across = float(np.mean(left @ right.T))
    return across, int(min(left.shape[0], right.shape[0]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=".env")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--clips", type=int, default=20, help="clips sampled per speaker")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--floor",
        type=float,
        default=0.5,
        help="flag a folder whose clips agree less than this",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_dir = args.raw_dir or (workspace_or_default(args.config).training_dir / "raw")
    if not raw_dir.is_dir():
        raise SystemExit(f"raw dir not found: {raw_dir}")

    encoder = VoiceEncoder()
    encoder.warmup()
    if not encoder.available:
        raise SystemExit("ECAPA encoder unavailable; nothing to compare")

    rng = random.Random(args.seed)
    folders = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    sampled: dict[str, list[list[float]]] = {}
    rows: list[tuple[str, int, float, float, int]] = []
    for folder in folders:
        embeddings = sample_embeddings(encoder, folder, args.clips, rng)
        if len(embeddings) < 3:
            print(f"[skip] {folder.name}: only {len(embeddings)} clips could be embedded")
            continue
        across, smaller = best_two_way_split(embeddings)
        rows.append((folder.name, len(embeddings), mean_pairwise(embeddings), across, smaller))
        sampled[folder.name] = embeddings

    # Measured clip-to-clip, exactly as coherence is: comparing averaged
    # centroids would flatter the separation and make every folder look tight.
    between: list[float] = []
    for left, right in itertools.combinations(sorted(sampled), 2):
        for a in sampled[left][:6]:
            for b in sampled[right][:6]:
                between.append(cosine_similarity(a, b))
    floor = float(np.mean(between)) if between else float("nan")

    print(f"\n{'speaker':<18}{'clips':>7}{'coherence':>11}{'split':>8}{'smaller':>9}")
    for name, count, score, across, smaller in sorted(rows, key=lambda row: row[2]):
        flag = "  <-- not one voice" if score < args.floor else ""
        gap = "       -" if math.isnan(across) else f"{across:>8.3f}"
        print(f"{name:<18}{count:>7}{score:>11.3f}{gap}{smaller:>9}{flag}")
    print(f"\nclips from different folders: {floor:.3f} -- what two strangers score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
