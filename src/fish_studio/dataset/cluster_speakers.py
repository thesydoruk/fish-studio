"""Split an imported dataset's speaker groups into voices by ECAPA embedding.

Fish samples the clone prompt from inside a speaker folder, so a folder that
holds several people teaches the model to ignore the prompt, and a corpus of
sixteen voices teaches it those sixteen. A podcast archive has hundreds of
episodes and no speaker labels: each episode becomes a group (see the
``--group-regex`` of ``hf-import``), and this step turns every group into
``{group}_s{k}`` voices by average-linkage clustering of the clip embeddings,
then drops voices too small to sample a prompt from.

Embeddings are cached next to the metadata so a threshold can be re-tried
without re-embedding thirty thousand clips.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

METADATA_FILES = ("metadata_train.csv", "metadata_eval.csv")
EMBEDDING_CACHE = "speaker_embeddings.npz"

# Calibrated on the combined corpus: clips of one speaker in one session sit
# at 0.35-0.8 against their own centroid; strangers at ~0.1.
DEFAULT_THRESHOLD = 0.35


@dataclass
class ClusterSpeakersStats:
    groups: int = 0
    voices_kept: int = 0
    voices_dropped: int = 0
    clips_total: int = 0
    clips_kept: int = 0
    clips_unembeddable: int = 0
    # group -> list of (voice, clips, seconds) including dropped ones
    per_group: dict[str, list[tuple[str, int, float]]] = field(default_factory=dict)


@dataclass
class _Row:
    file: str  # metadata_*.csv name
    audio_file: str
    text: str
    speaker: str
    duration: float = 0.0
    voice: str = ""


def _read_rows(dataset_dir: Path) -> list[_Row]:
    rows: list[_Row] = []
    for name in METADATA_FILES:
        path = dataset_dir / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle, delimiter="|"):
                rows.append(
                    _Row(
                        file=name,
                        audio_file=record["audio_file"],
                        text=record["text"],
                        speaker=record.get("speaker_name") or "speaker",
                    )
                )
    return rows


def _write_rows(dataset_dir: Path, rows: list[_Row]) -> None:
    for name in METADATA_FILES:
        subset = [r for r in rows if r.file == name]
        if not subset and not (dataset_dir / name).is_file():
            continue
        lines = ["audio_file|text|speaker_name"]
        lines.extend(f"{r.audio_file}|{r.text}|{r.voice or r.speaker}" for r in subset)
        (dataset_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_embedding_cache(dataset_dir: Path) -> dict[str, np.ndarray]:
    path = dataset_dir / EMBEDDING_CACHE
    if not path.is_file():
        return {}
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def save_embedding_cache(dataset_dir: Path, cache: dict[str, np.ndarray]) -> None:
    np.savez(dataset_dir / EMBEDDING_CACHE, **cache)


def cluster_embeddings(vectors: np.ndarray, threshold: float) -> list[int]:
    """Average-linkage clustering by centroid cosine; returns a cluster id per row.

    Greedy: the two clusters whose centroids are closest merge first, until no
    pair is above ``threshold``. Fine for the few hundred clips of one group.
    """
    n = vectors.shape[0]
    if n == 0:
        return []
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    members: list[list[int]] = [[i] for i in range(n)]
    centroids = [unit[i].copy() for i in range(n)]
    while len(members) > 1:
        c = np.stack(centroids)
        c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)
        sims = c @ c.T
        np.fill_diagonal(sims, -1.0)
        a, b = np.unravel_index(int(np.argmax(sims)), sims.shape)
        if sims[a, b] <= threshold:
            break
        a, b = (int(a), int(b)) if a < b else (int(b), int(a))
        members[a].extend(members[b])
        centroids[a] = unit[members[a]].mean(axis=0)
        del members[b]
        del centroids[b]
    labels = [0] * n
    for cluster_id, group in enumerate(members):
        for index in group:
            labels[index] = cluster_id
    return labels


def cluster_dataset_speakers(
    dataset_dir: Path,
    *,
    embed: Callable[[Path], np.ndarray | None],
    threshold: float = DEFAULT_THRESHOLD,
    min_clips: int = 20,
    min_seconds: float = 60.0,
    only_groups: set[str] | None = None,
    sample: int = 0,
    dry_run: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
) -> ClusterSpeakersStats:
    """Cluster every speaker group of ``dataset_dir`` into voices; rewrite metadata unless dry."""
    import soundfile as sf

    rows = _read_rows(dataset_dir)
    stats = ClusterSpeakersStats(clips_total=len(rows))
    groups: dict[str, list[_Row]] = {}
    for row in rows:
        groups.setdefault(row.speaker, []).append(row)
    if only_groups:
        groups = {k: v for k, v in groups.items() if k in only_groups}
    stats.groups = len(groups)

    cache = load_embedding_cache(dataset_dir)
    dirty = False
    kept_rows: list[_Row] = []
    for index, (group, members) in enumerate(sorted(groups.items())):
        if progress is not None:
            progress(group, index + 1, len(groups))
        chosen = members[:sample] if sample > 0 else members
        vectors: list[np.ndarray] = []
        embedded: list[_Row] = []
        for row in chosen:
            path = dataset_dir / row.audio_file
            key = row.audio_file
            if key not in cache:
                vector = embed(path)
                if vector is None:
                    stats.clips_unembeddable += 1
                    continue
                cache[key] = np.asarray(vector, dtype=np.float32)
                dirty = True
            try:
                row.duration = float(sf.info(str(path)).duration)
            except Exception:  # noqa: BLE001 - an unreadable clip is dropped below
                row.duration = 0.0
            vectors.append(cache[key])
            embedded.append(row)
        if not embedded:
            continue
        labels = cluster_embeddings(np.stack(vectors), threshold)
        totals: dict[int, list[float]] = {}
        for row, label in zip(embedded, labels, strict=True):
            totals.setdefault(label, []).append(row.duration)
        ranked = sorted(totals, key=lambda k: (-sum(totals[k]), k))
        voice_of = {label: f"{group}_s{rank}" for rank, label in enumerate(ranked)}
        report: list[tuple[str, int, float]] = []
        for label in ranked:
            count, seconds = len(totals[label]), sum(totals[label])
            report.append((voice_of[label], count, seconds))
            if count >= min_clips and seconds >= min_seconds:
                stats.voices_kept += 1
            else:
                stats.voices_dropped += 1
        stats.per_group[group] = report
        keep = {v for v, c, s in report if c >= min_clips and s >= min_seconds}
        for row, label in zip(embedded, labels, strict=True):
            row.voice = voice_of[label]
            if row.voice in keep:
                kept_rows.append(row)
    stats.clips_kept = len(kept_rows)

    if dirty and not dry_run:
        save_embedding_cache(dataset_dir, cache)
    elif dirty:
        save_embedding_cache(dataset_dir, cache)
    if not dry_run:
        untouched = [r for r in rows if r.speaker not in groups]
        _write_rows(dataset_dir, untouched + kept_rows)
        (dataset_dir / "speakers.json").write_text(
            json.dumps(
                {
                    "threshold": threshold,
                    "min_clips": min_clips,
                    "min_seconds": min_seconds,
                    "groups": {
                        g: [{"voice": v, "clips": c, "seconds": round(s, 1)} for v, c, s in rep]
                        for g, rep in stats.per_group.items()
                    },
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    return stats
