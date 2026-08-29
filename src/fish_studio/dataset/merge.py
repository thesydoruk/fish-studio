"""Merge multiple pipe-delimited datasets into one train/eval split for Fish training."""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MergeStats:
    total_clips: int
    train_clips: int
    eval_clips: int
    source_clips: dict[str, int]
    output_dir: str


def is_export_ready(dataset_dir: Path) -> bool:
    """True when a dataset folder has train metadata and a wavs directory."""
    path = Path(dataset_dir)
    return path.is_dir() and (path / "metadata_train.csv").is_file() and (path / "wavs").is_dir()


def list_export_ready_datasets(
    datasets_root: Path,
    *,
    exclude_names: Iterable[str] = (),
) -> list[Path]:
    """Return sorted export-ready dataset dirs under ``datasets_root``.

    Skips ``exclude_names`` (typically the merge output slug like ``combined``) so a
    previous merge is not fed back into itself.
    """
    root = Path(datasets_root)
    if not root.is_dir():
        return []

    excluded = {name.strip() for name in exclude_names if name and name.strip()}
    found: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name in excluded:
            continue
        if is_export_ready(path):
            found.append(path)
    return found


def _read_metadata_rows(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split("|")
        if header[:3] != ["audio_file", "text", "speaker_name"]:
            raise ValueError(f"{path}: expected header audio_file|text|speaker_name")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def _write_metadata(path: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = ["audio_file|text|speaker_name"]
    lines.extend(f"{audio}|{text}|{speaker}" for audio, text, speaker in rows)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def merge_datasets(
    source_dirs: list[Path],
    output_dir: Path,
    *,
    eval_split_size: float = 0.05,
    seed: int = 42,
    speaker_name: str = "speaker",
) -> MergeStats:
    """Copy WAVs with new sequential IDs and re-shuffle train/eval metadata.

    Per-row ``speaker_name`` values are preserved. ``speaker_name`` is only used
    as a fallback when a metadata row has an empty speaker field.
    """
    if not source_dirs:
        raise ValueError("at least one source dataset directory is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    merged: list[tuple[str, str, str]] = []
    source_clips: dict[str, int] = {}
    seq = 1

    for src in source_dirs:
        src = src.resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"dataset not found: {src}")

        rows: list[tuple[str, str, str]] = []
        for name in ("metadata_train.csv", "metadata_eval.csv"):
            meta = src / name
            if meta.is_file():
                rows.extend(_read_metadata_rows(meta))
        if not rows:
            raise ValueError(f"{src}: no metadata_train.csv / metadata_eval.csv rows found")

        count = 0
        for rel_audio, text, row_speaker in rows:
            audio_path = src / rel_audio
            if not audio_path.is_file():
                raise FileNotFoundError(f"{src}: missing audio file {rel_audio}")
            file_id = f"{seq:06d}"
            dst = wavs_dir / f"{file_id}.wav"
            shutil.copy2(audio_path, dst)
            kept_speaker = (row_speaker or "").strip() or speaker_name
            merged.append((f"wavs/{file_id}.wav", text, kept_speaker))
            seq += 1
            count += 1
        source_clips[src.name] = count

    shuffled = merged[:]
    random.Random(seed).shuffle(shuffled)
    eval_n = 0
    if len(shuffled) > 1:
        eval_n = int(len(shuffled) * eval_split_size)
        eval_n = max(1, eval_n)
        eval_n = min(eval_n, len(shuffled) - 1)

    eval_rows = shuffled[:eval_n]
    train_rows = shuffled[eval_n:]
    _write_metadata(output_dir / "metadata_train.csv", train_rows)
    _write_metadata(output_dir / "metadata_eval.csv", eval_rows)

    reference_src = next(
        # Reuse the first available reference clip for inference smoke tests.
        (src / "reference.wav" for src in source_dirs if (src / "reference.wav").is_file()),
        None,
    )
    if reference_src is not None:
        shutil.copy2(reference_src, output_dir / "reference.wav")

    stats = {
        "total_clips": len(merged),
        "train_clips": len(train_rows),
        "eval_clips": len(eval_rows),
        "source_clips": source_clips,
        "metadata_format": "pipe",
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return MergeStats(
        total_clips=len(merged),
        train_clips=len(train_rows),
        eval_clips=len(eval_rows),
        source_clips=source_clips,
        output_dir=str(output_dir),
    )
