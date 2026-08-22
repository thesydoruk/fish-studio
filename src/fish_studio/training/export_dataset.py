#!/usr/bin/env python3
"""Export a pipe-delimited dataset to Fish Speech fine-tuning layout (.wav + .lab)."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from fish_studio.config import StressConfig
from fish_studio.project_context import try_load_project
from fish_studio.stress import stressify
from fish_studio.training.layout import ensure_training_dirs


@dataclass(frozen=True)
class ExportStats:
    clips: int
    output_dir: Path
    metadata: Path
    speakers: dict[str, int]


def export_labeled_dataset(
    dataset_dir: Path,
    output_dir: Path,
    *,
    speaker_name: str,
    include_eval: bool = False,
    stress: StressConfig | None = None,
) -> ExportStats:
    """Copy WAVs into per-speaker folders with matching ``.lab`` transcripts for VQ extraction."""
    train_csv = dataset_dir / "metadata_train.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(f"metadata_train.csv not found: {train_csv}")

    wavs_dir = dataset_dir / "wavs"
    if not wavs_dir.is_dir():
        raise FileNotFoundError(f"wavs/ not found: {wavs_dir}")

    rows: list[tuple[str, str, str]] = []
    rows.extend(_read_metadata_csv(train_csv, speaker_name))
    if include_eval:
        eval_csv = dataset_dir / "metadata_eval.csv"
        if eval_csv.is_file():
            rows.extend(_read_metadata_csv(eval_csv, speaker_name))

    # Fish groups training prompts per speaker folder, so each voice needs its own.
    for speaker in {speaker for _, _, speaker in rows}:
        speaker_dir = output_dir / speaker
        if speaker_dir.exists():
            shutil.rmtree(speaker_dir)
        speaker_dir.mkdir(parents=True, exist_ok=True)

    speakers: dict[str, int] = {}
    for rel_audio, text, speaker in rows:
        src = dataset_dir / rel_audio
        if not src.is_file():
            src = wavs_dir / Path(rel_audio).name
        if not src.is_file():
            print(f"[warn] missing audio: {rel_audio}", file=sys.stderr)
            continue

        speaker_dir = output_dir / speaker
        stem = src.stem
        dst_wav = speaker_dir / f"{stem}.wav"
        shutil.copy2(src, dst_wav)
        if stress is None:
            label = text.strip()
        else:
            # Prefer already-marked dataset text; still fill gaps from this clip's WAV.
            label = stressify(text.strip(), stress, audio_path=dst_wav)
        (speaker_dir / f"{stem}.lab").write_text(label + "\n", encoding="utf-8")
        speakers[speaker] = speakers.get(speaker, 0) + 1

    exported = sum(speakers.values())
    if exported == 0:
        raise RuntimeError(f"No clips exported from {dataset_dir}")

    return ExportStats(
        clips=exported,
        output_dir=output_dir,
        metadata=train_csv,
        speakers=speakers,
    )


def _read_metadata_csv(path: Path, default_speaker: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        for row in reader:
            audio = (row.get("audio_file") or "").strip()
            text = (row.get("text") or "").strip()
            speaker = (row.get("speaker_name") or "").strip() or default_speaker
            if audio and text:
                rows.append((audio, text, speaker))
    return rows


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", default=".env")
    pre_args, _ = pre.parse_known_args()
    project = try_load_project(pre_args.config)

    defaults_dataset = "combined"
    defaults_speaker = "speaker"
    defaults_output: Path | None = None
    if project is not None:
        defaults_dataset = project.training.dataset_id
        defaults_speaker = project.training.speaker_name or project.export.speaker_name
        defaults_output = ensure_training_dirs(project.workspace())["raw"]

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--dataset-id", default=defaults_dataset)
    parser.add_argument("--speaker-name", default=defaults_speaker)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults_output,
        help="Fish raw dataset root (default: {data_root}/training/raw)",
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="Also export metadata_eval.csv clips",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = try_load_project(args.config)
    if project is None:
        print("[error] .env not found", file=sys.stderr)
        sys.exit(1)

    ws = project.workspace()
    dataset_dir = ws.dataset_dir(args.dataset_id)
    output_dir = args.output_dir or ensure_training_dirs(ws)["raw"]

    stats = export_labeled_dataset(
        dataset_dir,
        output_dir,
        speaker_name=args.speaker_name,
        include_eval=args.include_eval,
        stress=project.stress,
    )
    breakdown = ", ".join(f"{name}={count}" for name, count in sorted(stats.speakers.items()))
    print(f"[done] exported {stats.clips} clips to {stats.output_dir} ({breakdown})")


if __name__ == "__main__":
    main()
