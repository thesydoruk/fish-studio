#!/usr/bin/env python3
"""Export a pipe-delimited dataset to Fish Speech fine-tuning layout (.wav + .lab).

Two things make this fast enough to re-run. The audio is hard-linked into the
speaker folders when the dataset and the export share a filesystem: a copy of
the corpus is 76 GB and nothing downstream rewrites a wav. Stress marking runs
in a process pool, each worker with its own Stanza pipeline and its own CTC
aligner, because most clips carry at least one word the dictionary does not
know and the fill from audio is the bulk of the work: ~12 ms per clip on
``STRESS_ACOUSTIC_DEVICE=cuda`` against ~330 ms on CPU. Eight CUDA aligners
need ~18 GB, so stop the TTS stack first (``./run.sh stack stop``).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path

from fish_studio.config import StressConfig
from fish_studio.paths import link_or_copy
from fish_studio.project_context import try_load_project
from fish_studio.stress import stressify
from fish_studio.stress_align import CtcAligner, fill_stress_from_audio, needs_stress_fill
from fish_studio.training.layout import ensure_training_dirs


@dataclass(frozen=True)
class ExportStats:
    clips: int
    output_dir: Path
    metadata: Path
    speakers: dict[str, int]
    filled: int = 0


@dataclass(frozen=True)
class _Job:
    src: Path
    dst_wav: Path
    dst_lab: Path
    text: str
    speaker: str


@dataclass
class _Marker:
    """Everything one process needs to label a clip; built once per process."""

    stress: StressConfig | None
    aligner: CtcAligner | None = None

    @classmethod
    def build(cls, stress: StressConfig | None) -> _Marker:
        if stress is None or not stress.acoustic_fallback:
            return cls(stress=stress)
        return cls(stress=stress, aligner=CtcAligner(device=stress.acoustic_device))

    def label(self, job: _Job) -> tuple[str, bool]:
        """The transcript to write, and whether the audio contributed a mark."""
        if self.stress is None:
            return job.text.strip(), False
        # Prefer already-marked dataset text; the text stages never touch audio.
        marked = stressify(job.text.strip(), replace(self.stress, acoustic_fallback=False))
        if self.aligner is None or not needs_stress_fill(marked):
            return marked, False
        filled = fill_stress_from_audio(
            marked, job.dst_wav, min_margin=self.stress.acoustic_margin, aligner=self.aligner
        )
        return filled, filled != marked


_WORKER: _Marker | None = None


def _init_worker(stress: StressConfig | None, threads: int) -> None:
    """Cap each worker's thread pool to its share of the cores, then load its models."""
    global _WORKER
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass
    _WORKER = _Marker.build(stress)


def _export_one(job: _Job, marker: _Marker) -> tuple[str, bool]:
    link_or_copy(job.src, job.dst_wav)
    label, filled = marker.label(job)
    job.dst_lab.write_text(label + "\n", encoding="utf-8")
    return job.speaker, filled


def _export_in_worker(job: _Job) -> tuple[str, bool]:
    assert _WORKER is not None
    return _export_one(job, _WORKER)


def export_labeled_dataset(
    dataset_dir: Path,
    output_dir: Path,
    *,
    speaker_name: str,
    include_eval: bool = False,
    stress: StressConfig | None = None,
    workers: int = 1,
    progress_every: int = 5000,
) -> ExportStats:
    """Link WAVs into per-speaker folders with matching ``.lab`` transcripts for VQ extraction."""
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

    jobs: list[_Job] = []
    for rel_audio, text, speaker in rows:
        src = dataset_dir / rel_audio
        if not src.is_file():
            src = wavs_dir / Path(rel_audio).name
        if not src.is_file():
            print(f"[warn] missing audio: {rel_audio}", file=sys.stderr)
            continue
        speaker_dir = output_dir / speaker
        jobs.append(
            _Job(
                src=src,
                dst_wav=speaker_dir / f"{src.stem}.wav",
                dst_lab=speaker_dir / f"{src.stem}.lab",
                text=text,
                speaker=speaker,
            )
        )

    speakers: dict[str, int] = {}
    filled = 0
    started = time.monotonic()

    def note(speaker: str, did_fill: bool) -> None:
        nonlocal filled
        filled += did_fill
        speakers[speaker] = speakers.get(speaker, 0) + 1
        done = sum(speakers.values())
        if progress_every and done % progress_every == 0:
            rate = done / max(time.monotonic() - started, 1e-9)
            left = (len(jobs) - done) / max(rate, 1e-9)
            print(
                f"[export] {done}/{len(jobs)} clips, {rate:.0f}/s, filled {filled}, "
                f"~{left / 60:.0f} min left",
                flush=True,
            )

    if workers <= 1:
        marker = _Marker.build(stress)
        for job in jobs:
            note(*_export_one(job, marker))
    else:
        threads = max(1, (os.cpu_count() or workers) // workers)
        context = get_context("spawn")
        with context.Pool(workers, initializer=_init_worker, initargs=(stress, threads)) as pool:
            for speaker, did_fill in pool.imap_unordered(_export_in_worker, jobs, chunksize=32):
                note(speaker, did_fill)

    exported = sum(speakers.values())
    if exported == 0:
        raise RuntimeError(f"No clips exported from {dataset_dir}")

    return ExportStats(
        clips=exported,
        output_dir=output_dir,
        metadata=train_csv,
        speakers=speakers,
        filled=filled,
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
    defaults_workers = 8
    if project is not None:
        defaults_dataset = project.training.dataset_id
        defaults_speaker = project.training.speaker_name or project.export.speaker_name
        defaults_output = ensure_training_dirs(project.workspace())["raw"]
        defaults_workers = project.training.export_num_workers

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
    parser.add_argument(
        "--workers",
        type=int,
        default=defaults_workers,
        help="Stress-marking processes; each loads Stanza and, with the acoustic fill, "
        "an aligner (~2.3 GB of GPU memory on cuda)",
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
        workers=args.workers,
    )
    breakdown = ", ".join(f"{name}={count}" for name, count in sorted(stats.speakers.items()))
    print(
        f"[done] exported {stats.clips} clips to {stats.output_dir} "
        f"(filled stress from audio in {stats.filled}; {breakdown})"
    )


if __name__ == "__main__":
    main()
