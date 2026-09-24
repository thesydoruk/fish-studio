#!/usr/bin/env python3
"""Cut clone prompts for voices the model was never trained on.

Every speaker in ``training/raw`` is one the fine-tune has heard, so measuring
cloning against them only ever exercises the easy case. Voices that never made
it that far are already on disk: cross-video clustering assigns each channel's
diarized speakers to ``<base>_sN`` groups, and the ones that fall under the
``min_clips`` / ``min_speech_sec`` floors are dropped before export. Those are
real people -- usually guests -- recorded through the same chain, and the model
has not seen a single clip of them.

Segments are cut straight from the source audio by their transcript timestamps
rather than matched to the exported segment WAVs, whose indexes shift with the
merge and split passes.

The prompt kept per voice is the most typical of its candidates, by median
cosine to the rest, for the same reason ``build_uk_probe`` picks that way: a
clip that does not represent the speaker turns every later measurement into a
measurement of the clip.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import soundfile as sf

from fish_studio.ffmpeg import run_ffmpeg
from fish_studio.project_context import workspace_or_default
from fish_studio.server.voiceprint import VoiceEncoder, cosine_similarity

INDEX_HEADER = ("id", "wav", "text", "source", "seconds", "typicality")


@dataclass
class Candidate:
    cluster: str
    video_id: str
    source_audio: Path
    start: float
    end: float
    text: str

    @property
    def seconds(self) -> float:
        return self.end - self.start


def load_mapping(work_dir: Path) -> tuple[dict[str, str], str]:
    path = work_dir / "speaker_map.json"
    if not path.is_file():
        raise SystemExit(f"speaker map not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("mapping", {}), str(payload.get("base_speaker", ""))


def local_keys(base: str, video_id: str, speaker_id: str | None) -> list[str]:
    """The map is keyed by base__video__spk, or base__video when diarization was off."""
    keys = [f"{base}__{video_id}"]
    if speaker_id:
        keys.insert(0, f"{base}__{video_id}__{speaker_id}")
    return keys


def collect(
    work_dir: Path,
    mapping: dict[str, str],
    base: str,
    wanted: set[str],
    *,
    min_sec: float,
    max_sec: float,
    min_chars: int,
) -> dict[str, list[Candidate]]:
    found: dict[str, list[Candidate]] = {}
    for path in sorted((work_dir / "transcripts").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        video_id = str(payload.get("video_id") or path.stem)
        source_audio = Path(str(payload.get("source_audio") or ""))
        if not source_audio.is_file():
            continue
        for segment in payload.get("segments") or []:
            if segment.get("kind", "speech") != "speech":
                continue
            cluster = None
            for key in local_keys(base, video_id, segment.get("speaker_id")):
                if key in mapping:
                    cluster = mapping[key]
                    break
            if cluster not in wanted:
                continue
            text = str(segment.get("text") or "").strip()
            start = float(segment.get("start") or 0.0)
            end = float(segment.get("end") or 0.0)
            if len(text) < min_chars or not min_sec <= end - start <= max_sec:
                continue
            found.setdefault(cluster, []).append(
                Candidate(cluster, video_id, source_audio, start, end, text)
            )
    return found


def cut(candidate: Candidate, dest: Path, sample_rate: int = 44100) -> bool:
    result = run_ffmpeg(
        [
            "-y",
            "-ss",
            f"{candidate.start:.3f}",
            "-t",
            f"{candidate.seconds:.3f}",
            "-i",
            str(candidate.source_audio),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        text=True,
    )
    return result.returncode == 0 and dest.is_file()


def embed(encoder: VoiceEncoder, path: Path) -> list[float] | None:
    try:
        samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return None
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    return encoder.embed(np.asarray(samples, dtype=np.float32), int(rate))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=".env")
    parser.add_argument("--source", action="append", required=True, help="work/<slug>, repeatable")
    parser.add_argument("--out", type=Path, default=None, help="default {data_root}/eval/voices")
    parser.add_argument("--candidates", type=int, default=8, help="clips cut per voice")
    parser.add_argument("--min-sec", type=float, default=4.0)
    parser.add_argument("--max-sec", type=float, default=10.0)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument(
        "--floor",
        type=float,
        default=0.35,
        help="drop a voice whose own clips agree less than this -- it is not one person",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = workspace_or_default(args.config)
    data_root = workspace.data_root
    out_dir = args.out or (data_root / "eval" / "voices")
    out_dir.mkdir(parents=True, exist_ok=True)

    trained = {path.name for path in (workspace.training_dir / "raw").iterdir() if path.is_dir()}
    encoder = VoiceEncoder()
    encoder.warmup()
    if not encoder.available:
        raise SystemExit("ECAPA encoder unavailable; the typicality check needs it")

    rows: list[dict[str, str]] = []
    scratch = out_dir / "_cut"
    scratch.mkdir(exist_ok=True)
    for slug in args.source:
        work_dir = data_root / "work" / slug
        mapping, base = load_mapping(work_dir)
        held_out = {cluster for cluster in set(mapping.values()) if cluster not in trained}
        if not held_out:
            print(f"[skip] {slug}: every cluster is in training/raw")
            continue
        print(f"[{slug}] held out of training: {', '.join(sorted(held_out))}")

        found = collect(
            work_dir,
            mapping,
            base,
            held_out,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            min_chars=args.min_chars,
        )
        for cluster in sorted(found):
            candidates = sorted(found[cluster], key=lambda item: -item.seconds)[: args.candidates]
            cut_paths: list[tuple[Path, Candidate, list[float]]] = []
            for index, candidate in enumerate(candidates):
                dest = scratch / f"{cluster}_{index:02d}.wav"
                if not cut(candidate, dest):
                    continue
                vector = embed(encoder, dest)
                if vector:
                    cut_paths.append((dest, candidate, vector))
            if len(cut_paths) < 3:
                print(f"  {cluster}: only {len(cut_paths)} usable clips, skipped")
                continue

            best_index, best_score = 0, -2.0
            for index, (_, _, vector) in enumerate(cut_paths):
                others = [
                    cosine_similarity(vector, other)
                    for position, (_, _, other) in enumerate(cut_paths)
                    if position != index
                ]
                score = float(np.median(others))
                if score > best_score:
                    best_index, best_score = index, score
            if best_score < args.floor:
                print(f"  {cluster}: clips agree only {best_score:.3f}, not one voice -- skipped")
                continue

            path, candidate, _ = cut_paths[best_index]
            final = out_dir / f"{slug}__{cluster}.wav"
            path.replace(final)
            rows.append(
                {
                    "id": f"{slug}__{cluster}",
                    "wav": str(final.relative_to(data_root)).replace("\\", "/"),
                    "text": candidate.text,
                    "source": f"{slug}/{candidate.video_id}",
                    "seconds": f"{candidate.seconds:.2f}",
                    "typicality": f"{best_score:.3f}",
                }
            )
            print(
                f"  {cluster}: kept {final.name} ({candidate.seconds:.1f}s, typicality {best_score:.3f})"
            )

    for leftover in scratch.glob("*.wav"):
        leftover.unlink()
    scratch.rmdir()

    if not rows:
        raise SystemExit("no held-out voices could be extracted")
    index_path = out_dir / "index.tsv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_HEADER), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {index_path}: {len(rows)} voices the model has never heard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
