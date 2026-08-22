"""Export segmented clips to pipe-delimited dataset format (metadata_train/eval.csv + wavs/)."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from fish_studio.config import ExportConfig, StressConfig
from fish_studio.dataset.segment import AudioClip
from fish_studio.stress import stressify


@dataclass
class DatasetStats:
    total_clips: int
    total_duration_sec: float
    metadata_path: str
    wavs_dir: str
    reference_path: str | None
    eval_metadata_path: str | None = None
    train_clips: int = 0
    eval_clips: int = 0


@dataclass
class _ClipRecord:
    file_id: str
    text: str
    clip: AudioClip


def clear_dataset_dir(output_dir: str | Path) -> None:
    """Remove a pipe-delimited dataset directory (wavs, metadata, reference)."""
    path = Path(output_dir)
    if path.exists():
        shutil.rmtree(path)


class DatasetExporter:
    """Export clips in pipe-delimited dataset format (metadata_train/eval.csv + wavs/)."""

    def __init__(
        self,
        config: ExportConfig,
        stress: StressConfig | None = None,
    ) -> None:
        self.config = config
        self.stress = stress

    def export(
        self,
        clips: list[AudioClip],
        segments_dir: str | Path,
        start_index: int = 1,
        *,
        segment_roots: dict[str, Path] | None = None,
    ) -> DatasetStats:
        output_dir = Path(self.config.output_dir)
        wavs_dir = output_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

        segments_dir = Path(segments_dir)
        records: list[_ClipRecord] = []
        total_duration = 0.0

        for offset, clip in enumerate(clips):
            seq = start_index + offset
            file_id = f"{seq:06d}"
            src = self._clip_source_path(clip, segments_dir, segment_roots)
            dst = wavs_dir / f"{file_id}.wav"

            if not src.exists():
                # Manifest can list a clip whose WAV was never cut (or was deleted).
                continue

            shutil.copy2(src, dst)
            text = clip.text.strip()
            if self.stress is not None and self.stress.enabled:
                # WAV is on disk here, so acoustic fallback can mark leftover OOV words.
                text = stressify(text, self.stress, audio_path=dst)
            records.append(_ClipRecord(file_id=file_id, text=text, clip=clip))
            total_duration += clip.duration

        train_path, eval_path, train_n, eval_n = self._write_metadata(records, output_dir)
        clip_list = [r.clip for r in records]
        reference_path = self._create_reference(
            clip_list,
            segments_dir,
            output_dir,
            segment_roots=segment_roots,
        )
        self._create_speaker_references(
            clip_list,
            segments_dir,
            output_dir,
            segment_roots=segment_roots,
        )

        speaker_counts = Counter(
            (r.clip.speaker_name or self.config.speaker_name) for r in records
        )
        stats = {
            "total_clips": len(records),
            "train_clips": train_n,
            "eval_clips": eval_n,
            "total_duration_sec": round(total_duration, 2),
            "speaker_name": self.config.speaker_name,
            "speakers": dict(speaker_counts),
            "metadata_format": "pipe",
        }
        with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return DatasetStats(
            total_clips=len(records),
            total_duration_sec=total_duration,
            metadata_path=str(train_path),
            wavs_dir=str(wavs_dir),
            reference_path=reference_path,
            eval_metadata_path=str(eval_path),
            train_clips=train_n,
            eval_clips=eval_n,
        )

    def _clip_source_path(
        self,
        clip: AudioClip,
        segments_dir: Path,
        segment_roots: dict[str, Path] | None,
    ) -> Path:
        root = segment_roots.get(clip.video_id, segments_dir) if segment_roots else segments_dir
        return root / clip.video_id / f"{clip.clip_id}.wav"

    def _write_metadata(
        self, records: list[_ClipRecord], output_dir: Path
    ) -> tuple[Path, Path, int, int]:
        shuffled = records[:]
        random.Random(self.config.seed).shuffle(shuffled)

        eval_n = 0
        if len(shuffled) > 1:
            eval_n = int(len(shuffled) * self.config.eval_split_size)
            eval_n = max(1, eval_n)
            eval_n = min(eval_n, len(shuffled) - 1)  # always leave at least one train clip

        eval_records = shuffled[:eval_n]
        train_records = shuffled[eval_n:]

        train_path = output_dir / "metadata_train.csv"
        eval_path = output_dir / "metadata_eval.csv"
        self._write_metadata_file(train_path, train_records)
        self._write_metadata_file(eval_path, eval_records)

        return train_path, eval_path, len(train_records), len(eval_records)

    def _write_metadata_file(self, path: Path, records: list[_ClipRecord]) -> None:
        lines = ["audio_file|text|speaker_name"]
        lines.extend(
            f"wavs/{r.file_id}.wav|{r.text}|{r.clip.speaker_name or self.config.speaker_name}"
            for r in records
        )
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def append_clips(
        self,
        clips: list[AudioClip],
        segments_dir: str | Path,
    ) -> DatasetStats:
        output_dir = Path(self.config.output_dir)
        wavs_dir = output_dir / "wavs"
        existing = len(list(wavs_dir.glob("*.wav"))) if wavs_dir.exists() else 0
        return self.export(clips, segments_dir, start_index=existing + 1)

    def _create_reference(
        self,
        clips: list[AudioClip],
        segments_dir: str | Path,
        output_dir: Path,
        *,
        segment_roots: dict[str, Path] | None = None,
    ) -> str | None:
        if not clips:
            return None

        # Prefer the dominant speaker for the top-level reference.wav.
        counts = Counter(clip.speaker_name or self.config.speaker_name for clip in clips)
        dominant = counts.most_common(1)[0][0]
        dominant_clips = [
            clip for clip in clips if (clip.speaker_name or self.config.speaker_name) == dominant
        ]
        return self._write_reference_wav(
            dominant_clips or clips,
            Path(segments_dir),
            Path(output_dir) / "reference.wav",
            segment_roots=segment_roots,
        )

    def _create_speaker_references(
        self,
        clips: list[AudioClip],
        segments_dir: str | Path,
        output_dir: Path,
        *,
        segment_roots: dict[str, Path] | None = None,
    ) -> None:
        if not clips:
            return

        refs_dir = Path(output_dir) / "references"
        by_speaker: dict[str, list[AudioClip]] = {}
        for clip in clips:
            speaker = clip.speaker_name or self.config.speaker_name
            by_speaker.setdefault(speaker, []).append(clip)

        for speaker, speaker_clips in by_speaker.items():
            safe = speaker.replace("/", "_").replace("\\", "_")
            self._write_reference_wav(
                speaker_clips,
                Path(segments_dir),
                refs_dir / f"{safe}.wav",
                segment_roots=segment_roots,
            )

    def _write_reference_wav(
        self,
        clips: list[AudioClip],
        segments_dir: Path,
        ref_path: Path,
        *,
        segment_roots: dict[str, Path] | None = None,
    ) -> str | None:
        if not clips:
            return None

        target = self.config.reference_duration_sec

        def rank_key(clip: AudioClip) -> tuple[float, float, float]:
            """Prefer high word-confidence clips whose duration is close to the target."""
            score = clip.quality_score
            if score is None:
                if clip.avg_word_score is not None:
                    score = clip.avg_word_score
                elif clip.avg_logprob is not None:
                    score = max(0.0, clip.avg_logprob + 1.0)
                else:
                    score = 0.0
            return (-score, abs(clip.duration - target), -(clip.avg_word_score or 0.0))

        ranked = sorted(clips, key=rank_key)
        ref_clip = ranked[0]
        src = self._clip_source_path(ref_clip, segments_dir, segment_roots)
        if not src.exists():
            return None

        ref_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(self.config.reference_sample_rate),
            "-c:a",
            "pcm_s16le",
            str(ref_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None
        return str(ref_path)
