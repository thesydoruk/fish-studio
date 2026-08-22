"""Turn aligned transcripts into training clips and export per-clip WAV files."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from fish_studio.config import QualityConfig, SegmentationConfig
from fish_studio.dataset.audio_normalize import ffmpeg_output_args
from fish_studio.dataset.audio_quality import (
    analyze_segment_volume,
    avg_word_score,
    low_score_word_ratio,
    passes_audio_quality,
)
from fish_studio.dataset.junk_filter import filter_junk_segments
from fish_studio.dataset.script_split import refine_segment_language, split_segments_by_script
from fish_studio.dataset.segment_filters import filter_allowed_languages, filter_segments
from fish_studio.dataset.speakers import resolve_speaker_name
from fish_studio.dataset.stats import SegmentStats
from fish_studio.dataset.transcript import TranscriptResult, TranscriptSegment


@dataclass
class AudioClip:
    """One training utterance with quality metrics for export ranking."""

    clip_id: st
    video_id: st
    start: float
    end: float
    text: st
    duration: float
    source_audio: st
    speaker_name: str = "speaker"
    avg_word_score: float | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    low_score_word_ratio: float | None = None
    mean_volume_db: float | None = None
    max_volume_db: float | None = None
    quality_score: float | None = None


class AudioSegmenter:
    """Filter → merge adjacent segments → split long ones → cut WAV clips via ffmpeg."""

    def __init__(
        self,
        config: SegmentationConfig,
        quality: QualityConfig,
        target_language: str | None = None,
        *,
        filter_alignment_failed: bool = True,
        speaker_name: str = "speaker",
    ) -> None:
        self.config = config
        self.quality = quality
        self.target_language = target_language
        self.filter_alignment_failed = filter_alignment_failed
        self.speaker_name = speaker_name

    def build_clips(self, transcript: TranscriptResult) -> tuple[list[AudioClip], SegmentStats]:
        stats = SegmentStats()

        segments = filter_segments(
            transcript.segments,
            quality=self.quality,
            segmentation=self.config,
            target_language=self.target_language,
            aligned_at=transcript.aligned_at,
            stats=stats,
            filter_alignment_failed=self.filter_alignment_failed,
        )
        segments = filter_junk_segments(
            segments,
            transcript.sound_events,
            stats=stats,
        )
        if self.config.split_by_script:
            segments = split_segments_by_script(segments)
        else:
            segments = [refine_segment_language(segment) for segment in segments]
        segments = filter_allowed_languages(segments, self.config.allowed_languages, stats)
        merged = self._merge_segments(segments)
        # Split segments that exceed max duration (prefer word boundaries).
        normalized: list[TranscriptSegment] = []

        for seg in merged:
            normalized.extend(self._normalize_segment(seg))

        clips: list[AudioClip] = []

        for idx, seg in enumerate(normalized):
            text = self._clean_text(seg.text)
            if len(text) < self.config.min_chars or len(text) > self.config.max_chars:
                stats.rejected_text_length += 1
                continue

            duration = seg.end - seg.start
            if duration < self.config.min_duration_sec or duration > self.config.max_duration_sec:
                stats.rejected_duration += 1
                continue

            ok, (mean_volume_db, max_volume_db) = self._evaluate_audio_quality(
                seg,
                transcript.source_audio,
            )
            if not ok:
                stats.rejected_audio_quality += 1
                continue

            clip_id = f"{transcript.video_id}_{idx:04d}"
            clips.append(
                AudioClip(
                    clip_id=clip_id,
                    video_id=transcript.video_id,
                    start=seg.start,
                    end=seg.end,
                    text=text,
                    duration=duration,
                    source_audio=transcript.source_audio,
                    speaker_name=resolve_speaker_name(
                        self.speaker_name,
                        seg.speaker_id,
                        video_id=transcript.video_id,
                    ),
                    avg_word_score=avg_word_score(seg.words),
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                    compression_ratio=seg.compression_ratio,
                    low_score_word_ratio=low_score_word_ratio(
                        seg.words, self.quality.min_word_score
                    ),
                    mean_volume_db=mean_volume_db,
                    max_volume_db=max_volume_db,
                )
            )
            stats.clips_kept += 1
            stats.kept_duration_sec += duration

        return clips, stats

    def export_clips(
        self,
        clips: list[AudioClip],
        output_dir: str | Path,
        num_workers: int = 1,
    ) -> list[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pending = [clip for clip in clips if not (output_dir / f"{clip.clip_id}.wav").exists()]
        exported = [
            output_dir / f"{clip.clip_id}.wav"
            for clip in clips
            if (output_dir / f"{clip.clip_id}.wav").exists()
        ]

        if not pending:
            return exported

        if num_workers <= 1:
            for clip in pending:
                exported.append(self._export_one_clip(clip, output_dir))
            return exported

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(self._export_one_clip, clip, output_dir) for clip in pending]
            for future in as_completed(futures):
                exported.append(future.result())

        return exported

    def _export_one_clip(self, clip: AudioClip, output_dir: Path) -> Path:
        out_path = output_dir / f"{clip.clip_id}.wav"
        if out_path.exists():
            return out_path

        start = max(0.0, clip.start - self.config.padding_sec)
        end = clip.end + self.config.padding_sec
        cmd = [
            "ffmpeg",
            "-y",
            # Input seek: place -ss/-to before -i so ffmpeg does not decode the whole file.
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            clip.source_audio,
            *ffmpeg_output_args(self.config),
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {clip.clip_id}:\n{proc.stderr}")
        return out_path

    def save_clip_manifest(self, clips: list[AudioClip], output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "clips.jsonl"
        with manifest.open("w", encoding="utf-8") as f:
            for clip in clips:
                f.write(json.dumps(asdict(clip), ensure_ascii=False) + "\n")
        return manifest

    def _evaluate_audio_quality(
        self,
        segment: TranscriptSegment,
        source_audio: str,
    ) -> tuple[bool, tuple[float | None, float | None]]:
        if not self.quality.filter_audio_quality:
            return True, (None, None)
        if self.config.normalize_loudness:
            # loudnorm rewrites peak/mean on export; source YouTube levels must not drop clips.
            return True, (None, None)

        audio = analyze_segment_volume(
            source_audio,
            segment.start,
            segment.end,
            padding_sec=self.config.padding_sec,
        )
        if audio is None:
            return False, (None, None)

        return passes_audio_quality(audio, self.quality), (
            audio.mean_volume_db,
            audio.max_volume_db,
        )

    def _merge_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Join adjacent short gaps into one clip when duration/text limits allow."""
        if not segments:
            return []

        merged: list[TranscriptSegment] = []
        current = segments[0]

        for seg in segments[1:]:
            gap = seg.start - current.end
            combined_duration = seg.end - current.start
            combined_text = f"{current.text} {seg.text}".strip()

            should_merge = (
                gap <= self.config.merge_gap_sec
                and combined_duration <= self.config.max_duration_sec
                and len(combined_text) <= self.config.max_chars
                and current.speaker_id == seg.speaker_id
                and current.language == seg.language
            )

            if should_merge:
                # Conservative merge: take worst quality metrics from either side.
                current = TranscriptSegment(
                    id=current.id,
                    start=current.start,
                    end=seg.end,
                    text=combined_text,
                    avg_logprob=min(current.avg_logprob, seg.avg_logprob),
                    no_speech_prob=max(current.no_speech_prob, seg.no_speech_prob),
                    compression_ratio=max(current.compression_ratio, seg.compression_ratio),
                    words=current.words + seg.words,
                    language=current.language or seg.language,
                    language_probability=current.language_probability or seg.language_probability,
                    alignment_failed=current.alignment_failed or seg.alignment_failed,
                    speaker_id=current.speaker_id,
                )
            else:
                merged.append(current)
                current = seg

        merged.append(current)
        return merged

    def _normalize_segment(self, segment: TranscriptSegment) -> list[TranscriptSegment]:
        """Split over-long segments on word timestamps; fall back to a midpoint cut."""
        duration = segment.end - segment.start
        if duration <= self.config.max_duration_sec:
            return [segment]

        if segment.words:
            return self._split_by_words(segment)

        midpoint = segment.start + duration / 2
        text_parts = self._split_text_in_half(segment.text)
        return [
            TranscriptSegment(
                id=segment.id,
                start=segment.start,
                end=midpoint,
                text=text_parts[0],
                avg_logprob=segment.avg_logprob,
                no_speech_prob=segment.no_speech_prob,
                compression_ratio=segment.compression_ratio,
                words=[w for w in segment.words if w.end <= midpoint],
                language=segment.language,
                language_probability=segment.language_probability,
                alignment_failed=segment.alignment_failed,
                speaker_id=segment.speaker_id,
            ),
            TranscriptSegment(
                id=segment.id,
                start=midpoint,
                end=segment.end,
                text=text_parts[1],
                avg_logprob=segment.avg_logprob,
                no_speech_prob=segment.no_speech_prob,
                compression_ratio=segment.compression_ratio,
                words=[w for w in segment.words if w.start >= midpoint],
                language=segment.language,
                language_probability=segment.language_probability,
                alignment_failed=segment.alignment_failed,
                speaker_id=segment.speaker_id,
            ),
        ]

    def _split_by_words(self, segment: TranscriptSegment) -> list[TranscriptSegment]:
        chunks: list[TranscriptSegment] = []
        word_buffer: list = []
        text_buffer: list[str] = []
        chunk_start = segment.words[0].start

        for word in segment.words:
            candidate_duration = word.end - chunk_start

            if candidate_duration > self.config.max_duration_sec and text_buffer:
                # Close the current chunk *before* this word so we never overshoot max duration.
                chunks.append(
                    TranscriptSegment(
                        id=segment.id,
                        start=chunk_start,
                        end=word_buffer[-1].end,
                        text=" ".join(text_buffer).strip(),
                        avg_logprob=segment.avg_logprob,
                        no_speech_prob=segment.no_speech_prob,
                        compression_ratio=segment.compression_ratio,
                        words=list(word_buffer),
                        language=segment.language,
                        language_probability=segment.language_probability,
                        alignment_failed=segment.alignment_failed,
                        speaker_id=segment.speaker_id,
                    )
                )
                word_buffer = [word]
                text_buffer = [word.word]
                chunk_start = word.start
            else:
                word_buffer.append(word)
                text_buffer.append(word.word)

        if text_buffer:
            chunks.append(
                TranscriptSegment(
                    id=segment.id,
                    start=chunk_start,
                    end=word_buffer[-1].end,
                    text=" ".join(text_buffer).strip(),
                    avg_logprob=segment.avg_logprob,
                    no_speech_prob=segment.no_speech_prob,
                    compression_ratio=segment.compression_ratio,
                    words=word_buffer,
                    language=segment.language,
                    language_probability=segment.language_probability,
                    alignment_failed=segment.alignment_failed,
                    speaker_id=segment.speaker_id,
                )
            )

        return chunks

    @staticmethod
    def _split_text_in_half(text: str) -> tuple[str, str]:
        words = text.split()
        mid = len(words) // 2
        if mid == 0:
            return text, ""
        return " ".join(words[:mid]), " ".join(words[mid:])

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        # Collapse ellipses and strip glyphs Fish Speech was not trained to read.
        text = text.replace("...", ".")
        text = re.sub(r"[^\w\s\u0400-\u04FF\u0490-\u0491.,!?;:\-—'\"()«»]", "", text)
        return text.strip()
