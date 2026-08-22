"""Transcribe audio via the external audio-intel HTTP service."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx

from fish_studio.config import AudioIntelConfig
from fish_studio.dataset.stats import TranscriptStats
from fish_studio.dataset.transcript import (
    SoundEvent,
    SpeakerInfo,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
    load_transcript,
    parse_embedding,
    save_transcript,
)


class AudioIntelClient:
    """POST /v1/audio/transcriptions client for the dataset pipeline."""

    def __init__(self, config: AudioIntelConfig) -> None:
        self.config = config

    def describe_runtime(self) -> str:
        align = "align on" if self.config.align else "align off"
        diarize = "diarize on" if self.config.diarize else "diarize off"
        sound = "sound_events on" if self.config.sound_events else "sound_events off"
        language = self.config.language or "auto"
        return f"audio-intel @ {self.config.base_url} ({language}, {align}, {diarize}, {sound})"

    def health(self) -> dict:
        url = f"{self.config.base_url.rstrip('/')}/health"
        response = httpx.get(url, timeout=self.config.timeout_sec)
        response.raise_for_status()
        return response.json()

    def transcribe_file(self, video_id: str, audio_path: str | Path) -> TranscriptResult:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        url = f"{self.config.base_url.rstrip('/')}/v1/audio/transcriptions"
        data: dict[str, str] = {
            "response_format": "verbose_json",
            "align": "true" if self.config.align else "false",
            "diarize": "true" if self.config.diarize else "false",
            "sound_events": "true" if self.config.sound_events else "false",
        }
        language = (self.config.language or "").strip().lower()
        if language and language != "auto":
            data["language"] = language

        with audio_path.open("rb") as handle:
            files = {"file": (audio_path.name, handle)}
            response = httpx.post(
                url,
                data=data,
                files=files,
                timeout=self.config.timeout_sec,
            )
        response.raise_for_status()
        return self._from_api_payload(video_id, audio_path, response.json())

    def _from_api_payload(
        self,
        video_id: str,
        audio_path: Path,
        payload: dict,
    ) -> TranscriptResult:
        raw_segments = payload.get("segments", [])
        # audio-intel mixes speech and PANNs events in one list; clips only come from speech.
        speech_segments = [segment for segment in raw_segments if segment.get("kind") == "speech"]
        stats = TranscriptStats()
        segments: list[TranscriptSegment] = []

        for index, segment in enumerate(speech_segments):
            language = segment.get("language") or payload.get("language")
            if language:
                stats.detected_languages[language] = stats.detected_languages.get(language, 0) + 1

            words = [
                TranscriptWord(
                    word=str(word.get("word", "")).strip(),
                    start=float(word["start"]),
                    end=float(word["end"]),
                    probability=float(word.get("probability", word.get("score", 0.0))),
                )
                for word in segment.get("words") or []
                if word.get("word")
            ]
            alignment_failed = bool(segment.get("alignment_failed", False))
            if alignment_failed:
                stats.marked_alignment_failed += 1

            speaker_raw = segment.get("speaker_id")
            speaker_id = str(speaker_raw).strip() if speaker_raw is not None else None
            if speaker_id == "":
                speaker_id = None

            built = TranscriptSegment(
                id=index,
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=str(segment.get("text", "")).strip(),
                avg_logprob=float(segment.get("avg_logprob", -0.5)),
                no_speech_prob=float(segment.get("no_speech_prob", 0.0)),
                compression_ratio=float(segment.get("compression_ratio", 1.0)),
                words=words,
                language=language,
                alignment_failed=alignment_failed,
                speaker_id=speaker_id,
            )
            segments.append(built)
            stats.segments_kept += 1
            stats.kept_speech_duration_sec += built.end - built.start

        sound_events = [
            SoundEvent(
                label=str(segment.get("label", "")).strip(),
                start=float(segment["start"]),
                end=float(segment["end"]),
                score=float(segment.get("score", 0.0)),
                prompt_relevant=bool(segment.get("prompt_relevant", False)),
            )
            for segment in raw_segments
            if segment.get("kind") == "sound" and segment.get("label")
        ]

        speakers = [
            SpeakerInfo(
                id=str(speaker.get("id", "")).strip(),
                speech_seconds=float(speaker.get("speech_seconds", 0.0)),
                segment_count=int(speaker.get("segment_count", 0)),
                embedding=parse_embedding(speaker.get("embedding")),
            )
            for speaker in payload.get("speakers") or []
            if speaker.get("id")
        ]

        aligned_at = payload.get("aligned_at")
        if self.config.align and not aligned_at and payload.get("alignment_applied"):
            # Older audio-intel builds omit aligned_at; treat alignment_applied as success.
            aligned_at = datetime.now(timezone.utc).isoformat()

        return TranscriptResult(
            video_id=video_id,
            source_audio=str(audio_path),
            language=str(payload.get("language") or self.config.language or "unknown"),
            duration_sec=float(payload.get("duration", 0.0)),
            segments=segments,
            transcribed_at=datetime.now(timezone.utc).isoformat(),
            aligned_at=aligned_at if self.config.align else None,
            stats=stats,
            sound_events=sound_events,
            speakers=speakers,
        )

    save_transcript = staticmethod(save_transcript)
    load_transcript = staticmethod(load_transcript)
