"""Persist recent synthesis requests for offline analysis.

Each call writes a timestamped directory under ``{data_root}/logs/synthesis/``:

- ``request.json`` — texts, flags, durations, chunk list
- ``reference_0.wav`` (+ more if present)
- ``synth_raw.wav`` — model output before pause/tempo/loudness fit
- ``synth_final.wav`` — audio returned to the client

``LATEST`` points at the newest request id. Older dirs are pruned.
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

_LOCK = threading.Lock()
_LATEST_NAME = "LATEST"


@dataclass
class SynthesisLogRecord:
    """Payload written next to the WAVs for one synthesize call."""

    request_id: str
    created_at: str
    language: str
    match_timing: bool
    text_raw: str
    text_prepared: str
    chunks: list[str]
    reference_texts: list[str]
    sample_rate: int
    duration_raw_sec: float
    duration_final_sec: float
    duration_reference_0_sec: float | None = None
    active_speech_raw_sec: float | None = None
    active_speech_final_sec: float | None = None
    active_speech_reference_0_sec: float | None = None
    extra: dict = field(default_factory=dict)


class SynthesisRequestLogger:
    """Write and prune synthesis debug dumps under ``root``."""

    def __init__(self, root: Path, *, keep: int = 40, enabled: bool = True) -> None:
        self.root = Path(root)
        self.keep = max(1, int(keep))
        self.enabled = bool(enabled)

    def log(
        self,
        *,
        text_raw: str,
        text_prepared: str,
        chunks: list[str],
        language: str,
        match_timing: bool,
        reference_paths: list[Path],
        reference_texts: list[str],
        raw_audio: np.ndarray,
        final_audio: np.ndarray,
        sample_rate: int,
        extra: dict | None = None,
    ) -> Path | None:
        """Persist one request. Returns the request directory, or ``None`` if disabled."""
        if not self.enabled:
            return None
        created = datetime.now(timezone.utc)
        request_id = created.strftime("%Y%m%dT%H%M%S%fZ")
        dest = self.root / request_id

        with _LOCK:
            dest.mkdir(parents=True, exist_ok=False)
            try:
                _write_wav(dest / "synth_raw.wav", raw_audio, sample_rate)
                _write_wav(dest / "synth_final.wav", final_audio, sample_rate)
                for index, path in enumerate(reference_paths):
                    if path.is_file():
                        shutil.copy2(path, dest / f"reference_{index}.wav")

                duration_ref = None
                if reference_paths and reference_paths[0].is_file():
                    duration_ref = _wav_duration_sec(reference_paths[0])

                record = SynthesisLogRecord(
                    request_id=request_id,
                    created_at=created.isoformat(),
                    language=language,
                    match_timing=match_timing,
                    text_raw=text_raw,
                    text_prepared=text_prepared,
                    chunks=list(chunks),
                    reference_texts=list(reference_texts),
                    sample_rate=sample_rate,
                    duration_raw_sec=_audio_duration_sec(raw_audio, sample_rate),
                    duration_final_sec=_audio_duration_sec(final_audio, sample_rate),
                    duration_reference_0_sec=duration_ref,
                    extra=dict(extra or {}),
                )
                # Optional active-speech metrics when timing helpers are available.
                try:
                    from fish_studio.timing import measure_active_speech_sec

                    record.active_speech_raw_sec = measure_active_speech_sec(
                        raw_audio, sample_rate
                    )
                    record.active_speech_final_sec = measure_active_speech_sec(
                        final_audio, sample_rate
                    )
                    if reference_paths and reference_paths[0].is_file():
                        ref, ref_sr = sf.read(
                            str(reference_paths[0]), dtype="float32", always_2d=False
                        )
                        record.active_speech_reference_0_sec = measure_active_speech_sec(
                            np.asarray(ref, dtype=np.float32), int(ref_sr)
                        )
                except Exception:
                    pass

                (dest / "request.json").write_text(
                    json.dumps(asdict(record), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                (self.root / _LATEST_NAME).write_text(request_id + "\n", encoding="utf-8")
                self._prune_locked()
            except Exception:
                shutil.rmtree(dest, ignore_errors=True)
                raise
        return dest

    def latest_dir(self) -> Path | None:
        latest = self.root / _LATEST_NAME
        if not latest.is_file():
            return None
        request_id = latest.read_text(encoding="utf-8").strip()
        if not request_id:
            return None
        path = self.root / request_id
        return path if path.is_dir() else None

    def _prune_locked(self) -> None:
        if not self.root.is_dir():
            return
        dirs = sorted(
            (path for path in self.root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        for stale in dirs[: max(0, len(dirs) - self.keep)]:
            shutil.rmtree(stale, ignore_errors=True)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, format="WAV")


def _audio_duration_sec(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    samples = np.asarray(audio)
    if samples.ndim == 2:
        return float(samples.shape[0] / sample_rate)
    return float(samples.size / sample_rate)


def _wav_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames / info.samplerate)
