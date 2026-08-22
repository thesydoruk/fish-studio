"""Import Hugging Face speech datasets into the pipe-delimited dataset format.

Fish Speech packs training prompts per speaker folder and samples the reference
clip from within that group, so every source voice is kept as its own speaker
instead of being flattened into a single one.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from fish_studio.config import ExportConfig, SegmentationConfig
from fish_studio.dataset.audio_normalize import ffmpeg_output_args

TEXT_FIELDS = ("transcription", "text", "sentence")

# Sample rates are reported for stats only, so a handful of rows per source is
# enough and keeps ffprobe out of the per-row path.
_RATE_SAMPLE_ROWS = 20


@dataclass(frozen=True)
class HFSource:
    """One Hugging Face dataset repo mapped to a single speaker."""

    repo_id: str
    speaker: str
    config_name: str | None = None
    split: str = "train"

    @property
    def label(self) -> str:
        return f"{self.repo_id}[{self.config_name}]" if self.config_name else self.repo_id


@dataclass
class SourceStats:
    speaker: str
    repo_id: str
    kept: int = 0
    skipped_duration: int = 0
    skipped_text: int = 0
    skipped_audio: int = 0
    skipped_wer: int = 0
    duration_sec: float = 0.0
    sample_rates: dict[int, int] = field(default_factory=dict)
    resumed: bool = False


@dataclass
class ImportStats:
    output_dir: str
    total_clips: int
    train_clips: int
    eval_clips: int
    duration_sec: float
    sources: list[SourceStats]


def parse_source(spec: str) -> HFSource:
    """Parse ``repo_id[:config][@split]=speaker`` into an :class:`HFSource`.

    A trailing ``#…`` suffix is accepted for backward compatibility with older
    clip-cap specs and ignored — imports always consume the full split.
    """
    repo_part, sep, speaker_part = spec.partition("=")
    if not sep or not speaker_part.strip():
        raise ValueError(f"source must look like repo_id=speaker, got: {spec}")

    speaker, _, _ignored_limit = speaker_part.partition("#")

    repo_part, _, split = repo_part.partition("@")
    repo_id, _, config_name = repo_part.partition(":")
    if not repo_id.strip():
        raise ValueError(f"source is missing a repo id: {spec}")

    return HFSource(
        repo_id=repo_id.strip(),
        speaker=_slugify(speaker),
        config_name=config_name.strip() or None,
        split=split.strip() or "train",
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    if not slug:
        raise ValueError(f"speaker name must contain a letter or digit: {value!r}")
    return slug


def clean_text(text: str) -> str:
    """Normalize whitespace and drop characters the pipe format cannot carry."""
    text = re.sub(r"\s+", " ", text.replace("|", " ")).strip()
    text = text.replace("...", ".")
    return re.sub(r"[^\w\s\u0400-\u04FF\u0490-\u0491.,!?;:\-—'\"()«»]", "", text).strip()


def _pick_text(row: dict[str, Any]) -> str:
    for key in TEXT_FIELDS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return clean_text(value)
    return ""


def _audio_bytes(row: dict[str, Any]) -> bytes | None:
    audio = row.get("audio")
    if isinstance(audio, dict):
        data = audio.get("bytes")
        if isinstance(data, bytes) and data:
            return data
        path = audio.get("path")
        if isinstance(path, str) and Path(path).is_file():
            return Path(path).read_bytes()
    return None


def probe_audio(data: bytes) -> tuple[int | None, float | None]:
    """Return ``(sample_rate, duration_sec)`` of an encoded audio blob."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate:format=duration",
            "-of",
            "json",
            "pipe:0",
        ],
        input=data,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None, None
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, None

    streams = info.get("streams") or [{}]
    rate = streams[0].get("sample_rate")
    duration = (info.get("format") or {}).get("duration")
    return (int(rate) if rate else None, float(duration) if duration else None)


def _write_wav(data: bytes, dest: Path, segmentation: SegmentationConfig) -> bool:
    cmd = ["ffmpeg", "-y", "-i", "pipe:0", *ffmpeg_output_args(segmentation), str(dest)]
    proc = subprocess.run(cmd, input=data, capture_output=True, check=False)
    return proc.returncode == 0 and dest.is_file()


def wav_duration(path: Path) -> float | None:
    """Duration of a local wav file, read from its header."""
    try:
        with closing(wave.open(str(path), "rb")) as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
    except (OSError, wave.Error):
        return None
    return frames / rate if rate else None


def _load_rows(source: HFSource, *, streaming: bool = False) -> Iterator[dict[str, Any]]:
    from datasets import Audio, load_dataset

    dataset = load_dataset(
        source.repo_id, source.config_name, split=source.split, streaming=streaming
    )
    column_names = dataset.column_names or []
    if "audio" in column_names:
        # Keep the encoded bytes so ffmpeg controls resampling, not `datasets`.
        dataset = dataset.cast_column("audio", Audio(decode=False))
    return iter(dataset)


@dataclass
class _Record:
    file_id: str
    text: str
    speaker: str
    duration: float


def probe_sources(
    sources: list[HFSource], limit: int = 20, *, streaming: bool = False
) -> dict[str, dict[str, Any]]:
    """Report sample rates, durations and sample texts for the first rows per source."""
    report: dict[str, dict[str, Any]] = {}
    for source in sources:
        rates: dict[int, int] = {}
        durations: list[float] = []
        fields: list[str] = []
        texts: list[str] = []
        for index, row in enumerate(_load_rows(source, streaming=streaming)):
            if index >= limit:
                break
            if index == 0:
                fields = sorted(row.keys())
            text = _pick_text(row)
            if text:
                texts.append(text)
            data = _audio_bytes(row)
            if data is None:
                continue
            rate, duration = probe_audio(data)
            if rate:
                rates[rate] = rates.get(rate, 0) + 1
            row_duration = row.get("duration")
            if isinstance(row_duration, (int, float)) and row_duration > 0:
                durations.append(float(row_duration))
            elif duration:
                durations.append(duration)
        report[source.label] = {
            "speaker": source.speaker,
            "fields": fields,
            "sample_rates": rates,
            "mean_duration_sec": round(sum(durations) / len(durations), 2) if durations else None,
            "texts": texts[:3],
        }
    return report


def import_sources(
    sources: list[HFSource],
    output_dir: Path,
    *,
    segmentation: SegmentationConfig,
    export: ExportConfig,
    max_wer: float | None = None,
    num_workers: int = 8,
    streaming: bool = False,
    batch_size: int = 512,
    force: bool = False,
    progress: Any | None = None,
) -> ImportStats:
    """Download, filter and convert HF datasets into ``output_dir``."""
    if not sources:
        raise ValueError("at least one source is required")

    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    records: list[_Record] = []
    stats: list[SourceStats] = []

    min_sec = segmentation.min_duration_sec
    max_sec = segmentation.max_duration_sec

    def convert(item: tuple[Path, bytes, _Record]) -> tuple[_Record | None, str]:
        # ffprobe cannot report duration for compressed audio on a non-seekable
        # pipe, so the converted wav is the only reliable place to measure it.
        dest, data, record = item
        if not _write_wav(data, dest, segmentation):
            return None, "audio"
        duration = wav_duration(dest)
        if duration is None or not (min_sec <= duration <= max_sec):
            dest.unlink(missing_ok=True)
            return None, "duration"
        record.duration = duration
        return record, "kept"

    for source in sources:
        # A finished source is checkpointed so a later crash costs one source,
        # not the whole multi-hour conversion.
        manifest = manifests_dir / f"{source.speaker}.json"
        if not force:
            restored = _read_manifest(manifest)
            if restored is not None:
                restored_stats, restored_records = restored
                records.extend(restored_records)
                stats.append(restored_stats)
                if progress is not None:
                    progress(restored_stats)
                continue

        source_stats = SourceStats(speaker=source.speaker, repo_id=source.repo_id)
        converted: list[_Record] = []
        seq = 1
        sampled_rates = 0
        pending: list[tuple[Path, bytes, _Record]] = []

        def flush() -> None:
            # Convert in batches so streaming huge datasets stays bounded in memory.
            nonlocal pending
            if not pending:
                return
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                for record, reason in pool.map(convert, pending):
                    if record is not None:
                        converted.append(record)
                    elif reason == "duration":
                        source_stats.skipped_duration += 1
                    else:
                        source_stats.skipped_audio += 1
            pending = []

        for row in _load_rows(source, streaming=streaming):
            # Datasets that ship an ASR quality proxy let us drop misaligned rows.
            wer = row.get("wer")
            if max_wer is not None and isinstance(wer, (int, float)) and wer > max_wer:
                source_stats.skipped_wer += 1
                continue

            text = _pick_text(row)
            if not text or not (segmentation.min_chars <= len(text) <= segmentation.max_chars):
                source_stats.skipped_text += 1
                continue

            data = _audio_bytes(row)
            if data is None:
                source_stats.skipped_audio += 1
                continue

            # A declared duration lets us reject before paying for a conversion.
            declared = row.get("duration")
            if isinstance(declared, (int, float)) and declared > 0:
                if not (min_sec <= declared <= max_sec):
                    source_stats.skipped_duration += 1
                    continue

            if sampled_rates < _RATE_SAMPLE_ROWS:
                sampled_rates += 1
                rate, _ = probe_audio(data)
                if rate:
                    source_stats.sample_rates[rate] = source_stats.sample_rates.get(rate, 0) + 1

            file_id = f"{source.speaker}-{seq:06d}"
            seq += 1
            record = _Record(
                file_id=file_id,
                text=text,
                speaker=source.speaker,
                duration=0.0,
            )
            pending.append((wavs_dir / f"{file_id}.wav", data, record))
            if len(pending) >= batch_size:
                flush()

        flush()

        records.extend(converted)
        source_stats.kept = len(converted)
        source_stats.duration_sec = sum(r.duration for r in converted)

        _write_manifest(manifest, source_stats, converted)
        stats.append(source_stats)
        if progress is not None:
            progress(source_stats)

    if not records:
        raise RuntimeError("no clips imported; all rows were filtered out")

    train_rows, eval_rows = _split_rows(records, export)
    _write_metadata(output_dir / "metadata_train.csv", train_rows)
    _write_metadata(output_dir / "metadata_eval.csv", eval_rows)
    _create_reference(records, wavs_dir, output_dir, export)

    total_duration = sum(r.duration for r in records)
    payload = {
        "total_clips": len(records),
        "train_clips": len(train_rows),
        "eval_clips": len(eval_rows),
        "total_duration_sec": round(total_duration, 2),
        "sample_rate": segmentation.sample_rate,
        "metadata_format": "pipe",
        "speakers": {
            s.speaker: {
                "repo_id": s.repo_id,
                "clips": s.kept,
                "duration_sec": round(s.duration_sec, 2),
                "source_sample_rates": s.sample_rates,
            }
            for s in stats
        },
    }
    (output_dir / "stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return ImportStats(
        output_dir=str(output_dir),
        total_clips=len(records),
        train_clips=len(train_rows),
        eval_clips=len(eval_rows),
        duration_sec=total_duration,
        sources=stats,
    )


def _split_rows(records: list[_Record], export: ExportConfig) -> tuple[list[_Record], list[_Record]]:
    shuffled = records[:]
    random.Random(export.seed).shuffle(shuffled)

    eval_n = 0
    if len(shuffled) > 1:
        eval_n = max(1, int(len(shuffled) * export.eval_split_size))
        eval_n = min(eval_n, len(shuffled) - 1)
    return shuffled[eval_n:], shuffled[:eval_n]


def _write_manifest(path: Path, stats: SourceStats, records: list[_Record]) -> None:
    payload = {"stats": asdict(stats), "records": [asdict(r) for r in records]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read_manifest(path: Path) -> tuple[SourceStats, list[_Record]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_stats = dict(payload["stats"])
        raw_stats["sample_rates"] = {int(k): v for k, v in raw_stats["sample_rates"].items()}
        stats = SourceStats(**raw_stats)
        stats.resumed = True
        return stats, [_Record(**r) for r in payload["records"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_metadata(path: Path, records: list[_Record]) -> None:
    lines = ["audio_file|text|speaker_name"]
    lines.extend(f"wavs/{r.file_id}.wav|{r.text}|{r.speaker}" for r in records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_reference(
    records: list[_Record], wavs_dir: Path, output_dir: Path, export: ExportConfig
) -> None:
    target = export.reference_duration_sec
    ranked = sorted(records, key=lambda r: abs(r.duration - target))
    for record in ranked[:10]:
        src = wavs_dir / f"{record.file_id}.wav"
        if not src.is_file():
            continue
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(export.reference_sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_dir / "reference.wav"),
        ]
        if subprocess.run(cmd, capture_output=True, check=False).returncode == 0:
            return
