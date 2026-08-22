"""Project configuration loaded from environment variables (.env)."""

from __future__ import annotations

import json
import os
import re
import types
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from dotenv import load_dotenv

from fish_studio.paths import WorkspacePaths, resolve_data_root


def _slugify_source_id(source_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", source_id.strip()).strip("-").lower()
    if not slug:
        raise ValueError("source id must contain at least one letter or digit")
    return slug


@dataclass
class StorageConfig:
    """Root directory for all runtime artifacts (work, datasets, checkpoints, training)."""

    data_root: str = "./data"


@dataclass
class DataSourceConfig:
    """One audio source (YouTube channel/playlist or local folder)."""

    id: st
    kind: str = "youtube"  # youtube | local
    enabled: bool = True
    url: str = ""
    max_videos: int | None = None
    audio_format: str = "bestaudio/best"
    convert_on_download: bool = False
    min_video_duration_sec: int = 60
    max_video_duration_sec: int | None = None
    speaker_name: str | None = None  # overrides EXPORT_SPEAKER_NAME for this source


@dataclass
class YouTubeConfig:
    """Resolved YouTube options copied onto an :class:`AppConfig` for one source."""

    url: str = ""
    max_videos: int | None = None
    audio_format: str = "bestaudio/best"
    convert_on_download: bool = False
    min_video_duration_sec: int = 60
    max_video_duration_sec: int | None = None


@dataclass
class AudioIntelConfig:
    """Remote transcription service ([audio-intel](https://github.com/thesydoruk/audio-intel))."""

    base_url: str = "http://127.0.0.1:8081"
    language: str = "uk"
    align: bool = True  # word timestamps required for clip cuts
    diarize: bool = True
    sound_events: bool = True  # PANNs labels used by the junk filte
    timeout_sec: float = 7200.0
    drop_failed_segments: bool = True  # drop spans with no word alignment


@dataclass
class SegmentationConfig:
    """Clip length, merge, resample, and loudnorm settings for training WAVs."""

    min_duration_sec: float = 1.5
    max_duration_sec: float = 12.0
    target_duration_sec: float = 6.0
    min_speech_duration_sec: float = 0.5
    min_chars: int = 10
    max_chars: int = 220
    padding_sec: float = 0.05  # extra audio kept on each side of the aligned span
    sample_rate: int = 22050
    merge_gap_sec: float = 0.3  # join adjacent same-speaker segments across gaps this short
    filter_non_target_language: bool = True
    # Cut aligned words into uk/en runs so bilingual lessons become separate clips.
    split_by_script: bool = False
    # After script split, keep only these languages (ru and everything else is dropped).
    allowed_languages: list[str] = field(default_factory=lambda: ["uk", "en"])
    normalize_loudness: bool = True
    target_loudness_lufs: float = -18.0
    true_peak_db: float = -1.0
    loudness_range: float = 7.0


@dataclass
class QualityConfig:
    """Transcript and audio gates applied before a segment becomes a training clip."""

    min_avg_logprob: float = -1.0
    max_no_speech_prob: float = 0.6
    drop_compression_ratio_outliers: bool = True
    max_compression_ratio: float = 2.4  # Whisper repetition / hallucination proxy
    min_avg_word_score: float = 0.5
    min_word_score: float = 0.25
    max_low_score_word_ratio: float = 0.35
    filter_audio_quality: bool = True
    min_mean_volume_db: float = -40.0
    max_mean_volume_db: float = -8.0
    max_peak_volume_db: float = -1.0
    # Non-speech junk (music/noise/…) is always dropped when sound_events exist;
    # see fish_studio.dataset.junk_filter — not configurable.


@dataclass
class ExportConfig:
    """Pipe-delimited dataset layout written under ``datasets/<source>/``."""

    speaker_name: str = "speaker"
    eval_split_size: float = 0.05
    seed: int = 42
    reference_duration_sec: float = 8.0  # pick a clone clip near this length
    reference_sample_rate: int = 24000
    output_dir: str = ""  # filled from data_root + source slug


@dataclass
class PathsConfig:
    """Per-source work tree under ``data/work/<slug>/``. Empty strings are filled at resolve."""

    work_dir: str = ""
    downloads_dir: str = ""
    transcripts_dir: str = ""
    segments_dir: str = ""


@dataclass
class PipelineConfig:
    steps: list[str] = field(default_factory=lambda: ["all"])
    force: bool = False  # re-run steps even when outputs already exist
    num_workers: int = 6


@dataclass
class InferenceConfig:
    """HTTP TTS server bind options and defaults."""

    host: str = "0.0.0.0"
    port: int = 8080
    device: str = "cuda"
    language: str = "uk"
    max_upload_bytes: int = 15 * 1024 * 1024


@dataclass
class StressConfig:
    """Ukrainian stress marking applied to training text and synthesis input.

    The same settings must hold for both: the model learns to read the marks
    during fine-tuning, so marking only one side would reintroduce a
    train/inference mismatch.
    """

    enabled: bool = True
    # skip leaves heteronyms unmarked rather than guessing; see fish_studio.stress
    on_ambiguity: str = "skip"
    disambiguation: str = "dictionary"
    # Relative to the .env directory unless absolute. Empty disables the lexicon.
    lexicon_path: str = "configs/stress_lexicon.txt"
    # Keep Stanza on CPU so dataset prep / the API do not steal the TTS GPU.
    prefer_cpu: bool = True
    # When a clip WAV is available, mark remaining OOV / skipped words from energy.
    # Synthesis has no aligned audio, so this only affects dataset formation.
    acoustic_fallback: bool = True


@dataclass
class TrainingConfig:
    """Fish Speech s2-pro LoRA fine-tuning hyperparameters."""

    dataset_id: str = "combined"
    speaker_name: str = "speaker"
    project_name: str = "fish-uk"
    base_checkpoint: str = ""
    merged_checkpoint: str = ""
    max_steps: int = 10000
    batch_size: int = 2
    grad_accum: int = 1
    lr: float = 1e-4
    val_check_interval: int = 100
    # -1 keeps every checkpoint, so the best step can be picked after listening;
    # upstream keeps only the last few, which discards earlier candidates.
    save_top_k: int = -1
    lora_config: str = "fast_r_8_alpha_16"
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.01
    # Slow (text->semantic) targets drive pronunciation; fast targets shape acoustics.
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["fast_attention", "fast_mlp", "fast_embeddings", "fast_output"]
    )
    continue_path: str = ""
    vq_batch_size: int = 16
    vq_num_workers: int = 1
    proto_num_workers: int = 8
    proto_shard_size_mb: int = 10
    test_text: str = "Доброго дня! Вартість квитка 150 грн, знижка 10 відсотків."


@dataclass
class FishSpeechConfig:
    """Fish Speech serving (vLLM-Omni) and training checkpoint settings.

    Serving is always via an external vLLM process (``./run.sh vllm start``).
    ``model`` is relative to paths.data_root or a HuggingFace repo id.
    ``llama_checkpoint`` / decoder fields are used by LoRA training and CLI infer.
    """

    base_url: str = "http://127.0.0.1:8091"
    model: str = "checkpoints/fish-speech/s2-pro"
    gpu_memory_utilization: float = 0.72
    voice: str = "default"
    timeout_sec: float = 300.0
    llama_checkpoint: str = "checkpoints/fish-speech/s2-pro"
    decoder_checkpoint: str | None = None
    decoder_config_name: str = "modded_dac_vq"
    half: bool = True
    compile: bool = False
    chunk_length: int = 200  # chars per vLLM request; 0 disables splitting
    max_new_tokens: int = 0  # 0 = vLLM server default; training CLI uses runtime override
    max_concurrent_requests: int = 6
    default_reference_text: str = ""
    use_finetuned: bool = False


@dataclass
class SpeakerClusterConfig:
    """Cross-video speaker clustering within one SOURCES channel."""

    # Cosine similarity above this merges two video-local speakers.
    threshold: float = 0.75
    # After mapping, drop speakers that fail either floor (junk diarization tails).
    min_clips: int = 100
    min_speech_sec: float = 300.0


@dataclass
class AppConfig:
    """Runtime config for a single source — paths derived from data_root + source slug."""

    source: DataSourceConfig
    youtube: YouTubeConfig
    audio_intel: AudioIntelConfig = field(default_factory=AudioIntelConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    speaker_cluster: SpeakerClusterConfig = field(default_factory=SpeakerClusterConfig)

    @property
    def source_slug(self) -> str:
        return _slugify_source_id(self.source.id)

    def ensure_dirs(self) -> None:
        for path in (
            self.paths.work_dir,
            self.paths.downloads_dir,
            self.paths.transcripts_dir,
            self.paths.segments_dir,
            self.export.output_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)
        Path(self.export.output_dir, "wavs").mkdir(parents=True, exist_ok=True)


@dataclass
class ProjectConfig:
    """Top-level project config from .env: shared settings + list of sources."""

    sources: list[DataSourceConfig]
    storage: StorageConfig = field(default_factory=StorageConfig)
    audio_intel: AudioIntelConfig = field(default_factory=AudioIntelConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    fish_speech: FishSpeechConfig = field(default_factory=FishSpeechConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    speaker_cluster: SpeakerClusterConfig = field(default_factory=SpeakerClusterConfig)
    _config_path: Path = field(default=Path(".env"), repr=False)

    def workspace(self) -> WorkspacePaths:
        return WorkspacePaths(resolve_data_root(self._config_path, self.storage.data_root))

    def resolve_sources(
        self, source_ids: list[str] | tuple[str, ...] | None = None
    ) -> list[DataSourceConfig]:
        """Return enabled sources, optionally filtered by id (slug-matched)."""
        enabled = [s for s in self.sources if s.enabled]
        if not source_ids:
            return enabled
        wanted = {_slugify_source_id(sid) for sid in source_ids}
        selected = [s for s in enabled if _slugify_source_id(s.id) in wanted]
        missing = wanted - {_slugify_source_id(s.id) for s in selected}
        if missing:
            known = ", ".join(_slugify_source_id(s.id) for s in self.sources)
            raise ValueError(
                f"Unknown source id(s): {', '.join(sorted(missing))}. Configured: {known}"
            )
        return selected

    def app_config(self, source: DataSourceConfig) -> AppConfig:
        """Bind shared project settings to one source's work and dataset directories."""
        slug = _slugify_source_id(source.id)
        ws = self.workspace()
        sp = ws.source_paths(slug)
        speaker = source.speaker_name or self.export.speaker_name
        export = replace(
            self.export,
            speaker_name=speaker,
            output_dir=str(ws.dataset_dir(slug)),
        )
        paths = PathsConfig(
            work_dir=str(sp.work_dir),
            downloads_dir=str(sp.downloads_dir),
            transcripts_dir=str(sp.transcripts_dir),
            segments_dir=str(sp.segments_dir),
        )
        youtube = YouTubeConfig(
            url=source.url,
            max_videos=source.max_videos,
            audio_format=source.audio_format,
            convert_on_download=source.convert_on_download,
            min_video_duration_sec=source.min_video_duration_sec,
            max_video_duration_sec=source.max_video_duration_sec,
        )
        return AppConfig(
            source=source,
            youtube=youtube,
            audio_intel=self.audio_intel,
            segmentation=self.segmentation,
            quality=self.quality,
            export=export,
            paths=paths,
            pipeline=replace(self.pipeline),
            stress=self.stress,
            speaker_cluster=self.speaker_cluster,
        )


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _unwrap_optional(field_type: Any) -> Any:
    origin = get_origin(field_type)
    if origin in (types.UnionType, Union):
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        if args:
            return args[0]
    return field_type


def _parse_env_value(raw: str, field_type: Any) -> Any:
    if raw.strip().lower() in {"null", "none"}:
        return None

    origin = get_origin(field_type)
    if field_type is bool:
        return _parse_bool(raw)
    if origin is bool:
        return _parse_bool(raw)

    inner = _unwrap_optional(field_type)
    inner_origin = get_origin(inner)

    if inner is int:
        return int(raw)
    if inner is float:
        return float(raw)
    if inner_origin is list:
        if raw.strip().startswith("["):
            return json.loads(raw)
        item_type = get_args(inner)[0] if get_args(inner) else st
        return [
            _parse_env_value(part.strip(), item_type) for part in raw.split(",") if part.strip()
        ]
    return raw


def _dataclass_from_env(cls: type, prefix: str):
    # `from __future__ import annotations` makes field_def.type a string, which never
    # matches the real types in _parse_env_value, so every value would stay a str.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field_def in fields(cls):
        env_key = f"{prefix}_{field_def.name}".upper()
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        kwargs[field_def.name] = _parse_env_value(raw, hints[field_def.name])
    base = cls()
    merged = {field_def.name: getattr(base, field_def.name) for field_def in fields(cls)}
    merged.update(kwargs)
    return cls(**merged)


def _normalize_sources(entries: list[Any]) -> list[DataSourceConfig]:
    if not entries:
        return []

    sources: list[DataSourceConfig] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("each sources entry must be a mapping")
        source = DataSourceConfig(**entry)
        slug = _slugify_source_id(source.id)
        if slug in seen:
            raise ValueError(f"duplicate source id: {slug}")
        seen.add(slug)
        kind = source.kind.strip().lower()
        if kind not in {"youtube", "local"}:
            raise ValueError(f"source {slug}: kind must be youtube or local")
        if kind == "youtube" and not source.url.strip():
            raise ValueError(f"source {slug}: youtube kind requires url")
        sources.append(replace(source, kind=kind))
    return sources


def _sources_from_env() -> list[DataSourceConfig]:
    raw = os.environ.get("SOURCES", "[]").strip()
    if not raw:
        return []
    entries = json.loads(raw)
    if not isinstance(entries, list):
        raise TypeError("SOURCES must be a JSON array")
    return _normalize_sources(entries)


def _has_config_env() -> bool:
    return any(
        os.environ.get(key) for key in ("DATA_ROOT", "FISH_SPEECH_BASE_URL", "INFERENCE_PORT")
    )


def _load_dotenv(env_file: Path) -> Path:
    if env_file.is_file():
        load_dotenv(env_file)
        return env_file.resolve().parent
    if _has_config_env():
        return Path.cwd()
    raise FileNotFoundError(f"Config not found: {env_file}. Copy .env.example to .env")


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load ``.env`` into :class:`ProjectConfig`. Paths in the file are relative to its directory."""
    env_file = Path(path or ".env")
    anchor = _load_dotenv(env_file)

    fish_speech = _dataclass_from_env(FishSpeechConfig, "FISH_SPEECH")
    if os.environ.get("FISH_SPEECH_DECODER_CHECKPOINT", "").strip() == "":
        # Empty env means "use codec.pth next to the LLAMA checkpoint", not "".
        fish_speech = replace(fish_speech, decoder_checkpoint=None)

    stress = _dataclass_from_env(StressConfig, "STRESS")
    if stress.lexicon_path.strip():
        lexicon = Path(stress.lexicon_path)
        if not lexicon.is_absolute():
            # Resolve once so training export and synthesis share the same file.
            stress = replace(stress, lexicon_path=str((anchor / lexicon).resolve()))

    return ProjectConfig(
        sources=_sources_from_env(),
        storage=StorageConfig(data_root=os.environ.get("DATA_ROOT", "./data")),
        audio_intel=_dataclass_from_env(AudioIntelConfig, "AUDIO_INTEL"),
        segmentation=_dataclass_from_env(SegmentationConfig, "SEGMENTATION"),
        quality=_dataclass_from_env(QualityConfig, "QUALITY"),
        export=_dataclass_from_env(ExportConfig, "EXPORT"),
        pipeline=_dataclass_from_env(PipelineConfig, "PIPELINE"),
        training=_dataclass_from_env(TrainingConfig, "TRAINING"),
        inference=_dataclass_from_env(InferenceConfig, "INFERENCE"),
        fish_speech=fish_speech,
        stress=stress,
        speaker_cluster=_dataclass_from_env(SpeakerClusterConfig, "SPEAKER_CLUSTER"),
        _config_path=anchor / ".env",
    )


PIPELINE_STEPS = ("download", "transcribe", "segment", "cluster", "export")


def resolve_steps(steps: list[str]) -> list[str]:
    """Expand 'all' into the full pipeline."""
    if "all" in steps:
        return list(PIPELINE_STEPS)
    return [step for step in PIPELINE_STEPS if step in steps]
