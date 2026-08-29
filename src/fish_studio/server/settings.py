"""HTTP server settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fish_studio.config import ProjectConfig, StressConfig


@dataclass(frozen=True)
class FishSpeechSettings:
    """Connection settings for the external vLLM-Omni Fish Speech server."""

    base_url: str
    voice: str
    timeout_sec: float
    max_new_tokens: int
    chunk_length: int
    max_concurrent_requests: int
    default_reference_text: str
    default_language: str
    stress: StressConfig = field(default_factory=StressConfig)
    synth_log_dir: Path | None = None
    synth_log_keep: int = 40
    synth_log_enabled: bool = True

    @classmethod
    def from_project(cls, project: ProjectConfig) -> FishSpeechSettings:
        fish = project.fish_speech
        log_dir = project.workspace().logs_dir / "synthesis"
        return cls(
            base_url=fish.base_url.rstrip("/"),
            voice=fish.voice,
            timeout_sec=fish.timeout_sec,
            max_new_tokens=fish.max_new_tokens,
            chunk_length=fish.chunk_length,
            max_concurrent_requests=max(1, int(fish.max_concurrent_requests)),
            default_reference_text=fish.default_reference_text,
            default_language=project.inference.language,
            stress=project.stress,
            synth_log_dir=log_dir if fish.synth_log else None,
            synth_log_keep=max(1, int(fish.synth_log_keep)),
            synth_log_enabled=bool(fish.synth_log),
        )


@dataclass(frozen=True)
class ServerSettings:
    """Bind address plus the vLLM connection used by :class:`EngineManager`."""

    fish_speech: FishSpeechSettings
    host: str
    port: int
    max_upload_bytes: int
    default_language: str

    @classmethod
    def from_project(cls, project: ProjectConfig) -> ServerSettings:
        inf = project.inference
        return cls(
            fish_speech=FishSpeechSettings.from_project(project),
            host=inf.host,
            port=inf.port,
            max_upload_bytes=inf.max_upload_bytes,
            default_language=inf.language,
        )
