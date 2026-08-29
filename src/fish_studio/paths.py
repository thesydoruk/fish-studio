"""Central path resolution — all runtime artifacts live under config paths.data_root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def resolve_data_root(config_path: Path, data_root: str) -> Path:
    """Resolve data_root relative to the config file directory."""
    root = Path(data_root)
    if root.is_absolute():
        return root
    return (config_path.parent / root).resolve()


def resolve_under_data_root(data_root: Path, path: str | Path) -> Path:
    """Resolve a config path relative to data_root. Absolute paths are unchanged."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (data_root / candidate).resolve()


@dataclass(frozen=True)
class SourcePaths:
    """Per-source scratch dirs under ``data/work/<slug>/``."""

    work_dir: Path
    downloads_dir: Path
    transcripts_dir: Path
    segments_dir: Path


@dataclass(frozen=True)
class WorkspacePaths:
    """All filesystem writes are rooted here."""

    data_root: Path

    @property
    def work_root(self) -> Path:
        return self.data_root / "work"

    @property
    def datasets_root(self) -> Path:
        return self.data_root / "datasets"

    @property
    def checkpoints_dir(self) -> Path:
        return self.data_root / "checkpoints"

    @property
    def training_dir(self) -> Path:
        return self.data_root / "training"

    @property
    def logs_dir(self) -> Path:
        return self.data_root / "logs"

    def source_paths(self, slug: str) -> SourcePaths:
        base = self.work_root / slug
        return SourcePaths(
            work_dir=base,
            downloads_dir=base / "downloads",
            transcripts_dir=base / "transcripts",
            segments_dir=base / "segments",
        )

    def dataset_dir(self, slug: str) -> Path:
        return self.datasets_root / slug

    def ensure_layout(self) -> None:
        for path in (
            self.work_root,
            self.datasets_root,
            self.checkpoints_dir,
            self.training_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
