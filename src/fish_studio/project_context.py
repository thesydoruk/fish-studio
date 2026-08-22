"""Load project config for training and inference scripts."""

from __future__ import annotations

from pathlib import Path

from fish_studio.config import ProjectConfig, load_config
from fish_studio.paths import WorkspacePaths

DEFAULT_CONFIG_PATH = ".env"


def try_load_project(config_path: str | Path | None = None) -> ProjectConfig | None:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.is_file():
        return None
    return load_config(path)


def workspace_or_default(config_path: str | Path | None = None) -> WorkspacePaths:
    """Use the project data_root when ``.env`` exists, otherwise ``./data``."""
    project = try_load_project(config_path)
    if project is not None:
        return project.workspace()
    return WorkspacePaths((Path.cwd() / "data").resolve())
