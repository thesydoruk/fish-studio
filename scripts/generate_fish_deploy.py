#!/usr/bin/env python3
"""Generate vLLM deploy YAML with concurrency from .env."""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to generate the vLLM deploy config") from exc

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise SystemExit("python-dotenv is required to generate the vLLM deploy config") from exc

try:
    from fish_studio.runtime.fish_deploy import patch_deploy_concurrency
except ImportError as exc:
    raise SystemExit("fish_studio is required to generate the vLLM deploy config") from exc

ROOT = Path(__file__).resolve().parents[1]


def _load_env(env_path: Path) -> None:
    if env_path.is_file():
        load_dotenv(env_path)
    elif not os.environ.get("DATA_ROOT") and not os.environ.get("FISH_SPEECH_BASE_URL"):
        raise FileNotFoundError(f"Config not found: {env_path}. Copy .env.example to .env")


def _resolve_data_root(anchor: Path) -> Path:
    data_root = Path(os.environ.get("DATA_ROOT", "./data"))
    if not data_root.is_absolute():
        data_root = (anchor / data_root).resolve()
    return data_root


def _fish_max_concurrent() -> int:
    return max(int(os.environ.get("FISH_SPEECH_MAX_CONCURRENT_REQUESTS", "6")), 1)


def generate(env_path: Path) -> Path:
    _load_env(env_path)
    anchor = env_path.resolve().parent if env_path.is_file() else Path.cwd()
    template = ROOT / "configs" / "fish_speech_deploy.yaml"
    output = _resolve_data_root(anchor) / "work" / "fish_speech_deploy.generated.yaml"
    deploy = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    patch_deploy_concurrency(deploy, max_concurrent_requests=_fish_max_concurrent())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(deploy, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output


def main() -> int:
    env_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".env"
    print(generate(env_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
