"""Generate vLLM-Omni deploy YAML with concurrency from .env."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Codec streams are activation-heavy, so the stage-1 VRAM budget in
# configs/fish_speech_deploy.yaml is what limits this, not the AR stage.
CODEC_MAX_NUM_SEQS = 6


def codec_max_num_seqs(max_concurrent_requests: int) -> int:
    """Codec stage can decode a few streams in parallel without much VRAM."""
    return min(max(max_concurrent_requests, 1), CODEC_MAX_NUM_SEQS)


def patch_deploy_concurrency(
    deploy: dict[str, Any],
    *,
    max_concurrent_requests: int,
) -> dict[str, Any]:
    ar_max = max(max_concurrent_requests, 1)
    codec_max = codec_max_num_seqs(max_concurrent_requests)
    for stage in deploy.get("stages") or []:
        stage_id = stage.get("stage_id")
        if stage_id == 0:
            stage["max_num_seqs"] = ar_max
        elif stage_id == 1:
            stage["max_num_seqs"] = codec_max
    return deploy


def generate_fish_deploy_config(
    template_path: Path,
    output_path: Path,
    *,
    max_concurrent_requests: int,
) -> Path:
    deploy = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    patch_deploy_concurrency(deploy, max_concurrent_requests=max_concurrent_requests)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(deploy, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path
