"""Helpers for Fish speaker_name values derived from diarization IDs."""

from __future__ import annotations

import re


def sanitize_speaker_token(value: str) -> str:
    """Keep only filesystem-safe characters so speaker folders stay portable."""
    token = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return token.strip("_") or "speaker"


def resolve_speaker_name(
    base: str,
    speaker_id: str | None,
    *,
    video_id: str,
) -> str:
    """Map channel base + video + diarization id to a *local* Fish speaker name.

    Names are video-scoped so ``spk_0`` on two different videos never collide by
    string alone. Cross-video identity is applied later via ``speaker_map.json``.
    """
    base_name = sanitize_speaker_token(base) if base.strip() else "speaker"
    video_token = sanitize_speaker_token(video_id) if video_id.strip() else "video"
    if not speaker_id or not str(speaker_id).strip():
        return f"{base_name}__{video_token}"
    return f"{base_name}__{video_token}__{sanitize_speaker_token(str(speaker_id))}"


def global_speaker_name(base: str, cluster_index: int) -> str:
    """Stable channel-level id after embedding clustering (``{base}_s{k}``)."""
    base_name = sanitize_speaker_token(base) if base.strip() else "speaker"
    return f"{base_name}_s{int(cluster_index)}"
