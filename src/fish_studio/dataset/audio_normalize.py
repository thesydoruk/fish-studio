"""ffmpeg loudness normalization for exported clip WAVs."""

from __future__ import annotations

from fish_studio.config import SegmentationConfig


def build_loudness_filter(config: SegmentationConfig) -> str | None:
    """ffmpeg ``loudnorm`` filter, or None when clip-level normalization is disabled."""
    if not config.normalize_loudness:
        return None

    return (
        f"loudnorm=I={config.target_loudness_lufs}:"
        f"TP={config.true_peak_db}:"
        f"LRA={config.loudness_range}"
    )


def ffmpeg_output_args(config: SegmentationConfig) -> list[str]:
    """Mono PCM args, with loudnorm first so resampling sees the normalized signal."""
    args = ["-ac", "1", "-ar", str(config.sample_rate), "-c:a", "pcm_s16le"]
    loudness = build_loudness_filter(config)
    if loudness:
        args = ["-af", loudness, *args]
    return args
