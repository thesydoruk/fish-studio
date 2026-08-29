"""ffmpeg silence trimming and loudness normalization for exported clip WAVs."""

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


def build_silence_trim_filter(config: SegmentationConfig) -> str | None:
    """ffmpeg filter that strips leading and trailing silence, keeping a short margin.

    Both passes trim only the *start* of the stream — the second one runs on the
    reversed signal — so pauses inside an utterance are never touched.

    ``areverse`` has to buffer to end-of-stream, which only arrives if ffmpeg reads
    the input to its natural end. Give it a seeked span of a compressed container
    (``-ss X -to Y -i clip.webm``) and it never terminates, so this belongs only on
    call sites that feed a whole decoded stream. ``stop_periods`` is not a way
    around that: it also eats the pauses between sentences.
    """
    if not config.trim_silence:
        return None

    trim = (
        "silenceremove="
        "start_periods=1:"
        f"start_silence={config.trim_silence_keep_sec}:"
        f"start_threshold={config.trim_silence_db}dB:"
        "detection=rms"
    )
    return f"{trim},areverse,{trim},areverse"


def ffmpeg_output_args(config: SegmentationConfig, *, trim_silence: bool = True) -> list[str]:
    """Mono PCM args; trim runs before loudnorm so loudness is measured on speech only.

    Pass ``trim_silence=False`` when ffmpeg reads a seeked span rather than a whole
    stream — see :func:`build_silence_trim_filter` for why the filter cannot finish
    there.
    """
    args = ["-ac", "1", "-ar", str(config.sample_rate), "-c:a", "pcm_s16le"]
    trim = build_silence_trim_filter(config) if trim_silence else None
    filters = [chain for chain in (trim, build_loudness_filter(config)) if chain]
    if filters:
        args = ["-af", ",".join(filters), *args]
    return args
