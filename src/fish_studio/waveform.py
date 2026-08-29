"""Waveform helpers for synthesis post-processing."""

from __future__ import annotations

import numpy as np

_CROSSFADE_MS = 40.0


def concat_audio_chunks(
    pieces: list[np.ndarray],
    sample_rate: int,
    *,
    crossfade_ms: float = _CROSSFADE_MS,
) -> np.ndarray:
    """Join synthesized chunks with a short equal-power crossfade."""
    if not pieces:
        raise ValueError("no audio chunks to concatenate")
    if len(pieces) == 1:
        return pieces[0]

    current, _leading = _as_2d(pieces[0])
    overlap = max(1, int(sample_rate * crossfade_ms / 1000.0))
    for piece in pieces[1:]:
        nxt, _ = _as_2d(piece)
        current = _crossfade(current, nxt, overlap)
    return _restore_shape(current, pieces[0].ndim == 1, pieces[0].shape)


def _as_2d(samples: np.ndarray) -> tuple[np.ndarray, bool]:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return audio[:, None], True
    if audio.ndim == 2:
        return audio, False
    raise ValueError(f"expected 1-D or 2-D audio, got shape {audio.shape}")


def _restore_shape(audio: np.ndarray, leading: bool, original_shape: tuple[int, ...]) -> np.ndarray:
    if leading or len(original_shape) == 1:
        return audio[:, 0]
    return audio


def _crossfade(left: np.ndarray, right: np.ndarray, overlap: int) -> np.ndarray:
    overlap = min(overlap, left.shape[0], right.shape[0])
    if overlap <= 0:
        return np.concatenate([left, right], axis=0)
    # Equal-power ramps keep perceived level flat across the join.
    fade_out = np.sqrt(np.linspace(1.0, 0.0, overlap, dtype=np.float32))[:, None]
    fade_in = np.sqrt(np.linspace(0.0, 1.0, overlap, dtype=np.float32))[:, None]
    mixed = left[-overlap:] * fade_out + right[:overlap] * fade_in
    return np.concatenate([left[:-overlap], mixed, right[overlap:]], axis=0)
