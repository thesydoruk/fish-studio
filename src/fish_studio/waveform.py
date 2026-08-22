"""Waveform helpers for synthesis post-processing."""

from __future__ import annotations

import math

import numpy as np

# Only undo a clear downward trend; natural phrase-final drop is steeper but short.
_MIN_SLOPE_DB_S = -0.2
_MAX_GAIN = 4.0
_WINDOW_MS = 50.0
_EDGE_TRIM_MS = 200.0
_SPEECH_PEAK_RATIO = 0.08
_MIN_SPEECH_WINDOWS = 8
_MIN_DURATION_SEC = 1.5
_CROSSFADE_MS = 40.0
_SILENCE_PEAK_RATIO = 0.02


def compensate_fade(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Undo a slow, systematic RMS decay without flattening natural dynamics.

    Fish Speech (especially fine-tuned s2-pro) often loses energy toward the
    end of a long generation. A linear fit on log-RMS of speech windows is
    inverted so later samples are boosted; silence and short clips are left
    alone.
    """
    audio, leading = _as_2d(samples)
    n_samples = audio.shape[0]
    if sample_rate <= 0 or n_samples < int(sample_rate * _MIN_DURATION_SEC):
        return samples

    hop = max(1, int(sample_rate * _WINDOW_MS / 1000.0))
    rms = _window_rms(audio, hop)
    peak = float(np.max(rms)) if rms.size else 0.0
    if peak < 1e-6:
        return samples

    speech = rms >= peak * _SPEECH_PEAK_RATIO
    edge = max(1, int((_EDGE_TRIM_MS / _WINDOW_MS)))
    if rms.size > 2 * edge:
        # Onset/offset windows would flatten the slope fit toward zero.
        speech[:edge] = False
        speech[-edge:] = False
    if int(np.count_nonzero(speech)) < _MIN_SPEECH_WINDOWS:
        return samples

    times = (np.arange(rms.size, dtype=np.float64) + 0.5) * hop / sample_rate
    x = times[speech]
    y = np.log(np.maximum(rms[speech], 1e-8))
    slope, _intercept = np.polyfit(x, y, 1)
    slope_db_s = float(slope) * (20.0 / math.log(10.0))
    if slope_db_s > _MIN_SLOPE_DB_S:
        return samples

    t0 = float(x[0])
    sample_times = np.arange(n_samples, dtype=np.float64) / sample_rate
    max_gain_db = 20.0 * math.log10(_MAX_GAIN)
    gain_db = np.clip(-slope_db_s * (sample_times - t0), 0.0, max_gain_db)
    gain = np.power(10.0, gain_db / 20.0).astype(audio.dtype, copy=False)

    silence_thr = peak * _SILENCE_PEAK_RATIO
    window_gain = np.ones(rms.size, dtype=np.float64)
    for index, energy in enumerate(rms):
        if energy >= silence_thr:
            # Leave pauses unboosted so room tone does not swell with the fade fix.
            window_gain[index] = float(gain[min(n_samples - 1, index * hop)])
    window_gain = _smooth(window_gain, radius=2)
    sample_gain = np.repeat(window_gain, hop)[:n_samples]
    if sample_gain.size < n_samples:
        sample_gain = np.pad(sample_gain, (0, n_samples - sample_gain.size), mode="edge")

    boosted = audio * sample_gain[:, None]
    peak_out = float(np.max(np.abs(boosted)))
    if peak_out > 0.99:
        # Prevent the fade fix from introducing clips before loudness matching.
        boosted *= 0.99 / peak_out
    return _restore_shape(boosted, leading, samples.shape)


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


def _window_rms(audio: np.ndarray, hop: int) -> np.ndarray:
    n_windows = int(math.ceil(audio.shape[0] / hop))
    rms = np.empty(n_windows, dtype=np.float64)
    mono = audio.mean(axis=1)
    for index in range(n_windows):
        start = index * hop
        chunk = mono[start : start + hop]
        rms[index] = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
    return rms


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size == 0:
        return values
    kernel = np.ones(2 * radius + 1, dtype=np.float64)
    kernel /= kernel.size
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _crossfade(left: np.ndarray, right: np.ndarray, overlap: int) -> np.ndarray:
    overlap = min(overlap, left.shape[0], right.shape[0])
    if overlap <= 0:
        return np.concatenate([left, right], axis=0)
    # Equal-power ramps keep perceived loudness flat across the join.
    fade_out = np.sqrt(np.linspace(1.0, 0.0, overlap, dtype=np.float32))[:, None]
    fade_in = np.sqrt(np.linspace(0.0, 1.0, overlap, dtype=np.float32))[:, None]
    mixed = left[-overlap:] * fade_out + right[:overlap] * fade_in
    return np.concatenate([left[:-overlap], mixed, right[overlap:]], axis=0)
