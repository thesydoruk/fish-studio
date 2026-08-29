"""Match dubbed loudness to the original clip with a single linear gain.

After timing, the take is scaled so speech-gated K-weighted loudness
(ITU-R BS.1770) matches the first reference. That is one multiply — no
compressor, limiter, or EQ — so the waveform is not coloured. Gain is
capped so the peak stays under −0.1 dBFS instead of clipping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.signal import lfilter as _lfilter
except ImportError:
    _lfilter = None

_WINDOW_MS = 50.0
_SPEECH_FLOOR_RATIO = 0.1
_MIN_SPEECH_SEC = 0.15
_PEAK_EPS = 1.0 / 32_768
# Stay a hair under full scale so reconstruction does not clip on the DAC.
_PEAK_CEILING = 10 ** (-0.1 / 20.0)
# BS.1770 offset: LUFS = -0.691 + 10 log10(mean square of K-weighted signal).
_LUFS_OFFSET = -0.691


@dataclass(frozen=True)
class LoudnessFit:
    """Linear-gain result. ``audio`` is the scaled take."""

    audio: np.ndarray
    gain: float = 1.0
    gain_db: float = 0.0
    lufs_reference: float | None = None
    lufs_before: float | None = None
    lufs_after: float | None = None
    peak_limited: bool = False
    skip_reason: str | None = None

    def metrics(self) -> dict[str, float | bool | str | None]:
        return {
            "gain": round(self.gain, 6),
            "gain_db": round(self.gain_db, 3),
            "lufs_reference": _round_or_none(self.lufs_reference),
            "lufs_before": _round_or_none(self.lufs_before),
            "lufs_after": _round_or_none(self.lufs_after),
            "peak_limited": self.peak_limited,
            "skip_reason": self.skip_reason,
        }


def match_loudness_to_reference(
    samples: np.ndarray,
    sample_rate: int,
    reference: np.ndarray,
    reference_rate: int,
) -> LoudnessFit:
    """Scale ``samples`` so speech loudness matches ``reference``."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0:
        return LoudnessFit(audio=audio, skip_reason="empty_audio")
    ref = np.asarray(reference, dtype=np.float32)
    if ref.size == 0:
        return LoudnessFit(audio=audio, skip_reason="empty_reference")

    lufs_ref = _speech_lufs(ref, reference_rate)
    if lufs_ref is None:
        return LoudnessFit(audio=audio, skip_reason="no_speech_reference")
    lufs_before = _speech_lufs(audio, sample_rate)
    if lufs_before is None:
        return LoudnessFit(
            audio=audio,
            lufs_reference=lufs_ref,
            skip_reason="no_speech_synth",
        )

    gain = 10 ** ((lufs_ref - lufs_before) / 20.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    peak_limited = False
    if peak > _PEAK_EPS and peak * gain > _PEAK_CEILING:
        gain = _PEAK_CEILING / peak
        peak_limited = True

    scaled = (audio.astype(np.float64) * gain).astype(np.float32)
    lufs_after = _speech_lufs(scaled, sample_rate)
    return LoudnessFit(
        audio=scaled,
        gain=float(gain),
        gain_db=20.0 * math.log10(max(gain, 1e-12)),
        lufs_reference=lufs_ref,
        lufs_before=lufs_before,
        lufs_after=lufs_after,
        peak_limited=peak_limited,
    )


def speech_lufs(samples: np.ndarray, sample_rate: int) -> float | None:
    """Speech-gated K-weighted loudness of one clip, or ``None`` if it has too little speech."""
    return _speech_lufs(samples, sample_rate)


def scale_to_speech_lufs(
    samples: np.ndarray,
    sample_rate: int,
    target_lufs: float,
) -> np.ndarray:
    """Scale one clip so its speech loudness lands on ``target_lufs``, without clipping."""
    audio = np.asarray(samples, dtype=np.float32)
    lufs = _speech_lufs(audio, sample_rate)
    if lufs is None:
        return audio
    gain = 10 ** ((target_lufs - lufs) / 20.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > _PEAK_EPS and peak * gain > _PEAK_CEILING:
        gain = _PEAK_CEILING / peak
    return (audio.astype(np.float64) * gain).astype(np.float32)


def _speech_lufs(samples: np.ndarray, sample_rate: int) -> float | None:
    """K-weighted LUFS over active-speech windows, or None if too little speech."""
    if sample_rate <= 0:
        return None
    mono = _mono(samples)
    mask = _speech_mask(mono, sample_rate)
    if int(mask.sum()) < int(sample_rate * _MIN_SPEECH_SEC):
        return None
    weighted = _k_weight(mono, sample_rate)
    speech = weighted[mask]
    mean_square = float(np.mean(np.square(speech, dtype=np.float64)))
    if mean_square <= 0.0:
        return None
    return _LUFS_OFFSET + 10.0 * math.log10(mean_square)


def _speech_mask(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """True on windows whose RMS is at least a fraction of the peak window."""
    hop = max(1, int(sample_rate * _WINDOW_MS / 1000.0))
    n_windows = int(math.ceil(mono.size / hop))
    mask = np.zeros(mono.size, dtype=bool)
    if n_windows == 0:
        return mask
    rms = np.empty(n_windows, dtype=np.float64)
    for index in range(n_windows):
        start = index * hop
        chunk = mono[start : start + hop]
        energy = float(np.dot(chunk, chunk)) if chunk.size else 0.0
        rms[index] = math.sqrt(energy / max(1, chunk.size))
    peak = float(rms.max()) if rms.size else 0.0
    if peak < _PEAK_EPS:
        return mask
    floor = peak * _SPEECH_FLOOR_RATIO
    for index in range(n_windows):
        if rms[index] >= floor:
            start = index * hop
            mask[start : start + hop] = True
    return mask


def _k_weight(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """BS.1770 K-weighting: high shelf, then high-pass."""
    x = mono.astype(np.float64, copy=False)
    x = _sos_filter(x, *_high_shelf(sample_rate))
    return _sos_filter(x, *_high_pass(sample_rate))


def _high_shelf(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    # Coefficients from ITU-R BS.1770-4 / EBU R128 (pyloudnorm).
    f0 = 1681.974450955533
    gain_db = 3.999843853973347
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / sample_rate)
    vh = 10 ** (gain_db / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b = np.array(
        [
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
        ],
        dtype=np.float64,
    )
    a = np.array(
        [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        dtype=np.float64,
    )
    return b, a


def _high_pass(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    f0 = 38.13547087602444
    q = 0.5003270373238773
    k = math.tan(math.pi * f0 / sample_rate)
    denom = 1.0 + k / q + k * k
    b = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    a = np.array(
        [1.0, 2.0 * (k * k - 1.0) / denom, (1.0 - k / q + k * k) / denom],
        dtype=np.float64,
    )
    return b, a


def _sos_filter(samples: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Direct-form II transposed, one second-order section."""
    if _lfilter is not None:
        return np.asarray(_lfilter(b, a, samples), dtype=np.float64)
    out = np.empty_like(samples)
    w1 = 0.0
    w2 = 0.0
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a1, a2 = float(a[1]), float(a[2])
    for index, value in enumerate(samples):
        w0 = value - a1 * w1 - a2 * w2
        out[index] = b0 * w0 + b1 * w1 + b2 * w2
        w2 = w1
        w1 = w0
    return out


def _mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32, copy=False)
    raise ValueError(f"expected 1-D or 2-D audio, got shape {audio.shape}")


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 3)
