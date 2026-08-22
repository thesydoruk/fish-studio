"""One loudness stage: light remaster, then match the source clip level.

Callers pass the first ``speaker_wav`` as the level target. Fade undo and the
light remaster live here so they are not separate pipeline steps.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from fish_studio.waveform import compensate_fade

# How far a transient may rise above the reference peak (+4 dB).
_PEAK_ALLOWANCE = 1.6
# Never come closer than ~0.2 dB to full scale (32000/32768).
_ABSOLUTE_CEILING = 32_000 / 32_768
# Largest correction applied to reach the reference speech level (+12 dB).
_MAX_GAIN = 4.0
# Stop the corrective pass once the level is within ~0.3 dB of the target.
_GAIN_TOLERANCE = 1.035
_WINDOW_MS = 50.0
_SPEECH_FLOOR_RATIO = 0.1
_KNEE_RATIO = 0.7
_PEAK_EPS = 1.0 / 32_768


def finish_synthesis_audio(
    samples: np.ndarray,
    sample_rate: int,
    reference: np.ndarray,
    reference_rate: int,
    *,
    synthesis_text: str = "",
    reference_text: str = "",
    match_loudness: bool = True,
    match_timing: bool = True,
) -> np.ndarray:
    """Optionally remaster+match loudness, then optionally fit the ``reference`` slot."""
    from fish_studio.timing import match_timing_to_reference

    audio = samples
    if match_loudness:
        audio = match_loudness_to_reference(audio, sample_rate, reference, reference_rate)
    if match_timing:
        audio = match_timing_to_reference(
            audio,
            sample_rate,
            reference,
            reference_rate,
            text_uk=synthesis_text,
            text_en=reference_text,
        )
    return audio


def load_loudness_reference(path: str | Path) -> tuple[np.ndarray, int]:
    """Load the first ``speaker_wav`` as the loudness (and later timing) target."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if not isinstance(audio, np.ndarray) or audio.size == 0:
        raise ValueError(f"loudness reference is empty: {path}")
    return audio, int(rate)


def match_loudness_to_reference(
    samples: np.ndarray,
    sample_rate: int,
    reference: np.ndarray,
    reference_rate: int,
) -> np.ndarray:
    """Light remaster, then match speech level to the source."""
    from fish_studio.enhance import remaster_speech

    audio = compensate_fade(samples, sample_rate)
    audio = remaster_speech(audio, sample_rate)
    return fit_level_to_reference(audio, sample_rate, reference, reference_rate)


def fit_level_to_reference(
    samples: np.ndarray,
    sample_rate: int,
    reference: np.ndarray,
    reference_rate: int,
) -> np.ndarray:
    """Scale speech RMS to the reference without changing tone or dynamics."""
    ref_peak = measure_peak(reference)
    ref_rms = measure_speech_rms(reference, reference_rate)
    if ref_rms >= _PEAK_EPS:
        ceiling = min(_ABSOLUTE_CEILING, ref_peak * _PEAK_ALLOWANCE)
        return match_speech_level(samples, sample_rate, ref_rms, ceiling)
    if ref_peak > 0:
        # Near-silent reference: fall back to peak match so we still have a scale.
        return match_peak_to_target(samples, ref_peak)
    return samples


def measure_peak(samples: np.ndarray) -> float:
    """Absolute peak of the waveform (all channels)."""
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def measure_speech_rms(samples: np.ndarray, sample_rate: int) -> float:
    """RMS of active speech windows, ignoring pauses between phrases."""
    mono = _mono(samples)
    if mono.size == 0 or sample_rate <= 0:
        return 0.0

    hop = max(1, int(sample_rate * _WINDOW_MS / 1000.0))
    n_windows = int(math.ceil(mono.size / hop))
    sum_sq = np.empty(n_windows, dtype=np.float64)
    counts = np.empty(n_windows, dtype=np.int32)
    rms = np.empty(n_windows, dtype=np.float64)
    peak_rms = 0.0
    for index in range(n_windows):
        chunk = mono[index * hop : index * hop + hop]
        energy = float(np.dot(chunk, chunk))
        sum_sq[index] = energy
        counts[index] = chunk.size
        value = math.sqrt(energy / max(1, chunk.size))
        rms[index] = value
        if value > peak_rms:
            peak_rms = value
    if peak_rms < _PEAK_EPS:
        return 0.0

    # Ignore pauses: matching whole-clip RMS would treat a slow line as quieter speech.
    floor = peak_rms * _SPEECH_FLOOR_RATIO
    active_sum = 0.0
    active_count = 0
    for index in range(n_windows):
        if rms[index] < floor:
            continue
        active_sum += sum_sq[index]
        active_count += int(counts[index])
    if active_count <= 0:
        return 0.0
    return math.sqrt(active_sum / active_count)


def apply_peak_ceiling(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Clamp peaks after a later transform without changing speech gain."""
    ref_peak = measure_peak(reference)
    if ref_peak < _PEAK_EPS:
        ceiling = _ABSOLUTE_CEILING
    else:
        ceiling = min(_ABSOLUTE_CEILING, ref_peak * _PEAK_ALLOWANCE)
    return scale_with_soft_ceiling(samples, 1.0, ceiling)


def match_peak_to_target(samples: np.ndarray, target_peak: float) -> np.ndarray:
    """Linear peak match used when the reference has no measurable speech RMS."""
    current = measure_peak(samples)
    safe_target = min(1.0, max(0.0, float(target_peak)))
    if current < _PEAK_EPS or safe_target < _PEAK_EPS:
        return samples
    scale = safe_target / current
    if abs(scale - 1.0) < 0.001:
        return samples
    return np.clip(samples.astype(np.float32, copy=False) * scale, -1.0, 1.0)


def match_speech_level(
    samples: np.ndarray,
    sample_rate: int,
    target_rms: float,
    ceiling: float,
) -> np.ndarray:
    """Apply speech-RMS gain, then a second pass if the soft ceiling held the first one back."""
    current = measure_speech_rms(samples, sample_rate)
    if current < _PEAK_EPS:
        return samples
    gain = min(_MAX_GAIN, target_rms / current)
    matched = scale_with_soft_ceiling(samples, gain, ceiling)
    matched_rms = measure_speech_rms(matched, sample_rate)
    if matched_rms < _PEAK_EPS or target_rms / matched_rms < _GAIN_TOLERANCE:
        return matched
    return scale_with_soft_ceiling(matched, min(_MAX_GAIN, target_rms / matched_rms), ceiling)


def scale_with_soft_ceiling(samples: np.ndarray, gain: float, ceiling: float) -> np.ndarray:
    """Scale by ``gain``, rounding off anything that would pass ``ceiling``."""
    audio = np.asarray(samples, dtype=np.float32)
    limit = min(1.0, max(_PEAK_EPS, float(ceiling)))
    # Below the knee the scale is linear; above it tanh rounds off so transients do not clip hard.
    knee = limit * _KNEE_RATIO
    span = limit - knee
    scaled = audio * gain
    magnitude = np.abs(scaled)
    if span <= 0:
        shaped = np.minimum(magnitude, limit)
    else:
        excess = np.maximum(magnitude - knee, 0.0)
        compressed = knee + span * np.tanh(excess / span)
        shaped = np.where(magnitude <= knee, magnitude, compressed)
    signed = np.sign(scaled) * shaped
    return signed.astype(np.float32, copy=False)


def _mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1)
    raise ValueError(f"expected 1-D or 2-D audio, got shape {audio.shape}")
