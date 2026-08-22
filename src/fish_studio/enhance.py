"""Light remaster used by the loudness stage.

This is not a separate post-process. ``match_loudness_to_reference`` runs it
first (after undoing a slow fade), then fits speech level to the source clip.
Island flatten already fixes the real level swings; compression here only
tames band imbalance and peaks so the line stays natural.
"""

from __future__ import annotations

import math

import numpy as np

_HPF_HZ = 80.0
_CROSSOVERS_HZ = (250.0, 800.0, 2500.0, 6000.0)
# Keep body/mid slam; presence and air stay only a little above a game VO line.
_BAND_WEIGHTS = (0.80, 1.00, 0.75, 0.32, 0.08)
_MAX_BAND_BOOST = 10 ** (15.0 / 20.0)
_MAX_BAND_CUT = 10 ** (-12.0 / 20.0)
_PRESENCE_HZ = 3200.0
_PRESENCE_DB = 3.5
_AIR_HZ = 7500.0
_AIR_DB = 1.0
# Gentle makeup toward a speech RMS; the limiter is a safety net, not a crush.
_DRIVE_RMS = 10 ** (-16.0 / 20.0)
_MAX_DRIVE = 10 ** (8.0 / 20.0)
_LIMIT_CEILING = 10 ** (-0.3 / 20.0)
_WINDOW_MS = 4.0
_SPEECH_FLOOR_RATIO = 0.05
# Mid-phrase speech vs pause. Unstressed endings sit lower (~3% of peak).
_FLATTEN_FLOOR_RATIO = 0.12
_SOFT_FLOOR_RATIO = 0.03
# A below-floor run is still the same word when it is much shorter than a
# typical loud island. Phrase pauses are longer than that, so they stay put.
_PIT_TYPICAL_FRACTION = 0.2
# Quiet tails may fade out over this span; a still-loud ending only gets a click guard.
_END_FADE_MS = 20.0
_CLICK_FADE_MS = 8.0
_MAX_DYNAMIC_BOOST = 10 ** (12.0 / 20.0)
_MAX_DYNAMIC_CUT = 10 ** (-12.0 / 20.0)
# Quiet speech inside an island (dip / «ки» / whisper) can sit ~15–18 dB down.
_LEVEL_MAX_BOOST = 10 ** (18.0 / 20.0)
# Light upward only; flatten already lifts dips and whispered endings.
_BAND_RATIO_UP = (2.5, 2.5, 2.2, 1.4, 1.2)
_BAND_UP_CAP = (
    10 ** (10.0 / 20.0),
    10 ** (10.0 / 20.0),
    10 ** (8.0 / 20.0),
    10 ** (3.0 / 20.0),
    10 ** (1.5 / 20.0),
)
_MIN_DURATION_SEC = 0.08
_PEAK_EPS = 1.0 / 32_768


def remaster_speech(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Light multiband remaster. Duration is unchanged; final level is not."""
    audio, leading = _as_2d(samples)
    if sample_rate <= 0 or audio.shape[0] < int(sample_rate * _MIN_DURATION_SEC):
        return samples
    if float(np.max(np.abs(audio))) < _PEAK_EPS:
        return samples

    filtered = _highpass(audio, sample_rate, _HPF_HZ)
    bands = _split_bands(filtered, sample_rate, _CROSSOVERS_HZ)
    mono = filtered.mean(axis=1)
    speech_floor = _peak_rms(mono, sample_rate) * _SPEECH_FLOOR_RATIO
    body_rms = _speech_rms(bands[1].mean(axis=1), sample_rate)
    if body_rms < _PEAK_EPS:
        body_rms = _speech_rms(mono, sample_rate)
    if body_rms < _PEAK_EPS:
        return samples

    balanced = np.zeros_like(filtered)
    for index, (band, weight) in enumerate(zip(bands, _BAND_WEIGHTS)):
        current = _speech_rms(band.mean(axis=1), sample_rate)
        if current < _PEAK_EPS:
            continue
        static = min(_MAX_BAND_BOOST, max(_MAX_BAND_CUT, (body_rms * weight) / current))
        shaped = band * np.float32(static)
        high = index >= 3
        gain = _dynamic_gain(
            shaped.mean(axis=1),
            sample_rate,
            threshold=max(speech_floor * 1.5, body_rms * weight * 0.12),
            ratio_down=4.0,
            ratio_up=_BAND_RATIO_UP[index],
            attack_ms=8.0 if high else 5.0,
            release_ms=80.0 if high else 60.0,
            noise_floor=speech_floor * (3.0 if high else 1.0),
            max_boost=_BAND_UP_CAP[index],
        )
        balanced += shaped * gain[:, None]

    present = _peak_eq(balanced, sample_rate, _PRESENCE_HZ, _PRESENCE_DB, q=1.1)
    present = _peak_eq(present, sample_rate, _AIR_HZ, _AIR_DB, q=0.8)
    present = _deess(present, sample_rate, speech_floor)

    wide = _dynamic_gain(
        present.mean(axis=1),
        sample_rate,
        threshold=max(speech_floor * 2.0, _DRIVE_RMS * 0.12),
        ratio_down=3.0,
        ratio_up=1.8,
        attack_ms=12.0,
        release_ms=100.0,
        noise_floor=speech_floor,
        max_boost=10 ** (4.0 / 20.0),
    )
    remastered = present * wide[:, None]
    # Last pass: every speech island aims at the same RMS. Pauses stay put.
    remastered = _level_speech_windows(remastered, sample_rate)

    current_rms = _speech_rms(remastered.mean(axis=1), sample_rate)
    if current_rms >= _PEAK_EPS:
        remastered = remastered * np.float32(min(_MAX_DRIVE, _DRIVE_RMS / current_rms))

    limited = _lookahead_limit(remastered, sample_rate, _LIMIT_CEILING)
    limited = _fade_out(limited, sample_rate, _END_FADE_MS)
    return _restore_shape(limited, leading, samples.shape)


def _highpass(audio: np.ndarray, sample_rate: int, cutoff: float) -> np.ndarray:
    spec, freqs = _rfft(audio, sample_rate)
    hp = 1.0 / (1.0 + (cutoff / np.maximum(freqs, 1e-6)) ** 4)
    return _irfft(spec * hp[:, None], audio.shape[0])


def _split_bands(
    audio: np.ndarray, sample_rate: int, cuts: tuple[float, ...]
) -> list[np.ndarray]:
    spec, freqs = _rfft(audio, sample_rate)
    remaining = np.ones(freqs.shape, dtype=np.float64)
    bands: list[np.ndarray] = []
    for cut in cuts:
        low = 1.0 / (1.0 + (freqs / max(cut, 1.0)) ** 4)
        low = np.minimum(low, remaining)
        bands.append(_irfft(spec * low[:, None], audio.shape[0]))
        remaining = remaining - low
    bands.append(_irfft(spec * remaining[:, None], audio.shape[0]))
    return bands


def _peak_eq(
    audio: np.ndarray, sample_rate: int, center: float, gain_db: float, *, q: float
) -> np.ndarray:
    spec, freqs = _rfft(audio, sample_rate)
    ratio = np.maximum(freqs, 1.0) / max(center, 1.0)
    bell = np.exp(-0.5 * (np.log(ratio) * q) ** 2)
    gain = np.power(10.0, (gain_db * bell) / 20.0)
    return _irfft(spec * gain[:, None], audio.shape[0])


def _deess(audio: np.ndarray, sample_rate: int, noise_floor: float) -> np.ndarray:
    """Duck 6–9 kHz only when it outruns the 2–4 kHz presence band."""
    sibilant = _band_pass(audio, sample_rate, 6000.0, 9000.0)
    presence = _band_pass(audio, sample_rate, 2000.0, 4000.0)
    sib_env = _window_rms(sibilant.mean(axis=1), sample_rate)
    pre_env = _window_rms(presence.mean(axis=1), sample_rate)
    hop = _hop(sample_rate)
    ratio = sib_env / np.maximum(pre_env, noise_floor)
    duck = np.where((sib_env > noise_floor) & (ratio > 1.4), 1.0 / np.minimum(ratio, 4.0), 1.0)
    duck = _smooth_series(duck, hop, sample_rate, attack_ms=4.0, release_ms=40.0)
    gain = _interp_windows(duck, hop, audio.shape[0])
    # Blend: full-band minus a fraction of the sibilant band.
    return audio - sibilant * (1.0 - gain)[:, None]


def _band_pass(audio: np.ndarray, sample_rate: int, low: float, high: float) -> np.ndarray:
    spec, freqs = _rfft(audio, sample_rate)
    hp = 1.0 / (1.0 + (low / np.maximum(freqs, 1e-6)) ** 4)
    lp = 1.0 / (1.0 + (freqs / max(high, 1.0)) ** 4)
    return _irfft(spec * (hp * lp)[:, None], audio.shape[0])


def _dynamic_gain(
    mono: np.ndarray,
    sample_rate: int,
    *,
    threshold: float,
    ratio_down: float,
    ratio_up: float,
    attack_ms: float,
    release_ms: float,
    noise_floor: float,
    max_boost: float | None = None,
) -> np.ndarray:
    """Upward+downward compression from window RMS. Silence stays at unity."""
    hop = _hop(sample_rate)
    env = _window_rms(mono, sample_rate)
    desired = np.ones(env.size, dtype=np.float64)
    loud = env > threshold
    quiet = (env >= noise_floor) & ~loud
    desired[loud] = (env[loud] / threshold) ** (1.0 / ratio_down - 1.0)
    desired[quiet] = (env[quiet] / threshold) ** (1.0 / ratio_up - 1.0)
    boost = _MAX_DYNAMIC_BOOST if max_boost is None else max_boost
    desired = np.clip(desired, _MAX_DYNAMIC_CUT, boost)
    smoothed = _smooth_series(desired, hop, sample_rate, attack_ms=attack_ms, release_ms=release_ms)
    return _interp_windows(smoothed, hop, mono.size)


def _level_speech_windows(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Pull speech islands to one RMS. Phrase pauses and true silence stay put."""
    mono = audio.mean(axis=1)
    env = _window_rms(mono, sample_rate)
    target = _speech_rms(mono, sample_rate)
    if target < _PEAK_EPS or env.size == 0:
        return audio
    peak = float(np.max(env))
    if peak < _PEAK_EPS:
        return audio
    loud = env >= peak * _FLATTEN_FLOOR_RATIO
    soft = (env >= peak * _SOFT_FLOOR_RATIO) & ~loud
    take = _speech_level_mask(loud, soft)
    desired = np.ones(env.size, dtype=np.float64)
    desired[take] = np.minimum(
        _LEVEL_MAX_BOOST, target / np.maximum(env[take], _PEAK_EPS)
    )
    desired = np.maximum(desired, _MAX_DYNAMIC_CUT)
    hop = _hop(sample_rate)
    smoothed = _smooth_series(desired, hop, sample_rate, attack_ms=1.0, release_ms=8.0)
    gain = _interp_windows(smoothed, hop, audio.shape[0])
    return audio * gain[:, None]


def _speech_level_mask(loud: np.ndarray, soft: np.ndarray) -> np.ndarray:
    """Mark windows to flatten from energy and adjacency, not a trailing time window.

    Loud islands are speech. A short soft run between two islands is a word-internal
    dip and is leveled; a long quiet run is a phrase pause. Soft energy after the
    last island is the unstressed ending or whispered last word. True silence is
    never marked, so a stop-consonant hole is not pumped.
    """
    active = loud.copy()
    loud_runs = _true_runs(loud)
    if not loud_runs:
        return soft.copy()
    lengths = [end - start for start, end in loud_runs]
    typical = float(np.median(np.asarray(lengths, dtype=np.float64)))
    pit_limit = typical * _PIT_TYPICAL_FRACTION
    for left, right in zip(loud_runs, loud_runs[1:]):
        gap_start, gap_end = left[1], right[0]
        gap = gap_end - gap_start
        neighbor = min(left[1] - left[0], right[1] - right[0])
        if 0 < gap < pit_limit and gap < neighbor:
            active[gap_start:gap_end] = soft[gap_start:gap_end]
    last_end = loud_runs[-1][1]
    if last_end < loud.size:
        active[last_end:] = soft[last_end:]
    return active


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive-start, exclusive-end runs where ``mask`` is True."""
    runs: list[tuple[int, int]] = []
    index = 0
    size = mask.size
    while index < size:
        if not mask[index]:
            index += 1
            continue
        start = index
        index += 1
        while index < size and mask[index]:
            index += 1
        runs.append((start, index))
    return runs


def _fade_out(audio: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    """Clean stop at EOF without eating a still-loud last syllable.

    A full ``fade_ms`` ramp is only applied when the tail is already quiet.
    Otherwise a few milliseconds are enough to avoid a click.
    """
    n_quiet = max(1, int(sample_rate * fade_ms / 1000.0))
    n_click = max(1, int(sample_rate * _CLICK_FADE_MS / 1000.0))
    if audio.shape[0] <= n_quiet:
        return audio
    tail = audio[-n_quiet:]
    tail_peak = float(np.max(np.abs(tail)))
    full_peak = float(np.max(np.abs(audio)))
    n_fade = n_quiet if tail_peak < full_peak * _FLATTEN_FLOOR_RATIO else n_click
    ramp = np.linspace(1.0, 0.0, n_fade, dtype=np.float32)
    out = audio.copy()
    out[-n_fade:] *= ramp[:, None]
    return out


def _lookahead_limit(audio: np.ndarray, sample_rate: int, ceiling: float) -> np.ndarray:
    lookahead = max(1, int(sample_rate * 0.003))
    peak = np.max(np.abs(audio), axis=1)
    padded = np.pad(peak, (0, lookahead), mode="edge")
    future = np.lib.stride_tricks.sliding_window_view(padded, lookahead + 1).max(axis=1)
    gain = np.minimum(1.0, ceiling / np.maximum(future, _PEAK_EPS))
    hop = _hop(sample_rate)
    # Release-only smooth: attack is the lookahead itself.
    n_win = int(math.ceil(gain.size / hop))
    windowed = np.empty(n_win, dtype=np.float64)
    for index in range(n_win):
        chunk = gain[index * hop : index * hop + hop]
        windowed[index] = float(np.min(chunk)) if chunk.size else 1.0
    smoothed = _smooth_series(windowed, hop, sample_rate, attack_ms=1.0, release_ms=50.0)
    sample_gain = _interp_windows(smoothed, hop, audio.shape[0])
    return (audio * sample_gain[:, None]).astype(np.float32, copy=False)


def _rfft(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    spec = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[0], 1.0 / sample_rate).astype(np.float64)
    return spec, freqs


def _irfft(spec: np.ndarray, n_samples: int) -> np.ndarray:
    return np.fft.irfft(spec, n=n_samples, axis=0).astype(np.float32, copy=False)


def _hop(sample_rate: int) -> int:
    return max(1, int(sample_rate * _WINDOW_MS / 1000.0))


def _window_rms(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    hop = _hop(sample_rate)
    n_win = int(math.ceil(mono.size / hop))
    rms = np.empty(n_win, dtype=np.float64)
    for index in range(n_win):
        chunk = mono[index * hop : index * hop + hop]
        rms[index] = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
    return rms


def _speech_rms(mono: np.ndarray, sample_rate: int) -> float:
    """RMS of windows that look like speech, so pauses do not pull the target down."""
    rms = _window_rms(mono, sample_rate)
    peak = float(np.max(rms)) if rms.size else 0.0
    if peak < _PEAK_EPS:
        return 0.0
    hop = _hop(sample_rate)
    floor = peak * 0.1
    total = 0.0
    count = 0
    for index, value in enumerate(rms):
        if value < floor:
            continue
        chunk = mono[index * hop : index * hop + hop]
        total += float(np.dot(chunk, chunk))
        count += chunk.size
    if count <= 0:
        return 0.0
    return math.sqrt(total / count)


def _peak_rms(mono: np.ndarray, sample_rate: int) -> float:
    rms = _window_rms(mono, sample_rate)
    return float(np.max(rms)) if rms.size else 0.0


def _smooth_series(
    values: np.ndarray,
    hop: int,
    sample_rate: int,
    *,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    attack = math.exp(-hop / max(1.0, sample_rate * attack_ms / 1000.0))
    release = math.exp(-hop / max(1.0, sample_rate * release_ms / 1000.0))
    smoothed = np.empty(values.size, dtype=np.float64)
    current = float(values[0]) if values.size else 1.0
    for index, value in enumerate(values):
        coeff = attack if value < current else release
        current = coeff * current + (1.0 - coeff) * float(value)
        smoothed[index] = current
    return smoothed


def _interp_windows(values: np.ndarray, hop: int, n_samples: int) -> np.ndarray:
    if values.size == 0:
        return np.ones(n_samples, dtype=np.float32)
    times = (np.arange(values.size, dtype=np.float64) + 0.5) * hop
    return np.interp(np.arange(n_samples, dtype=np.float64), times, values).astype(np.float32)


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
