"""Fit a dubbed line into the reference slot without wrecking its tempo.

Cheapest artifact first:

1. Trim leading/trailing silence.
2. Edit phrase pauses toward the reference pause budget — with guards so quiet
   fricatives are not cut as silence.
3. Time-stretch (Praat PSOLA on CPU) only as far as articulation rate allows.
4. Absorb what is left with a further pause shrink; keep any residual overrun.

Tempo is measured as **syllables per second of active speech**, each side
against its own text. Comparing raw seconds across languages is wrong: a
Ukrainian line carries more syllables than the English original, so matching
absolute speech duration compresses it to an impossible rate. The stretch
target is the slot need, clamped into a plausible articulation band; when the
line still cannot fit, the fit reports ``needs_shorter_line`` instead of
speeding the voice up further.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

_WINDOW_MS = 50.0
_SPEECH_FLOOR_RATIO = 0.1
# Pause detection uses a lower floor: unvoiced fricatives (ш/с/х/ф) sit well
# below 10% of the peak window and must not be classified as silence.
_PAUSE_FLOOR_RATIO = 0.05
# Never edit audio closer than this to speech on either side of a pause —
# fricative onsets/tails leak past the RMS boundary by up to ~100 ms.
_PAUSE_GUARD_SEC = 0.10
_PEAK_EPS = 1.0 / 32_768
_EDGE_KEEP_MS = 120.0
_TAIL_KEEP_MS = 150.0
_TAIL_FLOOR_RATIO = 0.05
_EPS_REL = 0.05
_EPS_ABS_SEC = 0.35
_MIN_LINE_SEC = 1.2
_MIN_SYNTH_SEC = 0.8
_MIN_SYL = 4
_MIN_CHARS = 12
# Skip stretch for a tiny tempo residual — better leave alone than PSOLA for 3%.
_STRETCH_SKIP = 0.04
# Hard technical band for pitch-preserving Praat PSOLA.
_RATE_MIN = 0.5
_RATE_MAX = 2.0
# Plausible Ukrainian articulation, syllables per second of active speech.
# Slot pressure may push toward the top of this band, never past it.
_SYL_RATE_MIN = 4.0
_SYL_RATE_MAX = 6.0
# The band edges follow a reference that sits outside them; an original at
# 6.2 syl/s licenses a dub a little above the nominal ceiling.
_REF_RATE_HEADROOM = 1.1
# Tempo never slows a take (PSOLA below 1× smears pitch). Speed-up is
# capped at 1.3× — Praat can do 2×, but that reads as rushed on dubbed lines.
_TEMPO_RATE_MIN = 1.0
_TEMPO_RATE_MAX = 1.3
_PRAAT_F0_MIN = 60.0
_PRAAT_F0_MAX = 600.0
# Default per-pause ceiling; raised to the reference's longest phrase pause.
_MAX_PAUSE_SEC = 0.40
# Clause-level gaps only (~150–200 ms). Word-to-word dips are ~50–80 ms.
_MIN_PHRASE_PAUSE_SEC = 0.15
# Pauses at or above this may shrink; shorter word gaps stay put.
_MIN_KEEP_PAUSE_SEC = 0.10
# Never shrink a pause below this — a clipped clause gap sounds like a cut.
_MIN_SHRINK_PAUSE_SEC = 0.08
_PAD_FRONT = 0.30
# Short ramp at insert/cut seams so room tone does not click.
_PAUSE_FADE_MS = 6.0
# Pause-budget match slack (relative to ref pause total, with a floor).
_PAUSE_EPS_REL = 0.08
_PAUSE_EPS_ABS_SEC = 0.12

_ASTERISK_RE = re.compile(r"\*[^*]+\*")
_BRACKET_RE = re.compile(r"\[[^\[\]]*\]")
_UK_VOWEL_RE = re.compile(r"[аеєиіїоуюяАЕЄИІЇОУЮЯ]")
_LATIN_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]+")


def stretch_rate_bounds() -> tuple[float, float]:
    """Hard technical limits of Praat PSOLA."""
    return _RATE_MIN, _RATE_MAX


def syllable_rate_bounds() -> tuple[float, float]:
    """Plausible articulation band, syllables per second of active speech."""
    return _SYL_RATE_MIN, _SYL_RATE_MAX


def tempo_rate_bounds() -> tuple[float, float]:
    """Stretch factors the tempo fit is allowed to apply."""
    return _TEMPO_RATE_MIN, _TEMPO_RATE_MAX


class TimingStretchError(RuntimeError):
    """Praat PSOLA is missing or failed; no other stretch backend is allowed."""


def ensure_praat_psola() -> None:
    """Import Praat now so match_timing cannot silently fall through later."""
    try:
        import parselmouth  # noqa: F401
        from parselmouth.praat import call  # noqa: F401
    except ImportError as exc:
        raise TimingStretchError(
            "match_timing stretch requires praat-parselmouth (Praat PSOLA). "
            "Install with: pip install praat-parselmouth"
        ) from exc


@dataclass(frozen=True)
class TimingFit:
    """Fitted audio plus the numbers behind the decision."""

    audio: np.ndarray
    applied: bool = False
    slot_sec: float = 0.0
    duration_sec: float = 0.0
    overrun_sec: float = 0.0
    stretch_rate: float = 1.0
    syl_per_sec_synth: float = 0.0
    syl_per_sec_ref: float = 0.0
    syl_per_sec_final: float = 0.0
    needs_shorter_line: bool = False
    skip_reason: str = ""
    counts: dict = field(default_factory=dict)

    def metrics(self) -> dict:
        """Log-friendly view without the waveform."""
        return {
            "applied": self.applied,
            "slot_sec": round(self.slot_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "overrun_sec": round(self.overrun_sec, 3),
            "stretch_rate": round(self.stretch_rate, 4),
            "syl_per_sec_synth": round(self.syl_per_sec_synth, 3),
            "syl_per_sec_ref": round(self.syl_per_sec_ref, 3),
            "syl_per_sec_final": round(self.syl_per_sec_final, 3),
            "needs_shorter_line": self.needs_shorter_line,
            "skip_reason": self.skip_reason,
            "counts": dict(self.counts),
        }


def load_reference_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load the first ``speaker_wav`` used as the timing slot target."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if not isinstance(audio, np.ndarray) or audio.size == 0:
        raise ValueError(f"reference audio is empty: {path}")
    return audio, int(rate)


def strip_nonspeech(text: str) -> str:
    """Drop ``*...*`` / ``[...]`` stage-direction blocks, matching Transynth TTS prep."""
    cleaned = _ASTERISK_RE.sub(" ", text or "")
    cleaned = _BRACKET_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def count_syllables(text: str) -> int:
    """Ukrainian vowel letters plus Latin vowel groups."""
    if not text:
        return 0
    return len(_UK_VOWEL_RE.findall(text)) + len(_LATIN_VOWEL_RE.findall(text))


def measure_active_speech_sec(samples: np.ndarray, sample_rate: int) -> float:
    """Duration of windows whose RMS is above a fraction of the peak window."""
    rms, counts, peak = _window_rms(samples, sample_rate)
    if peak < _PEAK_EPS or sample_rate <= 0:
        return 0.0
    floor = peak * _SPEECH_FLOOR_RATIO
    active = 0
    for value, count in zip(rms, counts, strict=True):
        if value >= floor:
            active += int(count)
    return active / sample_rate


def duration_in_slot(duration_sec: float, slot_sec: float) -> bool:
    """True when ``duration_sec`` is close enough to the reference slot."""
    return _in_slot(duration_sec, slot_sec)


def match_timing_to_reference(
    samples: np.ndarray,
    sample_rate: int,
    reference: np.ndarray,
    reference_rate: int,
    *,
    text_uk: str,
    text_en: str,
) -> np.ndarray:
    """Audio-only view of :func:`fit_timing_to_reference`."""
    return fit_timing_to_reference(
        samples,
        sample_rate,
        reference,
        reference_rate,
        text_uk=text_uk,
        text_en=text_en,
    ).audio


def fit_timing_to_reference(
    samples: np.ndarray,
    sample_rate: int,
    reference: np.ndarray,
    reference_rate: int,
    *,
    text_uk: str,
    text_en: str,
) -> TimingFit:
    """Fit pauses, then tempo, then report whatever still does not fit."""
    audio = _mono(samples)
    if audio.size == 0 or sample_rate <= 0:
        return TimingFit(audio=samples, skip_reason="empty_audio")

    line = _mono(reference)
    slot_sec = _duration_sec(line, reference_rate)
    d_synth = _duration_sec(audio, sample_rate)
    if slot_sec < _MIN_LINE_SEC or d_synth < _MIN_SYNTH_SEC:
        return TimingFit(audio=samples, slot_sec=slot_sec, skip_reason="clip_too_short")

    en = strip_nonspeech(text_en)
    uk = strip_nonspeech(text_uk)
    s_en = count_syllables(en)
    s_uk = count_syllables(uk)
    if s_en < _MIN_SYL or s_uk < _MIN_SYL or len(en) < _MIN_CHARS or len(uk) < _MIN_CHARS:
        return TimingFit(audio=samples, slot_sec=slot_sec, skip_reason="text_too_thin")

    audio = _trim_edge_silence(audio, sample_rate)
    d_synth = _duration_sec(audio, sample_rate)
    a_line = measure_active_speech_sec(line, reference_rate)
    a_synth = measure_active_speech_sec(audio, sample_rate)
    if a_line <= 0 or a_synth <= 0 or d_synth < _MIN_SYNTH_SEC:
        return TimingFit(
            audio=_restore(samples, audio),
            slot_sec=slot_sec,
            duration_sec=d_synth,
            skip_reason="no_active_speech",
        )

    syl_ref = s_en / a_line
    # Pauses first: cheapest way to move duration, no timbre cost.
    audio = _fit_pauses(audio, sample_rate, reference=line, reference_rate=reference_rate)

    a_synth = measure_active_speech_sec(audio, sample_rate)
    syl_synth = s_uk / a_synth if a_synth > 0 else 0.0
    rate = _tempo_rate(
        syl_per_sec=syl_synth,
        syl_per_sec_ref=syl_ref,
        duration_sec=_duration_sec(audio, sample_rate),
        slot_sec=slot_sec,
    )
    if abs(rate - 1.0) >= _STRETCH_SKIP:
        audio = time_stretch(audio, rate, sample_rate)
    else:
        rate = 1.0

    # Stretch is capped, so a long line can still overrun; give the pauses a
    # second pass before admitting the residual.
    duration = _duration_sec(audio, sample_rate)
    if duration > slot_sec and not _in_slot(duration, slot_sec):
        audio = _shrink_pauses(audio, sample_rate, slot_sec)
        duration = _duration_sec(audio, sample_rate)

    a_final = measure_active_speech_sec(audio, sample_rate)
    overrun = max(0.0, duration - slot_sec)
    return TimingFit(
        audio=_restore(samples, audio),
        applied=True,
        slot_sec=slot_sec,
        duration_sec=duration,
        overrun_sec=overrun,
        stretch_rate=rate,
        syl_per_sec_synth=syl_synth,
        syl_per_sec_ref=syl_ref,
        syl_per_sec_final=(s_uk / a_final) if a_final > 0 else 0.0,
        needs_shorter_line=not _in_slot(duration, slot_sec) and duration > slot_sec,
        counts={"syllables_uk": s_uk, "syllables_ref": s_en},
    )


def time_stretch(
    samples: np.ndarray, rate: float, sample_rate: int | None = None
) -> np.ndarray:
    """Pitch-preserving Praat PSOLA. ``rate > 1`` is faster (shorter)."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0 or not math.isfinite(rate) or abs(rate - 1.0) < 1e-3:
        return audio
    if sample_rate is None or sample_rate <= 0:
        raise TimingStretchError("PSOLA stretch requires a positive sample_rate")
    rate = max(_RATE_MIN, min(float(rate), _RATE_MAX))
    if audio.ndim == 2:
        stretched = [
            time_stretch(audio[:, index], rate, sample_rate)
            for index in range(audio.shape[1])
        ]
        length = min(channel.size for channel in stretched)
        stacked = np.stack([channel[:length] for channel in stretched], axis=1)
        return stacked.astype(np.float32, copy=False)
    return _match_input_peak(audio, _praat_stretch(audio, rate, sample_rate))


def _fit_pauses(
    audio: np.ndarray,
    sample_rate: int,
    *,
    reference: np.ndarray,
    reference_rate: int,
) -> np.ndarray:
    """Match phrase-pause budget to the reference; speech samples stay untouched."""
    pause_ref = _phrase_pause_sec(reference, reference_rate)
    max_pause = max(_MAX_PAUSE_SEC, _max_phrase_pause_sec(reference, reference_rate))
    audio = _cap_pauses(audio, sample_rate, max_pause_sec=max_pause)

    pause_synth = _phrase_pause_sec(audio, sample_rate)
    slack = max(_PAUSE_EPS_REL * max(pause_ref, 0.25), _PAUSE_EPS_ABS_SEC)
    if abs(pause_synth - pause_ref) <= slack:
        return audio

    duration = _duration_sec(audio, sample_rate)
    if pause_synth > pause_ref:
        return _shrink_pauses(audio, sample_rate, duration - (pause_synth - pause_ref))
    return _expand_pauses(
        audio,
        sample_rate,
        duration + (pause_ref - pause_synth),
        max_pause_sec=max_pause,
    )


def _tempo_rate(
    *,
    syl_per_sec: float,
    syl_per_sec_ref: float,
    duration_sec: float,
    slot_sec: float,
) -> float:
    """Stretch factor the slot wants, clamped to a plausible articulation rate.

    The slot decides how much speed-up is *wanted*; the syllable band decides how
    much is *allowed*. Speed-up itself stops at 1.3×. A take is never slowed:
    PSOLA below 1× smears pitch more than it helps the slot. Both band edges
    yield to the reference: a deliberately unhurried delivery must not be
    pushed to conversational speed, and a brisk original licenses a brisk dub.
    """
    if syl_per_sec <= 0:
        return 1.0
    wanted = duration_sec / slot_sec if slot_sec > 0 and duration_sec > 0 else 1.0
    floor = min(_SYL_RATE_MIN, syl_per_sec_ref) if syl_per_sec_ref > 0 else _SYL_RATE_MIN
    ceiling = max(_SYL_RATE_MAX, syl_per_sec_ref * _REF_RATE_HEADROOM)
    target_syl_per_sec = min(max(wanted * syl_per_sec, floor), ceiling)
    rate = target_syl_per_sec / syl_per_sec
    return max(_TEMPO_RATE_MIN, min(rate, _TEMPO_RATE_MAX))


def _phrase_pause_sec(samples: np.ndarray, sample_rate: int) -> float:
    pauses = _list_phrase_pauses(samples, sample_rate, min_sec=_MIN_KEEP_PAUSE_SEC)
    if not pauses or sample_rate <= 0:
        return 0.0
    return sum(end - start for start, end in pauses) / sample_rate


def _max_phrase_pause_sec(samples: np.ndarray, sample_rate: int) -> float:
    pauses = _list_phrase_pauses(samples, sample_rate, min_sec=_MIN_KEEP_PAUSE_SEC)
    if not pauses or sample_rate <= 0:
        return 0.0
    return max(end - start for start, end in pauses) / sample_rate


def _list_phrase_pauses(
    samples: np.ndarray, sample_rate: int, *, min_sec: float
) -> list[tuple[int, int]]:
    rms, counts, peak = _window_rms(samples, sample_rate)
    if peak < _PEAK_EPS:
        return []
    floor = peak * _PAUSE_FLOOR_RATIO
    hop = max(1, int(sample_rate * _WINDOW_MS / 1000.0))
    return _internal_pauses(rms, counts, hop, floor, sample_rate, min_sec=min_sec)


def _cap_pauses(
    samples: np.ndarray, sample_rate: int, *, max_pause_sec: float
) -> np.ndarray:
    """Shrink any single phrase pause that exceeds ``max_pause_sec``."""
    max_pause = int(sample_rate * max_pause_sec)
    pauses = _list_phrase_pauses(samples, sample_rate, min_sec=_MIN_KEEP_PAUSE_SEC)
    if not pauses:
        return samples
    allocated = [max(0, (end - start) - max_pause) for start, end in pauses]
    if sum(allocated) <= 0:
        return samples
    return _remove_from_pauses(samples, pauses, allocated, sample_rate)


def _expand_pauses(
    samples: np.ndarray,
    sample_rate: int,
    target_sec: float,
    *,
    min_sec: float = _MIN_PHRASE_PAUSE_SEC,
    max_pause_sec: float = _MAX_PAUSE_SEC,
) -> np.ndarray:
    """Insert silence into phrase pauses, weighted by length, capped per pause."""
    need = int(round(target_sec * sample_rate)) - samples.size
    if need <= 0:
        return samples

    pauses = _list_phrase_pauses(samples, sample_rate, min_sec=min_sec)
    if not pauses:
        return _pad_to_duration(samples, sample_rate, target_sec)

    max_pause = int(sample_rate * max_pause_sec)
    capacities = [max(0, max_pause - (end - start)) for start, end in pauses]
    total_cap = sum(capacities)
    if total_cap <= 0:
        return _pad_to_duration(samples, sample_rate, target_sec)

    extra = min(need, total_cap)
    allocated = _allocate_by_weight(
        capacities, [end - start for start, end in pauses], extra
    )
    return _insert_into_pauses(samples, pauses, allocated, sample_rate)


def _shrink_pauses(
    samples: np.ndarray,
    sample_rate: int,
    target_sec: float,
) -> np.ndarray:
    """Remove silence from phrase pauses down to a guarded floor."""
    excess = samples.size - int(round(target_sec * sample_rate))
    if excess <= 0:
        return samples

    pauses = _list_phrase_pauses(samples, sample_rate, min_sec=_MIN_KEEP_PAUSE_SEC)
    if not pauses:
        return samples

    guard = int(sample_rate * _PAUSE_GUARD_SEC)
    min_pause = max(int(sample_rate * _MIN_SHRINK_PAUSE_SEC), 2 * guard)
    capacities = [max(0, (end - start) - min_pause) for start, end in pauses]
    total_cap = sum(capacities)
    if total_cap <= 0:
        return samples

    remove = min(excess, total_cap)
    allocated = _allocate_by_weight(
        capacities, [end - start for start, end in pauses], remove
    )
    return _remove_from_pauses(samples, pauses, allocated, sample_rate)


def _allocate_by_weight(
    capacities: list[int], weights: list[int], total: int
) -> list[int]:
    weight_sum = sum(weights)
    allocated = [0] * len(capacities)
    remaining = total
    for index, (capacity, weight) in enumerate(zip(capacities, weights, strict=True)):
        share = int(total * (weight / weight_sum)) if weight_sum else 0
        take = min(capacity, share, remaining)
        allocated[index] = take
        remaining -= take
    for index, capacity in enumerate(capacities):
        if remaining <= 0:
            break
        room = capacity - allocated[index]
        if room <= 0:
            continue
        take = min(room, remaining)
        allocated[index] += take
        remaining -= take
    return allocated


def _insert_into_pauses(
    samples: np.ndarray,
    pauses: list[tuple[int, int]],
    allocated: list[int],
    sample_rate: int,
) -> np.ndarray:
    """Insert faded silence at each pause's quietest guarded point."""
    fade = int(sample_rate * _PAUSE_FADE_MS / 1000.0)
    pieces: list[np.ndarray] = []
    cursor = 0
    for (start, end), insert in zip(pauses, allocated, strict=True):
        if insert <= 0:
            pieces.append(samples[cursor:end])
            cursor = end
            continue
        lo, hi = _pause_core(start, end, sample_rate)
        point = _quietest_point(samples, lo, hi, sample_rate)
        pre = samples[cursor:point]
        post = samples[point:end]
        ramp_n = min(fade, pre.size, post.size)
        if ramp_n > 0:
            ramp = np.linspace(1.0, 0.0, ramp_n, dtype=np.float32)
            pre = pre.copy()
            pre[-ramp_n:] *= ramp
            post = post.copy()
            post[:ramp_n] *= ramp[::-1]
        pieces.append(pre)
        pieces.append(np.zeros(insert, dtype=samples.dtype))
        pieces.append(post)
        cursor = end
    pieces.append(samples[cursor:])
    return np.concatenate(pieces) if pieces else samples


def _remove_from_pauses(
    samples: np.ndarray,
    pauses: list[tuple[int, int]],
    allocated: list[int],
    sample_rate: int,
) -> np.ndarray:
    """Drop samples from each pause's quiet core, with short fades."""
    fade = int(sample_rate * _PAUSE_FADE_MS / 1000.0)
    pieces: list[np.ndarray] = []
    cursor = 0
    for (start, end), cut in zip(pauses, allocated, strict=True):
        if cut <= 0:
            pieces.append(samples[cursor:end])
            cursor = end
            continue
        lo, hi = _pause_core(start, end, sample_rate)
        cut = min(cut, hi - lo)
        if cut <= 0:
            pieces.append(samples[cursor:end])
            cursor = end
            continue
        center = _quietest_point(samples, lo, hi, sample_rate)
        left_end = max(lo, min(center - cut // 2, hi - cut))
        right_start = left_end + cut
        pre = samples[cursor:left_end]
        post = samples[right_start:end]
        ramp_n = min(fade, pre.size, post.size)
        if ramp_n > 0:
            ramp = np.linspace(1.0, 0.0, ramp_n, dtype=np.float32)
            pre = pre.copy()
            pre[-ramp_n:] *= ramp
            post = post.copy()
            post[:ramp_n] *= ramp[::-1]
        pieces.append(pre)
        pieces.append(post)
        cursor = end
    pieces.append(samples[cursor:])
    return np.concatenate(pieces) if pieces else samples


def _pause_core(start: int, end: int, sample_rate: int) -> tuple[int, int]:
    """Editable range of a pause: guards keep edits away from speech boundaries."""
    guard = int(sample_rate * _PAUSE_GUARD_SEC)
    lo = start + guard
    hi = end - guard
    if hi <= lo:
        quarter = (end - start) // 4
        lo = start + quarter
        hi = end - quarter
    if hi <= lo:
        mid = (start + end) // 2
        return mid, mid + 1
    return lo, hi


def _quietest_point(
    samples: np.ndarray, lo: int, hi: int, sample_rate: int
) -> int:
    """Sample index with the lowest short-window energy inside ``[lo, hi)``."""
    if hi - lo <= 1:
        return lo
    segment = np.abs(np.asarray(samples[lo:hi], dtype=np.float64))
    win = max(1, int(sample_rate * 0.010))
    if segment.size > win:
        kernel = np.full(win, 1.0 / win)
        segment = np.convolve(segment, kernel, mode="same")
    return lo + int(np.argmin(segment))


def _internal_pauses(
    rms: np.ndarray,
    counts: np.ndarray,
    hop: int,
    floor: float,
    sample_rate: int,
    *,
    min_sec: float,
) -> list[tuple[int, int]]:
    """Return sample ranges of internal pauses, excluding leading/trailing silence."""
    min_len = int(sample_rate * min_sec)
    pauses: list[tuple[int, int]] = []
    index = 0
    n_windows = rms.size
    while index < n_windows:
        if rms[index] >= floor:
            index += 1
            continue
        begin = index
        while index < n_windows and rms[index] < floor:
            index += 1
        start = begin * hop
        end = _window_end(index - 1, counts, hop)
        if begin == 0 or index >= n_windows:
            continue
        if end - start < min_len:
            continue
        pauses.append((start, end))
    return pauses


def _pad_to_duration(samples: np.ndarray, sample_rate: int, target_sec: float) -> np.ndarray:
    target_n = int(round(target_sec * sample_rate))
    extra = target_n - samples.size
    if extra <= 0:
        return samples
    front = int(extra * _PAD_FRONT)
    back = extra - front
    return np.pad(samples, (front, back))


def _praat_stretch(samples: np.ndarray, rate: float, sample_rate: int) -> np.ndarray:
    """TD-PSOLA via Praat ``Lengthen (overlap-add)`` (parselmouth)."""
    ensure_praat_psola()
    import parselmouth
    from parselmouth.praat import call

    if not math.isfinite(rate) or abs(rate - 1.0) < 1e-3:
        return samples.astype(np.float32, copy=False)
    factor = 1.0 / rate
    if not math.isfinite(factor) or factor <= 0:
        raise TimingStretchError(f"invalid PSOLA lengthen factor for rate={rate}")
    try:
        sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sample_rate)
        lengthened = call(
            sound,
            "Lengthen (overlap-add)",
            _PRAAT_F0_MIN,
            _PRAAT_F0_MAX,
            float(factor),
        )
    except Exception as exc:
        raise TimingStretchError(f"Praat PSOLA failed: {exc}") from exc
    values = np.asarray(lengthened.values, dtype=np.float32)
    if values.ndim == 2:
        audio = values[0] if values.shape[0] <= values.shape[1] else values[:, 0]
    else:
        audio = values.reshape(-1)
    if audio.size == 0:
        raise TimingStretchError("Praat PSOLA returned empty audio")
    return audio


def _match_input_peak(original: np.ndarray, stretched: np.ndarray) -> np.ndarray:
    """Keep stretch from raising the peak above the input's."""
    peak_in = float(np.max(np.abs(original))) if original.size else 0.0
    peak_out = float(np.max(np.abs(stretched))) if stretched.size else 0.0
    if peak_in > _PEAK_EPS and peak_out > peak_in:
        return (stretched * np.float32(peak_in / peak_out)).astype(np.float32, copy=False)
    return stretched


def _in_slot(duration_sec: float, slot_sec: float) -> bool:
    return abs(duration_sec - slot_sec) <= max(_EPS_REL * slot_sec, _EPS_ABS_SEC)


def _trim_edge_silence(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Drop leading/trailing silence so duration uses the spoken span."""
    rms, counts, peak = _window_rms(samples, sample_rate)
    if peak < _PEAK_EPS:
        return samples
    lead_floor = peak * _TAIL_FLOOR_RATIO
    tail_floor = peak * _TAIL_FLOOR_RATIO
    first = next((index for index, value in enumerate(rms) if value >= lead_floor), None)
    last = next(
        (index for index in range(rms.size - 1, -1, -1) if rms[index] >= tail_floor),
        None,
    )
    if first is None or last is None:
        return samples
    hop = max(1, int(sample_rate * _WINDOW_MS / 1000.0))
    keep_lead = int(sample_rate * _EDGE_KEEP_MS / 1000.0)
    keep_tail = int(sample_rate * _TAIL_KEEP_MS / 1000.0)
    start = max(0, first * hop - keep_lead)
    end = min(samples.size, _window_end(last, counts, hop) + keep_tail)
    if end - start < int(sample_rate * _MIN_SYNTH_SEC):
        return samples
    return samples[start:end]


def _window_rms(
    samples: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, float]:
    mono = _mono(samples)
    if mono.size == 0 or sample_rate <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, np.zeros(0, dtype=np.int32), 0.0
    hop = max(1, int(sample_rate * _WINDOW_MS / 1000.0))
    n_windows = int(math.ceil(mono.size / hop))
    rms = np.empty(n_windows, dtype=np.float64)
    counts = np.empty(n_windows, dtype=np.int32)
    peak = 0.0
    for index in range(n_windows):
        chunk = mono[index * hop : index * hop + hop]
        counts[index] = chunk.size
        energy = float(np.dot(chunk, chunk)) if chunk.size else 0.0
        value = math.sqrt(energy / max(1, chunk.size))
        rms[index] = value
        if value > peak:
            peak = value
    return rms, counts, peak


def _window_end(window_index: int, counts: np.ndarray, hop: int) -> int:
    return window_index * hop + int(counts[window_index])


def _duration_sec(samples: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return _mono(samples).size / sample_rate


def _mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32, copy=False)
    raise ValueError(f"expected 1-D or 2-D audio, got shape {audio.shape}")


def _restore(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """Processed audio is mono; duplicate it across the original channel layout if needed."""
    if original.ndim == 1:
        return processed.astype(np.float32, copy=False)
    if processed.size == 0:
        return processed.astype(np.float32, copy=False)
    channels = original.shape[1]
    return np.repeat(processed[:, None], channels, axis=1).astype(np.float32, copy=False)
