"""Match synthesized timing to the first reference clip after loudness.

Fit the original slot: expand phrase pauses when short, time-stretch speech
islands when long (pauses ≥100 ms stay put), never rushing more than ~15% past
the actor's articulation rate.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from fish_studio.loudness import apply_peak_ceiling

# Same RMS hop as loudness.py so speech/pause decisions stay aligned.
_WINDOW_MS = 50.0
# A window is speech if its RMS is at least this fraction of the peak window.
_SPEECH_FLOOR_RATIO = 0.1
# 16-bit quantization floor; quieter than this is treated as silence.
_PEAK_EPS = 1.0 / 32_768
# Leave a short pad after trimming leading silence so attacks survive.
_EDGE_KEEP_MS = 50.0
# Trailing keep is longer: Ukrainian endings and Fish fade sit below the 10% floor.
_TAIL_KEEP_MS = 150.0
_TAIL_FLOOR_RATIO = 0.05
# Slot is a match if duration is within 5% or 350 ms of the reference, whichever is larger.
_EPS_REL = 0.05
_EPS_ABS_SEC = 0.35
# Skip matching on clips too short for a stable syllable-per-second estimate.
_MIN_LINE_SEC = 1.2
_MIN_SYNTH_SEC = 0.8
_MIN_SYL = 4
_MIN_CHARS = 12
# Plausible articulation rates (syllables / second of active speech).
_RATE_MIN = 1.2
_RATE_MAX = 8.0
# Skip tiny speed corrections so stretch does not run for a 1% mismatch.
_ATEMPO_SKIP = 0.02
# When the synth overruns the slot, do not rush more than 15% past the actor.
_RATE_VS_ACTOR_MAX = 1.15
_SLOT_ATEMPO_MAX = 1.18
# Phrase pauses may grow up to this length; longer would sound like a stall.
_MAX_PAUSE_SEC = 0.40
# Clause-level gaps only (~150–200 ms). Word-to-word dips are ~50–80 ms
# and must not be inflated toward _MAX_PAUSE_SEC.
_MIN_PHRASE_PAUSE_SEC = 0.15
# When speeding up, freeze any pause at or above this. Shorter gaps ride with
# speech; once a pause is this long, do not shrink it further.
_MIN_KEEP_PAUSE_SEC = 0.10
# Remaining slack after pause expansion is split 30% front / 70% back.
_PAD_FRONT = 0.30
_STRETCH_FRAME = 1024

_ASTERISK_RE = re.compile(r"\*[^*]+\*")
_BRACKET_RE = re.compile(r"\[[^\[\]]*\]")
_UK_VOWEL_RE = re.compile(r"[аеєиіїоуюяАЕЄИІЇОУЮЯ]")
_LATIN_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]+")


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


def match_timing_to_reference(
    samples: np.ndarray,
    sample_rate: int,
    reference: np.ndarray,
    reference_rate: int,
    *,
    text_uk: str,
    text_en: str,
) -> np.ndarray:
    """Try to fit the synthesized clip into the reference slot."""
    audio = _mono(samples)
    if audio.size == 0 or sample_rate <= 0:
        return samples

    line = _mono(reference)
    d_line = _duration_sec(line, reference_rate)
    d_synth = _duration_sec(audio, sample_rate)
    if d_line < _MIN_LINE_SEC or d_synth < _MIN_SYNTH_SEC:
        return samples

    en = strip_nonspeech(text_en)
    uk = strip_nonspeech(text_uk)
    # English reference vs Ukrainian synthesis: same line, different syllable counts.
    s_en = count_syllables(en)
    s_uk = count_syllables(uk)
    if s_en < _MIN_SYL or s_uk < _MIN_SYL or len(en) < _MIN_CHARS or len(uk) < _MIN_CHARS:
        return samples

    audio = _trim_edge_silence(audio, sample_rate)
    d_synth = _duration_sec(audio, sample_rate)
    a_line = measure_active_speech_sec(line, reference_rate)
    a_synth = measure_active_speech_sec(audio, sample_rate)
    if a_line <= 0 or a_synth <= 0 or d_synth < _MIN_SYNTH_SEC:
        return _restore(samples, audio)

    # Rate is syllables per second of *active* speech so pauses do not look like a slow actor.
    r_actor = s_en / a_line
    r_tts = s_uk / a_synth
    can_cap_rate = _RATE_MIN <= r_actor <= _RATE_MAX and _RATE_MIN <= r_tts <= _RATE_MAX

    audio = _fit_slot(
        audio,
        sample_rate,
        slot_sec=d_line,
        r_actor=r_actor if can_cap_rate else 0.0,
        r_tts=r_tts if can_cap_rate else 0.0,
    )
    limited = apply_peak_ceiling(audio, line)
    return _restore(samples, limited)


def time_stretch(
    samples: np.ndarray, rate: float, sample_rate: int | None = None
) -> np.ndarray:
    """Pitch-preserving stretch. ``rate > 1`` is faster (shorter).

    Prefers ffmpeg ``rubberband`` / ``atempo`` when ``sample_rate`` is known.
    Falls back to overlap-add if ffmpeg is missing or fails.
    """
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0 or not math.isfinite(rate) or abs(rate - 1.0) < 1e-3:
        return audio
    rate = max(0.5, min(rate, 2.0))
    if audio.ndim == 2:
        stretched = [
            time_stretch(audio[:, index], rate, sample_rate)
            for index in range(audio.shape[1])
        ]
        length = min(channel.size for channel in stretched)
        stacked = np.stack([channel[:length] for channel in stretched], axis=1)
        return stacked.astype(np.float32, copy=False)
    if sample_rate is not None and sample_rate > 0:
        ffmpeg_out = _ffmpeg_stretch(audio, rate, sample_rate)
        if ffmpeg_out is not None:
            return ffmpeg_out
    return _ola_stretch(audio, rate)


def _fit_slot(
    audio: np.ndarray,
    sample_rate: int,
    *,
    slot_sec: float,
    r_actor: float,
    r_tts: float,
) -> np.ndarray:
    """Grow internal pauses when short; stretch speech when long, never past the actor rate."""
    d_synth = _duration_sec(audio, sample_rate)
    if _in_slot(d_synth, slot_sec):
        return audio

    if d_synth < slot_sec:
        expanded = _expand_pauses(audio, sample_rate, slot_sec)
        if _duration_sec(expanded, sample_rate) < slot_sec * (1.0 - _EPS_REL):
            return _pad_to_duration(expanded, sample_rate, slot_sec)
        return expanded

    return _stretch_speech_islands(
        audio,
        sample_rate,
        slot_sec=slot_sec,
        r_actor=r_actor,
        r_tts=r_tts,
    )


def _in_slot(duration_sec: float, slot_sec: float) -> bool:
    return abs(duration_sec - slot_sec) <= max(_EPS_REL * slot_sec, _EPS_ABS_SEC)


def _trim_edge_silence(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Drop leading/trailing silence so duration and syllable-rate use the spoken span.

    The tail uses a lower floor and a longer pad than the onset: a quiet last
    syllable is still speech, and 50 ms after the last loud window eats it.
    """
    rms, counts, peak = _window_rms(samples, sample_rate)
    if peak < _PEAK_EPS:
        return samples
    lead_floor = peak * _SPEECH_FLOOR_RATIO
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


def _expand_pauses(samples: np.ndarray, sample_rate: int, target_sec: float) -> np.ndarray:
    """Insert silence into phrase pauses, weighted by current length, capped at ``_MAX_PAUSE_SEC``."""
    need = int(round(target_sec * sample_rate)) - samples.size
    if need <= 0:
        return samples

    rms, counts, peak = _window_rms(samples, sample_rate)
    if peak < _PEAK_EPS:
        return _pad_to_duration(samples, sample_rate, target_sec)
    floor = peak * _SPEECH_FLOOR_RATIO
    hop = max(1, int(sample_rate * _WINDOW_MS / 1000.0))
    pauses = _internal_pauses(
        rms, counts, hop, floor, sample_rate, min_sec=_MIN_PHRASE_PAUSE_SEC
    )
    if not pauses:
        return _pad_to_duration(samples, sample_rate, target_sec)

    max_pause = int(sample_rate * _MAX_PAUSE_SEC)
    capacities = [max(0, max_pause - (end - start)) for start, end in pauses]
    total_cap = sum(capacities)
    if total_cap <= 0:
        return _pad_to_duration(samples, sample_rate, target_sec)

    extra = min(need, total_cap)
    weights = [end - start for start, end in pauses]
    weight_sum = sum(weights)
    allocated = [0] * len(pauses)
    remaining = extra
    for index, (capacity, weight) in enumerate(zip(capacities, weights, strict=True)):
        share = int(extra * (weight / weight_sum)) if weight_sum else 0
        take = min(capacity, share, remaining)
        allocated[index] = take
        remaining -= take
    # Integer shares can leave leftover samples; fill remaining capacity in order.
    for index, capacity in enumerate(capacities):
        if remaining <= 0:
            break
        room = capacity - allocated[index]
        if room <= 0:
            continue
        take = min(room, remaining)
        allocated[index] += take
        remaining -= take

    pieces: list[np.ndarray] = []
    cursor = 0
    for (start, end), insert in zip(pauses, allocated, strict=True):
        mid = (start + end) // 2
        pieces.append(samples[cursor:mid])
        if insert > 0:
            pieces.append(np.zeros(insert, dtype=samples.dtype))
        pieces.append(samples[mid:end])
        cursor = end
    pieces.append(samples[cursor:])
    return np.concatenate(pieces) if pieces else samples


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
            # Edge silence is trimmed/padded separately; expanding it would shift onsets.
            continue
        if end - start < min_len:
            continue
        pauses.append((start, end))
    return pauses


def _stretch_speech_islands(
    audio: np.ndarray,
    sample_rate: int,
    *,
    slot_sec: float,
    r_actor: float,
    r_tts: float,
) -> np.ndarray:
    """Speed up speech only; leave pauses at or above ``_MIN_KEEP_PAUSE_SEC`` unchanged.

    Speech islands are concatenated, stretched once, then split back so short
    phrases are not sent through rubberband on their own.
    """
    regions = _speech_pause_regions(audio, sample_rate)
    pause_samples = sum(end - start for start, end, is_pause in regions if is_pause)
    speech_samples = audio.size - pause_samples
    if speech_samples <= 0:
        return audio

    pause_sec = pause_samples / sample_rate
    target_speech = slot_sec - pause_sec
    if target_speech <= 0:
        # Pauses already fill the slot; cannot fit without cutting them.
        return audio

    atempo = (speech_samples / sample_rate) / target_speech
    if r_actor > 0 and r_tts > 0:
        # Cap so the result cannot speak more than ~15% faster than the actor.
        atempo = min(atempo, _RATE_VS_ACTOR_MAX * r_actor / r_tts)
    atempo = _clamp(atempo, 1.0, _SLOT_ATEMPO_MAX)
    if atempo <= 1.0 + _ATEMPO_SKIP:
        return audio

    # One stretch on concatenated speech: short islands sound metallic on their own.
    speech_chunks = [audio[start:end] for start, end, is_pause in regions if not is_pause]
    if not speech_chunks:
        return audio
    lengths = [chunk.size for chunk in speech_chunks]
    stretched = time_stretch(np.concatenate(speech_chunks), atempo, sample_rate)
    split = _split_stretched(stretched, lengths)
    pieces: list[np.ndarray] = []
    speech_index = 0
    for start, end, is_pause in regions:
        if is_pause:
            pieces.append(audio[start:end])
            continue
        pieces.append(split[speech_index])
        speech_index += 1
    return np.concatenate(pieces) if pieces else audio


def _split_stretched(stretched: np.ndarray, lengths: list[int]) -> list[np.ndarray]:
    """Cut a concatenated stretch back into islands, proportional to the originals."""
    total = sum(lengths)
    if total <= 0 or stretched.size == 0:
        return [stretched[:0] for _ in lengths]
    pieces: list[np.ndarray] = []
    cursor = 0
    remaining_weight = total
    remaining = stretched.size
    for index, length in enumerate(lengths):
        if index == len(lengths) - 1:
            pieces.append(stretched[cursor:])
            break
        take = int(round(remaining * (length / remaining_weight))) if remaining_weight else 0
        take = max(0, min(take, remaining))
        pieces.append(stretched[cursor : cursor + take])
        cursor += take
        remaining -= take
        remaining_weight -= length
    return pieces


def _speech_pause_regions(
    samples: np.ndarray, sample_rate: int
) -> list[tuple[int, int, bool]]:
    """Alternate speech / phrase-pause ranges. ``True`` marks a frozen pause."""
    rms, counts, peak = _window_rms(samples, sample_rate)
    if peak < _PEAK_EPS or samples.size == 0:
        return [(0, samples.size, False)]
    floor = peak * _SPEECH_FLOOR_RATIO
    hop = max(1, int(sample_rate * _WINDOW_MS / 1000.0))
    pauses = _internal_pauses(
        rms, counts, hop, floor, sample_rate, min_sec=_MIN_KEEP_PAUSE_SEC
    )
    regions: list[tuple[int, int, bool]] = []
    cursor = 0
    for start, end in pauses:
        if start > cursor:
            regions.append((cursor, start, False))
        regions.append((start, end, True))
        cursor = end
    if cursor < samples.size:
        regions.append((cursor, samples.size, False))
    return regions or [(0, samples.size, False)]


def _pad_to_duration(samples: np.ndarray, sample_rate: int, target_sec: float) -> np.ndarray:
    target_n = int(round(target_sec * sample_rate))
    extra = target_n - samples.size
    if extra <= 0:
        return samples
    front = int(extra * _PAD_FRONT)
    back = extra - front
    return np.pad(samples, (front, back))


@lru_cache(maxsize=1)
def _ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


@lru_cache(maxsize=1)
def _ffmpeg_has_rubberband() -> bool:
    binary = _ffmpeg_bin()
    if binary is None:
        return False
    proc = subprocess.run(
        [binary, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "rubberband" in (proc.stdout or "")


def _ffmpeg_stretch(
    samples: np.ndarray, rate: float, sample_rate: int
) -> np.ndarray | None:
    """Return pitch-preserving stretch via ffmpeg, or ``None`` to fall back to OLA."""
    binary = _ffmpeg_bin()
    if binary is None:
        return None
    filters = []
    if _ffmpeg_has_rubberband():
        # Formant-preserve + short window is the least metallic option for speech.
        filters.append(
            f"rubberband=tempo={rate:.6f}:pitch=1:formant=preserved:"
            f"transients=crisp:window=short:pitchq=quality"
        )
        filters.append(f"rubberband=tempo={rate:.6f}:pitch=1:formant=preserved")
        filters.append(f"rubberband=tempo={rate:.6f}:pitch=1")
    filters.append(f"atempo={rate:.6f}")

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        dest = Path(tmp) / "out.wav"
        sf.write(str(src), samples, sample_rate, subtype="FLOAT")
        for filt in filters:
            dest.unlink(missing_ok=True)
            proc = subprocess.run(
                [
                    binary,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(src),
                    "-af",
                    filt,
                    "-c:a",
                    "pcm_f32le",
                    str(dest),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0 or not dest.is_file():
                continue
            out, _rate = sf.read(str(dest), dtype="float32", always_2d=False)
            audio = np.asarray(out, dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1).astype(np.float32, copy=False)
            if audio.size > 0:
                return audio
    return None


def _ola_stretch(samples: np.ndarray, rate: float) -> np.ndarray:
    """Pitch-preserving overlap-add. ``hop_a / hop_s == rate``; ``rate > 1`` consumes faster."""
    x = np.asarray(samples, dtype=np.float64)
    frame = min(_STRETCH_FRAME, max(64, x.size // 4 * 2 or 64))
    if frame % 2:
        frame += 1
    if x.size < frame:
        n_out = max(1, int(round(x.size / rate)))
        t = np.linspace(0.0, x.size - 1, n_out)
        return np.interp(t, np.arange(x.size), x).astype(np.float32)

    hop_s = max(1, frame // 4)
    hop_a = max(1, int(round(hop_s * rate)))
    window = _hann(frame)
    n_out = max(1, int(round(x.size / rate)))
    y = np.zeros(n_out + frame, dtype=np.float64)
    wsum = np.zeros_like(y)
    pos_in = 0
    pos_out = 0
    while pos_in + frame <= x.size and pos_out + frame <= y.size:
        chunk = x[pos_in : pos_in + frame] * window
        y[pos_out : pos_out + frame] += chunk
        wsum[pos_out : pos_out + frame] += window
        pos_in += hop_a
        pos_out += hop_s
    mask = wsum > 1e-8
    y[mask] /= wsum[mask]
    stretched = y[:n_out]
    peak_in = float(np.max(np.abs(x))) if x.size else 0.0
    peak_out = float(np.max(np.abs(stretched))) if stretched.size else 0.0
    if peak_in > _PEAK_EPS and peak_out > peak_in:
        # OLA window overlap can raise peaks; keep the input ceiling so loudness stays put.
        stretched *= peak_in / peak_out
    return stretched.astype(np.float32)


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


def _hann(n: int) -> np.ndarray:
    hann = getattr(np, "hann", np.hanning)
    return hann(n).astype(np.float64)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
