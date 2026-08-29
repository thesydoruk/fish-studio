"""Acoustic stress fallback for unmarked Ukrainian words.

Used only when a paired WAV is available (dataset / train export).
Synthesis requests have no aligned audio, so they never call this path.

Closed-class / clitic forms are skipped (filters remove the noisy cases that
heavy F0 scoring barely improved). Remaining multi-vowel words get a mark when
one syllable's RMS energy clearly outranks the rest.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf

from fish_studio.stress import COMBINING_ACUTE, strip_stress_marks

_VOWELS = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")
_WORD_RE = re.compile(
    r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ'’ʼ\u0301-]+",
    re.UNICODE,
)

# Require the winning syllable to beat the second by this factor, else skip.
_MIN_PROMINENCE_RATIO = 1.15
_MIN_WORD_SEC = 0.08

# Pronouns / clitics / closed-class forms where acoustic marking is unreliable.
# Casefolded, without stress marks.
_SKIP_WORDS = frozenset(
    {
        "не",
        "на",
        "що",
        "він",
        "до",
        "це",
        "як",
        "за",
        "так",
        "її",
        "ви",
        "про",
        "ти",
        "та",
        "був",
        "ще",
        "від",
        "чи",
        "ми",
        "все",
        "йому",
        "коли",
        "нього",
        "із",
        "тут",
        "вже",
        "то",
        "щоб",
        "там",
        "по",
        "того",
        "ні",
        "для",
        "те",
        "була",
        "їх",
        "той",
        "вам",
        "вас",
        "щось",
        "але",
        "або",
        "бо",
        "же",
        "би",
        "б",
        "і",
        "й",
        "у",
        "в",
        "з",
        "зі",
        "під",
        "над",
        "при",
        "без",
        "через",
        "після",
        "перед",
        "між",
        "коло",
        "біля",
        "ось",
        "ну",
        "хай",
        "нехай",
        "аби",
        "хоч",
        "хоча",
        "якщо",
        "тоді",
        "тепер",
        "потім",
        "тому",
        "цього",
        "цю",
        "ця",
        "цей",
        "ці",
        "цих",
        "цим",
        "свої",
        "свій",
        "своя",
        "своє",
        "мене",
        "мені",
        "мною",
        "тебе",
        "тобі",
        "тобою",
        "нас",
        "нам",
        "нами",
        "них",
        "ним",
        "ними",
        "нею",
        "неї",
        "його",
        "вона",
        "воно",
        "вони",
        "я",
        "а",
        "де",
        "куди",
        "звідки",
        "чому",
        "хто",
        "кого",
        "кому",
        "ким",
        "чим",
        "чого",
        "собі",
        "себе",
        "собою",
        "було",
        "були",
        "є",
        "буде",
        "може",
        "можна",
        "треба",
        "має",
        "ньому",
        "вами",
        "отож",
        "також",
    }
)


def _vowel_indices(word: str) -> list[int]:
    return [i for i, ch in enumerate(word) if ch in _VOWELS]


def _insert_stress(word: str, vowel_idx: int) -> str:
    if COMBINING_ACUTE in word:
        return word
    return word[: vowel_idx + 1] + COMBINING_ACUTE + word[vowel_idx + 1 :]


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float64)
    return audio, int(sr)


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def _should_skip_word(plain: str) -> bool:
    return plain.casefold() in _SKIP_WORDS


def _pick_stressed_vowel(word: str, samples: np.ndarray) -> int | None:
    """Return index of the stressed vowel in ``word``, or None if unsure."""
    plain = strip_stress_marks(word)
    vowels = _vowel_indices(plain)
    if len(vowels) < 2 or samples.size < 8:
        return None

    n = len(vowels)
    # No phone alignment: split the word WAV into equal-time vowel bins.
    chunk = samples.size / n
    scores: list[float] = []
    for i in range(n):
        start = int(i * chunk)
        end = int((i + 1) * chunk) if i + 1 < n else samples.size
        window = samples[start:end]
        # Onsets/offsets are often louder than the nucleus; down-weight the edges.
        edge_penalty = 0.92 if i == 0 or i == n - 1 else 1.0
        scores.append(_rms(window) * edge_penalty)

    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    best, second = order[0], order[1]
    if scores[best] <= 0:
        return None
    if scores[best] < scores[second] * _MIN_PROMINENCE_RATIO:
        return None
    return vowels[best]


def apply_acoustic_stress(
    text: str,
    audio_path: str | Path,
    *,
    min_vowels: int = 2,
    word_spans: list[tuple[float, float]] | None = None,
) -> str:
    """Fill combining acutes on still-unmarked multi-vowel words using WAV energy."""
    path = Path(audio_path)
    if not text.strip() or not path.is_file():
        return text

    try:
        audio, sr = _load_mono(path)
    except Exception:
        return text
    if audio.size == 0 or sr <= 0:
        return text

    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return text

    duration = audio.size / sr
    weights = [
        max(1, len(_vowel_indices(strip_stress_marks(m.group(0))))) for m in matches
    ]
    total_w = float(sum(weights))

    if word_spans is not None and len(word_spans) == len(matches):
        spans = [(max(0.0, s), min(duration, e)) for s, e in word_spans]
    else:
        # Without word timestamps, apportion the clip by vowel count (longer words get more time).
        spans = []
        cursor = 0.0
        for weight in weights:
            span = duration * (weight / total_w)
            spans.append((cursor, cursor + span))
            cursor += span

    pieces: list[str] = []
    last = 0
    for match, (word_start, word_end) in zip(matches, spans, strict=True):
        word = match.group(0)
        pieces.append(text[last : match.start()])
        last = match.end()

        plain = strip_stress_marks(word)
        span = max(0.0, word_end - word_start)
        if (
            COMBINING_ACUTE in word
            or len(_vowel_indices(plain)) < min_vowels
            or _should_skip_word(plain)
            or span < _MIN_WORD_SEC
        ):
            pieces.append(word)
            continue

        a0 = max(0, int(word_start * sr))
        a1 = min(audio.size, int(word_end * sr))
        vowel_idx = _pick_stressed_vowel(plain, audio[a0:a1])
        if vowel_idx is None:
            pieces.append(word)
            continue
        pieces.append(_insert_stress(plain, vowel_idx))

    pieces.append(text[last:])
    return "".join(pieces)
