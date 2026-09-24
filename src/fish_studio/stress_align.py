"""Where did the stress actually land? CTC forced alignment of known text.

The text is known on both sides here (probe line in, transcript of a dataset
clip), which turns the problem from recognition into forced alignment: a
Ukrainian wav2vec2 CTC model gives a time span per character, so each vowel gets
its real onset and offset instead of an equal share. Stress then reads off
those spans -- duration first, since a stressed Ukrainian vowel is mainly a
longer one, with intensity as the secondary cue.

CPU only and one model load per process: the GPU belongs to vLLM.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from fish_studio.stress import COMBINING_ACUTE, strip_stress_marks

logger = logging.getLogger(__name__)

ALIGN_MODEL_ID = "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm"
TARGET_RATE = 16_000
# wav2vec2 conv stack: one frame per 320 input samples at 16 kHz.
FRAME_SEC = 320 / TARGET_RATE

_VOWELS = set("аеєиіїоуюя")
# Same token shape as stress.py so word indexes line up across modules.
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ'’ʼ́-]+", re.UNICODE)
_APOSTROPHES = {"’", "‘", "ʼ", "＇", "`"}


@dataclass(frozen=True)
class VowelSpan:
    """One aligned character: which word, which vowel of that word, and when.

    ``vowel_ordinal`` is -1 for consonants and word separators. ``end`` is where
    CTC stopped emitting this character, which is a spike rather than the phone:
    use ``filled_end`` for anything that reads duration as a stress cue.
    """

    word_index: int
    vowel_ordinal: int
    char: str
    start: float
    end: float
    # Start of the next aligned character, so the blank run between spikes is
    # charged to the character that precedes it.
    filled_end: float = 0.0
    # Mean log-probability the CTC model gave this character over its frames.
    # The model is trained on native Ukrainian, so a low score on a letter the
    # take clearly contains is the model saying "that is not how this letter
    # sounds here" -- a segmental accent, which the stress estimator cannot see.
    score: float = 0.0
    # Mean log-probability of this character's phonemic rival over the same
    # frames (г vs ґ, и vs і), when the aligner was asked for one. The plain
    # score cannot see a softened р -- the model transcribes [rʲ] as р with
    # full confidence -- but the softening drags the following и toward і,
    # and that is a rival the vocabulary does distinguish.
    rival_score: float = float("nan")

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def filled_duration(self) -> float:
        return max(0.0, max(self.filled_end, self.end) - self.start)


def vowel_spans(spans: list[VowelSpan]) -> list[VowelSpan]:
    """Only the vowels, in reading order."""
    return [span for span in spans if span.vowel_ordinal >= 0]


def normalize_for_ctc(char: str) -> str:
    """Fold a source character to the model's vocabulary, or '' to drop it."""
    if char in _APOSTROPHES:
        return "'"
    lowered = char.lower()
    if lowered == "ё":
        return "е"
    return lowered


class CtcAligner:
    """Lazy CPU wav2vec2 aligner. Construction never loads weights."""

    def __init__(
        self,
        model_id: str = ALIGN_MODEL_ID,
        *,
        cache_dir: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device
        self._model = None
        self._vocab: dict[str, int] = {}
        self._blank = 0
        self._unavailable = ""

    @property
    def available(self) -> bool:
        return not self._unavailable and self._model is not None

    def _ensure(self) -> bool:
        if self._unavailable:
            return False
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
        except ImportError as exc:
            self._unavailable = f"transformers/torch missing: {exc}"
            logger.warning("stress alignment off: %s", self._unavailable)
            return False
        try:
            kwargs = {"cache_dir": str(self.cache_dir)} if self.cache_dir else {}
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_id, **kwargs)
            model = Wav2Vec2ForCTC.from_pretrained(self.model_id, **kwargs)
        except Exception as exc:
            self._unavailable = str(exc)
            logger.warning("stress alignment off: %s", self._unavailable)
            return False
        model.eval()
        try:
            model.to(self.device)
        except Exception as exc:
            logger.warning(
                "stress alignment: %s unavailable (%s), staying on CPU", self.device, exc
            )
            self.device = "cpu"
            model.to("cpu")
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self._model = model
        self._vocab = tokenizer.get_vocab()
        self._blank = int(tokenizer.pad_token_id)
        return True

    def _tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int, str]]]:
        """Return CTC target ids plus, per id, its (word index, vowel ordinal, char).

        Non-vowels carry ordinal -1. Characters outside the vocabulary are
        dropped rather than mapped to [UNK]: an unknown id would consume frames
        the neighbouring vowels need.
        """
        targets: list[int] = []
        origin: list[tuple[int, int, str]] = []
        separator = self._vocab.get("|")
        for word_index, match in enumerate(_WORD_RE.finditer(text)):
            word = strip_stress_marks(match.group(0))
            if targets and separator is not None:
                targets.append(separator)
                origin.append((-1, -1, " "))
            seen_vowels = 0
            for char in word:
                folded = normalize_for_ctc(char)
                token = self._vocab.get(folded)
                if token is None:
                    continue
                ordinal = -1
                if folded in _VOWELS:
                    ordinal = seen_vowels
                    seen_vowels += 1
                targets.append(token)
                origin.append((word_index, ordinal, folded))
        return targets, origin

    def align(
        self,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
        *,
        rival_of: dict[str, str] | None = None,
    ) -> list[VowelSpan] | None:
        """Character spans for ``text`` as spoken in ``audio``, or None if unusable.

        ``rival_of`` maps a character to the vocabulary token it is most likely
        to be confused with under an accent (``{"г": "ґ", "и": "і"}``); each
        such span also carries the rival's mean log-probability over its frames.
        """
        if not self._ensure():
            return None
        samples = _to_mono_16k(audio, sample_rate)
        if samples.size < TARGET_RATE // 10:
            return None
        targets, origin = self._tokenize(text)
        if not targets:
            return None

        import torch
        import torchaudio.functional as AF

        wav = torch.from_numpy(samples).unsqueeze(0)
        # The feature extractor only zero-means / unit-variances the waveform.
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)
        with torch.inference_mode():
            logits = self._model(wav.to(self.device)).logits
        # forced_align runs on CPU; the win from a GPU is in the forward pass.
        log_probs = torch.log_softmax(logits.float(), dim=-1).cpu()
        if log_probs.shape[1] < len(targets):
            # Fewer frames than characters: alignment has no valid path.
            return None
        target_tensor = torch.tensor([targets], dtype=torch.int32)
        try:
            aligned, scores = AF.forced_align(log_probs, target_tensor, blank=self._blank)
            # merge_tokens collapses blanks and frame repeats into one span per
            # target position, in order -- so it indexes straight into origin.
            token_spans = AF.merge_tokens(aligned[0], scores[0], blank=self._blank)
        except Exception:
            logger.exception("forced_align failed")
            return None
        if len(token_spans) != len(origin):
            logger.warning(
                "alignment produced %d spans for %d targets", len(token_spans), len(origin)
            )
            return None

        spans: list[VowelSpan] = []
        for position, (token_span, origin_entry) in enumerate(
            zip(token_spans, origin, strict=True)
        ):
            word_index, ordinal, char = origin_entry
            next_start = (
                token_spans[position + 1].start * FRAME_SEC
                if position + 1 < len(token_spans)
                else token_span.end * FRAME_SEC
            )
            rival_score = float("nan")
            rival = (rival_of or {}).get(char)
            rival_id = self._vocab.get(rival) if rival else None
            if rival_id is not None and token_span.end > token_span.start:
                rival_score = float(
                    log_probs[0, token_span.start : token_span.end, rival_id].mean()
                )
            spans.append(
                VowelSpan(
                    word_index=word_index,
                    vowel_ordinal=ordinal,
                    char=char,
                    start=token_span.start * FRAME_SEC,
                    end=token_span.end * FRAME_SEC,
                    filled_end=next_start,
                    score=float(token_span.score),
                    rival_score=rival_score,
                )
            )
        return spans


def _to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if int(sample_rate) == TARGET_RATE:
        return samples
    import torch
    import torchaudio

    wav = torch.from_numpy(samples).unsqueeze(0)
    return (
        torchaudio.functional.resample(wav, int(sample_rate), TARGET_RATE)
        .squeeze(0)
        .numpy()
        .astype(np.float32)
    )


def marked_ordinals(text: str) -> dict[int, int]:
    """Word index -> which vowel carries U+0301, for multi-vowel words only.

    Ordinals rather than string offsets, so a text whose apostrophes were
    repaired still lines up with one whose were not.
    """
    found: dict[int, int] = {}
    for index, match in enumerate(_WORD_RE.finditer(text)):
        word = match.group(0)
        position = word.find("́")
        if position < 0:
            continue
        plain = strip_stress_marks(word)
        if sum(1 for ch in plain.lower() if ch in _VOWELS) < 2:
            continue
        found[index] = sum(1 for ch in word[:position].lower() if ch in _VOWELS) - 1
    return found


def measure_stress_with_margin(
    text: str,
    samples: np.ndarray,
    sample_rate: int,
    aligner: CtcAligner,
) -> dict[int, tuple[int, float]] | None:
    """Word index -> (stressed vowel, how far it beat the runner-up).

    The margin is the winning score divided by the second-best, and it is what
    makes the estimator usable for *writing* marks rather than only grading
    them. Measured on 2637 dictionary-marked words: taking every verdict is
    72.1% correct, while requiring a 2x margin is 85.5% correct over 36% of
    words. A wrong mark teaches the wrong stress, so the caller usually wants
    the second trade.

    Words whose aligned vowel count disagrees with the text are left out rather
    than guessed: a dropped character shifts every later ordinal.
    """
    spans = aligner.align(text, samples, sample_rate)
    if spans is None:
        return None

    grouped: dict[int, list[VowelSpan]] = {}
    for span in vowel_spans(spans):
        grouped.setdefault(span.word_index, []).append(span)

    expected_counts: dict[int, int] = {}
    for index, match in enumerate(_WORD_RE.finditer(text)):
        plain = strip_stress_marks(match.group(0)).lower()
        expected_counts[index] = sum(1 for ch in plain if ch in _VOWELS)

    heard: dict[int, tuple[int, float]] = {}
    for word_index, word_spans in grouped.items():
        if len(word_spans) < 2 or len(word_spans) != expected_counts.get(word_index):
            continue
        word_spans.sort(key=lambda span: span.vowel_ordinal)
        scores = [
            span.filled_duration * vowel_energy(samples, sample_rate, span, filled=True)
            for span in word_spans
        ]
        if max(scores) <= 0.0:
            continue
        best = max(range(len(scores)), key=lambda i: scores[i])
        ranked = sorted(scores, reverse=True)
        margin = ranked[0] / ranked[1] if ranked[1] > 0 else float("inf")
        heard[word_index] = (best, margin)
    return heard


def measure_stress(
    text: str,
    samples: np.ndarray,
    sample_rate: int,
    aligner: CtcAligner,
) -> dict[int, int] | None:
    """Word index -> the vowel the speaker actually stressed. None if unalignable.

    Every verdict, regardless of confidence -- which is what grading wants,
    since dropping the uncertain words would score a checkpoint only on the
    easy ones. Use ``measure_stress_with_margin`` when writing marks.
    """
    heard = measure_stress_with_margin(text, samples, sample_rate, aligner)
    if heard is None:
        return None
    return {index: choice for index, (choice, _) in heard.items()}


def measure_gop(
    text: str,
    samples: np.ndarray,
    sample_rate: int,
    aligner: CtcAligner,
) -> dict[str, list[float]] | None:
    """Per character: the CTC log-probabilities of every aligned instance.

    Goodness of pronunciation, the plain form: how strongly a native-Ukrainian
    CTC model recognises each letter of the known text in the take. Averaged
    over a probe set and split by letter, it is the number the stress estimator
    cannot give -- a softened "р" or a fronted "и" leaves stress intact and
    shows up here instead. Comparable across checkpoints on the same probe set,
    like every other axis; never a nativeness score in absolute terms.
    """
    spans = aligner.align(text, samples, sample_rate)
    if spans is None:
        return None
    scores: dict[str, list[float]] = {}
    for span in spans:
        if span.word_index < 0 or not span.char.strip():
            continue
        scores.setdefault(span.char, []).append(span.score)
    return scores


# The confusions an accent produces that the CTC vocabulary can tell apart.
DEFAULT_RIVALS: dict[str, str] = {"г": "ґ", "и": "і", "е": "є", "ш": "щ"}
# Context keys: the vowel after these consonants reports separately, because a
# softened consonant shows up as its vowel drifting toward the front rival.
DEFAULT_AFTER: dict[str, tuple[str, ...]] = {"и": ("р", "л", "н", "т", "д", "с", "з")}


def measure_margins(
    text: str,
    samples: np.ndarray,
    sample_rate: int,
    aligner: CtcAligner,
    *,
    rivals: dict[str, str] | None = None,
    after: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, list[float]] | None:
    """Per letter: log P(letter) - log P(its rival) over the aligned frames.

    Plain per-letter confidence is blind to a softened р: the model transcribes
    [rʲ] as р with full confidence. What it does see is the neighbour that
    softening drags -- и toward і -- and the pairs it has tokens for, г against
    ґ above all. Positive means the native reading wins; the smaller, the
    closer the take is to the accented one. Keys are the letter, plus
    ``letter/consonant`` for the contexts in ``after``.
    """
    rivals = DEFAULT_RIVALS if rivals is None else rivals
    after = DEFAULT_AFTER if after is None else after
    spans = aligner.align(text, samples, sample_rate, rival_of=rivals)
    if spans is None:
        return None
    margins: dict[str, list[float]] = {}
    previous: str | None = None
    for span in spans:
        if span.word_index < 0 or not span.char.strip():
            previous = None
            continue
        if span.char in rivals and span.rival_score == span.rival_score:
            margin = span.score - span.rival_score
            margins.setdefault(span.char, []).append(margin)
            if previous is not None and previous in after.get(span.char, ()):
                margins.setdefault(f"{span.char}/{previous}", []).append(margin)
        previous = span.char
    return margins


# Consonants Ukrainian keeps hard before back vowels and an accent softens.
PALATAL_CONSONANTS = ("р", "л", "н", "т", "д", "с", "з")
# Back vowels: a hard consonant before these leaves the vowel onset alone; a
# softened one puts a [j]-like transition there, which raises F2 at the onset.
BACK_VOWELS = ("а", "о", "у", "и", "е")
_ONSET_SEC = 0.025


def f2_track(samples: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    """Second-formant values (Hz) over ``[start, end)``; empty when Praat finds none."""
    lo = max(0, int(start * sample_rate))
    hi = min(samples.size, int(end * sample_rate))
    if hi - lo < sample_rate // 50:
        return np.zeros(0)
    try:
        import parselmouth
    except ImportError:
        return np.zeros(0)
    try:
        sound = parselmouth.Sound(samples[lo:hi].astype(np.float64), sampling_frequency=sample_rate)
        formant = sound.to_formant_burg(time_step=0.005, max_number_of_formants=5)
        times = np.arange(0.0, formant.duration, 0.005)
        values = np.array([formant.get_value_at_time(2, t) for t in times], dtype=np.float64)
    except Exception:
        return np.zeros(0)
    return values[np.isfinite(values)]


def palatalization_index(
    samples: np.ndarray, sample_rate: int, consonant: VowelSpan, vowel: VowelSpan
) -> float | None:
    """F2 right after the consonant's release minus F2 at the vowel's middle, Hz.

    A hard consonant releases straight into the vowel, so F2 at the release
    sits near the vowel's steady state. A softened one carries a [j]-like
    transition there, and F2 starts high and falls. Larger is more
    palatalized.

    Where the window sits matters more than it looks. CTC spikes come late,
    so the vowel span's own start is already inside the vowel, and a window
    placed there measured the vowel's intrinsic drift instead (р before у read
    -355 Hz for the model and -485 for the human recordings -- the
    diphthongal у, not the р). The transition lives in the blank run after the
    consonant's spike, so the onset window starts at ``consonant.end``.
    """
    onset_start = consonant.end
    vowel_end = max(vowel.filled_end, vowel.end)
    if vowel_end - onset_start < 3 * _ONSET_SEC:
        return None
    onset = f2_track(samples, sample_rate, onset_start, onset_start + _ONSET_SEC)
    middle_at = vowel.start + (vowel_end - vowel.start) / 2.0
    middle = f2_track(samples, sample_rate, middle_at - _ONSET_SEC / 2, middle_at + _ONSET_SEC / 2)
    if onset.size == 0 or middle.size == 0:
        return None
    return float(np.median(onset) - np.median(middle))


def measure_palatalization(
    text: str,
    samples: np.ndarray,
    sample_rate: int,
    aligner: CtcAligner,
    *,
    consonants: tuple[str, ...] = PALATAL_CONSONANTS,
    vowels: tuple[str, ...] = BACK_VOWELS,
) -> dict[str, list[float]] | None:
    """Per consonant: palatalization index of every back vowel that follows it.

    The CTC model is trained to transcribe, so it is deliberately blind to
    allophony: it scores a softened р as р with full confidence, and the
    margin against a rival token saturates at 8-10 nats. This reads the
    acoustics directly instead. Keys are the consonant; ``consonant/vowel``
    keys break it down by what follows. Comparable across checkpoints, and
    against the human recordings of the same lines.
    """
    spans = aligner.align(text, samples, sample_rate)
    if spans is None:
        return None
    out: dict[str, list[float]] = {}
    for previous, span in itertools.pairwise(spans):
        if previous.word_index < 0 or span.word_index != previous.word_index:
            continue
        if previous.char not in consonants or span.char not in vowels:
            continue
        index = palatalization_index(samples, sample_rate, previous, span)
        if index is None:
            continue
        out.setdefault(previous.char, []).append(index)
        out.setdefault(f"{previous.char}/{span.char}", []).append(index)
    return out


_TRILL_HOP_SEC = 0.004
_TRILL_WIN_SEC = 0.008


def rms_envelope(samples: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    """Short-window RMS over ``[start, end)`` at a 4 ms hop; empty if too short."""
    lo, hi = max(0, int(start * sample_rate)), min(samples.size, int(end * sample_rate))
    hop, win = int(_TRILL_HOP_SEC * sample_rate), int(_TRILL_WIN_SEC * sample_rate)
    if hi - lo < win + hop:
        return np.zeros(0)
    window = samples[lo:hi].astype(np.float64)
    return np.array(
        [np.sqrt(np.mean(np.square(window[i : i + win]))) for i in range(0, window.size - win, hop)]
    )


def trill_features(
    samples: np.ndarray, sample_rate: int, consonant: VowelSpan, vowel: VowelSpan
) -> dict[str, float] | None:
    """How much of a trill this р is: duration, closures, and level against its vowel.

    What the listener called a soft р turned out, on the one phrase they
    labelled, to be a short, weak р with no closures -- an approximant where
    Ukrainian has a trill. The one р they accepted was the longest, the loudest
    against its vowel, and the only one with three or four dips in its
    envelope. Palatalization proper never showed: the formant index put that
    same р anywhere from +805 to -1565 Hz across takes, which is the tracker
    losing F2 on у, not the consonant.

    The consonant is taken from its own CTC spike to the vowel's, the vowel
    over its filled span. A dip is a local minimum of the 4 ms RMS envelope
    that sits below 85% of the envelope's peak -- a closure, loosely.
    """
    envelope_r = rms_envelope(samples, sample_rate, consonant.start, vowel.start)
    envelope_v = rms_envelope(samples, sample_rate, vowel.start, max(vowel.filled_end, vowel.end))
    if envelope_r.size < 3 or envelope_v.size < 3:
        return None
    inner = envelope_r[1:-1]
    dips = int(
        np.sum(
            (inner < envelope_r[:-2]) & (inner < envelope_r[2:]) & (inner < 0.85 * envelope_r.max())
        )
    )
    return {
        "dur_ms": 1000.0 * (vowel.start - consonant.start),
        "rv_ratio": float(envelope_r.mean() / (envelope_v.mean() + 1e-9)),
        "dips": float(dips),
    }


def measure_trill(
    text: str,
    samples: np.ndarray,
    sample_rate: int,
    aligner: CtcAligner,
    *,
    vowels: tuple[str, ...] = BACK_VOWELS,
) -> dict[str, list[dict[str, float]]] | None:
    """Trill features for every р before a back vowel, keyed ``р`` and ``р/<vowel>``."""
    spans = aligner.align(text, samples, sample_rate)
    if spans is None:
        return None
    out: dict[str, list[dict[str, float]]] = {}
    for previous, span in itertools.pairwise(spans):
        if previous.char != "р" or span.word_index != previous.word_index:
            continue
        if span.vowel_ordinal < 0 or span.char not in vowels:
            continue
        features = trill_features(samples, sample_rate, previous, span)
        if features is None:
            continue
        out.setdefault("р", []).append(features)
        out.setdefault(f"р/{span.char}", []).append(features)
    return out


def _formants_at(
    samples: np.ndarray, sample_rate: int, at: float, half_window: float = 0.015
) -> tuple[float, float] | None:
    """Median F1, F2 (Hz) over a short window centred on ``at``."""
    lo = max(0, int((at - half_window) * sample_rate))
    hi = min(samples.size, int((at + half_window) * sample_rate))
    if hi - lo < sample_rate // 100:
        return None
    try:
        import parselmouth
    except ImportError:
        return None
    try:
        sound = parselmouth.Sound(samples[lo:hi].astype(np.float64), sampling_frequency=sample_rate)
        formant = sound.to_formant_burg(time_step=0.005, max_number_of_formants=5)
        times = np.arange(0.0, formant.duration, 0.005)
        f1 = np.array([formant.get_value_at_time(1, t) for t in times], dtype=np.float64)
        f2 = np.array([formant.get_value_at_time(2, t) for t in times], dtype=np.float64)
    except Exception:
        return None
    f1, f2 = f1[np.isfinite(f1)], f2[np.isfinite(f2)]
    if f1.size == 0 or f2.size == 0:
        return None
    return float(np.median(f1)), float(np.median(f2))


def measure_vowel_formants(
    text: str,
    samples: np.ndarray,
    sample_rate: int,
    aligner: CtcAligner,
    *,
    vowels: tuple[str, ...] = ("и", "і"),
) -> dict[str, list[tuple[float, float]]] | None:
    """F1, F2 at the middle of every aligned instance of the listed vowels.

    Ukrainian и is [ɪ]: higher F1 and lower F2 than і [i], by hundreds of Hz.
    A model that reads и as і -- the listener's report -- collapses that
    distance. Unlike palatalization, this is a large vowel-quality contrast
    the formant tracker handles well, and the human recording of the same
    line gives the native distance for the same words.
    """
    spans = aligner.align(text, samples, sample_rate)
    if spans is None:
        return None
    out: dict[str, list[tuple[float, float]]] = {}
    for span in spans:
        if span.word_index < 0 or span.char not in vowels:
            continue
        end = max(span.filled_end, span.end)
        if end - span.start < 0.04:
            continue
        pair = _formants_at(samples, sample_rate, span.start + (end - span.start) / 2.0)
        if pair is None:
            continue
        out.setdefault(span.char, []).append(pair)
    return out


def measure_pitch_spread(samples: np.ndarray, sample_rate: int) -> dict[str, float] | None:
    """How much the pitch moves over a take: F0 std and 5-95 range, in semitones.

    The blind A/B set put the third axis on the table: the listener called the
    0-11 merge clearer but "machine-like", and in the two pairs where they
    named the served model's intonation as the more natural one, that model's
    pitch range was two to three times the candidate's. Semitones relative to
    the take's own median, so a low voice and a high one compare directly;
    the human recording of the same line is the natural reference. Needs no
    alignment. None when Praat finds too little voicing to say anything.
    """
    if samples.size < sample_rate // 4:
        return None
    try:
        import parselmouth
    except ImportError:
        return None
    try:
        sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sample_rate)
        f0 = sound.to_pitch(time_step=0.01).selected_array["frequency"]
    except Exception:
        return None
    voiced = f0[f0 > 0]
    if voiced.size < 10:
        return None
    semitones = 12.0 * np.log2(voiced / np.median(voiced))
    return {
        "f0_std_st": float(semitones.std()),
        "f0_range_st": float(np.percentile(semitones, 95) - np.percentile(semitones, 5)),
        "voiced": float(voiced.size / f0.size),
    }


def vowel_energy(
    samples: np.ndarray,
    sample_rate: int,
    span: VowelSpan,
    *,
    filled: bool = False,
) -> float:
    """RMS over the vowel's span. 0.0 when the span falls outside the clip."""
    stop = max(span.filled_end, span.end) if filled else span.end
    start = max(0, int(span.start * sample_rate))
    end = min(samples.size, int(stop * sample_rate))
    if end <= start:
        return 0.0
    window = samples[start:end]
    return float(np.sqrt(np.mean(np.square(window.astype(np.float64)))))


def vowel_pitch(
    samples: np.ndarray,
    sample_rate: int,
    span: VowelSpan,
    *,
    filled: bool = True,
) -> float:
    """Mean F0 over the vowel, in Hz. 0.0 when Praat finds no voicing.

    praat-parselmouth is already a hard server dependency (PSOLA), so pitch
    costs nothing extra to try as a third stress cue.
    """
    stop = max(span.filled_end, span.end) if filled else span.end
    start = max(0, int(span.start * sample_rate))
    end = min(samples.size, int(stop * sample_rate))
    if end - start < sample_rate // 100:
        return 0.0
    try:
        import parselmouth
    except ImportError:
        return 0.0
    try:
        sound = parselmouth.Sound(
            samples[start:end].astype(np.float64), sampling_frequency=sample_rate
        )
        pitch = sound.to_pitch()
        values = pitch.selected_array["frequency"]
    except Exception:
        return 0.0
    voiced = values[values > 0]
    return float(np.mean(voiced)) if voiced.size else 0.0


def needs_stress_fill(text: str) -> bool:
    """Is there a word of two or more vowels that still carries no mark?

    The export asks this before touching the audio: about nine clips in ten
    come out of the text stages fully marked, and reading and aligning them
    would be the whole cost of the fill for nothing.
    """
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        if COMBINING_ACUTE in word:
            continue
        if sum(1 for ch in word.lower() if ch in _VOWELS) >= 2:
            return True
    return False


def fill_stress_from_audio(
    text: str,
    audio_path: Path | str,
    *,
    min_margin: float = 2.0,
    aligner: CtcAligner | None = None,
    device: str = "cpu",
) -> str:
    """Mark still-unmarked multi-vowel words from the recording that goes with them.

    Only used where a paired WAV exists -- dataset export -- and only for words
    the dictionary, lexicon and Stanza all left bare. Those are the words the
    model would otherwise have to learn from audio alone.

    The margin gate is the point of this function. Filling every word is 72%
    accurate, which writes a wrong mark into roughly one transcript word in
    four, and a wrong mark is worse than none: it teaches the wrong stress
    instead of teaching nothing. At the default 2x margin the fill is 85.5%
    accurate over about a third of words. Those figures come from
    dictionary-known words; genuinely out-of-vocabulary ones tend to be longer,
    where the estimator is weaker, so treat them as the optimistic end.

    Returns ``text`` unchanged when the aligner is unavailable -- silently
    guessing with a weaker estimator would defeat the gate.
    """
    path = Path(audio_path)
    if not text.strip() or not path.is_file():
        return text
    engine = aligner if aligner is not None else _shared_aligner(device)
    try:
        samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        logger.warning("stress fill: cannot read %s", path)
        return text
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)

    heard = measure_stress_with_margin(
        text, np.asarray(samples, dtype=np.float32), int(rate), engine
    )
    if not heard:
        return text

    pieces: list[str] = []
    last = 0
    for index, match in enumerate(_WORD_RE.finditer(text)):
        word = match.group(0)
        pieces.append(text[last : match.start()])
        last = match.end()
        verdict = heard.get(index)
        if COMBINING_ACUTE in word or verdict is None:
            pieces.append(word)
            continue
        choice, margin = verdict
        if margin < min_margin:
            pieces.append(word)
            continue
        pieces.append(_insert_acute(word, choice))
    pieces.append(text[last:])
    return "".join(pieces)


def _insert_acute(word: str, vowel_ordinal: int) -> str:
    """Put U+0301 after the nth vowel of ``word``."""
    seen = -1
    for position, char in enumerate(word):
        if char.lower() in _VOWELS:
            seen += 1
            if seen == vowel_ordinal:
                return word[: position + 1] + COMBINING_ACUTE + word[position + 1 :]
    return word


@lru_cache(maxsize=2)
def _shared_aligner(device: str = "cpu") -> CtcAligner:
    """One model load per process: dataset export calls this per clip.

    On CPU the forward pass costs about 335 ms per clip, or 18 hours over a
    196k-clip corpus. Dataset preparation is the one time vLLM is not holding
    the GPU, so that is when ``device="cuda"`` is worth passing.
    """
    return CtcAligner(device=device)
