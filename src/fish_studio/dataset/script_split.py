"""Split aligned speech on writing system so EN and UK do not share a clip."""

from __future__ import annotations

import re
from dataclasses import replace

from fish_studio.dataset.transcript import TranscriptSegment, TranscriptWord

_CYRILLIC = re.compile(r"[\u0400-\u04FF\u0490-\u0491]")
_LATIN = re.compile(r"[A-Za-z]")
_RU_LETTERS = re.compile(r"[ыэёъЫЭЁЪ]")
_UK_LETTERS = re.compile(r"[ґєіїҐЄІЇ]")


def word_script(word: str) -> str | None:
    """Return ``uk`` / ``en`` from letters in ``word``, or None for punctuation-only."""
    has_cyrillic = bool(_CYRILLIC.search(word))
    has_latin = bool(_LATIN.search(word))
    if has_cyrillic and has_latin:
        cyrillic_n = len(_CYRILLIC.findall(word))
        latin_n = len(_LATIN.findall(word))
        return "uk" if cyrillic_n >= latin_n else "en"
    if has_cyrillic:
        return "uk"
    if has_latin:
        return "en"
    return None


def assign_clip_language(text: str, script: str | None, whisper_language: str | None) -> str | None:
    """Map a span to ``en`` / ``uk`` / ``ru`` (or another Whisper tag) for the allowlist."""
    detected = (whisper_language or "").strip().lower() or None
    if script == "en":
        return "en"
    if script != "uk":
        return detected

    if _RU_LETTERS.search(text):
        return "ru"
    if _UK_LETTERS.search(text):
        return "uk"
    if detected in {"ru", "uk"}:
        return detected
    return "uk"


def refine_segment_language(segment: TranscriptSegment) -> TranscriptSegment:
    """Re-label a span from letters first, Whisper LID only as a Cyrillic tie-break."""
    return replace(
        segment,
        language=assign_clip_language(segment.text, word_script(segment.text), segment.language),
    )


def split_segment_by_script(segment: TranscriptSegment) -> list[TranscriptSegment]:
    """Cut one span into same-script runs. Monolingual segments stay intact."""
    if not segment.words:
        return [refine_segment_language(segment)]

    runs: list[tuple[str, list[TranscriptWord]]] = []
    current_lang: str | None = None
    current_words: list[TranscriptWord] = []

    def flush() -> None:
        nonlocal current_lang, current_words
        if current_lang and current_words:
            runs.append((current_lang, current_words))
        current_lang = None
        current_words = []

    for word in segment.words:
        lang = word_script(word.word)
        if lang is None:
            if current_words:
                current_words.append(word)
            continue
        if current_lang is None:
            current_lang = lang
            current_words = [word]
            continue
        if lang == current_lang:
            current_words.append(word)
            continue
        flush()
        current_lang = lang
        current_words = [word]
    flush()

    if len(runs) <= 1:
        if runs:
            language = assign_clip_language(segment.text, runs[0][0], segment.language)
            return [replace(segment, language=language)]
        return [refine_segment_language(segment)]

    return [_segment_from_run(segment, script, words) for script, words in runs]


def split_segments_by_script(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Apply :func:`split_segment_by_script` to every span."""
    split: list[TranscriptSegment] = []
    for segment in segments:
        split.extend(split_segment_by_script(segment))
    return split


def _segment_from_run(
    source: TranscriptSegment,
    script: str,
    words: list[TranscriptWord],
) -> TranscriptSegment:
    text = " ".join(word.word for word in words).strip()
    return TranscriptSegment(
        id=source.id,
        start=words[0].start,
        end=words[-1].end,
        text=text,
        avg_logprob=source.avg_logprob,
        no_speech_prob=source.no_speech_prob,
        compression_ratio=source.compression_ratio,
        words=list(words),
        language=assign_clip_language(text, script, source.language),
        language_probability=source.language_probability,
        alignment_failed=source.alignment_failed,
        speaker_id=source.speaker_id,
    )
