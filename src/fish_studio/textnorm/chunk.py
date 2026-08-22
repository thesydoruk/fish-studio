"""Split long TTS text so each Fish Speech request stays in the trained length range."""

from __future__ import annotations

import re

# Sentence end, then whitespace. Ellipsis (… / ...) counts as an end mark.
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_RE = re.compile(r"(?<=[;:—–])\s+")
_COMMA_RE = re.compile(r"(?<=,)\s+")
_SPACE_RE = re.compile(r"\s+")


def split_synthesis_chunks(text: str, max_chars: int) -> list[str]:
    """Pack sentences into chunks of at most ``max_chars`` (0 disables splitting).

    Split order is sentence → clause → comma → spaces → hard wrap, so cuts stay
    at linguistic boundaries whenever the next level still fits.
    """
    prepared = text.strip()
    if not prepared:
        return []
    if max_chars <= 0 or len(prepared) <= max_chars:
        return [prepared]

    packed: list[str] = []
    current = ""
    for sentence in _split_keep_separators(prepared, _SENTENCE_RE):
        if len(sentence) > max_chars:
            if current:
                packed.append(current)
                current = ""
            packed.extend(_split_long_sentence(sentence, max_chars))
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            packed.append(current)
            current = sentence
    if current:
        packed.append(current)
    return packed


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Fall back through weaker punctuation when a single sentence exceeds ``max_chars``."""
    for pattern in (_CLAUSE_RE, _COMMA_RE):
        parts = _split_keep_separators(sentence, pattern)
        if len(parts) > 1 and all(len(part) <= max_chars for part in parts):
            return _pack_parts(parts, max_chars)
        if len(parts) > 1:
            packed: list[str] = []
            for part in parts:
                if len(part) <= max_chars:
                    packed.append(part)
                else:
                    packed.extend(_split_on_spaces(part, max_chars))
            return _pack_parts(packed, max_chars)
    return _split_on_spaces(sentence, max_chars)


def _split_on_spaces(text: str, max_chars: int) -> list[str]:
    words = [word for word in _SPACE_RE.split(text) if word]
    if not words:
        return [text[:max_chars]] if text else []
    return _pack_parts(words, max_chars)


def _pack_parts(parts: list[str], max_chars: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if len(part) > max_chars:
            if current:
                packed.append(current)
                current = ""
            packed.extend(_hard_wrap(part, max_chars))
            continue
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            packed.append(current)
            current = part
    if current:
        packed.append(current)
    return packed


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    """Last resort: cut mid-token so a request never exceeds the model length."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _split_keep_separators(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split on ``pattern`` and keep each piece stripped and non-empty."""
    return [piece.strip() for piece in pattern.split(text) if piece.strip()]
