"""Shared TTS text preparation: number expansion then stress marking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fish_studio.textnorm.expand_uk_numbers import expand_uk_numbers, should_expand_numbers
from fish_studio.stress import stressify

if TYPE_CHECKING:
    from fish_studio.config import StressConfig


def prepare_synthesis_text(
    text: str,
    *,
    language: str | None,
    stress: StressConfig,
) -> str:
    """Expand UK numerals, then stress-mark. Order matches training-export orthography."""
    prepared = text.strip()
    if should_expand_numbers(language):
        prepared = expand_uk_numbers(prepared)
    return stressify(prepared, stress)
