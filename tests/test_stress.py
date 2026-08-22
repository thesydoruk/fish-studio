"""Tests for Ukrainian stress marking."""

from __future__ import annotations

from pathlib import Path

import pytest

from fish_studio.config import StressConfig
from fish_studio.stress import (
    COMBINING_ACUTE,
    apply_lexicon,
    apply_stress_marks,
    has_stress_marks,
    normalize_uk_text,
    stressify,
    strip_stress_marks,
)

pytest.importorskip("ukrainian_word_stress")


def test_marks_are_combining_acute_not_spacing_acute() -> None:
    marked = apply_stress_marks("ліхтарик")

    # s2-pro honours U+0301 and ignores U+00B4, so the symbol choice is load-bearing.
    assert COMBINING_ACUTE in marked
    assert "\u00b4" not in marked
    assert marked.replace(COMBINING_ACUTE, "") == "ліхтарик"


def test_stress_lands_on_the_expected_vowel() -> None:
    assert apply_stress_marks("ліхтарик") == f"ліхта{COMBINING_ACUTE}рик"
    assert apply_stress_marks("ковпачків") == f"ковпачкі{COMBINING_ACUTE}в"


def test_marking_is_idempotent() -> None:
    once = apply_stress_marks("Я знайшов ліхтарик.")
    twice = apply_stress_marks(once)

    assert twice == once
    assert once.count(COMBINING_ACUTE) == twice.count(COMBINING_ACUTE)


def test_blank_text_is_passed_through() -> None:
    assert apply_stress_marks("") == ""
    assert apply_stress_marks("   ") == "   "


def test_heteronyms_stay_unmarked_by_default() -> None:
    # 'skip' is safer than guessing: an unmarked word falls back to the model's
    # own reading instead of being forced to a likely-wrong stress.
    assert not has_stress_marks(apply_stress_marks("броня"))


def test_disabled_config_leaves_text_alone() -> None:
    text = "Я знайшов ліхтарик."

    assert stressify(text, StressConfig(enabled=False, lexicon_path="")) == text
    assert has_stress_marks(stressify(text, StressConfig(enabled=True, lexicon_path="")))


def test_config_defaults_are_deterministic() -> None:
    # 'auto' would switch to Stanza whenever that package is present, which would
    # let training and synthesis mark the same sentence differently.
    config = StressConfig()

    assert config.disambiguation == "dictionary"
    assert config.on_ambiguity == "skip"
    assert config.enabled is True
    assert config.prefer_cpu is True
    assert config.acoustic_fallback is True
    assert config.lexicon_path == "configs/stress_lexicon.txt"


def test_normalize_restores_missing_apostrophes() -> None:
    assert normalize_uk_text("Памятаєте?") == "Пам'ятаєте?"
    assert normalize_uk_text("зявилася") == "з'явилася"
    assert normalize_uk_text("імя") == "ім'я"
    assert "'" in normalize_uk_text("зв’язок")  # curly → straight


def test_normalize_converts_spacing_acute_to_combining() -> None:
    assert normalize_uk_text("ліхта´рик") == f"ліхта{COMBINING_ACUTE}рик"


def test_force_remark_fills_previously_skipped_words() -> None:
    partial = f"Танцюва{COMBINING_ACUTE}ти вона не вміла"
    forced = apply_stress_marks(partial, force=True)

    assert has_stress_marks(forced)
    assert strip_stress_marks(forced) == strip_stress_marks(normalize_uk_text(partial))
    # Without force, partial marks freeze the sentence.
    assert apply_stress_marks(partial) == partial


def test_lexicon_overrides_dictionary(tmp_path: Path) -> None:
    lex = tmp_path / "lex.txt"
    lex.write_text(f"ральф\tра{COMBINING_ACUTE}льф\n", encoding="utf-8")

    marked = stressify(
        "Ральф увійшов.",
        StressConfig(enabled=True, lexicon_path=str(lex), disambiguation="dictionary"),
    )
    assert f"Ра{COMBINING_ACUTE}льф" in marked


def test_apply_lexicon_preserves_case() -> None:
    lexicon = {"ральф": f"ра{COMBINING_ACUTE}льф"}
    assert apply_lexicon("РАЛЬФ", lexicon) == f"РА{COMBINING_ACUTE}ЛЬФ"
    assert apply_lexicon("Ральф", lexicon) == f"Ра{COMBINING_ACUTE}льф"


def test_apostrophe_fix_enables_dictionary_mark() -> None:
    # Broken ASR form is OOV; restored form is in the stress dictionary.
    broken = apply_stress_marks("Памятаєте той столик?")
    assert "Пам'" in broken or "Пам'" in normalize_uk_text("Памятаєте")
    assert has_stress_marks(broken)
