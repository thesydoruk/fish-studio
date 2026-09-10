"""Tests for Ukrainian stress marking."""

from __future__ import annotations

from pathlib import Path

import pytest

from fish_studio.config import StressConfig
from fish_studio.stress import (
    COMBINING_ACUTE,
    _load_lexicon,
    _preferred_homonym_accent,
    apply_lexicon,
    apply_stress_marks,
    has_stress_marks,
    normalize_uk_text,
    stressify,
    strip_stress_marks,
)

_LEXICON = Path(__file__).resolve().parents[1] / "configs" / "stress_lexicon.txt"

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


def test_zviazok_masc_locative_prefers_final_vowel() -> None:
    parse = {
        "text": "зв'язку",
        "upos": "NOUN",
        "feats": "Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing",
    }
    assert _preferred_homonym_accent(parse) == 7


def test_zviazka_fem_accusative_prefers_stem() -> None:
    parse = {
        "text": "зв'язку",
        "upos": "NOUN",
        "feats": "Animacy=Inan|Case=Acc|Gender=Fem|Number=Sing",
    }
    assert _preferred_homonym_accent(parse) == 4


def test_zviazkamy_masc_instrumental_prefers_ending() -> None:
    parse = {
        "text": "зв'язками",
        "upos": "NOUN",
        "feats": "Animacy=Inan|Case=Ins|Gender=Masc|Number=Plur",
    }
    assert _preferred_homonym_accent(parse) == 7


def test_zviazkamy_fem_instrumental_prefers_stem() -> None:
    parse = {
        "text": "зв'язками",
        "upos": "NOUN",
        "feats": "Animacy=Inan|Case=Ins|Gender=Fem|Number=Plur",
    }
    assert _preferred_homonym_accent(parse) == 4


def test_zviazok_unrelated_parse_has_no_preference() -> None:
    assert _preferred_homonym_accent({"text": "броня", "feats": "Gender=Fem"}) is None


def test_stanza_marks_na_zviazku_on_the_ending() -> None:
    stanza = pytest.importorskip("stanza")
    try:
        stanza.Pipeline(
            "uk",
            processors="tokenize,pos,mwt",
            download_method=stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
        )
    except Exception:
        pytest.skip("Ukrainian Stanza models are not available")

    marked = apply_stress_marks(
        "Волт-Тек на зв'язку!",
        disambiguation="stanza",
        lexicon={},
    )
    assert f"зв'язку{COMBINING_ACUTE}" in marked
    assert f"зв'я{COMBINING_ACUTE}зку" not in marked


def test_stanza_keeps_zviazka_stress_on_the_stem() -> None:
    stanza = pytest.importorskip("stanza")
    try:
        stanza.Pipeline(
            "uk",
            processors="tokenize,pos,mwt",
            download_method=stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
        )
    except Exception:
        pytest.skip("Ukrainian Stanza models are not available")

    marked = apply_stress_marks("зв'язка ключів", disambiguation="stanza", lexicon={})
    assert f"зв'я{COMBINING_ACUTE}зка" in marked


def test_stanza_marks_dating_zviazkamy_on_the_ending() -> None:
    stanza = pytest.importorskip("stanza")
    try:
        stanza.Pipeline(
            "uk",
            processors="tokenize,pos,mwt",
            download_method=stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
        )
    except Exception:
        pytest.skip("Ukrainian Stanza models are not available")

    marked = apply_stress_marks(
        "Вона бавиться випадковими зв'язками.",
        disambiguation="stanza",
        lexicon={},
    )
    assert f"зв'язка{COMBINING_ACUTE}ми" in marked
    assert f"зв'я{COMBINING_ACUTE}зками" not in marked


def test_stanza_marks_exclamatory_yaka_on_the_ending() -> None:
    stanza = pytest.importorskip("stanza")
    try:
        stanza.Pipeline(
            "uk",
            processors="tokenize,pos,mwt",
            download_method=stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
        )
    except Exception:
        pytest.skip("Ukrainian Stanza models are not available")

    marked = apply_stress_marks(
        "Яка жахлива людина може таке сказати?",
        disambiguation="stanza",
        lexicon={},
    )
    assert f"Яка{COMBINING_ACUTE}" in marked
    assert f"Я{COMBINING_ACUTE}ка" not in marked


def test_stanza_marks_plural_ptakhy_on_the_ending() -> None:
    stanza = pytest.importorskip("stanza")
    try:
        stanza.Pipeline(
            "uk",
            processors="tokenize,pos,mwt",
            download_method=stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
        )
    except Exception:
        pytest.skip("Ukrainian Stanza models are not available")

    marked = apply_stress_marks(
        "Я бачила, як птахи сідали на Corvega і спалахували полум'ям.",
        disambiguation="stanza",
        lexicon={},
    )
    assert f"птахи{COMBINING_ACUTE}" in marked
    assert f"пта{COMBINING_ACUTE}хи" not in marked


def test_lexicon_file_marks_unambiguous_words() -> None:
    marked = stressify(
        "Ральф сказав йому про Емоджин. Стіна стоїть на заході. Не хочу бути стукачем. Все скінчено. Активуйте їх. Система має працювати. Я рятував усіх. Відтоді я присягнув.",
        StressConfig(enabled=True, lexicon_path=str(_LEXICON), disambiguation="dictionary"),
    )
    assert f"Ра{COMBINING_ACUTE}льф" in marked
    assert f"йому{COMBINING_ACUTE}" in marked
    assert f"Е{COMBINING_ACUTE}моджин" in marked
    assert f"Стіна{COMBINING_ACUTE}" in marked
    assert f"за{COMBINING_ACUTE}ході" in marked
    assert f"стукаче{COMBINING_ACUTE}м" in marked
    assert f"скі{COMBINING_ACUTE}нчено" in marked
    assert f"Активу{COMBINING_ACUTE}йте" in marked
    assert f"працюва{COMBINING_ACUTE}ти" in marked
    assert f"рятува{COMBINING_ACUTE}в" in marked
    assert f"Відто{COMBINING_ACUTE}ді" in marked


def test_lexicon_overrides_skincheno_even_when_sentence_already_marked() -> None:
    lexicon = _load_lexicon(str(_LEXICON))
    already = f"Все{COMBINING_ACUTE} скінчено, не буду стукачем. Активуйте їх. Має працювати. Я рятував. Відтоді."
    marked = apply_lexicon(already, lexicon)
    assert f"скі{COMBINING_ACUTE}нчено" in marked
    assert f"стукаче{COMBINING_ACUTE}м" in marked
    assert f"Активу{COMBINING_ACUTE}йте" in marked
    assert f"працюва{COMBINING_ACUTE}ти" in marked
    assert f"рятува{COMBINING_ACUTE}в" in marked
    assert f"Відто{COMBINING_ACUTE}ді" in marked


def test_lexicon_marks_tata_zhyvi_koly_zseredyny() -> None:
    marked = stressify(
        "Пошукати маму і тата. Вони живі. Зсередини немає ручки. Витягніть мене.",
        StressConfig(enabled=True, lexicon_path=str(_LEXICON), disambiguation="dictionary"),
    )
    assert f"та{COMBINING_ACUTE}та" in marked
    assert f"живі{COMBINING_ACUTE}" in marked
    assert f"Зсере{COMBINING_ACUTE}дини" in marked
    assert f"Ви{COMBINING_ACUTE}тягніть" in marked


def test_lexicon_marks_pokydok_vidstrilyty_and_swears() -> None:
    marked = stressify(
        "Покидьок. Вам доведеться їх відстрілити. Нахуй і пиздець. Манда і ссикло. Ссикуняка.",
        StressConfig(enabled=True, lexicon_path=str(_LEXICON), disambiguation="dictionary"),
    )
    assert f"По{COMBINING_ACUTE}кидьок" in marked
    assert f"відстріли{COMBINING_ACUTE}ти" in marked
    assert f"Наху{COMBINING_ACUTE}й" in marked
    assert f"пизде{COMBINING_ACUTE}ць" in marked
    assert f"Манда{COMBINING_ACUTE}" in marked
    assert f"ссикло{COMBINING_ACUTE}" in marked
    assert f"Ссикуня{COMBINING_ACUTE}ка" in marked


def test_lexicon_marks_napruzhenyi() -> None:
    marked = stressify(
        "Напружений день у вас, чи не так?",
        StressConfig(enabled=True, lexicon_path=str(_LEXICON), disambiguation="dictionary"),
    )
    assert f"Напру{COMBINING_ACUTE}жений" in marked
    lexicon = _load_lexicon(str(_LEXICON))
    already = f"Напруже{COMBINING_ACUTE}ний день."
    assert f"Напру{COMBINING_ACUTE}жений" in apply_lexicon(already, lexicon)


def test_lexicon_marks_blukachi() -> None:
    marked = stressify(
        "Блукачі!",
        StressConfig(enabled=True, lexicon_path=str(_LEXICON), disambiguation="dictionary"),
    )
    assert f"Блукачі{COMBINING_ACUTE}" in marked
    lexicon = _load_lexicon(str(_LEXICON))
    already = f"Блу{COMBINING_ACUTE}качі!"
    assert f"Блукачі{COMBINING_ACUTE}" in apply_lexicon(already, lexicon)


def test_lexicon_marks_bovdury() -> None:
    marked = stressify(
        "Бовдури.",
        StressConfig(enabled=True, lexicon_path=str(_LEXICON), disambiguation="dictionary"),
    )
    assert f"Бо{COMBINING_ACUTE}вдури" in marked
    lexicon = _load_lexicon(str(_LEXICON))
    already = f"Бовдури{COMBINING_ACUTE}."
    assert f"Бо{COMBINING_ACUTE}вдури" in apply_lexicon(already, lexicon)


def test_lexicon_marks_zamknenyi() -> None:
    marked = stressify(
        "Заходити у замкнений простір, переповнений болотниками?",
        StressConfig(enabled=True, lexicon_path=str(_LEXICON), disambiguation="dictionary"),
    )
    assert f"за{COMBINING_ACUTE}мкнений" in marked
    lexicon = _load_lexicon(str(_LEXICON))
    already = f"замкне{COMBINING_ACUTE}ний простір"
    assert f"за{COMBINING_ACUTE}мкнений" in apply_lexicon(already, lexicon)


def test_hyphen_stem_lexicon_marks_second_part() -> None:
    lexicon = _load_lexicon(str(_LEXICON))
    assert apply_lexicon(f"коли{COMBINING_ACUTE}-небудь", lexicon) == (
        f"коли{COMBINING_ACUTE}-не{COMBINING_ACUTE}будь"
    )
    assert apply_lexicon(f"півде{COMBINING_ACUTE}нно-західних", lexicon) == (
        f"півде{COMBINING_ACUTE}нно-за{COMBINING_ACUTE}хідних"
    )


def test_lexicon_file_covers_full_audit_names() -> None:
    marked = stressify(
        "Кейті і Еврару в Конкорда. Салемська відьма з Салема.",
        StressConfig(enabled=True, lexicon_path=str(_LEXICON), disambiguation="dictionary"),
    )
    assert f"Ке{COMBINING_ACUTE}йті" in marked
    assert f"Евра{COMBINING_ACUTE}ру" in marked
    assert f"Ко{COMBINING_ACUTE}нкорда" in marked
    assert f"Са{COMBINING_ACUTE}лемська" in marked
    assert f"Са{COMBINING_ACUTE}лема" in marked
    assert f"Сале{COMBINING_ACUTE}мська" not in marked


def test_ambiguous_former_lexicon_words_are_not_forced() -> None:
    # File used to pin one reading. Dictionary may still mark its own choice;
    # we must not override it to the old lexicon form.
    marked = apply_stress_marks("гуля тома будь-кого", disambiguation="dictionary", lexicon={})
    assert "гуля" in marked
    assert f"то{COMBINING_ACUTE}ма" not in marked
    assert f"будь-кого{COMBINING_ACUTE}" not in marked


def test_strilby_genitive_prefers_the_ending() -> None:
    parse = {
        "text": "стрільби",
        "upos": "NOUN",
        "feats": "Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing",
    }
    assert _preferred_homonym_accent(parse) == 8


def test_strilby_plural_prefers_the_stem() -> None:
    parse = {
        "text": "стрільби",
        "upos": "NOUN",
        "feats": "Animacy=Inan|Case=Nom|Gender=Fem|Number=Plur",
    }
    assert _preferred_homonym_accent(parse) == 4


def test_pomylky_plural_prefers_the_ending() -> None:
    parse = {
        "text": "помилки",
        "upos": "NOUN",
        "feats": "Animacy=Inan|Case=Acc|Gender=Fem|Number=Plur",
    }
    assert _preferred_homonym_accent(parse) == 7


def test_yaka_pronoun_prefers_the_ending() -> None:
    parse = {
        "text": "яка",
        "upos": "PRON",
        "feats": "Case=Nom|Gender=Fem|Number=Sing",
    }
    assert _preferred_homonym_accent(parse) == 3


def test_yaka_noun_yak_prefers_the_stem() -> None:
    parse = {
        "text": "яка",
        "upos": "NOUN",
        "feats": "Animacy=Anim|Case=Gen|Gender=Masc|Number=Sing",
    }
    assert _preferred_homonym_accent(parse) == 1


def test_ptakhy_plural_prefers_the_ending() -> None:
    parse = {
        "text": "птахи",
        "upos": "NOUN",
        "feats": "Animacy=Anim|Case=Nom|Gender=Masc|Number=Plur",
    }
    assert _preferred_homonym_accent(parse) == 5


def test_ptakhy_genitive_ptakha_prefers_the_stem() -> None:
    parse = {
        "text": "птахи",
        "upos": "NOUN",
        "feats": "Animacy=Anim|Case=Gen|Gender=Fem|Number=Sing",
    }
    assert _preferred_homonym_accent(parse) == 3


def test_zirky_plural_prefers_the_ending() -> None:
    parse = {
        "text": "зірки",
        "upos": "NOUN",
        "feats": "Animacy=Inan|Case=Nom|Gender=Fem|Number=Plur",
    }
    assert _preferred_homonym_accent(parse) == 5


def test_koly_conjunction_prefers_the_ending() -> None:
    parse = {
        "text": "коли",
        "upos": "SCONJ",
        "feats": "",
    }
    assert _preferred_homonym_accent(parse) == 4


def test_batkiv_noun_prefers_the_ending() -> None:
    parse = {
        "text": "батьків",
        "upos": "NOUN",
        "feats": "Animacy=Anim|Case=Gen|Gender=Masc|Number=Plur",
    }
    assert _preferred_homonym_accent(parse) == 6


def test_same_particle_prefers_the_first_vowel() -> None:
    parse = {
        "text": "саме",
        "upos": "PART",
        "feats": "",
    }
    assert _preferred_homonym_accent(parse) == 2


def test_vyslukhaly_plural_prefers_the_prefix() -> None:
    parse = {
        "text": "вислухали",
        "upos": "VERB",
        "feats": "Aspect=Perf|Mood=Ind|Number=Plur|Tense=Past|VerbForm=Fin",
    }
    assert _preferred_homonym_accent(parse) == 2


def test_povodylysya_plural_prefers_the_stem() -> None:
    parse = {
        "text": "поводилися",
        "upos": "VERB",
        "feats": "Aspect=Imp|Mood=Ind|Number=Plur|Tense=Past|VerbForm=Fin",
    }
    assert _preferred_homonym_accent(parse) == 4


def test_nasypaty_infinitive_prefers_the_thematic_vowel() -> None:
    parse = {
        "text": "насипати",
        "upos": "VERB",
        "feats": "Aspect=Perf|VerbForm=Inf",
    }
    assert _preferred_homonym_accent(parse) == 6


def test_stanza_marks_pochaly_nasypaty_on_the_ending() -> None:
    stanza = pytest.importorskip("stanza")
    try:
        stanza.Pipeline(
            "uk",
            processors="tokenize,pos,mwt",
            download_method=stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
        )
    except Exception:
        pytest.skip("Ukrainian Stanza models are not available")

    marked = apply_stress_marks(
        "щоб вони почали насипати болотникам гарячого свинцю",
        disambiguation="stanza",
        lexicon={},
    )
    assert f"насипа{COMBINING_ACUTE}ти" in marked
    assert f"наси{COMBINING_ACUTE}пати" not in marked


def test_buty_infinitive_prefers_first_vowel() -> None:
    parse = {
        "text": "бути",
        "upos": "VERB",
        "feats": "Aspect=Imp|VerbForm=Inf",
    }
    assert _preferred_homonym_accent(parse) == 2


def test_buty_without_infinitive_tag_stays_undecided() -> None:
    assert _preferred_homonym_accent({"text": "бути", "upos": "NOUN", "feats": ""}) is None


def test_overlay_file_can_still_force_a_reading(tmp_path: Path) -> None:
    lex = tmp_path / "lex.txt"
    lex.write_text(f"тома\tтома{COMBINING_ACUTE}\n", encoding="utf-8")

    marked = stressify(
        "том тома",
        StressConfig(enabled=True, lexicon_path=str(lex), disambiguation="dictionary"),
    )
    assert f"тома{COMBINING_ACUTE}" in marked


def test_apostrophe_fix_enables_dictionary_mark() -> None:
    # Broken ASR form is OOV; restored form is in the stress dictionary.
    broken = apply_stress_marks("Памятаєте той столик?")
    assert "Пам'" in broken or "Пам'" in normalize_uk_text("Памятаєте")
    assert has_stress_marks(broken)
