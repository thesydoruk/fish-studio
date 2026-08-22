"""Corpus-derived Ukrainian number rules for Fallout dialogue TTS.

Source: INFO\\NAM1 strings in the transynth ``localizer`` DB (ai-pipeline),
Ukrainian translations that contain digits (1538 distinct lines, 2026-08).

Rules are ordered by match priority (most specific first). Expansion strategies
document how ``expand_uk_numbers()`` rewrites each span for TTS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class NumberRule:
    """One number-normalization rule derived from voiced dialogue corpus."""

    id: str
    """Stable rule id used in logs and tests."""

    description: str
    """What the surface form means in dialogue."""

    pattern: re.Pattern[str]
    """Regex with named group ``num`` (and optional helpers)."""

    expand: str
    """Expansion strategy for TTS (human-readable contract)."""

    corpus_hits: int
    """Approximate token hits on UK NAM1 lines with digits."""

    examples: tuple[str, ...]
    """Short UK corpus examples (already translated)."""

    leave_as_is: bool = False
    """If true, do not expand to Ukrainian number-words (IDs, codes, caliber)."""


# Priority: first match wins for overlapping spans.
_RULES: tuple[NumberRule, ...] = (
    NumberRule(
        id="alnum_codename",
        description="Letter+digit callsigns / synth names (X6, G5, H2, Z1, C4, B5-92, 1R).",
        # Digit+letter branch is Latin-only so Ukrainian ordinals (15й, 2га) go
        # to hyphen_ordinal instead of being left as codenames.
        pattern=re.compile(
            r"(?i)\b(?P<num>[A-ZА-ЯІЇЄҐ]\d{1,3}(?:[-–—‑]\d{1,3})?|\d{1,3}[A-Za-z])\b"
        ),
        expand="Spell letters then digits separately, or keep spoken letter-names; "
        "never read as a cardinal year/amount.",
        corpus_hits=118,
        examples=(
            "Ви ставите під сумнів мої накази, X6?",
            "G5 була моєю подругою, розумієш?",
            "B5-92, ініціалізувати заводське скидання.",
        ),
        leave_as_is=True,
    ),
    NumberRule(
        id="date_day_month_year",
        description="Full date: day + Ukrainian month name + year (+ optional 'року').",
        pattern=re.compile(
            r"(?i)\b(?P<day>\d{1,2})\s+"
            r"(?P<month>січня|лютого|березня|квітня|травня|червня|липня|серпня|"
            r"вересня|жовтня|листопада|грудня)\s+"
            r"(?P<num>(?:19|20)\d{2})(?:\s*року)?\b"
        ),
        expand="Ordinal day + month + year as '… року' "
        "(e.g. 22 жовтня 2077 → двадцять друге жовтня дві тисячі сімдесят сьомого року).",
        corpus_hits=12,
        examples=(
            "22 жовтня 2077 року",
            "Автоматична активація була запланована на 1 січня 2078 року.",
            "Запис номер три… 26 лютого 2077 року. Час… 12:35.",
        ),
    ),
    NumberRule(
        id="year_with_roku",
        description="Year with explicit рік/року/році, 'у/в YYYY', or decade '1980-х'.",
        pattern=re.compile(
            r"(?i)(?:"
            r"(?:(?:\bу|\bв)\s+(?:19|20)\d{2}(?:-м[уы])?(?:\s*рок(?:у|і|ом)?)?)"
            r"|(?:(?:19|20)\d{2}\s*рок(?:у|і|ів|и|ом)?\b)"
            r"|(?:(?:19|20)\d{2}-х\b)"
            r"|(?:(?:19|20)\d{2}\s*р\.?\b)"
            r")"
        ),
        expand="Read as year: 'дві тисячі … року/році'; "
        "'у 2077-му' → 'у дві тисячі сімдесят сьомому'; "
        "'1980-х' → 'дев'ятнадцятивосьмих'.",
        corpus_hits=34,
        examples=(
            "Від імені всіх нас у 2077 році…",
            "Битви - це так у 2077-му.",
            "наприкінці 2044 року містер Бредбертон нарешті досяг успіху",
            "фільми 1980-х",
        ),
    ),
    NumberRule(
        id="time_hhmm",
        description="Clock time HH:MM.",
        pattern=re.compile(r"\b(?P<hour>\d{1,2}):(?P<num>\d{2})\b"),
        expand="Hours + minutes in Ukrainian "
        "(19:04 → дев'ятнадцята нуль чотири / дев'ятнадцята година нуль чотири).",
        corpus_hits=9,
        examples=(
            "Нагляд починається о, в біса, 19:04.",
            "Час… 12:35.",
            "Шоу починається рівно о 21:00.",
        ),
    ),
    NumberRule(
        id="radio_hour",
        description="Radio Freedom style 'N година …'.",
        pattern=re.compile(r"(?i)\b(?P<num>\d{1,2})\s+година\b"),
        expand="Cardinal/ordinal hour + 'година' as already in text "
        "(11 година ранку → одинадцята година ранку).",
        corpus_hits=20,
        examples=(
            "11 година вечора.",
            "11 година ранку на Радіо Свобода…",
            "2 години ночі на Радіо Свобода.",
        ),
    ),
    NumberRule(
        id="percent",
        description="Percentages with % or 'відсотк…'.",
        pattern=re.compile(
            r"(?i)(?P<num>\d+(?:[.,]\d+)?)\s*(?:%|відсотк(?:ів|и|а|у)?\b)"
        ),
        expand="Number + 'відсотків/відсотки' with correct genitive.",
        corpus_hits=73,
        examples=(
            "25% прибутку передається…",
            "35% алкоголю за об'ємом.",
            "Довірчий інтервал… 2,74%.",
            "25 відсотків... 50 відсотків... 75 відсотків...",
        ),
    ),
    NumberRule(
        id="caps_amount",
        description="Bottlecap prices / rewards.",
        pattern=re.compile(
            r"(?i)(?P<num>\d{1,6})\s*криш(?:ок|ки|ками|кам|ках|ка)?\b"
        ),
        expand="Cardinal + 'кришок' (genitive plural for amounts).",
        corpus_hits=334,
        examples=(
            "100 кришок плюс пиво.",
            "250 кришок... авансом.",
            "Мені потрібно 2000 кришок.",
        ),
    ),
    NumberRule(
        id="vault_id",
        description="Vault / Сховище numbers — identifiers, not years.",
        pattern=re.compile(
            r"(?i)\b(?:сховищ(?:е|а|у|ем|і)|vault)\s*[#№]?\s*(?P<num>\d{1,3})\b"
        ),
        expand="Keep noun, expand id as cardinal "
        "(Сховище 81 → Сховище вісімдесят один).",
        corpus_hits=145,
        examples=(
            "Чули про Сховище 81?",
            "записи зі Сховища 111",
            "вам завжди будуть раді у Сховищі 81",
        ),
    ),
    NumberRule(
        id="duration_unit",
        description="Count + time unit (years/hours/minutes/days/weeks).",
        pattern=re.compile(
            r"(?i)(?P<num>\d{1,4})\s*"
            r"(?P<unit>хв\.?|хвилин(?:и|у|ами)?|сек(?:унд(?:и|у|ами)?)?|"
            r"годин(?:и|у|ами)?|дн(?:і|ів|я)|день|тижн(?:і|ів|я)|"
            r"рок(?:и|ів|у|ом)|місяц(?:і|ів|я|ь))\b"
        ),
        expand="Cardinal agreeing with unit "
        "(200 років → двісті років; 24 години → двадцять чотири години).",
        corpus_hits=203,
        examples=(
            "200 років, а вона все ще працює.",
            "близько 8 чи 9 років тому",
            "24 години в добі, 24 пива в ящику.",
        ),
    ),
    NumberRule(
        id="distance_unit",
        description="Distance with metric unit.",
        pattern=re.compile(
            r"(?i)(?P<num>\d{1,5}(?:[.,]\d+)?)\s*"
            r"(?P<unit>м\b|метр(?:и|ів|а)|км\b|кілометр(?:и|ів|а))\b"
        ),
        expand="Cardinal/decimal + unit words.",
        corpus_hits=5,
        examples=(
            "в межах 300 метрів",
            "понад 100 метрів",
            "3 928 метрів",
        ),
    ),
    NumberRule(
        id="temperature",
        description="Temperature in degrees.",
        pattern=re.compile(
            r"(?i)(?P<num>-?\d{1,3}(?:[.,]\d+)?)\s*"
            r"(?:°|градус(?:и|ів|а)?)\b"
        ),
        expand="Cardinal + 'градусів'; keep scale words (Фаренгейт) as-is.",
        corpus_hits=5,
        examples=(
            "173,5 градуса за Фаренгейтом",
            "температурою до 56 градусів",
            "під кутом 45 градусів",
        ),
    ),
    NumberRule(
        id="deal_split",
        description="Percentage-like deals 'N на M' without % sign.",
        pattern=re.compile(r"(?i)\b(?P<a>\d{1,3})\s+на\s+(?P<num>\d{1,3})\b"),
        expand="Both cardinals: 'сімдесят на тридцять'.",
        corpus_hits=15,
        examples=(
            "70 на 30 і цуценя. Без обговорень.",
            "60 на 40, серйозно? Спробуйте 90 на 10.",
        ),
    ),
    NumberRule(
        id="ratio_slash",
        description="Slash ratios; special-case 24/7.",
        pattern=re.compile(r"\b(?P<a>\d{1,4})\s*/\s*(?P<num>\d{1,4})\b"),
        expand="24/7 → 'двадцять чотири на сім'; other ratios → 'N з M' / 'N на M'.",
        corpus_hits=5,
        examples=(
            "я маю бути на зв'язку 24/7",
            "Галерея відкрита 24/7! 24/7!",
        ),
    ),
    NumberRule(
        id="countdown",
        description="Dramatic countdown '10... 9... 8...'.",
        pattern=re.compile(r"\b(?P<num>\d{1,2})\.\.\."),
        expand="Each integer as cardinal; keep pause dots as short pauses.",
        corpus_hits=32,
        examples=(
            "10... 9... 8... 7...",
            "5... 4... 3... 2... 1...",
            "1...",
        ),
    ),
    NumberRule(
        id="hash_id",
        description="Hash / № identifiers (docking bay, reactor, rule numbers).",
        pattern=re.compile(r"[#№]\s*(?P<num>\d{1,6})\b"),
        expand="Optional 'номер' + cardinal "
        "(№113 → номер сто тринадцять).",
        corpus_hits=14,
        examples=(
            "Порада Підземки №113…",
            "Реактор №3 на нижньому рівні несправний.",
            "інтерв'ю Волт-Тек №03",
        ),
    ),
    NumberRule(
        id="hyphen_ordinal",
        description="Ordinal / case tails with optional ASCII/Unicode hyphen: "
        "15-го, 15го, 15‑го, 1ша, 2-ге, 15-х, 15-ти.",
        pattern=re.compile(
            r"(?i)\b(?P<num>\d{1,4})(?:[-–—‑])?"
            r"(?P<suf>ого|ому|ими|им|ої|ій|го|му|ї|й|ша|га|тя|та|ма|ше|ге|тє|те|ти|"
            r"я|е|а|і|х|м)\b"
        ),
        expand="Ordinal (or genitive cardinal for -ти/-х) from the written ending "
        "(15-го → п'ятнадцятого; 15го → same; 2-х → двох; 15-ти → п'ятнадцяти).",
        corpus_hits=80,
        examples=(
            "15-го",
            "15го",
            "з 15-го числа",
            "на 15-му рівні",
            "1-й",
            "1ша",
            "2-ге",
            "2-х",
            "15-ти",
            "15-ої",
            "15-ій",
        ),
    ),
    NumberRule(
        id="en_ordinal",
        description="English ordinal suffixes left in UK text: 1st, 2nd, 3rd, 15th.",
        pattern=re.compile(r"(?i)\b(?P<num>\d{1,4})(?P<suf>st|nd|rd|th)\b"),
        expand="Ukrainian masculine nominative ordinal (15th → п'ятнадцятий).",
        corpus_hits=11,
        examples=("1st", "2nd", "3rd", "15th"),
    ),
    NumberRule(
        id="numeric_range",
        description="Bare numeric ranges 10-15 / 70-30 (not ordinal tails).",
        pattern=re.compile(r"\b(?P<a>\d{1,4})[-–—‑](?P<num>\d{1,4})\b"),
        expand="Both ends as cardinals with a spoken dash "
        "(10-15 → десять — п'ятнадцять).",
        corpus_hits=20,
        examples=("10-15", "70-30", "10–15"),
    ),
    NumberRule(
        id="decimal_or_version",
        description="Decimals and version/model numbers (1.0, 5.56, 2,74).",
        pattern=re.compile(r"\b(?P<whole>\d+)(?P<sep>[.,])(?P<num>\d+)\b"),
        expand="Versions/calibers: digit-by-digit or 'п'ять п'ятдесят шість'; "
        "true decimals: 'ціла/цілих … десятих'. Context-dependent.",
        corpus_hits=19,
        examples=(
            "людина моделі 1.0",
            "пару коробок 5.56",
            "кричали 5,02 секунди",
            "0,04 відсотка",
        ),
    ),
    NumberRule(
        id="year_bare_fallout",
        description="Remaining Fallout-era years 19xx/20xx without 'року' (titles, product names).",
        # Exclude round thousands (2000 caps/volts) — those stay cardinal.
        pattern=re.compile(
            r"\b(?P<num>(?:19[1-9]\d|20(?:0[1-9]|[1-9]\d)\d|229\d))\b"
        ),
        expand="Prefer year reading ('дві тисячі сімдесят восьмий') unless "
        "nearby words imply a plain count/voltage/price (then cardinal).",
        corpus_hits=12,
        examples=(
            "голозаписі «Менеджер міста 2078»",
            "колонія 2291 на краю сектора",
        ),
    ),
    NumberRule(
        id="bare_thousands_plus",
        description="Bare integers ≥1000 (prices without 'кришок', large counts, store #).",
        pattern=re.compile(r"\b(?P<num>[1-9]\d{3,5})\b"),
        expand="Cardinal; if nearby currency words missing, still cardinal "
        "(not year unless year rule already matched).",
        corpus_hits=60,
        examples=(
            "1000. Або беріть, або йдіть.",
            "магазин номер 1518",
            "директиву 7395",
            "1162 обличчя",
        ),
    ),
    NumberRule(
        id="bare_int",
        description="Remaining bare integers (counts, prices, lottery numbers, levels).",
        pattern=re.compile(r"\b(?P<num>\d{1,3})\b"),
        expand="Default cardinal. Prefer genitive after 'з/із/до/близько'. "
        "Lottery/list forms 'Номер 13' → 'номер тринадцять'.",
        corpus_hits=1125,
        examples=(
            "4 із 5 людей страждають від радіаційного отруєння.",
            "13. Номер 13.",
            "150. Це все, що я можу заплатити.",
            "111? Це ж те Сховище…",
        ),
    ),
)


def iter_number_rules() -> Iterator[NumberRule]:
    """Yield rules in match priority order."""
    yield from _RULES


def classify_number_spans(text: str) -> list[tuple[NumberRule, re.Match[str]]]:
    """Return non-overlapping rule matches for ``text`` (priority order)."""
    if not text or not any(ch.isdigit() for ch in text):
        return []

    occupied: list[tuple[int, int]] = []
    hits: list[tuple[NumberRule, re.Match[str]]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in occupied)

    for rule in _RULES:
        for match in rule.pattern.finditer(text):
            start, end = match.span()
            if _overlaps(start, end):
                continue
            occupied.append((start, end))
            hits.append((rule, match))

    hits.sort(key=lambda item: item[1].start())
    return hits
