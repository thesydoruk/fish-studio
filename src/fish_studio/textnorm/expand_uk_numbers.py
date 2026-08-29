"""Expand Arabic numerals in Ukrainian dialogue to spoken words for TTS."""

from __future__ import annotations

import re
from typing import Callable

from fish_studio.textnorm.uk_dialog_number_rules import NumberRule, classify_number_spans
from fish_studio.textnorm.uk_numwords import (
    cardinal,
    cardinal_genitive,
    clock_minutes,
    decimal_words,
    half_words,
    ordinal,
    version_words,
    year_words,
)

_YEAR_DIGITS_RE = re.compile(r"(?:19|20)\d{2}|229\d")
_UK_LANGS = frozenset({"uk", "ua", "ukrainian", "uk-ua", "uk_ua"})


def should_expand_numbers(language: str | None) -> bool:
    """True for Ukrainian (and when language is unset, which is the training default)."""
    if not language:
        return True
    return language.strip().lower().replace("-", "_") in _UK_LANGS or language.strip().lower().startswith(
        "uk"
    )


_THOUSANDS_SPACE_RE = re.compile(r"(?<=\d)\s+(?=\d{3}\b)")


def expand_uk_numbers(text: str) -> str:
    """Replace digit spans covered by dialogue rules with Ukrainian words."""
    if not text or not any(ch.isdigit() for ch in text):
        return text

    # "3 928 метрів" → "3928 метрів" so distance/bare rules can match.
    text = _THOUSANDS_SPACE_RE.sub("", text)

    hits = classify_number_spans(text)
    if not hits:
        return text

    pieces: list[str] = []
    cursor = 0
    for rule, match in hits:
        pieces.append(text[cursor : match.start()])
        pieces.append(_expand_match(rule, match))
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces)


def _expand_match(rule: NumberRule, match: re.Match[str]) -> str:
    handler = _HANDLERS.get(rule.id)
    if handler is None or rule.leave_as_is:
        # Unknown or "leave as spoken digits" (IDs, calibers) — keep the original span.
        return match.group(0)
    try:
        return handler(match)
    except (ValueError, KeyError, TypeError):
        return match.group(0)


def _replace_first_number(span: str, words: str) -> str:
    return re.sub(r"-?\d+(?:[.,]\d+)?", words, span, count=1)


# ── Fractional quantities and unit agreement ────────────────────────────────

# (nom_sg, gen_sg, nom_pl, gen_pl) per unit; gen_sg follows decimals
# ("дві десятих градуса"), nom_pl/gen_pl follow "з половиною" by the whole part.
_UNIT_FORMS: dict[str, tuple[str, str, str, str]] = {
    "градус": ("градус", "градуса", "градуси", "градусів"),
    "відсоток": ("відсоток", "відсотка", "відсотки", "відсотків"),
    "секунда": ("секунда", "секунди", "секунди", "секунд"),
    "хвилина": ("хвилина", "хвилини", "хвилини", "хвилин"),
    "година": ("година", "години", "години", "годин"),
    "метр": ("метр", "метра", "метри", "метрів"),
    "кілометр": ("кілометр", "кілометра", "кілометри", "кілометрів"),
    "рік": ("рік", "року", "роки", "років"),
    "день": ("день", "дня", "дні", "днів"),
    "тиждень": ("тиждень", "тижня", "тижні", "тижнів"),
    "місяць": ("місяць", "місяця", "місяці", "місяців"),
}

_UNIT_GENDER: dict[str, str] = {
    "секунда": "fem",
    "хвилина": "fem",
    "година": "fem",
}

# Ordered prefixes: longer/specific first (кілометр before метр, місяц before м).
_UNIT_KEY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("кілометр", "кілометр"),
    ("км", "кілометр"),
    ("метр", "метр"),
    ("місяц", "місяць"),
    ("градус", "градус"),
    ("відсот", "відсоток"),
    ("секунд", "секунда"),
    ("сек", "секунда"),
    ("хвилин", "хвилина"),
    ("хв", "хвилина"),
    ("годин", "година"),
    ("рок", "рік"),
    ("рік", "рік"),
    ("тижн", "тиждень"),
    ("тиждень", "тиждень"),
    ("дн", "день"),
    ("день", "день"),
    ("м", "метр"),
)

# Full unit words we may rewrite in place (abbreviations м/км/хв/сек stay as-is).
_UNIT_TOKEN_RE = re.compile(
    r"(?i)\b(?:кілометр(?:ів|и|а|у)?|метр(?:ів|и|а|у)?|градус(?:ів|и|а|у)?|"
    r"відсотк(?:ів|и|а|у)|відсоток|секунд(?:и|у|ами)?|хвилин(?:и|у|ами)?|"
    r"годин(?:и|у|ами)?|рок(?:ів|и|у|ом)|тижн(?:ів|і|я)|тиждень|"
    r"місяц(?:ів|і|я|ь)|дн(?:ів|і|я)|день)\b"
)


def _unit_key(token: str) -> str | None:
    t = token.casefold()
    for prefix, key in _UNIT_KEY_PREFIXES:
        if t.startswith(prefix):
            return key
    return None


def _plural_index(n: int) -> int:
    """0 → one (1, 21…), 1 → few (2–4, 22–24…), 2 → many (0, 5–20…)."""
    n = abs(n) % 100
    if 11 <= n <= 14:
        return 2
    n %= 10
    if n == 1:
        return 0
    if 2 <= n <= 4:
        return 1
    return 2


def _quantity_words(raw: str, *, gender: str = "masc") -> tuple[str, str | None, int]:
    """Expand an integer/decimal token to words.

    Returns ``(words, kind, whole)`` where ``kind`` is ``None`` for integers,
    ``"half"`` for X,5 / X.5 ("три з половиною"), and ``"decimal"`` otherwise.
    """
    negative = raw.startswith("-")
    body = raw[1:] if negative else raw
    kind: str | None = None
    if "," in body or "." in body:
        sep = "," if "," in body else "."
        whole_s, frac_s = body.split(sep, 1)
        whole = int(whole_s)
        if frac_s == "5" and whole >= 1:
            words = half_words(whole, gender=gender)
            kind = "half"
        else:
            words = decimal_words(whole, frac_s)
            kind = "decimal"
    else:
        whole = int(body)
        words = cardinal(whole, gender=gender)
    if negative:
        words = f"мінус {words}"
    return words, kind, whole


def _generated_unit(key: str, kind: str | None, whole: int) -> str:
    """Unit word to emit when the source had only a symbol (%, °)."""
    nom_sg, gen_sg, nom_pl, gen_pl = _UNIT_FORMS[key]
    if kind == "decimal":
        return gen_sg
    one = gen_sg if kind == "half" else nom_sg  # півтора градуса / один градус
    return (one, nom_pl, gen_pl)[_plural_index(whole)]


def _inflect_unit(span: str, *, kind: str, whole: int) -> str:
    """Rewrite the unit word in ``span`` after a fractional quantity."""

    def repl(match: re.Match[str]) -> str:
        key = _unit_key(match.group(0))
        if key is None:
            return match.group(0)
        return _generated_unit(key, kind, whole)

    return _UNIT_TOKEN_RE.sub(repl, span, count=1)


def _expand_date(match: re.Match[str]) -> str:
    day = int(match.group("day"))
    month = match.group("month")
    year = int(match.group("num"))
    day_words = ordinal(day, gender="neut", case="nom")
    year_part = f"{year_words(year, case='gen')} року"
    return f"{day_words} {month} {year_part}"


def _expand_year_with_roku(match: re.Match[str]) -> str:
    span = match.group(0)
    year_m = _YEAR_DIGITS_RE.search(span)
    if not year_m:
        return span
    year = int(year_m.group(0))
    lower = span.lower()
    prefix = span[: year_m.start()]
    suffix = span[year_m.end() :]
    if re.match(r"-х\b", suffix, flags=re.IGNORECASE):
        base = year_words(year, case="gen")
        decade = base[:-3] + "их" if base.endswith("ого") else f"{base}их"
        return prefix + decade + suffix[2:]
    if re.match(r"-м[уы]\b", suffix, flags=re.IGNORECASE) or re.search(r"\bроці\b", lower):
        words = year_words(year, case="loc")
        suffix = re.sub(r"^-м[уы]\b", "", suffix, count=1, flags=re.IGNORECASE)
        return prefix + words + suffix
    if re.search(r"\bрок", lower):
        return prefix + year_words(year, case="gen") + suffix
    return prefix + year_words(year, case="loc") + suffix


def _expand_time(match: re.Match[str]) -> str:
    hour = int(match.group("hour"))
    if hour == 0:
        hour = 0
    elif hour == 24:
        hour = 24
    else:
        hour = hour % 24 or 24
    # 0:00 → дванадцята / нульова — keep literal midnight as "нульова"
    hour_words = (
        ordinal(12, gender="fem", case="nom")
        if hour == 0
        else ordinal(hour, gender="fem", case="nom")
    )
    minutes = int(match.group("num"))
    return f"{hour_words} {clock_minutes(minutes)}"


def _expand_radio_hour(match: re.Match[str]) -> str:
    hour = int(match.group("num"))
    return f"{ordinal(hour, gender='fem', case='nom')} година"


def _expand_percent(match: re.Match[str]) -> str:
    span = match.group(0)
    words, kind, whole = _quantity_words(match.group("num"))
    if "%" in span:
        return f"{words} {_generated_unit('відсоток', kind, whole)}"
    # Keep existing відсотк* word, reinflected after fractions.
    span = _replace_first_number(span, words)
    if kind is not None:
        span = _inflect_unit(span, kind=kind, whole=whole)
    return span


def _expand_caps(match: re.Match[str]) -> str:
    n = int(match.group("num"))
    span = match.group(0)
    return _replace_first_number(span, cardinal(n))


def _expand_vault(match: re.Match[str]) -> str:
    n = int(match.group("num"))
    span = match.group(0)
    return _replace_first_number(span, cardinal(n))


def _expand_with_unit(match: re.Match[str]) -> str:
    span = match.group(0)
    unit_key = _unit_key(match.group("unit"))
    gender = _UNIT_GENDER.get(unit_key or "", "masc")
    # A measurement unit implies a real quantity, so a dot reads as a decimal
    # too (13.2 години), never as a version/caliber.
    words, kind, whole = _quantity_words(match.group("num"), gender=gender)
    span = _replace_first_number(span, words)
    if kind is not None:
        span = _inflect_unit(span, kind=kind, whole=whole)
    return span


def _expand_temperature(match: re.Match[str]) -> str:
    span = match.group(0)
    words, kind, whole = _quantity_words(match.group("num"))
    # Prefer a spoken "градус…" form when only ° was present.
    if "°" in span and not re.search(r"градус", span, re.IGNORECASE):
        return f"{words} {_generated_unit('градус', kind, whole)}"
    span = _replace_first_number(span, words)
    if kind is not None:
        span = _inflect_unit(span, kind=kind, whole=whole)
    return span


def _expand_deal(match: re.Match[str]) -> str:
    a = int(match.group("a"))
    b = int(match.group("num"))
    return f"{cardinal(a)} на {cardinal(b)}"


def _expand_ratio(match: re.Match[str]) -> str:
    a = int(match.group("a"))
    b = int(match.group("num"))
    return f"{cardinal(a)} на {cardinal(b)}"


def _expand_countdown(match: re.Match[str]) -> str:
    n = int(match.group("num"))
    return f"{cardinal(n)}..."


def _expand_hash(match: re.Match[str]) -> str:
    n = int(match.group("num"))
    return f"номер {cardinal(n)}"


# Written ordinal tails → (gender, case). Longer suffixes are preferred in the regex.
_HYPHEN_ORDINAL_SUFFIX: dict[str, tuple[str, str]] = {
    "ого": ("masc", "gen"),
    "ому": ("masc", "loc"),
    "ими": ("masc", "instr"),  # approximate; plural rare in dialogue
    "им": ("masc", "instr"),
    "ої": ("fem", "gen"),
    "ій": ("fem", "loc"),
    "го": ("masc", "gen"),
    "му": ("masc", "loc"),
    "м": ("masc", "instr"),
    "й": ("masc", "nom"),
    "ї": ("fem", "gen"),
    "ша": ("fem", "nom"),
    "га": ("fem", "nom"),
    "тя": ("fem", "nom"),
    "та": ("fem", "nom"),
    "ма": ("fem", "nom"),
    "я": ("fem", "nom"),
    "а": ("fem", "nom"),
    "і": ("masc", "nom_pl"),
    "ше": ("neut", "nom"),
    "ге": ("neut", "nom"),
    "тє": ("neut", "nom"),
    "те": ("neut", "nom"),
    "е": ("neut", "nom"),
    "х": ("masc", "gen_pl"),
}

def _expand_hyphen_ordinal(match: re.Match[str]) -> str:
    n = int(match.group("num"))
    suf = match.group("suf").casefold()
    # -ти is always cardinal genitive. -х is cardinal genitive for small counts
    # (2-х, 3-х); larger values read as ordinal genitive plural (15-х → …их),
    # except years which year_with_roku already consumed (1980-х).
    if suf == "ти" or (suf == "х" and n <= 4):
        return cardinal_genitive(n)
    if suf == "х":
        return ordinal(n, gender="masc", case="gen_pl")
    gender, case = _HYPHEN_ORDINAL_SUFFIX.get(suf, ("masc", "nom"))
    return ordinal(n, gender=gender, case=case)


def _expand_en_ordinal(match: re.Match[str]) -> str:
    return ordinal(int(match.group("num")), gender="masc", case="nom")


def _expand_numeric_range(match: re.Match[str]) -> str:
    a = int(match.group("a"))
    b = int(match.group("num"))
    return f"{cardinal(a)} — {cardinal(b)}"


def _expand_decimal_or_version(match: re.Match[str]) -> str:
    frac = match.group("num")
    sep = match.group("sep")
    if sep == ",":
        # Comma is always a true decimal: 3,5 → три з половиною; 2,74 → дві цілих … сотих.
        words, _, _ = _quantity_words(f"{match.group('whole')},{frac}")
        return words
    return version_words(int(match.group("whole")), frac)


def _expand_year_bare(match: re.Match[str]) -> str:
    return year_words(int(match.group("num")), case="nom")


def _expand_bare(match: re.Match[str]) -> str:
    return cardinal(int(match.group("num")))


_HANDLERS: dict[str, Callable[[re.Match[str]], str]] = {
    # Keys must match NumberRule.id in uk_dialog_number_rules.py.
    "date_day_month_year": _expand_date,
    "year_with_roku": _expand_year_with_roku,
    "time_hhmm": _expand_time,
    "radio_hour": _expand_radio_hour,
    "percent": _expand_percent,
    "caps_amount": _expand_caps,
    "vault_id": _expand_vault,
    "duration_unit": _expand_with_unit,
    "distance_unit": _expand_with_unit,
    "temperature": _expand_temperature,
    "deal_split": _expand_deal,
    "ratio_slash": _expand_ratio,
    "countdown": _expand_countdown,
    "hash_id": _expand_hash,
    "hyphen_ordinal": _expand_hyphen_ordinal,
    "en_ordinal": _expand_en_ordinal,
    "numeric_range": _expand_numeric_range,
    "decimal_or_version": _expand_decimal_or_version,
    "year_bare_fallout": _expand_year_bare,
    "bare_thousands_plus": _expand_bare,
    "bare_int": _expand_bare,
}
