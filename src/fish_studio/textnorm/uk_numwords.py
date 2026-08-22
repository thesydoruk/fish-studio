"""Ukrainian number-words used by dialogue TTS normalization.

Covers cardinals and the ordinal genders/cases needed for dates, years, and
clock hours in Fallout-style lines. Not a general morphological engine.
"""

from __future__ import annotations

_ONES_MASC = (
    "нуль",
    "один",
    "два",
    "три",
    "чотири",
    "п'ять",
    "шість",
    "сім",
    "вісім",
    "дев'ять",
)
_ONES_FEM = (
    "нуль",
    "одна",
    "дві",
    "три",
    "чотири",
    "п'ять",
    "шість",
    "сім",
    "вісім",
    "дев'ять",
)
_ONES_NEUT = (
    "нуль",
    "одне",
    "два",
    "три",
    "чотири",
    "п'ять",
    "шість",
    "сім",
    "вісім",
    "дев'ять",
)
_TEENS = (
    "десять",
    "одинадцять",
    "дванадцять",
    "тринадцять",
    "чотирнадцять",
    "п'ятнадцять",
    "шістнадцять",
    "сімнадцять",
    "вісімнадцять",
    "дев'ятнадцять",
)
_TENS = (
    "",
    "",
    "двадцять",
    "тридцять",
    "сорок",
    "п'ятдесят",
    "шістдесят",
    "сімдесят",
    "вісімдесят",
    "дев'яносто",
)
_HUNDREDS = (
    "",
    "сто",
    "двісті",
    "триста",
    "чотириста",
    "п'ятсот",
    "шістсот",
    "сімсот",
    "вісімсот",
    "дев'ятсот",
)

_ORD_MASC_NOM = {
    1: "перший",
    2: "другий",
    3: "третій",
    4: "четвертий",
    5: "п'ятий",
    6: "шостий",
    7: "сьомий",
    8: "восьмий",
    9: "дев'ятий",
    10: "десятий",
    11: "одинадцятий",
    12: "дванадцятий",
    13: "тринадцятий",
    14: "чотирнадцятий",
    15: "п'ятнадцятий",
    16: "шістнадцятий",
    17: "сімнадцятий",
    18: "вісімнадцятий",
    19: "дев'ятнадцятий",
    20: "двадцятий",
    30: "тридцятий",
    40: "сороковий",
    50: "п'ятдесятий",
    60: "шістдесятий",
    70: "сімдесятий",
    80: "вісімдесятий",
    90: "дев'яностий",
}

_ORD_FEM_NOM = {
    1: "перша",
    2: "друга",
    3: "третя",
    4: "четверта",
    5: "п'ята",
    6: "шоста",
    7: "сьома",
    8: "восьма",
    9: "дев'ята",
    10: "десята",
    11: "одинадцята",
    12: "дванадцята",
    13: "тринадцята",
    14: "чотирнадцята",
    15: "п'ятнадцята",
    16: "шістнадцята",
    17: "сімнадцята",
    18: "вісімнадцята",
    19: "дев'ятнадцята",
    20: "двадцята",
    30: "тридцята",
    40: "сорокова",
    50: "п'ятдесята",
    60: "шістдесята",
    70: "сімдесята",
    80: "вісімдесята",
    90: "дев'яноста",
}

_ORD_NEUT_NOM = {
    1: "перше",
    2: "друге",
    3: "третє",
    4: "четверте",
    5: "п'яте",
    6: "шосте",
    7: "сьоме",
    8: "восьме",
    9: "дев'яте",
    10: "десяте",
    11: "одинадцяте",
    12: "дванадцяте",
    13: "тринадцяте",
    14: "чотирнадцяте",
    15: "п'ятнадцяте",
    16: "шістнадцяте",
    17: "сімнадцяте",
    18: "вісімнадцяте",
    19: "дев'ятнадцяте",
    20: "двадцяте",
    30: "тридцяте",
    40: "сорокове",
    50: "п'ятдесяте",
    60: "шістдесяте",
    70: "сімдесяте",
    80: "вісімдесяте",
    90: "дев'яносте",
}

_HUNDRED_ORD_MASC = {
    1: "сотий",
    2: "двохсотий",
    3: "трьохсотий",
    4: "чотирьохсотий",
    5: "п'ятисотий",
    6: "шестисотий",
    7: "семисотий",
    8: "восьмисотий",
    9: "дев'ятисотий",
}


def _ones(gender: str) -> tuple[str, ...]:
    if gender == "fem":
        return _ONES_FEM
    if gender == "neut":
        return _ONES_NEUT
    return _ONES_MASC


def _under_thousand(n: int, gender: str) -> str:
    if n < 0 or n >= 1000:
        raise ValueError(n)
    if n < 10:
        return _ones(gender)[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return _TENS[tens]
        return f"{_TENS[tens]} {_ones(gender)[ones]}"
    hundreds, rest = divmod(n, 100)
    if rest == 0:
        return _HUNDREDS[hundreds]
    return f"{_HUNDREDS[hundreds]} {_under_thousand(rest, gender)}"


def _thousand_word(n: int) -> str:
    n = abs(n) % 100
    if 11 <= n <= 14:
        return "тисяч"
    n %= 10
    if n == 1:
        return "тисяча"
    if 2 <= n <= 4:
        return "тисячі"
    return "тисяч"


def _million_word(n: int) -> str:
    n = abs(n) % 100
    if 11 <= n <= 14:
        return "мільйонів"
    n %= 10
    if n == 1:
        return "мільйон"
    if 2 <= n <= 4:
        return "мільйони"
    return "мільйонів"


def cardinal(n: int, *, gender: str = "masc") -> str:
    """Nominative cardinal. ``gender`` affects 1/2 (and compounds ending in them)."""
    if n < 0:
        return f"мінус {cardinal(-n, gender=gender)}"
    if n < 1000:
        return _under_thousand(n, gender)
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        th_num = _under_thousand(thousands, "fem")
        th_word = _thousand_word(thousands)
        if rest == 0:
            return f"{th_num} {th_word}"
        return f"{th_num} {th_word} {_under_thousand(rest, gender)}"
    millions, rest = divmod(n, 1_000_000)
    head = f"{_under_thousand(millions, 'fem')} {_million_word(millions)}"
    if rest == 0:
        return head
    return f"{head} {cardinal(rest, gender=gender)}"


def _inflect_ordinal_masc(nom: str, case: str) -> str:
    if case == "nom":
        return nom
    if case == "nom_pl":
        if nom.endswith(("ий", "ій")):
            return nom[:-2] + "і"
        return nom
    if nom.endswith("ій"):  # третій
        stem = nom[:-2]
        if case == "gen":
            return stem + "ього"
        if case == "loc":
            return stem + "ьому"
        if case == "instr":
            return stem + "ім"
        if case == "gen_pl":
            return stem + "іх"
    if nom.endswith("ий"):
        stem = nom[:-2]
        if case == "gen":
            return stem + "ого"
        if case == "loc":
            return stem + "ому"
        if case == "instr":
            return stem + "им"
        if case == "gen_pl":
            return stem + "их"
    return nom


def _inflect_ordinal_fem(nom: str, case: str) -> str:
    if case == "nom":
        return nom
    if case == "nom_pl":
        if nom.endswith(("а", "я")):
            return nom[:-1] + "і"
        return nom
    if nom.endswith(("я", "а")):
        stem = nom[:-1]
        soft = nom.endswith("я")
        if case == "gen":
            return stem + ("ьої" if soft else "ої")
        if case == "loc":
            return stem + "ій"
        if case == "instr":
            return stem + ("ьою" if soft else "ою")
        if case == "gen_pl":
            return stem + ("іх" if soft else "их")
    return nom


def _inflect_ordinal_neut(nom: str, case: str) -> str:
    if case == "nom":
        return nom
    if case == "nom_pl":
        if nom.endswith(("е", "є")):
            return nom[:-1] + "і"
        return nom
    if nom.endswith("є"):  # третє
        stem = nom[:-1]
        if case == "gen":
            return stem + "ього"
        if case == "loc":
            return stem + "ьому"
        if case == "instr":
            return stem + "ім"
        if case == "gen_pl":
            return stem + "іх"
    if nom.endswith("е"):
        stem = nom[:-1]
        if case == "gen":
            return stem + "ого"
        if case == "loc":
            return stem + "ому"
        if case == "instr":
            return stem + "им"
        if case == "gen_pl":
            return stem + "их"
    return nom


def _gender_table(gender: str) -> dict[int, str]:
    if gender == "fem":
        return _ORD_FEM_NOM
    if gender == "neut":
        return _ORD_NEUT_NOM
    return _ORD_MASC_NOM


def _inflect(nom: str, gender: str, case: str) -> str:
    if gender == "fem":
        return _inflect_ordinal_fem(nom, case)
    if gender == "neut":
        return _inflect_ordinal_neut(nom, case)
    return _inflect_ordinal_masc(nom, case)


def _ordinal_under_hundred(n: int, gender: str, case: str) -> str:
    table = _gender_table(gender)
    if n in table:
        return _inflect(table[n], gender, case)
    if n < 1 or n > 99:
        raise ValueError(n)
    tens, ones = divmod(n, 10)
    return f"{_TENS[tens]} {_inflect(table[ones], gender, case)}"


def _hundred_ord(hundreds: int, gender: str, case: str) -> str:
    base = _HUNDRED_ORD_MASC[hundreds]
    if gender == "fem":
        base = base[:-2] + "а"
    elif gender == "neut":
        base = base[:-2] + "е"
    return _inflect(base, gender, case)


def ordinal(n: int, *, gender: str = "masc", case: str = "nom") -> str:
    """Ordinal in ``gender`` (masc/fem/neut) and ``case`` (nom/gen/loc)."""
    if n <= 0:
        return cardinal(n)
    if n < 100:
        return _ordinal_under_hundred(n, gender, case)
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if rest == 0:
            return _hundred_ord(hundreds, gender, case)
        return f"{_HUNDREDS[hundreds]} {_ordinal_under_hundred(rest, gender, case)}"
    thousands, rest = divmod(n, 1000)
    # Years sound more natural as "тисяча дев'ятсот…" than "одна тисяча…".
    if thousands == 1:
        head = "тисяча"
    else:
        head = f"{_under_thousand(thousands, 'fem')} {_thousand_word(thousands)}"
    if rest == 0:
        return head
    if rest < 100:
        return f"{head} {_ordinal_under_hundred(rest, gender, case)}"
    hundreds, low = divmod(rest, 100)
    if low == 0:
        return f"{head} {_hundred_ord(hundreds, gender, case)}"
    return f"{head} {_HUNDREDS[hundreds]} {_ordinal_under_hundred(low, gender, case)}"


def year_words(n: int, *, case: str = "nom") -> str:
    """Year as masculine ordinal phrase."""
    return ordinal(n, gender="masc", case=case)


def decimal_words(whole: int, frac: str) -> str:
    """TTS-friendly decimal: ``2,74`` → ``дві цілих сімдесят чотири``."""
    if whole == 1 or (whole % 10 == 1 and whole % 100 != 11):
        whole_part = f"{cardinal(whole, gender='fem')} ціла"
    elif whole % 10 == 2 and whole % 100 != 12:
        whole_part = f"{cardinal(whole, gender='fem')} цілих"
    else:
        whole_part = f"{cardinal(whole)} цілих"
    return f"{whole_part} {cardinal(int(frac))}"


def version_words(whole: int, frac: str) -> str:
    """Model/caliber style ``1.0`` / ``5.56`` → ``один нуль`` / ``п'ять п'ятдесят шість``."""
    return f"{cardinal(whole)} {cardinal(int(frac))}"


def clock_minutes(n: int) -> str:
    """Minutes for HH:MM — ``04`` → ``нуль чотири``, ``35`` → ``тридцять п'ять``."""
    if n < 10:
        return f"нуль {cardinal(n)}"
    return cardinal(n)


_GEN_ONES = (
    "нуля",
    "одного",
    "двох",
    "трьох",
    "чотирьох",
    "п'яти",
    "шести",
    "семи",
    "восьми",
    "дев'яти",
)
_GEN_TEENS = (
    "десяти",
    "одинадцяти",
    "дванадцяти",
    "тринадцяти",
    "чотирнадцяти",
    "п'ятнадцяти",
    "шістнадцяти",
    "сімнадцяти",
    "вісімнадцяти",
    "дев'ятнадцяти",
)
_GEN_TENS = (
    "",
    "",
    "двадцяти",
    "тридцяти",
    "сорока",
    "п'ятдесяти",
    "шістдесяти",
    "сімдесяти",
    "вісімдесяти",
    "дев'яноста",
)
_GEN_HUNDREDS = (
    "",
    "ста",
    "двохсот",
    "трьохсот",
    "чотирьохсот",
    "п'ятисот",
    "шестисот",
    "семисот",
    "восьмисот",
    "дев'ятисот",
)


def _genitive_under_thousand(n: int) -> str:
    if n < 10:
        return _GEN_ONES[n]
    if n < 20:
        return _GEN_TEENS[n - 10]
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return _GEN_TENS[tens]
        return f"{_GEN_TENS[tens]} {_GEN_ONES[ones]}"
    hundreds, rest = divmod(n, 100)
    if rest == 0:
        return _GEN_HUNDREDS[hundreds]
    return f"{_GEN_HUNDREDS[hundreds]} {_genitive_under_thousand(rest)}"


def cardinal_genitive(n: int) -> str:
    """Genitive cardinal: ``15`` → ``п'ятнадцяти``, ``2`` → ``двох``."""
    if n < 0:
        return f"мінус {cardinal_genitive(-n)}"
    if n < 1000:
        return _genitive_under_thousand(n)
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        # Genitive of "N тисяч"
        if thousands == 1:
            head = "тисячі"
        else:
            head = f"{_genitive_under_thousand(thousands)} тисяч"
        if rest == 0:
            return head
        return f"{head} {_genitive_under_thousand(rest)}"
    return cardinal(n)  # rare in dialogue; fall back
