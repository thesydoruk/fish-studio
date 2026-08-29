"""Ukrainian dialogue number expansion for TTS."""

from __future__ import annotations

from fish_studio.textnorm.expand_uk_numbers import expand_uk_numbers, should_expand_numbers
from fish_studio.textnorm.uk_numwords import (
    cardinal,
    decimal_words,
    half_words,
    ordinal,
    year_words,
)


def test_should_expand_only_uk() -> None:
    assert should_expand_numbers("uk")
    assert should_expand_numbers("uk-UA")
    assert should_expand_numbers(None)
    assert not should_expand_numbers("en")
    assert not should_expand_numbers("zh")


def test_cardinals() -> None:
    assert cardinal(0) == "нуль"
    assert cardinal(81) == "вісімдесят один"
    assert cardinal(111) == "сто одинадцять"
    assert cardinal(2000) == "дві тисячі"
    assert cardinal(2, gender="fem") == "дві"


def test_vault_and_caps() -> None:
    assert expand_uk_numbers("Чули про Сховище 81?") == "Чули про Сховище вісімдесят один?"
    assert (
        expand_uk_numbers("Мені потрібно 2000 кришок.")
        == "Мені потрібно дві тисячі кришок."
    )


def test_date_and_year() -> None:
    out = expand_uk_numbers("22 жовтня 2077 року")
    assert "жовтня" in out
    assert "року" in out
    assert not any(ch.isdigit() for ch in out)

    assert "2077" not in expand_uk_numbers("Битви - це так у 2077-му.")
    assert year_words(2077, case="loc").endswith("ому")


def test_time_percent_countdown() -> None:
    assert expand_uk_numbers("Шоу починається рівно о 21:00.") == (
        "Шоу починається рівно о двадцять перша нуль нуль."
    )
    assert expand_uk_numbers("35% алкоголю.") == "тридцять п'ять відсотків алкоголю."
    assert expand_uk_numbers("5... 4... 3...") == "п'ять... чотири... три..."


def test_decimal_and_half_words() -> None:
    assert decimal_words(2, "74") == "дві цілих сімдесят чотири сотих"
    assert decimal_words(135, "2") == "сто тридцять п'ять цілих дві десятих"
    assert decimal_words(0, "0013") == "нуль цілих тринадцять десятитисячних"
    assert decimal_words(21, "1") == "двадцять одна ціла одна десята"
    assert half_words(3) == "три з половиною"
    assert half_words(1) == "півтора"
    assert half_words(1, gender="fem") == "півтори"


def test_temperature_decimals() -> None:
    assert expand_uk_numbers("Ідеальна температура - 135.5 градусів за фаренгейтом") == (
        "Ідеальна температура - сто тридцять п'ять з половиною градусів за фаренгейтом"
    )
    assert expand_uk_numbers("Ідеальна температура - 135.2 градусів за фаренгейтом") == (
        "Ідеальна температура - сто тридцять п'ять цілих дві десятих градуса за фаренгейтом"
    )
    assert expand_uk_numbers("Ваша кава. 173,5 градуса за Фаренгейтом.") == (
        "Ваша кава. сто сімдесят три з половиною градуси за Фаренгейтом."
    )
    assert expand_uk_numbers("температурою до 56 градусів") == (
        "температурою до п'ятдесят шість градусів"
    )
    assert expand_uk_numbers("надворі -2 градуси") == "надворі мінус два градуси"


def test_percent_decimals_and_agreement() -> None:
    assert expand_uk_numbers("Довірчий інтервал... 2,74%.") == (
        "Довірчий інтервал... дві цілих сімдесят чотири сотих відсотка."
    )
    assert expand_uk_numbers("на 0,04 відсотка") == "на нуль цілих чотири сотих відсотка"
    assert expand_uk_numbers("ймовірність 95,23 відсотка") == (
        "ймовірність дев'яносто п'ять цілих двадцять три сотих відсотка"
    )
    assert expand_uk_numbers("Резервна енергія: 34%.") == (
        "Резервна енергія: тридцять чотири відсотки."
    )
    assert expand_uk_numbers("41%") == "сорок один відсоток"
    assert expand_uk_numbers("на 100% сумісні") == "на сто відсотків сумісні"


def test_duration_and_bare_decimals() -> None:
    assert expand_uk_numbers("Кричали 5,02 секунди.") == (
        "Кричали п'ять цілих дві сотих секунди."
    )
    assert expand_uk_numbers("будуть готові за 13.2 години") == (
        "будуть готові за тринадцять цілих дві десятих години"
    )
    assert expand_uk_numbers("у 3,5 раза точніша") == "у три з половиною раза точніша"
    assert expand_uk_numbers("2 години ночі") == "дві години ночі"
    assert expand_uk_numbers("відстань 1.5 км") == "відстань півтора км"


def test_versions_stay_spoken_digits() -> None:
    assert expand_uk_numbers("людина моделі 1.0") == "людина моделі один нуль"
    assert expand_uk_numbers("пару коробок 5.56") == "пару коробок п'ять п'ятдесят шість"


def test_alnum_left_alone() -> None:
    assert expand_uk_numbers("Накази для X6.") == "Накази для X6."


def test_deal_and_ratio() -> None:
    assert expand_uk_numbers("70 на 30 і цуценя.") == "сімдесят на тридцять і цуценя."
    assert "двадцять чотири на сім" in expand_uk_numbers("на зв'язку 24/7")


def test_radio_hour() -> None:
    assert expand_uk_numbers("11 година ранку.") == "одинадцята година ранку."


def test_hash_and_bare() -> None:
    assert expand_uk_numbers("Реактор №3 несправний.") == "Реактор номер три несправний."
    assert expand_uk_numbers("150.") == "сто п'ятдесят."


def test_ordinal_day() -> None:
    assert ordinal(22, gender="neut", case="nom") == "двадцять друге"
    assert ordinal(1, gender="fem", case="nom") == "перша"


def test_hyphen_ordinals() -> None:
    assert expand_uk_numbers("15-го") == "п'ятнадцятого"
    assert expand_uk_numbers("15го") == "п'ятнадцятого"
    assert expand_uk_numbers("15‑го") == "п'ятнадцятого"  # U+2011
    assert expand_uk_numbers("з 15-го числа") == "з п'ятнадцятого числа"
    assert expand_uk_numbers("1-го") == "першого"
    assert expand_uk_numbers("2-го") == "другого"
    assert expand_uk_numbers("3-го") == "третього"
    assert expand_uk_numbers("21-го") == "двадцять першого"
    assert expand_uk_numbers("15-й") == "п'ятнадцятий"
    assert expand_uk_numbers("15й") == "п'ятнадцятий"
    assert expand_uk_numbers("на 15-му рівні") == "на п'ятнадцятому рівні"
    assert expand_uk_numbers("2-ге") == "друге"
    assert expand_uk_numbers("7-ма") == "сьома"
    assert expand_uk_numbers("1ша") == "перша"
    assert expand_uk_numbers("15-ої") == "п'ятнадцятої"
    assert expand_uk_numbers("15-ій") == "п'ятнадцятій"
    assert expand_uk_numbers("2-х") == "двох"
    assert expand_uk_numbers("15-ти") == "п'ятнадцяти"
    assert expand_uk_numbers("15th") == "п'ятнадцятий"
    assert expand_uk_numbers("10-15") == "десять — п'ятнадцять"
    # Codenames stay; years keep their rules.
    assert expand_uk_numbers("X6") == "X6"
    assert expand_uk_numbers("1R") == "1R"
    assert "2077" not in expand_uk_numbers("у 2077-му")
    assert expand_uk_numbers("у 2077-му").endswith("ому")
    assert "1980" not in expand_uk_numbers("1980-х")
