"""Coverage checks for corpus-derived Ukrainian dialogue number rules."""

from __future__ import annotations

from fish_studio.textnorm.uk_dialog_number_rules import (
    classify_number_spans,
    iter_number_rules,
)


def test_rule_ids_are_unique() -> None:
    ids = [rule.id for rule in iter_number_rules()]
    assert len(ids) == len(set(ids))


def test_priority_vault_not_year() -> None:
    hits = classify_number_spans("Чули про Сховище 81?")
    assert [rule.id for rule, _ in hits] == ["vault_id"]


def test_priority_caps_not_year() -> None:
    hits = classify_number_spans("Мені потрібно 2000 кришок.")
    assert [rule.id for rule, _ in hits] == ["caps_amount"]


def test_date_and_time() -> None:
    hits = classify_number_spans(
        "Запис номер три, 26 лютого 2077 року. Час... 12:35."
    )
    ids = [rule.id for rule, _ in hits]
    assert "date_day_month_year" in ids
    assert "time_hhmm" in ids


def test_year_locative() -> None:
    hits = classify_number_spans("Битви - це так у 2077-му.")
    assert hits[0][0].id == "year_with_roku"


def test_alnum_codename_left_alone() -> None:
    hits = classify_number_spans("Ви ставите під сумнів мої накази, X6?")
    assert len(hits) == 1
    assert hits[0][0].id == "alnum_codename"
    assert hits[0][0].leave_as_is is True


def test_percent_and_countdown() -> None:
    assert classify_number_spans("Довірчий інтервал... 2,74%.")[0][0].id == "percent"
    ids = [rule.id for rule, _ in classify_number_spans("5... 4... 3... 2... 1...")]
    assert ids == ["countdown"] * 5


def test_deal_split() -> None:
    hits = classify_number_spans("70 на 30 і цуценя.")
    assert [rule.id for rule, _ in hits] == ["deal_split"]


def test_examples_on_each_rule_match_themselves() -> None:
    """Every documented example must trigger its owning rule at least once."""
    for rule in iter_number_rules():
        matched_any = False
        for example in rule.examples:
            if not any(ch.isdigit() for ch in example):
                continue
            ids = {hit.id for hit, _ in classify_number_spans(example)}
            if rule.id in ids:
                matched_any = True
                break
        assert matched_any, f"rule {rule.id} examples did not self-match"
