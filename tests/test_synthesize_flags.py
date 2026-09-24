"""Parse match_timing form flags (default true)."""

from fish_studio.server.app import _parse_flag


def test_parse_flag_defaults_to_true() -> None:
    assert _parse_flag(None) is True
    assert _parse_flag("") is True


def test_parse_flag_can_default_to_false() -> None:
    assert _parse_flag(None, default=False) is False
    assert _parse_flag("", default=False) is False


def test_parse_flag_accepts_common_false_values() -> None:
    assert _parse_flag("false") is False
    assert _parse_flag("0") is False
    assert _parse_flag("off") is False
    assert _parse_flag(False) is False


def test_parse_flag_accepts_common_true_values() -> None:
    assert _parse_flag("true") is True
    assert _parse_flag("1") is True
    assert _parse_flag(True) is True


def test_parse_attempts_defaults_to_one_take() -> None:
    from fish_studio.server.app import _parse_attempts

    assert _parse_attempts(None) == 1
    assert _parse_attempts("") == 1
    assert _parse_attempts("5") == 5


def test_parse_attempts_rejects_nonsense_and_excess() -> None:
    import pytest

    from fish_studio.server.app import _parse_attempts

    with pytest.raises(ValueError):
        _parse_attempts("0")
    with pytest.raises(ValueError):
        _parse_attempts("99")
    with pytest.raises(ValueError):
        _parse_attempts("many")


def test_parse_retry_below_defaults_to_never_judging_the_voice() -> None:
    import pytest

    from fish_studio.server.app import _parse_retry_below

    assert _parse_retry_below(None) == 0.0
    assert _parse_retry_below("0.3") == 0.3
    with pytest.raises(ValueError):
        _parse_retry_below("1.5")
