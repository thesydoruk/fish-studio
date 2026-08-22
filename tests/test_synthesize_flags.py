"""Parse match_loudness / match_timing form flags (default true)."""

from fish_studio.server.app import _parse_flag


def test_parse_flag_defaults_to_true() -> None:
    assert _parse_flag(None) is True
    assert _parse_flag("") is True


def test_parse_flag_accepts_common_false_values() -> None:
    assert _parse_flag("false") is False
    assert _parse_flag("0") is False
    assert _parse_flag("off") is False
    assert _parse_flag(False) is False


def test_parse_flag_accepts_common_true_values() -> None:
    assert _parse_flag("true") is True
    assert _parse_flag("1") is True
    assert _parse_flag(True) is True
