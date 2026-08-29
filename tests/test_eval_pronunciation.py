import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "eval_pronunciation",
    Path(__file__).resolve().parents[1] / "scripts" / "eval_pronunciation.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_mod)


def test_cer_identical_after_punct() -> None:
    assert _mod.character_error_rate("Привіт, світе!", "привіт світе") == 0.0


def test_cer_substitution() -> None:
    assert abs(_mod.character_error_rate("кіт", "кот") - 1 / 3) < 1e-9
