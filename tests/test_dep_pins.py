"""Keep extras from re-introducing the fish-speech / audiotools resolver fight."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
INSTALL_SH = (ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")


def _requirement_names(block: str) -> list[str]:
    names: list[str] = []
    for raw in block.splitlines():
        line = raw.split("#", 1)[0].strip().strip(",").strip('"').strip("'")
        if not line:
            continue
        name = line.split("@", 1)[0].split(">", 1)[0].split("<", 1)[0].split("=", 1)[0].split("[", 1)[0]
        names.append(name.strip().lower())
    return names


def test_declared_deps_do_not_pin_protobuf() -> None:
    start = PYPROJECT.index("dependencies = [")
    end = PYPROJECT.index("]", start)
    names = _requirement_names(PYPROJECT[start:end])
    assert "protobuf" not in names


def test_extras_do_not_fight_fish_speech_pins() -> None:
    start = PYPROJECT.index("[project.optional-dependencies]")
    extras = PYPROJECT[start : PYPROJECT.index("[tool.uv]", start)]
    names = _requirement_names(extras)
    # fish-speech v2.0.0-beta already pins these; a second range makes pip fail.
    assert "datasets" not in names
    assert "tokenizers" not in names
    assert "pandas" not in names
    assert "torch" not in names
    assert "torchaudio" not in names


def test_tensorboard_stays_below_protobuf_6_pin() -> None:
    assert '"tensorboard>=2.20,<2.21"' in PYPROJECT
    assert "tensorboard>=2.21" not in PYPROJECT


def test_install_sh_overrides_protobuf_after_resolve() -> None:
    assert "force_protobuf_for_fish_speech" in INSTALL_SH
    assert "--no-deps" in INSTALL_SH
    assert "protobuf>=4.25.3,<6" in INSTALL_SH
    assert "|| true" not in INSTALL_SH
    assert "setuptools>=68,<81" in INSTALL_SH
