"""A new --project-name must land in a new run directory, not the .env project's."""

from __future__ import annotations

from pathlib import Path

from fish_studio.training.train_lora import resolve_run_dir


def test_a_new_project_name_gets_its_own_directory():
    """The failure this guards: v17 resumed v16's step_1000 and stopped at zero steps."""
    runs = Path("/data/training/runs")
    assert resolve_run_dir("fish-uk-v17-tau2", None, runs_root=runs) == runs / "fish-uk-v17-tau2"


def test_an_explicit_run_dir_always_wins():
    runs = Path("/data/training/runs")
    explicit = Path("/elsewhere/run")
    assert resolve_run_dir("fish-uk-v17-tau2", explicit, runs_root=runs) == explicit


def test_without_a_project_and_without_a_run_dir_there_is_nowhere_to_write():
    assert resolve_run_dir("fish-uk", None, runs_root=None) is None
