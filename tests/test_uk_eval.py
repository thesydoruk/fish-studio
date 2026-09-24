"""Probe-set loading and the two axes the report puts a number on."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fish_studio.server.uk_eval import (
    Axis,
    Report,
    Voice,
    VoiceReport,
    evaluate_voices,
    load_probes,
    load_voices,
    score_audio,
    trill_weak_share,
    vowel_summary,
)
from fish_studio.stress_align import CtcAligner, VowelSpan

pytest.importorskip("numpy")

_ROWS = (
    "id\tspeaker\ttext\ttruth\thuman_wav\tref_wav\tref_text\n"
    "a_1\tanna\tсьогодні погода\tсього́дні пого́да\tdatasets/x/1.wav\ttraining/raw/anna/9.wav\tрефере́нс\n"
    "a_2\tanna\tгарна погода\tга́рна пого́да\tdatasets/x/2.wav\ttraining/raw/anna/9.wav\tрефере́нс\n"
)


def _write_probe(tmp_path: Path) -> Path:
    path = tmp_path / "uk_probe.tsv"
    path.write_text(_ROWS, encoding="utf-8")
    return path


def test_load_probes_resolves_paths_under_the_data_root(tmp_path):
    """The set is frozen with relative paths so it travels between hosts."""
    probes = load_probes(_write_probe(tmp_path), Path("/data"))
    assert [probe.id for probe in probes] == ["a_1", "a_2"]
    assert probes[0].human_wav == Path("/data/datasets/x/1.wav")
    assert probes[0].ref_wav == Path("/data/training/raw/anna/9.wav")


def test_load_probes_honours_the_limit(tmp_path):
    assert len(load_probes(_write_probe(tmp_path), Path("/data"), limit=1)) == 1


def test_load_probes_rejects_an_empty_set(tmp_path):
    path = tmp_path / "empty.tsv"
    path.write_text("id\tspeaker\ttext\ttruth\thuman_wav\tref_wav\tref_text\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_probes(path, Path("/data"))


def test_axis_reports_zero_rather_than_dividing_by_nothing():
    assert Axis().value == 0.0


class _FixedAligner(CtcAligner):
    def __init__(self, spans):
        super().__init__()
        self._spans = spans

    def align(self, text, audio, sample_rate, **kwargs):  # type: ignore[override]
        return self._spans


def _two_vowel_spans(second_is_stressed: bool) -> list[VowelSpan]:
    long_span = (0.05, 0.10, 0.30)
    short_span = (0.05, 0.08, 0.12)
    first = short_span if second_is_stressed else long_span
    second = long_span if second_is_stressed else short_span
    return [
        VowelSpan(
            word_index=0, vowel_ordinal=0, char="о", start=0.0, end=first[1], filled_end=first[2]
        ),
        VowelSpan(
            word_index=0,
            vowel_ordinal=1,
            char="а",
            start=0.35,
            end=0.35 + second[1],
            filled_end=0.35 + second[2],
        ),
    ]


def test_score_audio_counts_truth_and_follow_separately():
    """A lexicon miss must lower stress while leaving the model's follow intact."""
    rate = 16_000
    samples = np.full(rate, 0.4, dtype=np.float32)
    aligner = _FixedAligner(_two_vowel_spans(second_is_stressed=True))
    stress, follow = Axis(), Axis()

    score_audio(
        samples,
        rate,
        expected={0: 0},  # curated truth says the first vowel
        marked={0: 1},  # the server marked the second, and the take obeyed
        aligner=aligner,
        text="пора",
        stress_axis=stress,
        follow_axis=follow,
    )
    assert (stress.hits, stress.total) == (0, 1)
    assert (follow.hits, follow.total) == (1, 1)


def test_score_audio_reports_failure_when_nothing_aligned():
    """An unalignable take is counted as such, not as a wrong answer."""

    class _Dead(CtcAligner):
        def align(self, text, audio, sample_rate, **kwargs):  # type: ignore[override]
            return None

    stress = Axis()
    ok = score_audio(
        np.zeros(16_000, dtype=np.float32),
        16_000,
        expected={0: 0},
        marked={},
        aligner=_Dead(),
        text="пора",
        stress_axis=stress,
        follow_axis=None,
    )
    assert ok is False
    assert stress.total == 0


def test_report_shows_the_human_ceiling_next_to_the_model():
    """The model is read against what the estimator scores on real speech."""
    report = Report(label="scale-0.5")
    report.stress.hits, report.stress.total = 70, 100
    report.human_stress.hits, report.human_stress.total = 75, 100
    report.voice.append(0.55)
    report.human_voice.append(0.50)
    line = report.format()
    assert "stress= 70.0% (human 75.0%)" in line
    assert "voice=0.550 (human 0.500)" in line


def test_report_omits_the_ceiling_when_human_scoring_was_skipped():
    report = Report(label="scale-0.5")
    report.stress.hits, report.stress.total = 70, 100
    assert "human" not in report.format()


_VOICE_ROWS = (
    "id\twav\ttext\tseen_sim_min\tseen_sim_max\tseconds\n"
    "eb4a9fad\teval/voices/eb4a9fad.wav\tи я знаю\t-0.033\t0.155\t4.22\n"
    "missing\teval/voices/nope.wav\tтекст\t0.5\t0.6\t4.0\n"
)


def test_load_voices_skips_clips_that_are_not_on_disk(tmp_path):
    """A rotated-away log entry must not abort a sweep that can still run."""
    (tmp_path / "eval" / "voices").mkdir(parents=True)
    (tmp_path / "eval" / "voices" / "eb4a9fad.wav").write_bytes(b"RIFF")
    index = tmp_path / "index.tsv"
    index.write_text(_VOICE_ROWS, encoding="utf-8")

    voices = load_voices(index, tmp_path)
    assert [voice.id for voice in voices] == ["eb4a9fad"]
    assert voices[0].text == "и я знаю"


def test_load_voices_rejects_an_index_with_nothing_usable(tmp_path):
    index = tmp_path / "index.tsv"
    index.write_text(_VOICE_ROWS, encoding="utf-8")
    with pytest.raises(SystemExit):
        load_voices(index, tmp_path)


def test_voice_report_means_zero_before_any_take():
    assert VoiceReport(voice_id="eb4a9fad").mean == 0.0


def test_voice_lines_zero_skips_the_cohort_instead_of_running_every_line():
    """0 means off. Reading it as "no limit" would replay the whole set per voice."""
    reports = evaluate_voices(
        voices=[Voice(id="eb4a9fad", wav=Path("nope.wav"), text="текст")],
        probes=[],
        client=None,
        encoder=None,
        aligner=None,
        language="uk",
        lines=0,
        out_dir=None,
    )
    assert reports == []


def test_report_prints_a_gop_line_with_per_letter_means():
    """The letter breakdown is what turns 'there is an accent' into 'it is on р'."""
    report = Report(label="cand")
    report.stress.hits, report.stress.total = 70, 100
    report.gop = {"р": [-2.0, -1.0], "а": [-0.2]}
    report.human_gop = {"р": [-0.5], "а": [-0.1]}
    line = report.format()
    assert "gop=-1.07" in line
    assert "(human -0.30)" in line
    assert "р:-1.50" in line


def test_score_audio_collects_gop_into_the_sink():
    class _Scored(CtcAligner):
        def align(self, text, audio, sample_rate, **kwargs):  # type: ignore[override]
            return [
                VowelSpan(
                    word_index=0, vowel_ordinal=-1, char="р", start=0.0, end=0.05, score=-1.5
                ),
                VowelSpan(
                    word_index=0,
                    vowel_ordinal=0,
                    char="а",
                    start=0.05,
                    end=0.1,
                    filled_end=0.2,
                    score=-0.1,
                ),
                VowelSpan(
                    word_index=0,
                    vowel_ordinal=1,
                    char="о",
                    start=0.3,
                    end=0.4,
                    filled_end=0.6,
                    score=-0.3,
                ),
            ]

    sink: dict[str, list[float]] = {}
    score_audio(
        np.full(16_000, 0.4, dtype=np.float32),
        16_000,
        expected={0: 1},
        marked={},
        aligner=_Scored(),
        text="рао",
        stress_axis=Axis(),
        follow_axis=None,
        gop_sink=sink,
    )
    assert sink == {"р": [-1.5], "а": [-0.1], "о": [-0.3]}


def test_report_prints_margins_with_the_human_ceiling_per_key():
    report = Report(label="cand")
    report.stress.hits, report.stress.total = 70, 100
    report.margins = {"г": [0.2, 0.6], "и/р": [1.0]}
    report.human_margins = {"г": [2.0]}
    line = report.format()
    assert "margin  г:+0.40(h+2.00)  и/р:+1.00" in line


def test_score_audio_collects_margins_into_the_sink():
    class _Rivals(CtcAligner):
        def align(self, text, audio, sample_rate, **kwargs):  # type: ignore[override]
            return [
                VowelSpan(word_index=0, vowel_ordinal=-1, char="р", start=0.0, end=0.05, score=0.0),
                VowelSpan(
                    word_index=0,
                    vowel_ordinal=0,
                    char="и",
                    start=0.05,
                    end=0.1,
                    filled_end=0.2,
                    score=-0.3,
                    rival_score=-1.3,
                ),
                VowelSpan(
                    word_index=0,
                    vowel_ordinal=1,
                    char="а",
                    start=0.3,
                    end=0.4,
                    filled_end=0.6,
                    score=-0.1,
                ),
            ]

    sink: dict[str, list[float]] = {}
    score_audio(
        np.full(16_000, 0.4, dtype=np.float32),
        16_000,
        expected={0: 1},
        marked={},
        aligner=_Rivals(),
        text="риа",
        stress_axis=Axis(),
        follow_axis=None,
        margin_sink=sink,
    )
    assert sink["и"] == pytest.approx([1.0])
    assert sink["и/р"] == pytest.approx([1.0])


def test_report_prints_the_palatalization_line_with_the_human_ceiling():
    report = Report(label="cand")
    report.stress.hits, report.stress.total = 70, 100
    report.palatal = {"р": [120.0, 180.0], "р/а": [150.0]}
    report.human_palatal = {"р": [40.0]}
    line = report.format()
    assert "palatal р:+150(h+40)  р/а:+150" in line


def test_score_audio_collects_palatalization_into_the_sink(monkeypatch):
    import fish_studio.server.uk_eval as module

    monkeypatch.setattr(
        module, "measure_palatalization", lambda text, samples, rate, aligner: {"р": [95.0]}
    )
    sink: dict[str, list[float]] = {}
    score_audio(
        np.full(16_000, 0.4, dtype=np.float32),
        16_000,
        expected={0: 1},
        marked={},
        aligner=_FixedAligner(_two_vowel_spans(second_is_stressed=True)),
        text="пора",
        stress_axis=Axis(),
        follow_axis=None,
        palatal_sink=sink,
    )
    assert sink == {"р": [95.0]}


def test_report_prints_trill_medians_with_the_human_ceiling():
    report = Report(label="cand")
    report.stress.hits, report.stress.total = 70, 100
    report.trill = {
        "р": [
            {"dur_ms": 40.0, "rv_ratio": 0.8, "dips": 1.0},
            {"dur_ms": 80.0, "rv_ratio": 2.0, "dips": 3.0},
        ],
    }
    report.human_trill = {"р": [{"dur_ms": 40.0, "rv_ratio": 1.0, "dips": 1.0}]}
    line = report.format()
    assert "trill   р:60ms/1.40/2/w0%(h40/1.00/1/w0%)" in line


def test_score_audio_collects_trill_features_into_the_sink(monkeypatch):
    import fish_studio.server.uk_eval as module

    monkeypatch.setattr(
        module,
        "measure_trill",
        lambda text, samples, rate, aligner: {
            "р": [{"dur_ms": 60.0, "rv_ratio": 1.5, "dips": 2.0}]
        },
    )
    sink: dict[str, list[dict[str, float]]] = {}
    score_audio(
        np.full(16_000, 0.4, dtype=np.float32),
        16_000,
        expected={0: 1},
        marked={},
        aligner=_FixedAligner(_two_vowel_spans(second_is_stressed=True)),
        text="пора",
        stress_axis=Axis(),
        follow_axis=None,
        trill_sink=sink,
    )
    assert sink == {"р": [{"dur_ms": 60.0, "rv_ratio": 1.5, "dips": 2.0}]}


def test_trill_weak_share_counts_only_the_short_quiet_closureless_tail():
    """The median hid the deficit; the tail is where a weak р lives."""
    feats = [
        {"dur_ms": 40.0, "rv_ratio": 0.6, "dips": 0.0},  # weak
        {"dur_ms": 40.0, "rv_ratio": 0.6, "dips": 1.0},  # has a closure
        {"dur_ms": 60.0, "rv_ratio": 0.6, "dips": 0.0},  # long enough
        {"dur_ms": 40.0, "rv_ratio": 1.2, "dips": 0.0},  # loud enough
    ]
    assert trill_weak_share(feats) == pytest.approx(0.25)
    assert trill_weak_share([]) != trill_weak_share([])  # nan for nothing measured


def test_report_prints_pitch_spread_with_the_human_reference():
    report = Report(label="cand")
    report.stress.hits, report.stress.total = 70, 100
    report.pitch = [{"f0_std_st": 5.0, "f0_range_st": 11.0, "voiced": 0.6}]
    report.human_pitch = [{"f0_std_st": 6.0, "f0_range_st": 15.0, "voiced": 0.6}]
    assert "pitch   σ=5.0st range=11.0st (h6.0/15.0)" in report.format()


def test_vowel_summary_reports_the_и_і_distance_and_needs_both_vowels():
    vowels = {"и": [(550.0, 1700.0), (530.0, 1750.0)], "і": [(300.0, 2300.0)]}
    summary = vowel_summary(vowels)
    assert summary is not None
    assert summary["dF2"] == pytest.approx(2300.0 - 1725.0)
    assert summary["dF1"] == pytest.approx(540.0 - 300.0)
    assert vowel_summary({"и": [(550.0, 1700.0)]}) is None


def test_report_prints_the_vowel_line_with_the_human_distance():
    report = Report(label="cand")
    report.stress.hits, report.stress.total = 70, 100
    report.vowels = {"и": [(540.0, 1725.0)], "і": [(300.0, 2300.0)]}
    report.human_vowels = {"и": [(560.0, 1650.0)], "і": [(290.0, 2350.0)]}
    line = report.format()
    assert "vowel   и:540/1725  і:300/2300  dF2=+575 dF1=+240  (h dF2=+700 dF1=+270)" in line
