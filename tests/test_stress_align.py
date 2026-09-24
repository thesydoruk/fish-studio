"""Reading stress off aligned audio: ordinals, filled spans, and the predictor."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from fish_studio.stress_align import (
    CtcAligner,
    VowelSpan,
    fill_stress_from_audio,
    marked_ordinals,
    measure_gop,
    measure_margins,
    measure_palatalization,
    measure_pitch_spread,
    measure_stress,
    measure_stress_with_margin,
    measure_trill,
    measure_vowel_formants,
    normalize_for_ctc,
    palatalization_index,
    trill_features,
    vowel_energy,
    vowel_spans,
)

pytest.importorskip("numpy")


def test_marked_ordinals_counts_vowels_not_characters():
    """The acute's string offset moves with spelling; its vowel ordinal does not."""
    assert marked_ordinals("пого́да") == {0: 1}
    assert marked_ordinals("зв'язо́к") == {0: 1}


def test_marked_ordinals_skips_unmarked_and_single_vowel_words():
    """Only multi-vowel words carrying a mark can be scored."""
    found = marked_ordinals("він ішо́в до лі́су")
    assert found == {1: 1, 3: 0}


def test_marked_ordinals_indexes_every_word_including_unmarked_ones():
    """Word indexes must survive gaps, or a later word is scored against the wrong target."""
    assert marked_ordinals("сього́дні гарна пого́да") == {0: 1, 2: 1}


def test_normalize_for_ctc_folds_apostrophes_and_case():
    assert normalize_for_ctc("’") == "'"
    assert normalize_for_ctc("П") == "п"
    assert normalize_for_ctc("ё") == "е"


def test_filled_duration_charges_the_blank_run_to_the_preceding_character():
    """A CTC spike is not a phone: without the blank run, duration reads backwards."""
    span = VowelSpan(word_index=0, vowel_ordinal=0, char="а", start=0.1, end=0.14, filled_end=0.30)
    assert span.duration == pytest.approx(0.04)
    assert span.filled_duration == pytest.approx(0.20)


def test_filled_duration_never_falls_below_the_spike():
    """A missing or stale filled_end must not shrink the span to nothing."""
    span = VowelSpan(word_index=0, vowel_ordinal=0, char="а", start=0.1, end=0.25, filled_end=0.0)
    assert span.filled_duration == pytest.approx(0.15)


def test_vowel_spans_drops_consonants_and_separators():
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=-1, char="п", start=0.0, end=0.1),
        VowelSpan(word_index=0, vowel_ordinal=0, char="о", start=0.1, end=0.2),
    ]
    assert [span.char for span in vowel_spans(spans)] == ["о"]


def test_vowel_energy_reads_the_filled_span_when_asked():
    """The spike covers the onset; the quiet tail only shows up in the filled span."""
    rate = 16_000
    samples = np.concatenate(
        [np.full(rate // 10, 0.5, dtype=np.float32), np.zeros(rate // 10, dtype=np.float32)]
    )
    span = VowelSpan(word_index=0, vowel_ordinal=0, char="а", start=0.0, end=0.1, filled_end=0.2)
    assert vowel_energy(samples, rate, span) == pytest.approx(0.5, abs=1e-3)
    assert vowel_energy(samples, rate, span, filled=True) == pytest.approx(0.354, abs=1e-2)


class _StubAligner(CtcAligner):
    """Returns fixed spans so the predictor can be tested without the CTC model."""

    def __init__(self, spans: list[VowelSpan]) -> None:
        super().__init__()
        self._spans = spans

    def align(self, text, audio, sample_rate, **kwargs):  # type: ignore[override]
        return self._spans


def _loud(rate: int, seconds: float, level: float) -> np.ndarray:
    return np.full(int(rate * seconds), level, dtype=np.float32)


def test_measure_stress_picks_the_long_loud_vowel():
    """Duration times energy -- the second vowel wins on both cues here."""
    rate = 16_000
    samples = np.concatenate([_loud(rate, 0.05, 0.2), _loud(rate, 0.20, 0.6)])
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=0, char="о", start=0.0, end=0.03, filled_end=0.05),
        VowelSpan(word_index=0, vowel_ordinal=1, char="а", start=0.05, end=0.10, filled_end=0.25),
    ]
    assert measure_stress("пора", samples, rate, _StubAligner(spans)) == {0: 1}


def test_measure_stress_skips_words_whose_alignment_lost_a_vowel():
    """Two vowels in the text, one aligned: every later ordinal would be wrong."""
    rate = 16_000
    samples = _loud(rate, 0.3, 0.4)
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=0, char="о", start=0.0, end=0.05, filled_end=0.1),
    ]
    assert measure_stress("пора", samples, rate, _StubAligner(spans)) == {}


def test_measure_stress_returns_none_when_alignment_is_unavailable():
    """A missing model disables the metric; it must not silently score zeros."""

    class _Dead(CtcAligner):
        def align(self, text, audio, sample_rate, **kwargs):  # type: ignore[override]
            return None

    assert measure_stress("пора", np.zeros(16_000, dtype=np.float32), 16_000, _Dead()) is None


def test_measure_stress_ignores_silent_words():
    """No energy anywhere means no verdict, rather than a vote for vowel zero."""
    rate = 16_000
    samples = np.zeros(rate, dtype=np.float32)
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=0, char="о", start=0.0, end=0.05, filled_end=0.1),
        VowelSpan(word_index=0, vowel_ordinal=1, char="а", start=0.1, end=0.15, filled_end=0.2),
    ]
    assert measure_stress("пора", samples, rate, _StubAligner(spans)) == {}


def test_measure_stress_with_margin_reports_how_clear_the_winner_was():
    """The margin is what makes the estimator safe to write marks with."""
    rate = 16_000
    samples = np.concatenate([_loud(rate, 0.05, 0.2), _loud(rate, 0.20, 0.6)])
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=0, char="о", start=0.0, end=0.03, filled_end=0.05),
        VowelSpan(word_index=0, vowel_ordinal=1, char="а", start=0.05, end=0.10, filled_end=0.25),
    ]
    heard = measure_stress_with_margin("пора", samples, rate, _StubAligner(spans))
    assert heard is not None
    choice, margin = heard[0]
    assert choice == 1
    assert margin > 2.0


def test_fill_stress_from_audio_leaves_the_text_alone_without_an_aligner(tmp_path):
    """No model means no marks -- guessing with a weaker estimator defeats the gate."""

    class _Dead(CtcAligner):
        def align(self, text, audio, sample_rate, **kwargs):  # type: ignore[override]
            return None

    wav = tmp_path / "clip.wav"
    sf.write(str(wav), np.zeros(16_000, dtype=np.float32), 16_000)
    assert fill_stress_from_audio("пора додому", wav, aligner=_Dead()) == "пора додому"


def test_fill_stress_from_audio_skips_words_below_the_margin(tmp_path):
    """Two vowels of equal weight is exactly the case that must stay unmarked."""
    rate = 16_000
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), np.full(rate, 0.4, dtype=np.float32), rate)
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=0, char="о", start=0.0, end=0.10, filled_end=0.20),
        VowelSpan(word_index=0, vowel_ordinal=1, char="а", start=0.20, end=0.30, filled_end=0.40),
    ]
    assert fill_stress_from_audio("пора", wav, aligner=_StubAligner(spans)) == "пора"


def test_fill_stress_from_audio_marks_a_confident_word(tmp_path):
    rate = 16_000
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), np.full(rate, 0.4, dtype=np.float32), rate)
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=0, char="о", start=0.0, end=0.02, filled_end=0.04),
        VowelSpan(word_index=0, vowel_ordinal=1, char="а", start=0.10, end=0.30, filled_end=0.60),
    ]
    assert fill_stress_from_audio("пора", wav, aligner=_StubAligner(spans)) == "пора́"


def test_fill_stress_from_audio_never_touches_an_already_marked_word(tmp_path):
    """Dictionary marks win; the recording only fills what is still bare."""
    rate = 16_000
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), np.full(rate, 0.4, dtype=np.float32), rate)
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=0, char="о", start=0.0, end=0.02, filled_end=0.04),
        VowelSpan(word_index=0, vowel_ordinal=1, char="а", start=0.10, end=0.30, filled_end=0.60),
    ]
    marked = "по́ра"
    assert fill_stress_from_audio(marked, wav, aligner=_StubAligner(spans)) == marked


def test_vowel_span_score_defaults_to_zero():
    assert VowelSpan(word_index=0, vowel_ordinal=0, char="а", start=0.0, end=0.1).score == 0.0


def test_measure_gop_groups_aligned_scores_by_character():
    """Two 'р' in different words land in one bucket; separators are left out."""
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=-1, char="р", start=0.0, end=0.1, score=-0.2),
        VowelSpan(word_index=0, vowel_ordinal=0, char="а", start=0.1, end=0.2, score=-0.1),
        VowelSpan(word_index=-1, vowel_ordinal=-1, char=" ", start=0.2, end=0.25, score=-3.0),
        VowelSpan(word_index=1, vowel_ordinal=-1, char="р", start=0.25, end=0.35, score=-1.4),
    ]
    gop = measure_gop("ра р", np.zeros(16_000, dtype=np.float32), 16_000, _StubAligner(spans))
    assert gop == {"р": [-0.2, -1.4], "а": [-0.1]}


def test_measure_gop_returns_none_when_alignment_is_unavailable():
    class _Dead(CtcAligner):
        def align(self, text, audio, sample_rate, **kwargs):  # type: ignore[override]
            return None

    assert measure_gop("ра", np.zeros(16_000, dtype=np.float32), 16_000, _Dead()) is None


class _RivalAligner(CtcAligner):
    """Spans carrying rival scores, as align() fills them when asked for rivals."""

    def __init__(self, spans: list[VowelSpan]) -> None:
        super().__init__()
        self._spans = spans

    def align(self, text, audio, sample_rate, **kwargs):  # type: ignore[override]
        assert kwargs.get("rival_of"), "measure_margins must ask the aligner for rivals"
        return self._spans


def test_measure_margins_reports_letter_minus_rival_and_the_context_after_r():
    """и after р reports twice: as и, and as и/р -- the softened-р signature."""
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=-1, char="р", start=0.0, end=0.05, score=-0.0),
        VowelSpan(
            word_index=0,
            vowel_ordinal=0,
            char="и",
            start=0.05,
            end=0.15,
            score=-0.4,
            rival_score=-1.2,
        ),
        VowelSpan(
            word_index=1,
            vowel_ordinal=-1,
            char="г",
            start=0.3,
            end=0.4,
            score=-0.9,
            rival_score=-0.5,
        ),
        VowelSpan(
            word_index=1,
            vowel_ordinal=0,
            char="и",
            start=0.4,
            end=0.5,
            score=-0.2,
            rival_score=-2.0,
        ),
    ]
    margins = measure_margins(
        "ри ги", np.zeros(16_000, dtype=np.float32), 16_000, _RivalAligner(spans)
    )
    assert margins is not None
    assert margins["и"] == pytest.approx([0.8, 1.8])
    assert margins["и/р"] == pytest.approx([0.8])
    assert margins["г"] == pytest.approx([-0.4])


def test_measure_margins_skips_spans_without_a_rival_score():
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=-1, char="г", start=0.0, end=0.1, score=-0.3),
    ]
    margins = measure_margins("г", np.zeros(16_000, dtype=np.float32), 16_000, _RivalAligner(spans))
    assert margins == {}


def test_palatalization_index_needs_a_vowel_long_enough_to_have_an_onset_and_a_middle():
    short = VowelSpan(word_index=0, vowel_ordinal=0, char="а", start=0.0, end=0.03, filled_end=0.05)
    consonant = VowelSpan(word_index=0, vowel_ordinal=-1, char="р", start=0.0, end=0.0)
    assert (
        palatalization_index(np.zeros(16_000, dtype=np.float32), 16_000, consonant, short) is None
    )


def test_measure_palatalization_pairs_each_softenable_consonant_with_its_back_vowel(monkeypatch):
    """р+а inside one word reports under р and р/а; a vowel across a word gap does not."""
    import fish_studio.stress_align as module

    monkeypatch.setattr(
        module, "palatalization_index", lambda samples, rate, consonant, vowel: 150.0
    )
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=-1, char="р", start=0.0, end=0.05),
        VowelSpan(word_index=0, vowel_ordinal=0, char="а", start=0.05, end=0.1, filled_end=0.25),
        VowelSpan(word_index=-1, vowel_ordinal=-1, char=" ", start=0.25, end=0.27),
        VowelSpan(word_index=1, vowel_ordinal=-1, char="л", start=0.27, end=0.3),
        VowelSpan(word_index=1, vowel_ordinal=0, char="і", start=0.3, end=0.35, filled_end=0.5),
        VowelSpan(word_index=1, vowel_ordinal=-1, char="т", start=0.5, end=0.55),
    ]
    spans.append(
        VowelSpan(word_index=2, vowel_ordinal=0, char="о", start=0.6, end=0.7, filled_end=0.9)
    )
    out = measure_palatalization(
        "ра лі т о", np.zeros(16_000, dtype=np.float32), 16_000, _StubAligner(spans)
    )
    assert out == {"р": [150.0], "р/а": [150.0]}


def _spike_train(rate: int, seconds: float, closures: int, level: float) -> np.ndarray:
    """A voiced segment with ``closures`` short dips, like a trilled р."""
    n = int(rate * seconds)
    out = np.full(n, level, dtype=np.float32)
    if closures:
        for k in range(closures):
            centre = int(n * (k + 1) / (closures + 1))
            out[centre - rate // 500 : centre + rate // 500] = level * 0.2
    return out


def test_trill_features_count_closures_and_level_against_the_vowel():
    """Three dips and a р louder than its vowel: what the accepted р looked like."""
    rate = 16_000
    r = _spike_train(rate, 0.08, closures=3, level=0.5)
    v = np.full(int(rate * 0.2), 0.3, dtype=np.float32)
    samples = np.concatenate([r, v])
    consonant = VowelSpan(word_index=0, vowel_ordinal=-1, char="р", start=0.0, end=0.02)
    vowel = VowelSpan(word_index=0, vowel_ordinal=0, char="у", start=0.08, end=0.1, filled_end=0.28)
    features = trill_features(samples, rate, consonant, vowel)
    assert features is not None
    assert features["dur_ms"] == pytest.approx(80.0)
    assert features["dips"] == 3
    assert features["rv_ratio"] > 1.0


def test_trill_features_see_a_weak_flat_r_as_no_closures():
    rate = 16_000
    r = _spike_train(rate, 0.03, closures=0, level=0.1)
    v = np.full(int(rate * 0.2), 0.3, dtype=np.float32)
    samples = np.concatenate([r, v])
    consonant = VowelSpan(word_index=0, vowel_ordinal=-1, char="р", start=0.0, end=0.01)
    vowel = VowelSpan(
        word_index=0, vowel_ordinal=0, char="о", start=0.03, end=0.05, filled_end=0.23
    )
    features = trill_features(samples, rate, consonant, vowel)
    assert features is not None
    assert features["dips"] == 0
    assert features["rv_ratio"] < 0.5


def test_measure_trill_keys_by_r_and_by_following_vowel(monkeypatch):
    import fish_studio.stress_align as module

    monkeypatch.setattr(
        module,
        "trill_features",
        lambda samples, rate, c, v: {"dur_ms": 60.0, "rv_ratio": 1.5, "dips": 2.0},
    )
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=-1, char="р", start=0.0, end=0.02),
        VowelSpan(word_index=0, vowel_ordinal=0, char="у", start=0.06, end=0.08, filled_end=0.2),
        VowelSpan(word_index=1, vowel_ordinal=-1, char="р", start=0.3, end=0.32),
        VowelSpan(word_index=1, vowel_ordinal=0, char="і", start=0.36, end=0.38, filled_end=0.5),
    ]
    out = measure_trill("ру рі", np.zeros(16_000, dtype=np.float32), 16_000, _StubAligner(spans))
    assert out is not None
    assert set(out) == {"р", "р/у"}
    assert out["р/у"][0]["dips"] == 2.0


def _tone(rate: int, seconds: float, f0_start: float, f0_end: float) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    f0 = np.linspace(f0_start, f0_end, t.size)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    return (0.5 * np.sin(phase)).astype(np.float32)


def test_pitch_spread_is_near_zero_for_a_steady_tone():
    spread = measure_pitch_spread(_tone(16_000, 1.0, 150.0, 150.0), 16_000)
    assert spread is not None
    assert spread["f0_range_st"] < 1.0


def test_pitch_spread_reads_an_octave_sweep_as_about_twelve_semitones():
    spread = measure_pitch_spread(_tone(16_000, 1.5, 110.0, 220.0), 16_000)
    assert spread is not None
    assert 9.0 < spread["f0_range_st"] < 13.0


def test_pitch_spread_is_none_for_silence():
    assert measure_pitch_spread(np.zeros(16_000, dtype=np.float32), 16_000) is None


def test_measure_vowel_formants_takes_each_aligned_и_and_і_and_skips_the_rest(monkeypatch):
    import fish_studio.stress_align as module

    monkeypatch.setattr(module, "_formants_at", lambda samples, rate, at: (500.0, 1800.0))
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=-1, char="м", start=0.0, end=0.02),
        VowelSpan(word_index=0, vowel_ordinal=0, char="и", start=0.02, end=0.05, filled_end=0.12),
        VowelSpan(word_index=-1, vowel_ordinal=-1, char=" ", start=0.12, end=0.13),
        VowelSpan(word_index=1, vowel_ordinal=0, char="і", start=0.13, end=0.15, filled_end=0.25),
        VowelSpan(word_index=1, vowel_ordinal=1, char="а", start=0.25, end=0.27, filled_end=0.4),
        VowelSpan(word_index=2, vowel_ordinal=0, char="и", start=0.4, end=0.41, filled_end=0.42),
    ]
    out = measure_vowel_formants(
        "ми іа и", np.zeros(16_000, dtype=np.float32), 16_000, _StubAligner(spans)
    )
    assert out == {"и": [(500.0, 1800.0)], "і": [(500.0, 1800.0)]}


def test_needs_stress_fill_only_for_bare_words_with_two_or_more_vowels():
    from fish_studio.stress_align import needs_stress_fill

    assert needs_stress_fill("dobryi den, hora vysoka") is False  # latin: no vowels counted
    assert needs_stress_fill("\u0433\u043e\u0440\u0430\u0301") is False  # marked
    assert needs_stress_fill("\u0433\u043e\u0440\u0430") is True  # bare, two vowels
    assert needs_stress_fill("\u0442\u0430\u043a \u0456") is False  # one-vowel words only


def test_measure_rhotic_reports_f3_of_each_r_against_its_vowel(monkeypatch):
    import fish_studio.stress_align as module

    def fake_f3(samples, rate, t0, t1):
        # р spans start at 0.0 / 0.30; vowels follow. Make the first р an approximant.
        return 1600.0 if (t0 == 0.0) else (2400.0 if t0 == 0.30 else 2500.0)

    monkeypatch.setattr(module, "_f3_between", fake_f3)
    spans = [
        VowelSpan(word_index=0, vowel_ordinal=-1, char="р", start=0.0, end=0.02),
        VowelSpan(word_index=0, vowel_ordinal=0, char="а", start=0.05, end=0.07, filled_end=0.2),
        VowelSpan(word_index=-1, vowel_ordinal=-1, char=" ", start=0.2, end=0.21),
        VowelSpan(word_index=1, vowel_ordinal=-1, char="р", start=0.30, end=0.32),
        VowelSpan(word_index=1, vowel_ordinal=0, char="у", start=0.35, end=0.37, filled_end=0.5),
        VowelSpan(word_index=1, vowel_ordinal=-1, char="р", start=0.5, end=0.52),  # word-final р
    ]
    out = module.measure_rhotic(
        "ра ру", np.zeros(16_000, dtype=np.float32), 16_000, _StubAligner(spans)
    )
    assert out == [1600.0 / 2500.0, 2400.0 / 2500.0]
