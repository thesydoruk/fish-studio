"""Grade a served checkpoint on the two axes the merge scale trades between.

A merge buys Ukrainian pronunciation with speaker identity, and
until now the dose was picked on-ear, which cannot say whether a change moved
the curve or moved along it. This puts a number on each side of that trade, from
one run over the frozen probe set:

``stress``   share of dictionary-marked words the take stresses correctly
``follow``   share of the marks the server put in the text that the take honours
``voice``    ECAPA cosine to the clone prompt -- the same encoder as the serve gate

``stress`` is the end-to-end answer and ``follow`` isolates the model's part of
it: a lexicon miss lowers ``stress`` while leaving ``follow`` untouched.

Every probe line also carries the human recording of itself, and scoring that
with the same estimator gives the ceiling for these numbers on these words. A
checkpoint is read against that ceiling, never against 100%.

Timing fit is off for the probe: PSOLA would move the very vowel durations the
stress estimator reads.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from fish_studio.config import StressConfig
from fish_studio.project_context import try_load_project, workspace_or_default
from fish_studio.server.voiceprint import VoiceEncoder
from fish_studio.stress_align import (
    RHOTIC_APPROXIMANT_RATIO,
    CtcAligner,
    marked_ordinals,
    measure_gop,
    measure_margins,
    measure_palatalization,
    measure_pitch_spread,
    measure_rhotic,
    measure_stress,
    measure_trill,
    measure_vowel_formants,
)
from fish_studio.textnorm.prepare import prepare_synthesis_text

# Margin keys worth a column: the pairs an accent lands on, and и after р.
MARGIN_KEYS = ("г", "и", "и/р", "е", "ш")
# Palatalization keys: the consonant an accent softens first, by following vowel.
PALATAL_KEYS = ("р", "р/а", "р/о", "р/у", "р/и", "л", "н", "т", "д", "с")


@dataclass
class Probe:
    id: str
    speaker: str
    text: str
    truth: str
    human_wav: Path
    ref_wav: Path
    ref_text: str


@dataclass
class Axis:
    """One counted proportion."""

    hits: int = 0
    total: int = 0

    def add(self, *, hit: bool) -> None:
        self.total += 1
        if hit:
            self.hits += 1

    @property
    def value(self) -> float:
        return self.hits / self.total if self.total else 0.0


@dataclass
class Voice:
    """A clone prompt from outside the training set.

    The probe set's own references are all dataset speakers, which is the case
    the model finds easiest. Production logs show the failures land elsewhere:
    a Russian-language prompt scored -0.03 to 0.16 against a 0.30 gate. Those
    clips are kept under ``{data_root}/eval/voices`` and replayed here, because
    a dose that costs nothing on a familiar voice can still cost everything on
    an unfamiliar one.
    """

    id: str
    wav: Path
    text: str


@dataclass
class VoiceReport:
    voice_id: str
    similarity: list[float] = field(default_factory=list)
    stress: Axis | None = None

    @property
    def mean(self) -> float:
        return float(np.mean(self.similarity)) if self.similarity else 0.0


# Letters worth reporting on their own: the ones an accent lands on first.
GOP_LETTERS = ("р", "л", "г", "в", "ч", "щ", "и", "і")


def gop_mean(scores: dict[str, list[float]], letters: tuple[str, ...] | None = None) -> float:
    picked = [
        value
        for char, values in scores.items()
        if letters is None or char in letters
        for value in values
    ]
    return float(np.mean(picked)) if picked else float("nan")


TRILL_KEYS = ("р", "р/а", "р/о", "р/у", "р/и")


def trill_medians(feats: list[dict[str, float]]) -> tuple[float, float, float] | None:
    """(duration ms, level against vowel, closures) -- medians over the р tokens."""
    if not feats:
        return None
    return (
        float(np.median([f["dur_ms"] for f in feats])),
        float(np.median([f["rv_ratio"] for f in feats])),
        float(np.median([f["dips"] for f in feats])),
    )


def trill_weak_share(feats: list[dict[str, float]]) -> float:
    """Share of р tokens that are short, quiet and closure-less at once.

    Medians over a probe set put the candidate level with native speech while
    a listener still heard weak р in one phrase, so the deficit, if it is
    there, lives in a tail the median hides. This is the tail: no closure,
    40 ms or less, and quieter than the vowel that follows.
    """
    if not feats:
        return float("nan")
    weak = sum(1 for f in feats if f["dips"] == 0 and f["dur_ms"] <= 40.0 and f["rv_ratio"] < 0.8)
    return weak / len(feats)


def rhotic_summary(ratios: list[float]) -> dict[str, float] | None:
    """Share of р tokens that are English-style approximants, plus the median F3 ratio."""
    if not ratios:
        return None
    values = np.asarray(ratios, dtype=np.float64)
    return {
        "approximant": float((values < RHOTIC_APPROXIMANT_RATIO).mean()),
        "ratio": float(np.median(values)),
        "n": int(values.size),
    }


def vowel_summary(
    vowels: dict[str, list[tuple[float, float]]],
) -> dict[str, float] | None:
    """Median F1/F2 of и and і, and the distance between them.

    ``dF2`` is F2(і) - F2(и) and ``dF1`` is F1(и) - F1(і): both positive in
    native Ukrainian, both shrink toward zero when и is read as і.
    """
    if not vowels.get("и") or not vowels.get("і"):
        return None
    f1_y = float(np.median([f1 for f1, _ in vowels["и"]]))
    f2_y = float(np.median([f2 for _, f2 in vowels["и"]]))
    f1_i = float(np.median([f1 for f1, _ in vowels["і"]]))
    f2_i = float(np.median([f2 for _, f2 in vowels["і"]]))
    return {
        "и_F1": f1_y,
        "и_F2": f2_y,
        "і_F1": f1_i,
        "і_F2": f2_i,
        "dF2": f2_i - f2_y,
        "dF1": f1_y - f1_i,
        "n_и": len(vowels["и"]),
        "n_і": len(vowels["і"]),
    }


@dataclass
class Report:
    label: str
    stress: Axis = field(default_factory=Axis)
    follow: Axis = field(default_factory=Axis)
    human_stress: Axis = field(default_factory=Axis)
    voice: list[float] = field(default_factory=list)
    human_voice: list[float] = field(default_factory=list)
    # Goodness of pronunciation: CTC log-prob per aligned letter. Stress is
    # blind to a softened "р"; this is where it shows.
    gop: dict[str, list[float]] = field(default_factory=dict)
    human_gop: dict[str, list[float]] = field(default_factory=dict)
    # log P(letter) - log P(rival): г/ґ, и/і, е/є, ш/щ, plus и after a
    # consonant. Plain confidence cannot see a softened р; this can.
    margins: dict[str, list[float]] = field(default_factory=dict)
    human_margins: dict[str, list[float]] = field(default_factory=dict)
    # F2 onset minus F2 middle of the vowel after a softenable consonant, Hz.
    # The CTC-based measures saturate on a softened р; the formant does not.
    palatal: dict[str, list[float]] = field(default_factory=dict)
    human_palatal: dict[str, list[float]] = field(default_factory=dict)
    # Trill features per р before a back vowel: duration, level against the
    # vowel, closures. The only measure so far that ranked the listener's one
    # accepted р above the four rejected ones, in every take.
    trill: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    human_trill: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    # Pitch spread per take, semitones: the axis the listener called
    # "machine-like". The human recording of the same line is the reference.
    pitch: list[dict[str, float]] = field(default_factory=list)
    human_pitch: list[dict[str, float]] = field(default_factory=list)
    # (F1, F2) at the middle of every и and і: the vowel-quality contrast a
    # model that reads и as і collapses. Human lines give the native distance.
    vowels: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    human_vowels: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    # F3 of each р against its vowel: below 0.8 the р is an English approximant,
    # the one thing a listener hears as "soft р" that no other measure ranked.
    rhotic: list[float] = field(default_factory=list)
    human_rhotic: list[float] = field(default_factory=list)
    lines: int = 0
    align_failed: int = 0
    synth_failed: int = 0

    @property
    def mean_voice(self) -> float:
        return float(np.mean(self.voice)) if self.voice else 0.0

    @property
    def mean_human_voice(self) -> float:
        return float(np.mean(self.human_voice)) if self.human_voice else 0.0

    def format(self) -> str:
        ceiling = f" (human {self.human_stress.value:.1%})" if self.human_stress.total else ""
        voice_ceiling = f" (human {self.mean_human_voice:.3f})" if self.human_voice else ""
        gop_line = ""
        if self.gop:
            overall = gop_mean(self.gop)
            human = f" (human {gop_mean(self.human_gop):.2f})" if self.human_gop else ""
            letters = " ".join(
                f"{char}:{gop_mean(self.gop, (char,)):.2f}"
                for char in GOP_LETTERS
                if char in self.gop
            )
            gop_line = f"\n{'':<14} gop={overall:.2f}{human}  {letters}"
        if self.margins:
            cells = []
            for key in MARGIN_KEYS:
                if key not in self.margins:
                    continue
                cell = f"{key}:{gop_mean(self.margins, (key,)):+.2f}"
                if key in self.human_margins:
                    cell += f"(h{gop_mean(self.human_margins, (key,)):+.2f})"
                cells.append(cell)
            gop_line += f"\n{'':<14} margin  {'  '.join(cells)}"
        if self.palatal:
            cells = []
            for key in PALATAL_KEYS:
                if key not in self.palatal:
                    continue
                cell = f"{key}:{gop_mean(self.palatal, (key,)):+.0f}"
                if key in self.human_palatal:
                    cell += f"(h{gop_mean(self.human_palatal, (key,)):+.0f})"
                cells.append(cell)
            gop_line += f"\n{'':<14} palatal {'  '.join(cells)}  [F2 onset-mid, Hz]"
        if self.trill:
            cells = []
            for key in TRILL_KEYS:
                mine = trill_medians(self.trill.get(key, []))
                if mine is None:
                    continue
                weak = trill_weak_share(self.trill.get(key, []))
                cell = f"{key}:{mine[0]:.0f}ms/{mine[1]:.2f}/{mine[2]:.0f}/w{weak:.0%}"
                theirs = trill_medians(self.human_trill.get(key, []))
                if theirs is not None:
                    human_weak = trill_weak_share(self.human_trill.get(key, []))
                    cell += f"(h{theirs[0]:.0f}/{theirs[1]:.2f}/{theirs[2]:.0f}/w{human_weak:.0%})"
                cells.append(cell)
            gop_line += (
                f"\n{'':<14} trill   {'  '.join(cells)}  [dur/r-v level/closures/weak share]"
            )
        mine_rh = rhotic_summary(self.rhotic)
        if mine_rh is not None:
            theirs_rh = rhotic_summary(self.human_rhotic)
            human = (
                ""
                if theirs_rh is None
                else f"  (h {100 * theirs_rh['approximant']:.0f}%/{theirs_rh['ratio']:.2f})"
            )
            gop_line += (
                f"\n{'':<14} rhotic  approximant={100 * mine_rh['approximant']:.0f}% "
                f"ratio={mine_rh['ratio']:.2f}{human}  [р with F3/F3(vowel)<0.8; n={mine_rh['n']}]"
            )
        if self.pitch:
            std = float(np.median([p["f0_std_st"] for p in self.pitch]))
            rng = float(np.median([p["f0_range_st"] for p in self.pitch]))
            human = ""
            if self.human_pitch:
                h_std = float(np.median([p["f0_std_st"] for p in self.human_pitch]))
                h_rng = float(np.median([p["f0_range_st"] for p in self.human_pitch]))
                human = f" (h{h_std:.1f}/{h_rng:.1f})"
            gop_line += f"\n{'':<14} pitch   σ={std:.1f}st range={rng:.1f}st{human}"
        mine = vowel_summary(self.vowels)
        if mine is not None:
            line = (
                f"и:{mine['и_F1']:.0f}/{mine['и_F2']:.0f}  і:{mine['і_F1']:.0f}/{mine['і_F2']:.0f}  "
                f"dF2={mine['dF2']:+.0f} dF1={mine['dF1']:+.0f}"
            )
            theirs = vowel_summary(self.human_vowels)
            if theirs is not None:
                line += f"  (h dF2={theirs['dF2']:+.0f} dF1={theirs['dF1']:+.0f})"
            gop_line += f"\n{'':<14} vowel   {line}  [F1/F2 Hz; и-і distance]"
        return (
            f"{self.label:<14} "
            f"stress={self.stress.value:>6.1%}{ceiling} n={self.stress.total:<4} "
            f"follow={self.follow.value:>6.1%} n={self.follow.total:<4} "
            f"voice={self.mean_voice:>5.3f}{voice_ceiling} "
            f"lines={self.lines} align_fail={self.align_failed} synth_fail={self.synth_failed}"
            f"{gop_line}"
        )


def load_probes(path: Path, data_root: Path, limit: int = 0) -> list[Probe]:
    if not path.is_file():
        raise SystemExit(f"probe set not found: {path}")
    probes: list[Probe] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            probes.append(
                Probe(
                    id=row["id"],
                    speaker=row["speaker"],
                    text=row["text"],
                    truth=row["truth"],
                    human_wav=data_root / row["human_wav"],
                    ref_wav=data_root / row["ref_wav"],
                    ref_text=row["ref_text"],
                )
            )
            if limit and len(probes) >= limit:
                break
    if not probes:
        raise SystemExit(f"probe set is empty: {path}")
    return probes


def read_audio(source: Path | io.BytesIO) -> tuple[np.ndarray, int]:
    """Mono float32 from a path or an in-memory WAV."""
    handle = source if isinstance(source, io.BytesIO) else str(source)
    samples, rate = sf.read(handle, dtype="float32", always_2d=False)
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    return np.asarray(samples, dtype=np.float32), int(rate)


def synthesize(
    client: httpx.Client,
    *,
    text: str,
    ref_wav: Path,
    ref_text: str,
    language: str,
) -> bytes:
    files = {"speaker_wav": (ref_wav.name, ref_wav.read_bytes(), "audio/wav")}
    data = {
        "text": text,
        "language": language,
        "speaker_text": ref_text,
        "match_timing": "false",
    }
    response = client.post("/v1/synthesize", data=data, files=files)
    response.raise_for_status()
    return response.content


def load_voices(path: Path, data_root: Path) -> list[Voice]:
    """Out-of-domain clone prompts kept from production requests."""
    if not path.is_file():
        raise SystemExit(f"voice index not found: {path}")
    voices: list[Voice] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="	"):
            wav = data_root / row["wav"]
            if not wav.is_file():
                print(f"[warn] voice clip missing: {wav}", file=sys.stderr)
                continue
            voices.append(Voice(id=row["id"], wav=wav, text=row.get("text", "")))
    if not voices:
        raise SystemExit(f"voice index has no usable clips: {path}")
    return voices


def score_audio(
    samples: np.ndarray,
    rate: int,
    *,
    expected: dict[int, int],
    marked: dict[int, int],
    aligner: CtcAligner,
    text: str,
    stress_axis: Axis,
    follow_axis: Axis | None,
    gop_sink: dict[str, list[float]] | None = None,
    margin_sink: dict[str, list[float]] | None = None,
    palatal_sink: dict[str, list[float]] | None = None,
    trill_sink: dict[str, list[dict[str, float]]] | None = None,
    vowel_sink: dict[str, list[tuple[float, float]]] | None = None,
    rhotic_sink: list[float] | None = None,
) -> bool:
    """Score one take. False when alignment gave nothing to score."""
    heard = measure_stress(text, samples, rate, aligner)
    if heard is None:
        return False
    if gop_sink is not None:
        gop = measure_gop(text, samples, rate, aligner)
        for char, values in (gop or {}).items():
            gop_sink.setdefault(char, []).extend(values)
    if margin_sink is not None:
        margins = measure_margins(text, samples, rate, aligner)
        for key, values in (margins or {}).items():
            margin_sink.setdefault(key, []).extend(values)
    if palatal_sink is not None:
        palatal = measure_palatalization(text, samples, rate, aligner)
        for key, values in (palatal or {}).items():
            palatal_sink.setdefault(key, []).extend(values)
    if trill_sink is not None:
        trill = measure_trill(text, samples, rate, aligner)
        for key, values in (trill or {}).items():
            trill_sink.setdefault(key, []).extend(values)
    if vowel_sink is not None:
        vowels = measure_vowel_formants(text, samples, rate, aligner)
        for key, values in (vowels or {}).items():
            vowel_sink.setdefault(key, []).extend(values)
    if rhotic_sink is not None:
        rhotic_sink.extend(measure_rhotic(text, samples, rate, aligner) or [])
    for word_index, want in expected.items():
        got = heard.get(word_index)
        if got is not None:
            stress_axis.add(hit=got == want)
    if follow_axis is not None:
        for word_index, want in marked.items():
            got = heard.get(word_index)
            if got is not None:
                follow_axis.add(hit=got == want)
    return True


def evaluate_voices(
    *,
    voices: list[Voice],
    probes: list[Probe],
    client: httpx.Client,
    encoder: VoiceEncoder,
    aligner: CtcAligner,
    language: str,
    lines: int,
    out_dir: Path | None,
) -> list[VoiceReport]:
    """Replay the same probe lines through each unfamiliar prompt."""
    reports: list[VoiceReport] = []
    if lines <= 0:
        return reports
    selected = probes[:lines]
    for voice in voices:
        report = VoiceReport(voice_id=voice.id, stress=Axis())
        ref_samples, ref_rate = read_audio(voice.wav)
        embedding = encoder.embed(ref_samples, ref_rate)
        for probe in selected:
            try:
                payload = synthesize(
                    client,
                    text=probe.text,
                    ref_wav=voice.wav,
                    ref_text=voice.text,
                    language=language,
                )
            except Exception as exc:  # noqa: BLE001 - one bad line must not end the run
                print(f"[warn] {voice.id}/{probe.id}: synthesis failed: {exc}", file=sys.stderr)
                continue
            if out_dir is not None:
                (out_dir / f"{voice.id}__{probe.id}.wav").write_bytes(payload)
            samples, rate = read_audio(io.BytesIO(payload))
            if embedding:
                similarity = encoder.similarity(samples, rate, embedding)
                if similarity is not None:
                    report.similarity.append(similarity)
            score_audio(
                samples,
                rate,
                expected=marked_ordinals(probe.truth),
                marked={},
                aligner=aligner,
                text=probe.text,
                stress_axis=report.stress,
                follow_axis=None,
            )
        reports.append(report)
    return reports


def evaluate(
    *,
    probes: list[Probe],
    base_url: str,
    label: str,
    language: str,
    stress: StressConfig,
    out_dir: Path | None,
    timeout: float,
    with_human: bool,
    voices: list[Voice] | None = None,
    voice_lines: int = 0,
    voices_only: bool = False,
) -> tuple[Report, list[VoiceReport]]:
    report = Report(label=label)
    aligner = CtcAligner()
    encoder = VoiceEncoder()
    encoder.warmup()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    reference_embeddings: dict[str, list[float] | None] = {}
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        for probe in [] if voices_only else probes:
            expected = marked_ordinals(probe.truth)
            if not expected:
                continue
            # What the server itself will mark, from the same code and lexicon.
            marked = marked_ordinals(
                prepare_synthesis_text(probe.text, language=language, stress=stress)
            )

            try:
                payload = synthesize(
                    client,
                    text=probe.text,
                    ref_wav=probe.ref_wav,
                    ref_text=probe.ref_text,
                    language=language,
                )
            except Exception as exc:  # noqa: BLE001 - one bad line must not end the run
                report.synth_failed += 1
                print(f"[warn] {probe.id}: synthesis failed: {exc}", file=sys.stderr)
                continue
            if out_dir is not None:
                (out_dir / f"{probe.id}.wav").write_bytes(payload)
            samples, rate = read_audio(io.BytesIO(payload))

            report.lines += 1
            spread = measure_pitch_spread(samples, rate)
            if spread is not None:
                report.pitch.append(spread)
            if not score_audio(
                samples,
                rate,
                expected=expected,
                marked=marked,
                aligner=aligner,
                text=probe.text,
                stress_axis=report.stress,
                follow_axis=report.follow,
                gop_sink=report.gop,
                margin_sink=report.margins,
                palatal_sink=report.palatal,
                trill_sink=report.trill,
                vowel_sink=report.vowels,
                rhotic_sink=report.rhotic,
            ):
                report.align_failed += 1

            if probe.speaker not in reference_embeddings:
                ref_samples, ref_rate = read_audio(probe.ref_wav)
                reference_embeddings[probe.speaker] = encoder.embed(ref_samples, ref_rate)
            reference = reference_embeddings[probe.speaker]
            if reference:
                similarity = encoder.similarity(samples, rate, reference)
                if similarity is not None:
                    report.voice.append(similarity)

            if with_human and probe.human_wav.is_file():
                human_samples, human_rate = read_audio(probe.human_wav)
                human_spread = measure_pitch_spread(human_samples, human_rate)
                if human_spread is not None:
                    report.human_pitch.append(human_spread)
                score_audio(
                    human_samples,
                    human_rate,
                    expected=expected,
                    marked=marked,
                    aligner=aligner,
                    text=probe.text,
                    stress_axis=report.human_stress,
                    follow_axis=None,
                    gop_sink=report.human_gop,
                    margin_sink=report.human_margins,
                    palatal_sink=report.human_palatal,
                    trill_sink=report.human_trill,
                    vowel_sink=report.human_vowels,
                    rhotic_sink=report.human_rhotic,
                )
                if reference:
                    similarity = encoder.similarity(human_samples, human_rate, reference)
                    if similarity is not None:
                        report.human_voice.append(similarity)

        voice_reports: list[VoiceReport] = []
        if voices:
            voice_reports = evaluate_voices(
                voices=voices,
                probes=probes,
                client=client,
                encoder=encoder,
                aligner=aligner,
                language=language,
                lines=voice_lines,
                out_dir=out_dir,
            )
    return report, voice_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=".env")
    parser.add_argument("--probe", type=Path, default=Path("configs/uk_probe.tsv"))
    parser.add_argument("--label", required=True, help="name shown in the report line")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--out-dir", type=Path, default=None, help="keep the synthesized WAVs")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="first N probe lines only")
    parser.add_argument("--language", default="uk")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--voices",
        type=Path,
        default=None,
        help="TSV of out-of-domain clone prompts (default: {data_root}/eval/voices/index.tsv)",
    )
    parser.add_argument(
        "--voice-lines",
        type=int,
        default=10,
        help="probe lines replayed through each out-of-domain prompt; 0 skips the cohort",
    )
    parser.add_argument(
        "--voices-only",
        action="store_true",
        help="skip the dataset cohort and measure only the out-of-domain prompts",
    )
    parser.add_argument(
        "--skip-human",
        action="store_true",
        help="skip the human-recording ceiling (same for every checkpoint)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project = try_load_project(args.config)
    stress = project.stress if project is not None else StressConfig()
    data_root = workspace_or_default(args.config).data_root

    probes = load_probes(args.probe, data_root, args.limit)
    voices_path = args.voices or (data_root / "eval" / "voices" / "index.tsv")
    voices = load_voices(voices_path, data_root) if voices_path.is_file() else []
    if args.voices and not voices:
        raise SystemExit(f"voice index not found: {args.voices}")

    report, voice_reports = evaluate(
        probes=probes,
        base_url=args.base_url,
        label=args.label,
        language=args.language,
        stress=stress,
        out_dir=args.out_dir,
        timeout=args.timeout,
        with_human=not args.skip_human,
        voices=voices,
        voice_lines=args.voice_lines,
        voices_only=args.voices_only,
    )
    if not args.voices_only:
        print(report.format(), flush=True)
    else:
        print(f"{report.label:<14} out-of-domain prompts only", flush=True)
    for voice_report in voice_reports:
        stress_axis = voice_report.stress
        share = f"{stress_axis.value:>6.1%}" if stress_axis else "     -"
        total = stress_axis.total if stress_axis else 0
        print(
            f"  voice {voice_report.voice_id:<10} clone={voice_report.mean:>6.3f} "
            f"n={len(voice_report.similarity):<3} stress={share} n={total}",
            flush=True,
        )

    if args.json_out is not None:
        payload = {
            "label": report.label,
            "lines": report.lines,
            "align_failed": report.align_failed,
            "synth_failed": report.synth_failed,
            "stress": {"value": round(report.stress.value, 4), "n": report.stress.total},
            "follow": {"value": round(report.follow.value, 4), "n": report.follow.total},
            "human_stress": {
                "value": round(report.human_stress.value, 4),
                "n": report.human_stress.total,
            },
            "voice": {"mean": round(report.mean_voice, 4), "n": len(report.voice)},
            "human_voice": {
                "mean": round(report.mean_human_voice, 4),
                "n": len(report.human_voice),
            },
            "gop": {
                "mean": round(gop_mean(report.gop), 4) if report.gop else None,
                "letters": {
                    char: round(gop_mean(report.gop, (char,)), 4)
                    for char in GOP_LETTERS
                    if char in report.gop
                },
                "n": sum(len(values) for values in report.gop.values()),
            },
            "human_gop": {
                "mean": round(gop_mean(report.human_gop), 4) if report.human_gop else None,
                "letters": {
                    char: round(gop_mean(report.human_gop, (char,)), 4)
                    for char in GOP_LETTERS
                    if char in report.human_gop
                },
            },
            "margins": {
                key: round(gop_mean(report.margins, (key,)), 4)
                for key in MARGIN_KEYS
                if key in report.margins
            },
            "human_margins": {
                key: round(gop_mean(report.human_margins, (key,)), 4)
                for key in MARGIN_KEYS
                if key in report.human_margins
            },
            "palatal": {
                key: round(gop_mean(report.palatal, (key,)), 1)
                for key in PALATAL_KEYS
                if key in report.palatal
            },
            "human_palatal": {
                key: round(gop_mean(report.human_palatal, (key,)), 1)
                for key in PALATAL_KEYS
                if key in report.human_palatal
            },
            "trill": {
                key: {
                    **dict(zip(("dur_ms", "rv_ratio", "dips"), medians, strict=True)),
                    "weak_share": round(trill_weak_share(report.trill.get(key, [])), 4),
                    "n": len(report.trill.get(key, [])),
                }
                for key in TRILL_KEYS
                if (medians := trill_medians(report.trill.get(key, []))) is not None
            },
            "human_trill": {
                key: {
                    **dict(zip(("dur_ms", "rv_ratio", "dips"), medians, strict=True)),
                    "weak_share": round(trill_weak_share(report.human_trill.get(key, [])), 4),
                    "n": len(report.human_trill.get(key, [])),
                }
                for key in TRILL_KEYS
                if (medians := trill_medians(report.human_trill.get(key, []))) is not None
            },
            # Raw per-token features, so a tail or a per-speaker cut can be
            # computed later without another synthesis pass.
            "trill_tokens": report.trill,
            "human_trill_tokens": report.human_trill,
            "vowels": vowel_summary(report.vowels),
            "human_vowels": vowel_summary(report.human_vowels),
            "rhotic": rhotic_summary(report.rhotic),
            "human_rhotic": rhotic_summary(report.human_rhotic),
            "pitch": {
                "f0_std_st": round(float(np.median([p["f0_std_st"] for p in report.pitch])), 2),
                "f0_range_st": round(float(np.median([p["f0_range_st"] for p in report.pitch])), 2),
                "n": len(report.pitch),
            }
            if report.pitch
            else None,
            "human_pitch": {
                "f0_std_st": round(
                    float(np.median([p["f0_std_st"] for p in report.human_pitch])), 2
                ),
                "f0_range_st": round(
                    float(np.median([p["f0_range_st"] for p in report.human_pitch])), 2
                ),
                "n": len(report.human_pitch),
            }
            if report.human_pitch
            else None,
            "out_of_domain_voices": [
                {
                    "id": voice_report.voice_id,
                    "clone": round(voice_report.mean, 4),
                    "n": len(voice_report.similarity),
                    "stress": round(voice_report.stress.value, 4) if voice_report.stress else None,
                    "stress_n": voice_report.stress.total if voice_report.stress else 0,
                }
                for voice_report in voice_reports
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
