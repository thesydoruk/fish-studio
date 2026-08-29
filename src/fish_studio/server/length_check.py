"""Screen a served model for early termination and for level collapse.

Two failures make a checkpoint unusable and neither is obvious on a short sample.
The model can stop generating before the text is spoken, so anything long is cut
off mid-sentence. It can also return speech tens of decibels below the level of
its own reference, which sounds like the voice fading away and cannot be rescued
by turning the volume up: at -68 dBFS there is almost nothing left of a 16-bit
signal to amplify. A run measuring duration alone reports a healthy number while
the audio is inaudible, so both are measured here.

Post-processing is disabled for the probe: ``match_timing`` would fit output
into the reference clip's slot and hide duration failures under test.
"""

from __future__ import annotations

import argparse
import array
import io
import math
import statistics
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Unhurried Ukrainian narration. Only used to turn a character count into an
# expected duration, so it needs to be representative rather than exact.
CHARS_PER_SEC = 15.0

# Increasing lengths: truncation shows up as a ratio that falls as text grows.
SAMPLE_TEXTS = (
    "Сьогодні гарна погода.",
    "Сьогодні гарна погода, і ми нарешті можемо вийти на прогулянку разом.",
    "Сьогодні гарна погода, і ми нарешті можемо вийти на прогулянку разом. "
    "Минулого тижня дощ ішов майже щодня, тому всі втомилися сидіти вдома.",
    "Сьогодні гарна погода, і ми нарешті можемо вийти на прогулянку разом. "
    "Минулого тижня дощ ішов майже щодня, тому всі втомилися сидіти вдома. "
    "Сподіваюся, що наступні кілька днів будуть такими ж теплими і сонячними.",
)

OK = "OK"
WEAK = "WEAK"
BROKEN = "BROKEN"


@dataclass
class LengthReport:
    label: str
    ratios: list[float]
    per_text: list[tuple[int, list[float]]]
    # How far each sample sits below the reference clip, in dB. Positive = quieter.
    level_drops: list[float] = field(default_factory=list)

    @property
    def median(self) -> float:
        return statistics.median(self.ratios)

    @property
    def worst(self) -> float:
        return min(self.ratios)

    @property
    def worst_level_drop(self) -> float:
        return max(self.level_drops) if self.level_drops else 0.0


def verdict(worst: float, *, fail_under: float, level_drop: float = 0.0,
            max_level_drop: float = 12.0) -> str:
    """Grade a run by its worst sample, not its average.

    A model that truncates does so intermittently, so the mean stays
    respectable long after the output has become unusable. A level collapse is
    graded the same way and on its own: correct duration at -60 dBFS is still a
    dead checkpoint.
    """
    if level_drop > max_level_drop + 8:
        return BROKEN
    if worst >= fail_under:
        return WEAK if level_drop > max_level_drop else OK
    if worst >= fail_under - 0.15:
        return WEAK
    return BROKEN


def format_report(report: LengthReport, *, fail_under: float,
                  max_level_drop: float = 12.0) -> str:
    detail = "  ".join(
        f"{chars}ch:" + "/".join(f"{ratio:.0%}" for ratio in ratios)
        for chars, ratios in report.per_text
    )
    grade = verdict(
        report.worst,
        fail_under=fail_under,
        level_drop=report.worst_level_drop,
        max_level_drop=max_level_drop,
    )
    return (
        f"{report.label:<12} median={report.median:>5.0%} worst={report.worst:>5.0%} "
        f"quiet={report.worst_level_drop:>5.1f}dB {grade:<7} {detail}"
    )


def wav_seconds(payload: bytes) -> float:
    with wave.open(io.BytesIO(payload), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def wav_level_db(payload: bytes | Path) -> float:
    """Full-clip RMS in dBFS. Crude next to LUFS, but a 20 dB hole is a 20 dB hole."""
    handle = (
        wave.open(io.BytesIO(payload), "rb")
        if isinstance(payload, bytes)
        else wave.open(str(payload), "rb")
    )
    with handle as source:
        if source.getsampwidth() != 2:
            raise ValueError(
                f"expected 16-bit PCM, got {8 * source.getsampwidth()}-bit — "
                "the RMS math below would read garbage"
            )
        frames = source.readframes(source.getnframes())
    if not frames:
        return -120.0
    samples = array.array("h")
    samples.frombytes(frames)
    if not samples:
        return -120.0
    mean_square = sum(float(s) * s for s in samples) / (len(samples) * 32768.0 * 32768.0)
    return 10 * math.log10(max(mean_square, 1e-12))


def synthesize(
    client: httpx.Client,
    *,
    reference: Path,
    reference_text: str,
    text: str,
    language: str,
) -> tuple[float, float]:
    """Return the duration in seconds and the RMS level in dBFS."""
    files = {"speaker_wav": (reference.name, reference.read_bytes(), "audio/wav")}
    data = {
        "text": text,
        "language": language,
        "speaker_text": reference_text,
        "match_timing": "false",
    }
    response = client.post("/v1/synthesize", data=data, files=files)
    response.raise_for_status()
    return wav_seconds(response.content), wav_level_db(response.content)


def check_length(
    *,
    base_url: str,
    reference: Path,
    reference_text: str,
    label: str,
    runs: int,
    language: str,
    timeout: float,
) -> LengthReport:
    ratios: list[float] = []
    per_text: list[tuple[int, list[float]]] = []
    drops: list[float] = []
    reference_level = wav_level_db(reference)
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        for text in SAMPLE_TEXTS:
            expected = len(text) / CHARS_PER_SEC
            measured = [
                synthesize(
                    client,
                    reference=reference,
                    reference_text=reference_text,
                    text=text,
                    language=language,
                )
                for _ in range(runs)
            ]
            got = [seconds / expected for seconds, _ in measured]
            drops.extend(reference_level - level for _, level in measured)
            ratios.extend(got)
            per_text.append((len(text), got))
    return LengthReport(label=label, ratios=ratios, per_text=per_text, level_drops=drops)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=Path, required=True, help="reference clip WAV")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ref-text", help="transcript of the reference clip")
    group.add_argument("--ref-text-file", type=Path, help="file holding that transcript")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--label", default="model", help="name shown in the report line")
    parser.add_argument("--runs", type=int, default=2, help="repeats per sample text")
    parser.add_argument("--language", default="uk")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.70,
        help="exit non-zero when the worst sample falls below this share",
    )
    parser.add_argument(
        "--max-level-drop",
        type=float,
        default=12.0,
        help="exit non-zero when output sits more than this many dB under the reference",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference_text = (
        args.ref_text
        if args.ref_text is not None
        else args.ref_text_file.read_text(encoding="utf-8")
    ).strip()
    if not reference_text:
        raise SystemExit("reference transcript is empty")
    if not args.ref.is_file():
        raise SystemExit(f"reference clip not found: {args.ref}")

    report = check_length(
        base_url=args.base_url,
        reference=args.ref,
        reference_text=reference_text,
        label=args.label,
        runs=args.runs,
        language=args.language,
        timeout=args.timeout,
    )
    print(
        format_report(
            report, fail_under=args.fail_under, max_level_drop=args.max_level_drop
        ),
        flush=True,
    )
    healthy = (
        report.worst >= args.fail_under
        and report.worst_level_drop <= args.max_level_drop
    )
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
