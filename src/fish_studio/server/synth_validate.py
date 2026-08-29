"""Reject obviously broken raw synthesis: silence or a mid-line cutoff.

Judged on the model WAV *before* pause/tempo fit, against the chunk text.
``match_timing`` must not run first — stretch would hide a short take.

The implied rate is syllables(text) / active_speech(wav). A cutoff leaves the
full text in the numerator and only the spoken prefix in the denominator, so
the rate jumps well above any real articulation. Valid fast speech stays under
the ceiling; thin one-word lines are not scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fish_studio.timing import count_syllables, measure_active_speech_sec, strip_nonspeech

# Same thin-text floor as the timing fit — one-word lines are too noisy to score.
_MIN_SYL = 4
_MIN_CHARS = 12
# Below this, a scored line is empty / a click, not speech.
_SILENCE_ACTIVE_SEC = 0.25
# 6 syl/s is the timing band ceiling; 1.3× stretch is ~7.8. 10 is past any
# plausible raw take and still below a half-spoken line at a normal rate.
_CUTOFF_SYL_PER_SEC = 10.0
_PEAK_EPS = 1.0 / 32_768

MAX_SYNTH_ATTEMPTS = 3
# ASCII header on the WAV response so a client can log a kept-bad take.
SYNTH_WARNING_HEADER = "X-Synth-Warning"


@dataclass(frozen=True)
class SynthCheck:
    """One raw-take verdict."""

    ok: bool
    reason: str
    syllables: int
    active_speech_sec: float
    implied_syl_per_sec: float

    def metrics(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "syllables": self.syllables,
            "active_speech_sec": round(self.active_speech_sec, 3),
            "implied_syl_per_sec": round(self.implied_syl_per_sec, 3),
        }


def judge_raw_synth(audio: np.ndarray, sample_rate: int, text: str) -> SynthCheck:
    """Return whether ``audio`` can contain ``text`` as spoken Ukrainian."""
    line = strip_nonspeech(text)
    syllables = count_syllables(line)
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0 or sample_rate <= 0:
        return SynthCheck(False, "silence", syllables, 0.0, 0.0)

    peak = float(np.max(np.abs(samples)))
    active = measure_active_speech_sec(samples, sample_rate)
    implied = (syllables / active) if active > 0 else 0.0

    if syllables < _MIN_SYL or len(line) < _MIN_CHARS:
        return SynthCheck(True, "", syllables, active, implied)

    if peak < _PEAK_EPS or active < _SILENCE_ACTIVE_SEC:
        return SynthCheck(False, "silence", syllables, active, implied)
    if implied >= _CUTOFF_SYL_PER_SEC:
        return SynthCheck(False, "cutoff", syllables, active, implied)
    return SynthCheck(True, "", syllables, active, implied)


def pick_best_attempt(
    attempts: list[tuple[np.ndarray, int, SynthCheck]],
) -> tuple[np.ndarray, int, SynthCheck]:
    """Prefer a passing take; otherwise the most complete failing one."""
    if not attempts:
        raise ValueError("no synthesis attempts")
    passing = [item for item in attempts if item[2].ok]
    if passing:
        return passing[0]

    def _key(item: tuple[np.ndarray, int, SynthCheck]) -> tuple[int, float, float]:
        check = item[2]
        not_silence = 0 if check.reason == "silence" else 1
        return (not_silence, check.active_speech_sec, -check.implied_syl_per_sec)

    return max(attempts, key=_key)


def quality_warning(reports: list[dict]) -> str:
    """Human-readable warning when a returned take is still silence or a cutoff."""
    parts: list[str] = []
    multi = len(reports) > 1
    for index, report in enumerate(reports, start=1):
        if report.get("ok", True):
            continue
        reason = str(report.get("reason") or "invalid")
        attempts = int(report.get("attempts") or MAX_SYNTH_ATTEMPTS)
        implied = float(report.get("implied_syl_per_sec") or 0.0)
        active = float(report.get("active_speech_sec") or 0.0)
        if reason == "silence":
            detail = f"silence after {attempts} attempts ({active:.2f}s active speech)"
        elif reason == "cutoff":
            detail = (
                f"cutoff after {attempts} attempts "
                f"({implied:.1f} syl/s, {active:.2f}s active speech)"
            )
        else:
            detail = f"{reason} after {attempts} attempts"
        parts.append(f"chunk {index}: {detail}" if multi else detail)
    return "; ".join(parts)
