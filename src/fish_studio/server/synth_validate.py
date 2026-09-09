"""Reject obviously broken raw synthesis: silence, cutoff, or a drifted voice.

Judged on the model WAV *before* pause/tempo fit, against the chunk text and
the clone-prompt embedding. ``match_timing`` must not run first — stretch
would hide a short take and would not fix a wrong speaker.

The implied rate is syllables(text) / active_speech(wav). A cutoff leaves the
full text in the numerator and only the spoken prefix in the denominator, so
the rate jumps well above any real articulation. Valid fast speech stays under
the ceiling; thin one-word lines skip the rate check but are still voice-scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fish_studio.server.voiceprint import VOICE_RETRY_BELOW, VOICE_WARN_BELOW
from fish_studio.timing import count_syllables, measure_active_speech_sec, strip_nonspeech

# Same thin-text floor as the timing fit — one-word lines are too noisy for rate.
_MIN_SYL = 4
_MIN_CHARS = 12
# Below this, a scored line is empty / a click, not speech.
_SILENCE_ACTIVE_SEC = 0.25
# 6 syl/s is the timing band ceiling; 1.3× stretch is ~7.8. 10 is past any
# plausible raw take and still below a half-spoken line at a normal rate.
_CUTOFF_SYL_PER_SEC = 10.0
_PEAK_EPS = 1.0 / 32_768

MAX_SYNTH_ATTEMPTS = 5
# ASCII headers on the WAV response so a client can log a kept-bad take.
SYNTH_WARNING_HEADER = "X-Synth-Warning"


@dataclass(frozen=True)
class SynthCheck:
    """One raw-take verdict."""

    ok: bool
    reason: str
    syllables: int
    active_speech_sec: float
    implied_syl_per_sec: float
    voice_similarity: float | None = None

    def metrics(self) -> dict:
        payload = {
            "ok": self.ok,
            "reason": self.reason,
            "syllables": self.syllables,
            "active_speech_sec": round(self.active_speech_sec, 3),
            "implied_syl_per_sec": round(self.implied_syl_per_sec, 3),
        }
        if self.voice_similarity is not None:
            payload["voice_similarity"] = round(self.voice_similarity, 3)
        return payload


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


def attach_voice(
    check: SynthCheck,
    similarity: float | None,
    *,
    retry_below: float = VOICE_RETRY_BELOW,
) -> SynthCheck:
    """Fold clone cosine into a quality verdict. Short lines are scored too.

    ``None`` means the clip could not be embedded (too short / encoder off) —
    quality is left unchanged. Below ``retry_below`` a passing take becomes a
    voice fail so the chunk loop retries.
    """
    tagged = SynthCheck(
        ok=check.ok,
        reason=check.reason,
        syllables=check.syllables,
        active_speech_sec=check.active_speech_sec,
        implied_syl_per_sec=check.implied_syl_per_sec,
        voice_similarity=similarity,
    )
    if not tagged.ok or similarity is None:
        return tagged
    if similarity < retry_below:
        return SynthCheck(
            False,
            "voice",
            tagged.syllables,
            tagged.active_speech_sec,
            tagged.implied_syl_per_sec,
            similarity,
        )
    return tagged


def pick_best_attempt(
    attempts: list[tuple[np.ndarray, int, SynthCheck]],
) -> tuple[np.ndarray, int, SynthCheck]:
    """Prefer a passing take; otherwise speech with the strongest clone match."""
    if not attempts:
        raise ValueError("no synthesis attempts")
    passing = [item for item in attempts if item[2].ok]
    if passing:
        return max(passing, key=_voice_key)

    return max(attempts, key=_fail_key)


def _voice_key(item: tuple[np.ndarray, int, SynthCheck]) -> tuple[float, float]:
    check = item[2]
    similarity = check.voice_similarity if check.voice_similarity is not None else -1.0
    return (similarity, check.active_speech_sec)


def _fail_key(item: tuple[np.ndarray, int, SynthCheck]) -> tuple[int, float, float, float]:
    check = item[2]
    if check.reason == "silence":
        tier = 0
    elif check.reason == "cutoff":
        tier = 1
    else:
        # voice miss (or unknown) still has speech — keep it over a cutoff.
        tier = 2
    similarity = check.voice_similarity if check.voice_similarity is not None else -1.0
    return (tier, similarity, check.active_speech_sec, -check.implied_syl_per_sec)


def line_voice_similarity(reports: list[dict]) -> float | None:
    """Weakest scored chunk — one drifted sentence should not hide in the mean."""
    scores = [
        float(report["voice_similarity"])
        for report in reports
        if report.get("voice_similarity") is not None
    ]
    return min(scores) if scores else None


def quality_warning(
    reports: list[dict],
    *,
    warn_below: float = VOICE_WARN_BELOW,
    default_attempts: int = MAX_SYNTH_ATTEMPTS,
) -> str:
    """Human-readable warning when a returned take is still broken or weakly cloned."""
    parts: list[str] = []
    multi = len(reports) > 1
    for index, report in enumerate(reports, start=1):
        detail = _warning_detail(report, warn_below=warn_below, default_attempts=default_attempts)
        if not detail:
            continue
        parts.append(f"chunk {index}: {detail}" if multi else detail)
    return "; ".join(parts)


def _warning_detail(
    report: dict,
    *,
    warn_below: float = VOICE_WARN_BELOW,
    default_attempts: int = MAX_SYNTH_ATTEMPTS,
) -> str:
    attempts = int(report.get("attempts") or default_attempts)
    similarity = report.get("voice_similarity")
    sim_txt = "" if similarity is None else f", similarity {float(similarity):.2f}"
    if not report.get("ok", True):
        reason = str(report.get("reason") or "invalid")
        implied = float(report.get("implied_syl_per_sec") or 0.0)
        active = float(report.get("active_speech_sec") or 0.0)
        if reason == "silence":
            return f"silence after {attempts} attempts ({active:.2f}s active speech{sim_txt})"
        if reason == "cutoff":
            return (
                f"cutoff after {attempts} attempts "
                f"({implied:.1f} syl/s, {active:.2f}s active speech{sim_txt})"
            )
        if reason == "voice":
            score = 0.0 if similarity is None else float(similarity)
            return f"voice after {attempts} attempts (similarity {score:.2f})"
        return f"{reason} after {attempts} attempts{sim_txt}"
    if similarity is not None and float(similarity) < warn_below:
        return f"weak voice ({float(similarity):.2f})"
    return ""
