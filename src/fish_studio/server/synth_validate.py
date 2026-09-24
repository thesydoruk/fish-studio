"""Flag an obviously broken raw take: silence or a cutoff.

Judged on the model WAV *before* pause/tempo fit, against the chunk text.
``match_timing`` must not run first — stretch would hide a short take.

The implied rate is syllables(text) / active_speech(wav). A cutoff leaves the
full text in the numerator and only the spoken prefix in the denominator, so
the rate jumps well above any real articulation. Valid fast speech stays under
the ceiling; thin one-word lines skip the rate check.

The server makes one take per chunk and reports what it saw: the verdict goes
out as ``X-Synth-Warning`` and the clone cosine as ``X-Voice-Similarity``. What
to do with a bad take — regenerate, keep, pick between takes — is the client's
decision, with its own thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fish_studio.timing import count_syllables, measure_active_speech_sec, strip_nonspeech

# Same thin-text floor as the timing fit — one-word lines are too noisy for rate.
_MIN_SYL = 4
_MIN_CHARS = 12
# Below this, a scored line is empty / a click, not speech.
_SILENCE_ACTIVE_SEC = 0.25
# 5.6 syl/s is the timing band ceiling; 1.25× stretch is ~7.0. 10 is past any
# plausible raw take and still below a half-spoken line at a normal rate.
_CUTOFF_SYL_PER_SEC = 10.0
_PEAK_EPS = 1.0 / 32_768

# ASCII header on the WAV response so a client can see a broken take.
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


def attach_voice(check: SynthCheck, similarity: float | None) -> SynthCheck:
    """Record the clone cosine on a verdict. ``None`` = the clip could not be embedded."""
    return SynthCheck(
        ok=check.ok,
        reason=check.reason,
        syllables=check.syllables,
        active_speech_sec=check.active_speech_sec,
        implied_syl_per_sec=check.implied_syl_per_sec,
        voice_similarity=similarity,
    )


def line_voice_similarity(reports: list[dict]) -> float | None:
    """Weakest scored chunk — one drifted sentence should not hide in the mean."""
    scores = [
        float(report["voice_similarity"])
        for report in reports
        if report.get("voice_similarity") is not None
    ]
    return min(scores) if scores else None


def quality_warning(reports: list[dict]) -> str:
    """Human-readable warning when a returned take is silence or a cutoff."""
    parts: list[str] = []
    multi = len(reports) > 1
    for index, report in enumerate(reports, start=1):
        detail = _warning_detail(report)
        if not detail:
            continue
        parts.append(f"chunk {index}: {detail}" if multi else detail)
    return "; ".join(parts)


def _warning_detail(report: dict) -> str:
    if report.get("ok", True):
        return ""
    reason = str(report.get("reason") or "invalid")
    implied = float(report.get("implied_syl_per_sec") or 0.0)
    active = float(report.get("active_speech_sec") or 0.0)
    similarity = report.get("voice_similarity")
    sim_txt = "" if similarity is None else f", similarity {float(similarity):.2f}"
    if reason == "silence":
        return f"silence ({active:.2f}s active speech{sim_txt})"
    if reason == "cutoff":
        return f"cutoff ({implied:.1f} syl/s, {active:.2f}s active speech{sim_txt})"
    return f"{reason}{sim_txt}"
