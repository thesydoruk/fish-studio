"""Shared synthesis result and parameter types."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SynthesisResult:
    """WAV payload returned by every synthesis backend."""

    wav_bytes: bytes
    sample_rate: int
    language: str


@dataclass(frozen=True)
class FishSynthesisDefaults:
    """s2-pro sampling defaults shared by CLI infer and the HTTP API schema."""

    temperature: float = 0.8
    top_p: float = 0.8
    repetition_penalty: float = 1.1
    chunk_length: int = 200
    max_new_tokens: int = 1024


FISH_SYNTHESIS_DEFAULTS = FishSynthesisDefaults()


def fish_synthesis_defaults_dict() -> dict[str, float | int]:
    return asdict(FISH_SYNTHESIS_DEFAULTS)


@dataclass(frozen=True)
class FishSynthesisParams:
    """Optional per-request overrides. ``None`` keeps the backend default."""

    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None

    def resolve(
        self,
        defaults: FishSynthesisDefaults = FISH_SYNTHESIS_DEFAULTS,
    ) -> dict[str, float]:
        """Fill missing fields from ``defaults`` (used by the native engine only)."""
        return {
            "temperature": (
                self.temperature if self.temperature is not None else defaults.temperature
            ),
            "top_p": self.top_p if self.top_p is not None else defaults.top_p,
            "repetition_penalty": (
                self.repetition_penalty
                if self.repetition_penalty is not None
                else defaults.repetition_penalty
            ),
        }
