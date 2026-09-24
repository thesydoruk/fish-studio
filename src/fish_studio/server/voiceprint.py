"""ECAPA speaker embeddings for the synthesis voice gate.

Compares the raw take to the clone prompt (same model as the FO4 bench:
``speechbrain/spkrec-ecapa-voxceleb``). CPU by default so vLLM keeps the GPU.

Calibrated on FO4 vanilla + ear: 0.30 is the usable-clone floor, but the
server only reports the score (``X-Voice-Similarity``); what counts as too low
is the client's threshold. Short lines are scored too.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_RATE = 16_000
MIN_EMBED_SEC = 0.2
VOICE_SIMILARITY_HEADER = "X-Voice-Similarity"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine of two vectors; 0.0 when either side is empty or zero."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    denom = (left_norm**0.5) * (right_norm**0.5)
    return 0.0 if denom == 0.0 else dot / denom


def l2_normalize(vector: list[float]) -> list[float]:
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


def to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample to the ECAPA rate. Empty input stays empty."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if int(sample_rate) == TARGET_RATE:
        return samples
    import torch
    import torchaudio

    wav = torch.from_numpy(samples).unsqueeze(0)
    return (
        torchaudio.functional.resample(wav, int(sample_rate), TARGET_RATE)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


class VoiceEncoder:
    """Lazy SpeechBrain ECAPA encoder. Safe to construct when the extra is missing."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else _default_model_dir()
        self._classifier = None
        self._unavailable = ""

    @property
    def available(self) -> bool:
        return not self._unavailable and self._classifier is not None

    def warmup(self) -> None:
        """Load weights once (server startup). Failures disable the gate, not synth."""
        self._ensure()

    def embed(self, audio: np.ndarray, sample_rate: int) -> list[float] | None:
        """Return an L2-normalized embedding, or None when the clip cannot be scored."""
        samples = to_mono_16k(audio, sample_rate)
        if samples.size < int(TARGET_RATE * MIN_EMBED_SEC):
            return None
        if float(np.max(np.abs(samples))) < 1e-4:
            return None
        classifier = self._ensure()
        if classifier is None:
            return None
        import torch

        wav = torch.from_numpy(samples).unsqueeze(0)
        try:
            with torch.inference_mode():
                embedding = classifier.encode_batch(wav)
        except Exception:
            logger.exception("voice embedding failed")
            return None
        vector = embedding.squeeze().detach().cpu().float().numpy().tolist()
        return l2_normalize(vector)

    def similarity(
        self,
        audio: np.ndarray,
        sample_rate: int,
        reference: list[float],
    ) -> float | None:
        embedding = self.embed(audio, sample_rate)
        if embedding is None:
            return None
        return cosine_similarity(embedding, reference)

    def _ensure(self):
        if self._unavailable:
            return None
        if self._classifier is not None:
            return self._classifier
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            self._unavailable = "speechbrain is not installed"
            logger.warning("voice gate off: %s", self._unavailable)
            return None
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self._classifier = EncoderClassifier.from_hparams(
                source=MODEL_ID,
                savedir=str(self.model_dir),
                run_opts={"device": "cpu"},
            )
        except Exception as exc:
            self._unavailable = str(exc)
            logger.warning("voice gate off: %s", self._unavailable)
            return None
        return self._classifier


def _default_model_dir() -> Path:
    return Path.home() / ".cache" / "fish-studio" / "ecapa-voxceleb"
