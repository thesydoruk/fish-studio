"""Fish Speech HTTP proxy to vLLM-Omni."""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf

from fish_studio.loudness import match_loudness_to_reference
from fish_studio.server.references import (
    MAX_REFERENCES,
    ReferenceClip,
    concat_reference_audio,
    format_reference_text,
)
from fish_studio.server.settings import FishSpeechSettings
from fish_studio.server.synth_log import SynthesisRequestLogger
from fish_studio.server.synth_validate import (
    attach_voice,
    judge_raw_synth,
    line_voice_similarity,
    quality_warning,
)
from fish_studio.server.voiceprint import VoiceEncoder
from fish_studio.speech_norm import normalize_speech
from fish_studio.synthesis import FISH_SYNTHESIS_DEFAULTS, SynthesisResult
from fish_studio.textnorm import prepare_synthesis_text, split_synthesis_chunks
from fish_studio.timing import (
    ensure_praat_psola,
    fit_timing_to_reference,
    load_reference_audio,
)
from fish_studio.waveform import concat_audio_chunks

logger = logging.getLogger(__name__)


class VllmFishProxy:
    """Forwards synthesis requests to a vLLM-Omni server running fishaudio/s2-pro."""

    def __init__(self, settings: FishSpeechSettings) -> None:
        self.settings = settings
        self._available = False
        self._sample_rate = 44100
        self._client: httpx.Client | None = None
        self._synth_log = SynthesisRequestLogger(
            settings.synth_log_dir or Path("_unused"),
            keep=settings.synth_log_keep,
            enabled=bool(settings.synth_log_enabled and settings.synth_log_dir),
        )
        self._voice = VoiceEncoder()

    @property
    def is_loaded(self) -> bool:
        return self._available

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        base = self.settings.base_url.rstrip("/")
        try:
            response = httpx.get(f"{base}/health", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"vLLM-Omni server is not reachable at {base}. Start it with: ./run.sh vllm start"
            ) from exc
        self._client = httpx.Client(
            base_url=base,
            timeout=self.settings.timeout_sec,
            limits=httpx.Limits(
                max_connections=self.settings.max_concurrent_requests,
                max_keepalive_connections=self.settings.max_concurrent_requests,
            ),
        )
        ensure_praat_psola()
        self._voice.warmup()
        self._available = True

    def unload(self) -> None:
        self._available = False
        if self._client is not None:
            self._client.close()
            self._client = None

    def synthesize(
        self,
        *,
        text: str,
        language: str,
        references: list[ReferenceClip],
        match_timing: bool = True,
    ) -> SynthesisResult:
        """Synthesize ``text`` in the voice of ``references``: one take per chunk.

        The response carries what the server saw -- ``X-Synth-Warning`` for a
        silent or cut-off take, ``X-Voice-Similarity`` for the weakest chunk's
        clone cosine -- and nothing more. Whether to regenerate is the
        client's call.
        """
        client = self._client
        if not self._available or client is None:
            raise RuntimeError("Model is not loaded")
        if not references:
            raise ValueError("at least one reference clip is required")

        text_raw = text
        # Expand UK numerals before stress marks so both match training orthography.
        text = prepare_synthesis_text(
            text,
            language=language,
            stress=self.settings.stress,
        )
        if not text:
            raise ValueError("text must not be empty")

        max_chars = (
            self.settings.chunk_length
            if self.settings.chunk_length > 0
            else FISH_SYNTHESIS_DEFAULTS.chunk_length
        )
        chunks = split_synthesis_chunks(text, max_chars)
        if not chunks:
            raise ValueError("text must not be empty")

        reference_texts = [clip.text for clip in references]
        reference_text = format_reference_text(reference_texts)
        audio_paths = [clip.audio_path for clip in references]
        combined_path: Path | None = None
        clone_path = audio_paths[0]
        try:
            if len(audio_paths) > 1:
                # vLLM clone prompt is one concatenated WAV; slot matching stays on clip 0.
                combined_path = concat_reference_audio(audio_paths)
                clone_path = combined_path

            if not clone_path.is_file():
                raise FileNotFoundError(f"speaker sample not found: {clone_path}")

            mime = mimetypes.guess_type(clone_path.name)[0] or "audio/wav"
            ref_b64 = base64.b64encode(clone_path.read_bytes()).decode("ascii")
            slot_ref, slot_rate = load_reference_audio(audio_paths[0])
            clone_wav, clone_rate = (
                (slot_ref, slot_rate)
                if clone_path == audio_paths[0]
                else load_reference_audio(clone_path)
            )
            ref_embedding = self._voice.embed(clone_wav, clone_rate)

            pieces, sample_rate, quality = self._synthesize_chunks(
                client, chunks, mime, ref_b64, reference_text, ref_embedding
            )
            self._sample_rate = sample_rate
            raw = concat_audio_chunks(pieces, sample_rate) if len(pieces) > 1 else pieces[0]
            speech = normalize_speech(raw, sample_rate)
            leveled = speech.audio
            timed = leveled
            timing: dict[str, Any] = {"enabled": match_timing}
            if match_timing:
                fit = fit_timing_to_reference(
                    leveled,
                    sample_rate,
                    slot_ref,
                    slot_rate,
                    text_uk=text,
                    text_en=references[0].text,
                )
                timed = fit.audio
                timing.update(fit.metrics())
                if fit.needs_shorter_line:
                    logger.warning(
                        "needs_shorter_line: %.2fs over a %.2fs slot at %.1f syl/s "
                        "(stretch %.2fx capped); text=%r",
                        fit.overrun_sec,
                        fit.slot_sec,
                        fit.syl_per_sec_final,
                        fit.stretch_rate,
                        text_raw,
                    )
            loudness = match_loudness_to_reference(timed, sample_rate, slot_ref, slot_rate)
            final = loudness.audio
            warning = quality_warning(quality)
            voice_similarity = line_voice_similarity(quality)
            try:
                self._synth_log.log(
                    text_raw=text_raw,
                    text_prepared=text,
                    chunks=chunks,
                    language=language,
                    match_timing=match_timing,
                    reference_paths=audio_paths,
                    reference_texts=reference_texts,
                    raw_audio=raw,
                    final_audio=final,
                    sample_rate=sample_rate,
                    extra={
                        "chunk_count": len(chunks),
                        "quality": quality,
                        "warning": warning,
                        "voice_similarity": voice_similarity,
                        "timing": timing,
                        "speech_norm": speech.metrics(),
                        "loudness": loudness.metrics(),
                    },
                )
            except Exception:
                # Logging must never fail the client response.
                pass
            return SynthesisResult(
                wav_bytes=_encode_wav(final, sample_rate),
                sample_rate=sample_rate,
                language=language,
                warning=warning,
                voice_similarity=voice_similarity,
            )
        finally:
            if combined_path is not None:
                combined_path.unlink(missing_ok=True)

    def _synthesize_chunks(
        self,
        client: httpx.Client,
        chunks: list[str],
        mime: str,
        ref_b64: str,
        reference_text: str,
        ref_embedding: list[float] | None,
    ) -> tuple[list[np.ndarray], int, list[dict]]:
        """Generate each sentence chunk once and judge it."""
        pieces: list[np.ndarray] = []
        quality: list[dict] = []
        sample_rate = self._sample_rate
        for chunk in chunks:
            audio, sample_rate, report = self._synthesize_chunk_judged(
                client, chunk, mime, ref_b64, reference_text, ref_embedding
            )
            pieces.append(audio)
            quality.append(report)
        return pieces, sample_rate, quality

    def _synthesize_chunk_judged(
        self,
        client: httpx.Client,
        chunk: str,
        mime: str,
        ref_b64: str,
        reference_text: str,
        ref_embedding: list[float] | None = None,
    ) -> tuple[np.ndarray, int, dict]:
        """One raw take, judged for silence / cutoff and scored against the clone prompt."""
        audio, sample_rate = _decode_wav(
            self._request_speech(client, chunk, mime, ref_b64, reference_text)
        )
        check = judge_raw_synth(audio, sample_rate, chunk)
        similarity = (
            None
            if ref_embedding is None
            else self._voice.similarity(audio, sample_rate, ref_embedding)
        )
        check = attach_voice(check, similarity)
        if not check.ok:
            logger.warning(
                "synth %s (implied %.1f syl/s, active %.2fs, sim=%s): %r",
                check.reason,
                check.implied_syl_per_sec,
                check.active_speech_sec,
                _sim_label(check.voice_similarity),
                chunk,
            )
        return audio, sample_rate, check.metrics()

    def _request_speech(
        self,
        client: httpx.Client,
        text: str,
        mime: str,
        ref_b64: str,
        reference_text: str,
    ) -> bytes:
        payload: dict[str, Any] = {
            "input": text,
            "voice": self.settings.voice,
            "response_format": "wav",
            # vLLM-Omni Fish plugin expects a data-URI, not a raw path or multipart file.
            "ref_audio": f"data:{mime};base64,{ref_b64}",
            "ref_text": reference_text,
        }
        if self.settings.max_new_tokens > 0:
            payload["max_new_tokens"] = self.settings.max_new_tokens

        try:
            response = client.post("/v1/audio/speech", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise RuntimeError(
                f"vLLM-Omni synthesis failed ({exc.response.status_code}): {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"vLLM-Omni request failed: {exc}") from exc

        wav_bytes = response.content
        if not wav_bytes:
            raise RuntimeError("vLLM-Omni returned no audio")
        return wav_bytes

    def info(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "base_url": self.settings.base_url,
            "voice": self.settings.voice,
            "max_concurrent_requests": self.settings.max_concurrent_requests,
            "chunk_length": self.settings.chunk_length,
            "sample_rate": self._sample_rate,
            "default_language": self.settings.default_language,
            "synth_log": {
                "enabled": self._synth_log.enabled,
                "dir": str(self.settings.synth_log_dir) if self.settings.synth_log_dir else None,
                "keep": self.settings.synth_log_keep,
                "latest": (
                    None
                    if not self._synth_log.enabled
                    else (
                        str(latest)
                        if (latest := self._synth_log.latest_dir()) is not None
                        else None
                    )
                ),
            },
            "notes": {
                "server": (
                    "Requires a running vLLM-Omni server (./run.sh vllm start). "
                    "The model is selected at vLLM startup, not per request."
                ),
                "speaker_text": (
                    "Optional reference transcript improves cloning quality. "
                    "Pass speaker_text (one value) or speaker_texts (one per clip). "
                    f"Up to {MAX_REFERENCES} speaker_wav files are concatenated in order."
                ),
                "long_form": (
                    "Text longer than FISH_SPEECH_CHUNK_LENGTH is split on sentence "
                    "boundaries and concatenated. Chunks of one request run in "
                    "sequence so they do not multiply GPU load. match_timing "
                    "(default true) fits the first speaker_wav slot: pause "
                    "budget first, then Praat PSOLA, limited to a plausible "
                    "syllables-per-second rate and at most 1.25×, never slower. "
                    "A line that still overruns is logged as needs_shorter_line "
                    "rather than sped up further. Each chunk is generated once; "
                    "a take that is silence or implies an impossible syllable "
                    "rate (cutoff) is returned with X-Synth-Warning, and "
                    "X-Voice-Similarity carries the weakest chunk's ECAPA cosine "
                    "vs the clone prompt, so the client can judge and regenerate. "
                    "Right after synthesis, ffmpeg dynaudnorm evens ragged "
                    "syllable loudness; timing runs on that take; a linear "
                    "speech-gated LUFS gain then matches the first speaker_wav "
                    "(ffmpeg loudnorm is not used — short lines make I unstable "
                    "and it would recompress what dynaudnorm just evened). "
                ),
            },
        }


def _decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    except RuntimeError as exc:
        raise RuntimeError(f"vLLM-Omni returned invalid audio: {exc}") from exc
    if not isinstance(audio, np.ndarray) or audio.size == 0:
        raise RuntimeError("vLLM-Omni returned no audio")
    return audio, int(sample_rate)


def _sim_label(similarity: float | None) -> str:
    return "n/a" if similarity is None else f"{similarity:.2f}"


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()
