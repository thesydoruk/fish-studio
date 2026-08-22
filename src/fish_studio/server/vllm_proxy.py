"""Fish Speech HTTP proxy to vLLM-Omni."""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf

from fish_studio.loudness import finish_synthesis_audio, load_loudness_reference
from fish_studio.server.references import (
    MAX_REFERENCES,
    ReferenceClip,
    concat_reference_audio,
    format_reference_text,
    normalize_reference_texts,
)
from fish_studio.server.settings import FishSpeechSettings
from fish_studio.synthesis import FISH_SYNTHESIS_DEFAULTS, SynthesisResult
from fish_studio.textnorm import prepare_synthesis_text, split_synthesis_chunks
from fish_studio.waveform import concat_audio_chunks


class VllmFishProxy:
    """Forwards synthesis requests to a vLLM-Omni server running fishaudio/s2-pro."""

    def __init__(self, settings: FishSpeechSettings) -> None:
        self.settings = settings
        self._available = False
        self._sample_rate = 44100
        self._client: httpx.Client | None = None

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
        match_loudness: bool = True,
        match_timing: bool = True,
    ) -> SynthesisResult:
        client = self._client
        if not self._available or client is None:
            raise RuntimeError("Model is not loaded")
        if not references:
            raise ValueError("at least one reference clip is required")

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

        reference_text = format_reference_text([clip.text for clip in references])
        audio_paths = [clip.audio_path for clip in references]
        combined_path: Path | None = None
        clone_path = audio_paths[0]
        try:
            if len(audio_paths) > 1:
                # vLLM clone prompt is one concatenated WAV; loudness/timing stay on clip 0.
                combined_path = concat_reference_audio(audio_paths)
                clone_path = combined_path

            if not clone_path.is_file():
                raise FileNotFoundError(f"speaker sample not found: {clone_path}")

            mime = mimetypes.guess_type(clone_path.name)[0] or "audio/wav"
            ref_b64 = base64.b64encode(clone_path.read_bytes()).decode("ascii")
            loudness_ref, loudness_rate = load_loudness_reference(audio_paths[0])
            pieces, sample_rate = self._synthesize_chunks(
                client, chunks, mime, ref_b64, reference_text
            )
            self._sample_rate = sample_rate
            combined = concat_audio_chunks(pieces, sample_rate) if len(pieces) > 1 else pieces[0]
            combined = finish_synthesis_audio(
                combined,
                sample_rate,
                loudness_ref,
                loudness_rate,
                synthesis_text=text,
                reference_text=references[0].text,
                match_loudness=match_loudness,
                match_timing=match_timing,
            )
            return SynthesisResult(
                wav_bytes=_encode_wav(combined, sample_rate),
                sample_rate=sample_rate,
                language=language,
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
    ) -> tuple[list[np.ndarray], int]:
        """Generate each sentence chunk in order.

        Loudness and slot matching run on the concatenated line so they see
        the same span as the actor reference.
        """

        pieces: list[np.ndarray] = []
        sample_rate = self._sample_rate
        for chunk in chunks:
            audio, sample_rate = _decode_wav(
                self._request_speech(client, chunk, mime, ref_b64, reference_text)
            )
            pieces.append(audio)
        return pieces, sample_rate

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
                    "sequence so they do not multiply GPU load. match_loudness "
                    "(default true) applies a light remaster, then "
                    "matches speech level to the first speaker_wav. match_timing "
                    "(default true) fits the clip to that slot by stretching "
                    "speech only; stretch never rushes more than ~15% past "
                    "the actor's articulation rate."
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


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()
