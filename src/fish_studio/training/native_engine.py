"""In-process Fish Speech engine for CLI training smoke tests (not the HTTP server)."""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from fish_studio.loudness import finish_synthesis_audio, load_loudness_reference
from fish_studio.runtime.fish_config import FishSpeechRuntimeSettings
from fish_studio.stress import stressify
from fish_studio.synthesis import FISH_SYNTHESIS_DEFAULTS, FishSynthesisParams, SynthesisResult
from fish_studio.textnorm import prepare_synthesis_text


def _ensure_fish_speech_project_root() -> None:
    """fish-speech uses pyrootutils with a `.project-root` marker in site-packages."""
    import importlib.util

    spec = importlib.util.find_spec("fish_speech")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("fish_speech package not found")
    site_packages = Path(spec.submodule_search_locations[0]).resolve().parent
    (site_packages / ".project-root").touch()


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class NativeFishSpeechEngine:
    """In-process fish-speech TTS for ``./run.sh train infer``, not the HTTP server.

    Serving always goes through vLLM. This path loads Dual-AR + codec here so a
    merged LoRA checkpoint can be smoke-tested without standing up vLLM-Omni.
    """

    def __init__(self, settings: FishSpeechRuntimeSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._engine = None
        self._sample_rate = 44100

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    def load(self) -> None:
        llama = self.settings.llama_checkpoint
        decoder = self.settings.decoder_checkpoint
        for path, label in ((llama, "llama checkpoint"), (decoder, "decoder checkpoint")):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        _ensure_fish_speech_project_root()

        try:
            from tools.server.model_manager import ModelManager
        except ImportError as exc:
            raise ImportError(
                "Fish Speech is not installed. Install with: pip install -e '.[training]'"
            ) from exc

        manager = ModelManager(
            mode="tts",
            device=self.settings.device,
            half=self.settings.half,
            compile=self.settings.compile,
            llama_checkpoint_path=str(llama),
            decoder_checkpoint_path=str(decoder),
            decoder_config_name=self.settings.decoder_config_name,
        )
        self._engine = manager.tts_inference_engine
        self._sample_rate = int(manager.decoder_model.sample_rate)

    def unload(self) -> None:
        with self._lock:
            self._engine = None

    def synthesize(
        self,
        *,
        text: str,
        language: str,
        speaker_path: str | Path,
        synthesis_params: FishSynthesisParams | None = None,
        speaker_text: str | None = None,
    ) -> SynthesisResult:
        if self._engine is None:
            raise RuntimeError("Model is not loaded")

        text = prepare_synthesis_text(
            text,
            language=language,
            stress=self.settings.stress,
        )
        if not text:
            raise ValueError("text must not be empty")

        speaker_path = Path(speaker_path)
        if not speaker_path.is_file():
            raise FileNotFoundError(f"speaker sample not found: {speaker_path}")

        # Reference transcript stays as recorded (often English); only stress-mark it.
        reference = (speaker_text or self.settings.default_reference_text).strip()
        reference_text = stressify(reference, self.settings.stress)
        resolved = (synthesis_params or FishSynthesisParams()).resolve()

        try:
            from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
            from tools.server.inference import inference_wrapper
        except ImportError as exc:
            raise ImportError(
                "Fish Speech is not installed. Install with: pip install -e '.[training]'"
            ) from exc

        request = ServeTTSRequest(
            text=text,
            references=[
                ServeReferenceAudio(
                    audio=speaker_path.read_bytes(),
                    text=reference_text,
                )
            ],
            temperature=_clamp(float(resolved["temperature"]), 0.1, 1.0),
            top_p=_clamp(float(resolved["top_p"]), 0.1, 1.0),
            repetition_penalty=_clamp(float(resolved["repetition_penalty"]), 0.9, 2.0),
            max_new_tokens=self.settings.max_new_tokens,
            chunk_length=(
                self.settings.chunk_length
                if self.settings.chunk_length > 0
                else FISH_SYNTHESIS_DEFAULTS.chunk_length
            ),
            format="wav",
            streaming=False,
        )

        # inference_wrapper is not re-entrant; serialize concurrent synthesize calls.
        with self._lock:
            try:
                audio = next(inference_wrapper(request, self._engine))
            except StopIteration as exc:
                raise RuntimeError("Fish Speech returned no audio") from exc

        if not isinstance(audio, np.ndarray) or audio.size == 0:
            raise RuntimeError("Fish Speech returned no audio")

        loudness_ref, loudness_rate = load_loudness_reference(speaker_path)
        # Same post-process as HTTP: loudness (remaster + source level), then slot.
        audio = finish_synthesis_audio(
            audio,
            self._sample_rate,
            loudness_ref,
            loudness_rate,
            synthesis_text=text,
            reference_text=reference_text,
        )
        buffer = io.BytesIO()
        sf.write(buffer, audio, self._sample_rate, format="WAV")
        return SynthesisResult(
            wav_bytes=buffer.getvalue(),
            sample_rate=self._sample_rate,
            language=language,
        )

    def info(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "device": self.settings.device,
            "llama_checkpoint": str(self.settings.llama_checkpoint),
            "decoder_checkpoint": str(self.settings.decoder_checkpoint),
        }
