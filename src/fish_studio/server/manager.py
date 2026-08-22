"""Fish Speech synthesis via vLLM-Omni."""

from __future__ import annotations

import threading
from typing import Any

from fish_studio.runtime.vllm_health import check_vllm_http
from fish_studio.server.references import MAX_REFERENCES, ReferenceClip
from fish_studio.server.vllm_proxy import VllmFishProxy
from fish_studio.synthesis import FishSynthesisParams, SynthesisResult


class EngineManager:
    """Lazy-load the vLLM proxy and serialize first-load against concurrent requests."""

    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._loaded = False
        self._proxy = VllmFishProxy(settings.fish_speech)

    def ensure_loaded(self) -> VllmFishProxy:
        with self._lock:
            if not self._loaded:
                self._proxy.load()
                self._loaded = True
            return self._proxy

    def synthesize(
        self,
        *,
        text: str,
        language: str,
        references: list[ReferenceClip],
        fish_params: FishSynthesisParams | None = None,
        match_loudness: bool = True,
        match_timing: bool = True,
    ) -> SynthesisResult:
        del fish_params  # HTTP serving ignores these; vLLM owns sampling at process start.
        proxy = self.ensure_loaded()
        return proxy.synthesize(
            text=text,
            language=language,
            references=references,
            match_loudness=match_loudness,
            match_timing=match_timing,
        )

    def info(self) -> dict[str, Any]:
        data = self._proxy.info()
        data["active"] = self._loaded
        return data

    def health(self) -> dict[str, Any]:
        try:
            vllm = check_vllm_http(self.settings.fish_speech.base_url)
        except Exception:
            return {"status": "unavailable"}
        return {"status": "ok" if vllm.get("ok") else "unavailable"}
