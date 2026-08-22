"""FastAPI HTTP layer for Fish Speech."""

from __future__ import annotations

import asyncio
import base64
import binascii
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

from fish_studio.server.manager import EngineManager
from fish_studio.server.references import (
    MAX_REFERENCES,
    ReferenceClip,
    normalize_reference_texts,
    validate_reference_count,
)
from fish_studio.server.settings import ServerSettings
from fish_studio.synthesis import FishSynthesisParams


class JsonSynthesisRequest(BaseModel):
    """JSON synthesis body. First ``speaker_wav_b64`` is also the loudness/timing target."""

    text: str = Field(..., min_length=1, max_length=5000)
    language: str | None = None
    speaker_wav_b64: str | None = Field(default=None, min_length=1)
    speaker_wav_b64_list: list[str] | None = Field(default=None, min_length=1, max_length=MAX_REFERENCES)
    speaker_wav_format: str = "wav"
    speaker_text: str | None = None
    speaker_texts: list[str] | None = Field(default=None, min_length=1, max_length=MAX_REFERENCES)
    temperature: float | None = Field(None, ge=0.01, le=1.5)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(None, ge=0.9, le=20.0)
    match_loudness: bool = True
    match_timing: bool = True

    @model_validator(mode="after")
    def _validate_reference_payload(self) -> JsonSynthesisRequest:
        if self.speaker_wav_b64_list is None and self.speaker_wav_b64 is None:
            raise ValueError("speaker_wav_b64 or speaker_wav_b64_list is required")
        if self.speaker_wav_b64_list is not None and self.speaker_wav_b64 is not None:
            raise ValueError("pass either speaker_wav_b64 or speaker_wav_b64_list, not both")
        if self.speaker_texts is not None and self.speaker_text is not None:
            raise ValueError("pass either speaker_text or speaker_texts, not both")
        return self


def _fish_params(
    *,
    temperature: float | None,
    top_p: float | None,
    repetition_penalty: float | None,
) -> FishSynthesisParams | None:
    data = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    filtered = {key: value for key, value in data.items() if value is not None}
    if not filtered:
        return None
    return FishSynthesisParams(**filtered)


def _parse_flag(value: bool | str | None, *, default: bool = True) -> bool:
    """Parse a JSON/form flag. Omitted values keep ``default`` (true)."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean flag, got {value!r}")


def _default_language(settings: ServerSettings, language: str | None) -> str:
    if language:
        return language.strip().lower()
    return settings.default_language


def _resolve_raw_speaker_texts(
    *,
    count: int,
    speaker_text: str | None,
    speaker_texts: list[str] | None,
) -> list[str | None]:
    """One transcript per clip. A single ``speaker_text`` is only valid with one WAV."""
    if speaker_texts is not None:
        if speaker_text is not None:
            raise ValueError("pass either speaker_text or speaker_texts, not both")
        if len(speaker_texts) != count:
            raise ValueError(f"expected {count} speaker_texts value(s), got {len(speaker_texts)}")
        return list(speaker_texts)
    if count > 1:
        raise ValueError(
            f"multiple speaker_wav files require {count} speaker_texts values "
            "(or a single speaker_wav with speaker_text)"
        )
    return [speaker_text]


def _build_references(
    settings: ServerSettings,
    *,
    audio_paths: list[Path],
    speaker_text: str | None,
    speaker_texts: list[str] | None,
) -> list[ReferenceClip]:
    validate_reference_count(len(audio_paths))
    texts = normalize_reference_texts(
        _resolve_raw_speaker_texts(
            count=len(audio_paths),
            speaker_text=speaker_text,
            speaker_texts=speaker_texts,
        ),
        count=len(audio_paths),
        default_text=settings.fish_speech.default_reference_text,
        stress=settings.fish_speech.stress,
    )
    return [
        ReferenceClip(audio_path=path, text=text)
        for path, text in zip(audio_paths, texts, strict=True)
    ]


async def _save_uploads(files: list[UploadFile], *, max_upload_bytes: int) -> list[Path]:
    paths: list[Path] = []
    for upload in files:
        payload = await upload.read()
        if not payload:
            raise ValueError("speaker_wav is empty")
        if len(payload) > max_upload_bytes:
            raise ValueError("speaker_wav is too large")
        suffix = Path(upload.filename or "speaker.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(payload)
        tmp.close()
        paths.append(Path(tmp.name))
    return paths


def _decode_reference_bytes(
    body: JsonSynthesisRequest,
    *,
    max_upload_bytes: int,
) -> list[bytes]:
    payloads = (
        body.speaker_wav_b64_list
        if body.speaker_wav_b64_list is not None
        else [body.speaker_wav_b64 or ""]
    )
    decoded: list[bytes] = []
    for entry in payloads:
        try:
            speaker_bytes = base64.b64decode(entry, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid speaker_wav_b64") from exc
        if not speaker_bytes:
            raise ValueError("speaker_wav_b64 is empty")
        if len(speaker_bytes) > max_upload_bytes:
            raise ValueError("speaker sample is too large")
        decoded.append(speaker_bytes)
    return decoded


def create_app(settings: ServerSettings) -> FastAPI:
    """Build the FastAPI app. Synthesis runs in a worker thread so the event loop stays free."""
    manager = EngineManager(settings)

    app = FastAPI(
        title="Fish Speech TTS Server",
        version="1.0.0",
        description="HTTP API for Fish Speech voice cloning via vLLM-Omni.",
    )

    @app.get("/health", response_model=None)
    def health() -> JSONResponse:
        body = manager.health()
        status_code = 200 if body.get("status") == "ok" else 503
        return JSONResponse(status_code=status_code, content=body)

    @app.get("/v1/info")
    def info() -> dict:
        return manager.info()

    @app.post("/v1/synthesize")
    async def synthesize_multipart(
        text: Annotated[str, Form()],
        speaker_wav: Annotated[list[UploadFile], File()],
        language: Annotated[str | None, Form()] = None,
        speaker_text: Annotated[str | None, Form()] = None,
        speaker_texts: Annotated[list[str] | None, Form()] = None,
        temperature: Annotated[float | None, Form()] = None,
        top_p: Annotated[float | None, Form()] = None,
        repetition_penalty: Annotated[float | None, Form()] = None,
        match_loudness: Annotated[str | None, Form()] = None,
        match_timing: Annotated[str | None, Form()] = None,
    ) -> Response:
        lang = _default_language(settings, language)
        if not text.strip():
            raise HTTPException(status_code=400, detail="text must not be empty")
        if not speaker_wav:
            raise HTTPException(status_code=400, detail="at least one speaker_wav is required")

        speaker_paths: list[Path] = []
        try:
            speaker_paths = await _save_uploads(
                speaker_wav,
                max_upload_bytes=settings.max_upload_bytes,
            )
            references = _build_references(
                settings,
                audio_paths=speaker_paths,
                speaker_text=speaker_text,
                speaker_texts=speaker_texts,
            )
            result = await asyncio.to_thread(
                manager.synthesize,
                text=text,
                language=lang,
                references=references,
                fish_params=_fish_params(
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                ),
                match_loudness=_parse_flag(match_loudness),
                match_timing=_parse_flag(match_timing),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"synthesis failed: {exc}") from exc
        finally:
            for path in speaker_paths:
                path.unlink(missing_ok=True)

        return Response(
            content=result.wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Sample-Rate": str(result.sample_rate),
                "X-Language": result.language,
            },
        )

    @app.post("/v1/synthesize/json")
    async def synthesize_json(body: JsonSynthesisRequest) -> Response:
        lang = _default_language(settings, body.language)

        speaker_paths: list[Path] = []
        try:
            payloads = _decode_reference_bytes(body, max_upload_bytes=settings.max_upload_bytes)
            suffix = f".{body.speaker_wav_format.lstrip('.')}"
            for payload in payloads:
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp.write(payload)
                tmp.close()
                speaker_paths.append(Path(tmp.name))

            references = _build_references(
                settings,
                audio_paths=speaker_paths,
                speaker_text=body.speaker_text,
                speaker_texts=body.speaker_texts,
            )
            result = await asyncio.to_thread(
                manager.synthesize,
                text=body.text,
                language=lang,
                references=references,
                fish_params=_fish_params(
                    temperature=body.temperature,
                    top_p=body.top_p,
                    repetition_penalty=body.repetition_penalty,
                ),
                match_loudness=body.match_loudness,
                match_timing=body.match_timing,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"synthesis failed: {exc}") from exc
        finally:
            for path in speaker_paths:
                path.unlink(missing_ok=True)

        return Response(
            content=result.wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Sample-Rate": str(result.sample_rate),
                "X-Language": result.language,
            },
        )

    @app.exception_handler(FileNotFoundError)
    def missing_model_handler(_request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return app
