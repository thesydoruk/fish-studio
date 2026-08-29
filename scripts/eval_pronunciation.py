#!/usr/bin/env python3
"""Probe a served checkpoint: synth → audio-intel → CER / LID / speaker cosine.

Post-processing stays off so timing/FX cannot hide pronunciation or clone drift.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fish_studio.dataset.speaker_cluster import cosine_similarity  # noqa: E402


_LETTERS = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_for_cer(text: str) -> str:
    folded = text.casefold().replace("ё", "е")
    folded = _LETTERS.sub(" ", folded)
    return _SPACE.sub(" ", folded).strip()


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance / max(len(ref), 1) on normalized letters."""
    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rch in enumerate(ref, start=1):
        cur = [i]
        for j, hch in enumerate(hyp, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (rch != hch)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1] / len(ref)


def _first_embedding(payload: dict) -> list[float] | None:
    for speaker in payload.get("speakers") or []:
        raw = speaker.get("embedding")
        if isinstance(raw, list) and raw:
            return [float(x) for x in raw]
    return None


def synthesize(
    client: httpx.Client,
    *,
    text: str,
    refs: list[tuple[Path, str]],
    language: str,
) -> bytes:
    import base64

    payload: dict = {
        "text": text,
        "language": language,
        "match_timing": False,
        "speaker_wav_format": "wav",
    }
    encoded = [base64.b64encode(path.read_bytes()).decode("ascii") for path, _ in refs]
    texts = [ref_text for _, ref_text in refs]
    if len(refs) == 1:
        payload["speaker_wav_b64"] = encoded[0]
        payload["speaker_text"] = texts[0]
    else:
        payload["speaker_wav_b64_list"] = encoded
        payload["speaker_texts"] = texts
    response = client.post("/v1/synthesize/json", json=payload)
    response.raise_for_status()
    return response.content


def transcribe(intel_url: str, wav_path: Path, timeout: float) -> dict:
    url = f"{intel_url.rstrip('/')}/v1/audio/transcriptions"
    with wav_path.open("rb") as handle:
        response = httpx.post(
            url,
            data={
                "response_format": "verbose_json",
                "align": "false",
                "diarize": "true",
                "sound_events": "false",
            },
            files={"file": (wav_path.name, handle, "audio/wav")},
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json()


def run_case(
    *,
    synth: httpx.Client,
    intel_url: str | None,
    out_dir: Path,
    label: str,
    text: str,
    refs: list[tuple[Path, str]],
    language: str,
    timeout: float,
) -> dict:
    wav = synthesize(synth, text=text, refs=refs, language=language)
    wav_path = out_dir / f"{label}.wav"
    wav_path.write_bytes(wav)
    row: dict = {
        "label": label,
        "text": text,
        "wav": str(wav_path),
        "n_refs": len(refs),
    }
    if not intel_url:
        return row
    payload = transcribe(intel_url, wav_path, timeout)
    asr = str(payload.get("text") or "").strip()
    if not asr:
        asr = " ".join(
            str(seg.get("text") or "").strip()
            for seg in payload.get("segments") or []
            if seg.get("kind", "speech") == "speech"
        ).strip()
    langs = [
        str(seg.get("language") or payload.get("language") or "")
        for seg in payload.get("segments") or []
        if seg.get("kind", "speech") == "speech" or "language" in seg
    ]
    lid = payload.get("language") or (langs[0] if langs else "")
    ref_asr = transcribe(intel_url, refs[0][0], timeout)
    spk = None
    left = _first_embedding(ref_asr)
    right = _first_embedding(payload)
    if left and right:
        spk = cosine_similarity(left, right)
    row.update(
        {
            "asr": asr,
            "cer": round(character_error_rate(text, asr), 3),
            "lid": lid,
            "spk_cos": None if spk is None else round(spk, 3),
        }
    )
    return row


def parse_ref(raw: str) -> tuple[Path, str]:
    path_s, sep, text = raw.partition("::")
    if not sep:
        raise ValueError(f"--ref must be path::transcript, got {raw!r}")
    path = Path(path_s)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--intel-url", default="", help="audio-intel; empty skips ASR")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--ref", action="append", required=True, help="path::transcript")
    parser.add_argument("--language", default="uk")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    refs = [parse_ref(item) for item in args.ref]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    intel = args.intel_url.strip() or None
    rows = []
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        for index, text in enumerate(args.text):
            label = f"{args.label}_{index:02d}"
            row = run_case(
                synth=client,
                intel_url=intel,
                out_dir=args.out_dir,
                label=label,
                text=text,
                refs=refs,
                language=args.language,
                timeout=args.timeout,
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.out_dir / f"{args.label}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
