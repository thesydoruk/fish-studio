"""Readiness checks for vLLM-Omni without sending a speech request."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

EXPECTED_STAGES = (0, 1)
STAGE_MARK = "StageEngineCoreProc_stage"


def vllm_port(base_url: str | None = None) -> int:
    raw = (base_url or os.environ.get("FISH_SPEECH_BASE_URL") or "http://127.0.0.1:8091").strip()
    return urlparse(raw).port or int(os.environ.get("VLLM_PORT", "8091"))


def vllm_base_url(base_url: str | None = None) -> str:
    raw = (base_url or os.environ.get("FISH_SPEECH_BASE_URL") or "http://127.0.0.1:8091").rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme and parsed.hostname:
        return raw
    return f"http://127.0.0.1:{vllm_port(raw)}"


def fetch(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return int(response.getcode() or 0), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ConnectionError(str(exc.reason) if isinstance(exc, urllib.error.URLError) else str(exc)) from exc


def models_ready(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    return isinstance(data, list) and any(isinstance(item, dict) and item.get("id") for item in data)


def stage_ids_from_text(text: str) -> set[int]:
    found: set[int] = set()
    marker = STAGE_MARK
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            break
        tail = text[index + len(marker) :]
        digits = ""
        for char in tail:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            found.add(int(digits))
        start = index + len(marker)
    return found


def running_stage_ids(proc_root: str = "/proc") -> set[int]:
    found: set[int] = set()
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return found
    for pid in entries:
        if not pid.isdigit():
            continue
        for name in ("comm", "cmdline"):
            path = os.path.join(proc_root, pid, name)
            try:
                raw = open(path, "rb").read()
            except OSError:
                continue
            text = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
            found.update(stage_ids_from_text(text))
    return found


def check_http(base_url: str | None = None, timeout: float = 5.0) -> dict[str, Any]:
    base = vllm_base_url(base_url)
    try:
        health_code, _health_body = fetch(f"{base}/health", timeout=timeout)
    except ConnectionError as exc:
        return {"ok": False, "reason": f"health unreachable: {exc}", "base_url": base}
    if health_code != 200:
        return {"ok": False, "reason": f"health HTTP {health_code}", "base_url": base}

    try:
        models_code, models_body = fetch(f"{base}/v1/models", timeout=timeout)
    except ConnectionError as exc:
        return {"ok": False, "reason": f"models unreachable: {exc}", "base_url": base}
    if models_code != 200:
        return {"ok": False, "reason": f"models HTTP {models_code}", "base_url": base}
    try:
        payload = json.loads(models_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": f"models JSON: {exc}", "base_url": base}
    if not models_ready(payload):
        return {"ok": False, "reason": "no models listed", "base_url": base}
    return {"ok": True, "base_url": base, "models": len(payload["data"])}


def check_stages(expected: tuple[int, ...] = EXPECTED_STAGES) -> dict[str, Any]:
    running = running_stage_ids()
    missing = [stage for stage in expected if stage not in running]
    if missing:
        return {
            "ok": False,
            "reason": f"missing stage processes: {missing}",
            "stages": sorted(running),
        }
    return {"ok": True, "stages": sorted(running)}


def check_vllm_http(base_url: str | None = None, timeout: float = 5.0) -> dict[str, Any]:
    return check_http(base_url, timeout=timeout)


def check_vllm_container(base_url: str | None = None, timeout: float = 5.0) -> dict[str, Any]:
    http = check_http(base_url, timeout=timeout)
    if not http.get("ok"):
        return http
    stages = check_stages()
    if not stages.get("ok"):
        return {**http, **stages, "ok": False}
    return {**http, **stages}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--container",
        action="store_true",
        help="Require Fish Speech stage-0 and stage-1 engine processes.",
    )
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args(argv)
    result = (
        check_vllm_container(args.base_url)
        if args.container
        else check_vllm_http(args.base_url)
    )
    if not result.get("ok"):
        print(result.get("reason", "unhealthy"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
