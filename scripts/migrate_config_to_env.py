#!/usr/bin/env python3
"""One-time migration: config.yaml -> .env (keeps config.yaml.bak)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML required for migration") from exc

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config.yaml"
ENV = ROOT / ".env"
EXAMPLE = ROOT / ".env.example"


def main() -> int:
    if not EXAMPLE.is_file():
        raise SystemExit(".env.example not found")

    if CFG.is_file():
        raw = yaml.safe_load(CFG.read_text(encoding="utf-8")) or {}
        fish = raw.get("fish_speech") or {}
        inf = raw.get("inference") or {}
        train = raw.get("fish_training") or raw.get("training") or {}
        sources = raw.get("sources") or []

        out: dict[str, str] = {}
        for line in EXAMPLE.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                key, val = s.split("=", 1)
                out[key] = val

        out["DATA_ROOT"] = str((raw.get("paths") or {}).get("data_root", "./data"))
        out["SOURCES"] = json.dumps(sources, ensure_ascii=False)

        mapping = {
            "INFERENCE_HOST": inf.get("host"),
            "INFERENCE_PORT": inf.get("port"),
            "INFERENCE_DEVICE": inf.get("device"),
            "INFERENCE_LANGUAGE": inf.get("language"),
            "INFERENCE_MAX_UPLOAD_BYTES": inf.get("max_upload_bytes"),
            "FISH_SPEECH_MODEL": fish.get("model"),
            "FISH_SPEECH_GPU_MEMORY_UTILIZATION": fish.get("gpu_memory_utilization"),
            "FISH_SPEECH_VOICE": fish.get("voice"),
            "FISH_SPEECH_TIMEOUT_SEC": fish.get("timeout_sec"),
            "FISH_SPEECH_LLAMA_CHECKPOINT": fish.get("llama_checkpoint"),
            "FISH_SPEECH_DECODER_CHECKPOINT": fish.get("decoder_checkpoint") or "",
            "FISH_SPEECH_DECODER_CONFIG_NAME": fish.get("decoder_config_name"),
            "FISH_SPEECH_HALF": fish.get("half"),
            "FISH_SPEECH_COMPILE": fish.get("compile"),
            "FISH_SPEECH_CHUNK_LENGTH": fish.get("chunk_length"),
            "FISH_SPEECH_MAX_NEW_TOKENS": fish.get("max_new_tokens"),
            "FISH_SPEECH_MAX_CONCURRENT_REQUESTS": fish.get("max_concurrent_requests"),
            "FISH_SPEECH_DEFAULT_REFERENCE_TEXT": fish.get("default_reference_text", ""),
            "FISH_SPEECH_USE_FINETUNED": fish.get("use_finetuned"),
            "TRAINING_DATASET_ID": train.get("dataset_id"),
            "TRAINING_SPEAKER_NAME": train.get("speaker_name"),
            "TRAINING_PROJECT_NAME": train.get("project_name"),
            "TRAINING_MAX_STEPS": train.get("max_steps"),
            "TRAINING_BATCH_SIZE": train.get("batch_size"),
            "TRAINING_GRAD_ACCUM": train.get("grad_accum"),
            "TRAINING_LR": train.get("lr"),
            "TRAINING_VAL_CHECK_INTERVAL": train.get("val_check_interval"),
            "TRAINING_LORA_CONFIG": train.get("lora_config"),
        }
        for key, val in mapping.items():
            if val is None:
                continue
            out[key] = (
                "true"
                if isinstance(val, bool) and val
                else "false"
                if isinstance(val, bool)
                else str(val)
            )

        out["FISH_SPEECH_BASE_URL"] = "http://vllm:8091"

        rendered: list[str] = []
        for line in EXAMPLE.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                rendered.append(line)
                continue
            key = s.split("=", 1)[0]
            rendered.append(f"{key}={out.get(key, s.split('=', 1)[1])}")
        ENV.write_text("\n".join(rendered) + "\n", encoding="utf-8")
        shutil.copy2(CFG, CFG.with_suffix(".yaml.bak"))
        print(f"migrated {CFG} -> {ENV}")
    elif ENV.is_file():
        print(f"{ENV} already exists")
    else:
        shutil.copy2(EXAMPLE, ENV)
        print(f"created {ENV} from example")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
