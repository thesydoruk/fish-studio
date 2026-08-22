# Fish Speech TTS Serve

HTTP TTS server with voice cloning via [Fish Speech](https://github.com/fishaudio/fish-speech) s2-pro, plus a dataset builder and LoRA fine-tuning pipeline for Ukrainian.

The HTTP server proxies synthesis to an external [vLLM-Omni](https://github.com/vllm-project/vllm-omni) process.

> **Ukrainian LoRA from scratch:** follow the step-by-step playbook
> [`docs/ukrainian-lora-playbook.md`](docs/ukrainian-lora-playbook.md)
> (audio-intel → bilingual YouTube → HF import → full `combined` merge → `bg-train` → serve).

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (recommended)
- Fish Speech `s2-pro` checkpoints (downloaded by `./install.sh all`)

## Quick start

```bash
./run.sh install all
cp .env.example .env
./run.sh vllm install
./run.sh vllm start
./run.sh serve
```

CLI entry point: `fish-server` or `python -m fish_studio.server.serve`

Default port: `8080` (`INFERENCE_PORT` in `.env`).

## HTTP API

| Method | Path                  | Description             |
| ------ | --------------------- | ----------------------- |
| `GET`  | `/health`             | Ready to synthesize; 503 if the engine is down |
| `GET`  | `/v1/info`            | Server and model info   |
| `POST` | `/v1/synthesize`      | Multipart synthesis     |
| `POST` | `/v1/synthesize/json` | JSON + base64 reference |

### Fields

| Field                             | Required | Default              | Description                        |
| --------------------------------- | -------- | -------------------- | ---------------------------------- |
| `text`                            | yes      | —                    | Text to synthesize                 |
| `speaker_wav` / `speaker_wav_b64` | yes      | —                    | Voice clone clip; **first file** is also the loudness and timing target |
| `speaker_text`                    | no       | —                    | Reference transcript (recommended) |
| `language`                        | no       | `INFERENCE_LANGUAGE` | Language code                      |
| `match_loudness`                  | no       | `true`               | Light remaster, then match level to the first `speaker_wav` |
| `match_timing`                    | no       | `true`               | Fit duration to the first `speaker_wav` slot |

Fish Speech params: `temperature`, `top_p`, `repetition_penalty`

### Example

```bash
curl -X POST http://localhost:8080/v1/synthesize \
  -F "text=Привіт, як справи?" \
  -F "speaker_wav=@reference.wav" \
  -F "speaker_text=Привіт, як справи?" \
  --output out.wav
```

## Fish Speech (vLLM-Omni)

Either run both services in containers:

```bash
docker compose up -d --build   # vllm (:8091) + server (:8080)
docker compose logs -f vllm
docker compose down
```

Or run vLLM on the host in its own venv:

```bash
./run.sh vllm install   # one-time: create .venv-vllm, install vllm-omni + fish-speech
./run.sh vllm start     # serve FISH_SPEECH_MODEL from .env
./run.sh vllm status
./run.sh vllm stop
```

To serve the fine-tuned model, convert the merged checkpoint to the HF layout first:

```bash
./run.sh train export-vllm       # → {data_root}/training/vllm/
# .env: FISH_SPEECH_MODEL=training/vllm
./run.sh vllm restart
```

## Project scripts

Single entry point: **`./run.sh`**

| Command                     | Description                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| `./run.sh install [target]` | Create venv, install deps, download checkpoints                                             |
| `./run.sh server`           | Start Fish Speech TTS HTTP server                                                           |
| `./run.sh vllm <cmd>`       | vLLM-Omni server: `install`, `start`, `stop`, `restart`, `status`                           |
| `./run.sh dataset <cmd>`    | Dataset builder (`run`, `merge`, `all`, `sources`, `datasets`, `hf-import`, …)               |
| `./run.sh dataset-build`    | Run all enabled sources + merge (shortcut)                                                  |
| `./run.sh train <step>`     | Fish Speech LoRA: `export`, `vq`, `protos`, `train`, `merge`, `export-vllm`, `infer`, `all` |
| `./run.sh bg-train`         | LoRA training in background (`data/logs/training.log`)                                      |
| `./run.sh tensorboard`      | TensorBoard for training runs: `start`, `stop`, `status`                                    |
| `./run.sh server start`     | Background server → `data/logs/server.log`                                                  |
| `./run.sh status`           | Datasets, checkpoints, server, GPU, audio-intel warning                                     |
| `./run.sh logs <name>`      | `server`, `training`, `pipeline`                                                            |
| `./run.sh analyze <kind>`   | Quality helpers: `transcripts` \| `clips`                                                   |
| `./run.sh synthesize`       | Quick HTTP synthesis test                                                                   |
| `./run.sh init`             | Create `.env` from `.env.example`                                                           |

`install` targets: `server` (default), `dataset`, `training`, `all`, `dev`

## Dataset builde

Build pipe-delimited training datasets from YouTube or local audio via [audio-intel](https://github.com/thesydoruk/audio-intel) transcription.

For the full Ukrainian bilingual + HF + LoRA path, use
[`docs/ukrainian-lora-playbook.md`](docs/ukrainian-lora-playbook.md).

```bash
./run.sh install all
cp .env.example .env
# 1) Start audio-intel with align + diarize + sound_events (see playbook)
# 2) Fill SOURCES=[{...}] in .env; for bilingual lessons set:
#    AUDIO_INTEL_LANGUAGE=auto
#    AUDIO_INTEL_SOUND_EVENTS=true
#    SEGMENTATION_SPLIT_BY_SCRIPT=true
#    SEGMENTATION_ALLOWED_LANGUAGES=uk,en
#    SEGMENTATION_FILTER_NON_TARGET_LANGUAGE=false

./run.sh dataset run --source game-vo-1
# Merge every export-ready folder under data/datasets/ (YouTube + HF imports, …)
./run.sh dataset merge
# Or only enabled SOURCES entries:
./run.sh dataset merge --from-sources

# Or full build + on-disk merge in one step:
./run.sh dataset all
```

| Command                    | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| `./run.sh dataset run`       | Full pipeline: download → transcribe → segment → cluster → export |
| `./run.sh dataset all`       | `run` (all enabled sources) + full on-disk `merge` → `combined` |
| `./run.sh dataset sources`   | List `SOURCES` **and** export-ready folders on disk     |
| `./run.sh dataset datasets`  | List export-ready `data/datasets/*` only                |
| `./run.sh dataset cluster` / `speakers-cluster` | Cross-video embedding merge → `work/<source>/speaker_map.json` |
| `./run.sh dataset speakers-remap` | Manual `speaker_name` rewrite in an exported dataset |
| `./run.sh dataset merge`     | Merge all export-ready `datasets/*` → `combined` (use `--from-sources` for SOURCES-only) |
| `./run.sh dataset hf-import` | Import Hugging Face speech datasets                     |

Dataset settings use env prefixes: `SOURCES`, `AUDIO_INTEL_*`, `SEGMENTATION_*`, `QUALITY_*`, `EXPORT_*`, `PIPELINE_*`.

**Pipeline contract**

- `work/<source>/transcripts/*.json` = **full** audio-intel ASR (unfiltered).
- `segment` / `export` apply quality, language, duration, char, and junk gates.
- Junk filtering is **hardcoded**: any overlap with a non-speech `sound_event` drops the clip (requires `AUDIO_INTEL_SOUND_EVENTS=true` and server `SOUND_EVENTS_ENABLED=1`).
- `merge` (default) scans **all** export-ready dirs under `data/datasets/`, not only `SOURCES` — so HF imports like `uk-mix` are included automatically. Exclude the output slug (`-o combined`) so a previous merge is not fed into itself.
- Enabled `SOURCES` are always processed **sequentially**.

Transcription requests audio-intel with `align`, `diarize`, and `sound_events` (see
`AUDIO_INTEL_DIARIZE`, `AUDIO_INTEL_SOUND_EVENTS`). On the audio-intel host enable
`SPEAKERS_ENABLED=1` (needs `HF_TOKEN` for pyannote), `SOUND_EVENTS_ENABLED=1`, and
alignment if you use word scores. Diarized `speaker_id` values become **video-scoped**
local Fish names (`{EXPORT_SPEAKER_NAME}__{video_id}__{speaker_id}`). The `cluster`
step merges matching voices across videos of the same source into `{name}_s{k}` via
roster embeddings (`SPEAKER_CLUSTER_THRESHOLD`, default `0.75`). Export applies
`work/<source>/speaker_map.json`, then drops speakers under
`SPEAKER_CLUSTER_MIN_CLIPS` / `SPEAKER_CLUSTER_MIN_SPEECH_SEC` (defaults
`100` / `300`). Cross-channel merges stay manual
(`./run.sh dataset speakers-remap`). Merge keeps per-clip speakers so `train export`
can write separate `training/raw/{speaker}/` folders.

Keep `SEGMENTATION_MAX_CHARS` aligned with Fish training (default `220` in `.env.example`).

`SEGMENTATION_SAMPLE_RATE` should match the s2-pro codec (`44100`); lower values
throw away band the codec can still represent.

### Bilingual YouTube (UK/EN script split)

Monolingual Ukrainian data improves UK pronunciation, but does not teach
**EN reference → UK text** cloning: the model never sees the same speaker in
both languages. Bilingual lessons (script-split into EN/UK clips under one
`speaker_name`) supply those pairs; keep the overall mix UK-heavy via HF imports.
Details: [`docs/ukrainian-lora-playbook.md`](docs/ukrainian-lora-playbook.md#why-bilingual-uken-data-matters).

```bash
# .env
AUDIO_INTEL_LANGUAGE=auto
SEGMENTATION_SPLIT_BY_SCRIPT=true
SEGMENTATION_ALLOWED_LANGUAGES=uk,en
SEGMENTATION_FILTER_NON_TARGET_LANGUAGE=false
```

Aligned words are cut into Latin vs Cyrillic runs so English phrases and Ukrainian
explanations become separate clips (same speaker id). Latin-only tokens such as
`ACP-125` count as English. Russian LID / letters are dropped by the allowlist.

### Importing Hugging Face datasets

```bash
./run.sh dataset hf-import -o uk-mix \
  --source speech-uk/opentts-mykyta=mykyta \
  --source speech-uk/opentts-lada=lada \
  --source patriotyk/filatov_24000=filatov#40000 \
  --streaming --max-wer 0.15
```

Source spec: `repo_id[:config][@split]=speaker[#limit]`. Each source becomes its
own speaker, because Fish groups training prompts per speaker folder and samples
the reference clip from inside that group — flattening voices together teaches
the model to ignore the reference when cloning.

| Option             | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| `--probe`          | Report sample rates, durations and sample text, no import |
| `--streaming`      | Stream rows instead of downloading the whole repo         |
| `--max-wer`        | Drop rows whose dataset-provided `wer` is too high        |
| `--max-per-source` | Default clip cap (a `#limit` in the spec wins)            |

## Fine-tuning Fish Speech s2-pro

Stock s2-pro treats Ukrainian as a lower-priority language (Russian is higher tier), which can sound too Russian on Ukrainian text. This project adds a LoRA fine-tuning pipeline on your exported dataset.

Full checklist: [`docs/ukrainian-lora-playbook.md`](docs/ukrainian-lora-playbook.md).

```bash
./run.sh install all
cp .env.example .env
# Build datasets (YouTube + optional hf-import), then:
./run.sh dataset merge -o combined
# TRAINING_DATASET_ID=combined in .env
# Stop audio-intel first — it holds GPU VRAM needed for VQ / LoRA / vLLM

./run.sh bg-train all            # recommended: logs to data/logs/training.log
# Or foreground:
./run.sh train all
# Or step by step:
./run.sh train export   # writes .lab with stress marks
./run.sh train vq
./run.sh train protos
./run.sh train train
./run.sh train merge

# Test merged checkpoint:
./run.sh train infer \
  --text "Доброго дня!" \
  --speaker-wav data/datasets/combined/reference.wav \
  --speaker-text "Доброго дня!" \
  --out synthesized.wav
```

| Command                      | Description                                          |
| ---------------------------- | ---------------------------------------------------- |
| `./run.sh train export`      | Dataset → Fish `.wav` + stressed `.lab` under `training/raw/` |
| `./run.sh train vq`          | Extract semantic tokens with the stock s2-pro codec  |
| `./run.sh train protos`      | Pack tokens into protobuf shards                     |
| `./run.sh train train`       | LoRA fine-tune LLAMA weights → `training/runs/`      |
| `./run.sh train merge`       | Merge LoRA → `training/merged/`                      |
| `./run.sh train export-vllm` | Convert merged checkpoint for vLLM-Omni              |
| `./run.sh train infer`       | CLI synthesis with merged checkpoint                 |

Stock s2-pro weights under `checkpoints/fish-speech/` are never modified.

### Choosing LoRA targets

`TRAINING_LORA_TARGET_MODULES` decides which half of the Dual-AR model adapts:

| Target                                   | Trains                                             |
| ---------------------------------------- | -------------------------------------------------- |
| `attention`, `mlp`, `embeddings`, `output` | Slow text→semantic stack — pronunciation, prosody |
| `fast_*` counterparts                    | Acoustic decoder — timbre and delivery              |

Slow targets imply their `fast_*` counterpart. Fixing "Ukrainian that sounds
Russian" needs the slow targets; the fast-only default only reshapes acoustics.

`./run.sh train merge` regenerates a matching hydra LoRA config from these
settings, since upstream merges with `strict=True` and would otherwise fail on
mismatched target modules.

### Ukrainian stress marks

Ukrainian stress is lexical, so a model can only place it on word forms it
memorised. s2-pro honours a combining acute (U+0301) in its input and ignores the
spacing acute (U+00B4), which makes `ліхта́рик` a reliable way to correct words it
reads wrong — including domain vocabulary absent from any audiobook corpus.

Dataset export (`./run.sh dataset …` and `./run.sh train export`) marks
transcripts as it writes them: apostrophe normalisation → dictionary/Stanza →
lexicon (`configs/stress_lexicon.txt`) → acoustic fallback from the clip WAV fo
words still unmarked. Synthesis uses the same text pipeline without the acoustic
step (no aligned audio on the request). Marked text sounds natural only afte
fine-tuning on marked transcripts, so keep `STRESS_*` settings identical between
training and serving.

Changing stress settings (or the lexicon) means re-exporting — there is no
in-place backfill step:

```bash
./run.sh train export   # rewrites .wav/.lab (drops existing VQ .npy next to them)
./run.sh train vq
./run.sh train protos
# then retrain LoRA from the base checkpoint (continue is a weaker option)
```

`STRESS_ON_AMBIGUITY=skip` leaves unresolved heteronyms unmarked.
`STRESS_DISAMBIGUATION=stanza` resolves them from context (~500 MB Stanza
models, forced onto CPU via `STRESS_PREFER_CPU` so the TTS GPU stays free).

### Monitoring a run

```bash
./run.sh bg-train            # training → data/logs/training.log
./run.sh tensorboard start   # curves on :6006
./run.sh logs training       # tail the log
./run.sh status              # datasets, checkpoints, GPU
```

To serve the fine-tuned model via HTTP:

```bash
# .env
FISH_SPEECH_USE_FINETUNED=true
```

Training hyperparameters: `TRAINING_*` variables in `.env`.

## Code layout

Single Python package `fish_studio`:

```
src/fish_studio/
  config.py, paths.py, cli.py, synthesis.py   # shared config and types
  dataset/          # YouTube/local → transcribe → segment → export
  training/         # LoRA pipeline (export, vq, protos, train, merge, infer)
  runtime/          # inference checkpoint paths, vLLM deploy helpers
  server/           # FastAPI HTTP server + vLLM proxy
```

## Tests

```bash
pip install -e ".[training,dev]"
pytest
```

## Configuration

Copy `.env.example` → `.env`. All settings are environment variables:

| Prefix / variable                                          | Purpose                                       |
| ---------------------------------------------------------- | --------------------------------------------- |
| `DATA_ROOT`                                                | Work, datasets, checkpoints, training, logs   |
| `SOURCES`                                                  | JSON array of YouTube / local audio sources   |
| `AUDIO_INTEL_*`                                            | Remote transcription service                  |
| `SEGMENTATION_*` / `QUALITY_*` / `EXPORT_*` / `PIPELINE_*` | Dataset pipeline tuning                       |
| `TRAINING_*`                                               | Fish Speech s2-pro LoRA fine-tuning           |
| `INFERENCE_*`                                              | HTTP server bind address and upload limit     |
| `FISH_SPEECH_*`                                            | vLLM-Omni serving + training checkpoint paths |

Docker Compose reads the same `.env` (`docker compose up`) and overrides `DATA_ROOT`
and `FISH_SPEECH_BASE_URL` so the `server` container reaches the `vllm` container.

### Data layout (`DATA_ROOT`, default `./data`)

| Path                       | Contents                            |
| -------------------------- | ----------------------------------- |
| `work/{source}/`           | Downloads, transcripts, segments    |
| `datasets/{source}/`       | Exported training datasets          |
| `checkpoints/fish-speech/` | Fish Speech s2-pro weights          |
| `training/raw/`            | Fish export (`.wav` + `.lab`)       |
| `training/protos/`         | Protobuf shards for training        |
| `training/runs/`           | LoRA training checkpoints           |
| `training/merged/`         | Merged Fish Speech weights          |
| `training/vllm/`           | HF layout for vLLM-Omni             |
| `logs/`                    | Pipeline, server, and training logs |

## License

MIT
