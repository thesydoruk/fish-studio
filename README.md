# Fish Studio

HTTP TTS server with voice cloning via [Fish Speech](https://github.com/fishaudio/fish-speech) s2-pro, plus a dataset builder and LoRA fine-tuning pipeline for Ukrainian.

**Author:** [Valerii Sydoruk](https://github.com/thesydoruk)

The HTTP server proxies synthesis to an external [vLLM-Omni](https://github.com/vllm-project/vllm-omni) process.

> **Built with Fish Audio.** Fish Speech and the s2-pro weights are **not** MIT.
> They are under the [Fish Audio Research License](licenses/FISH-AUDIO-RESEARCH-LICENSE.txt)
> (research and non-commercial use). Commercial use of those materials needs a
> separate license from [Fish Audio](https://fish.audio). See [NOTICE](licenses/NOTICE).

> **Ukrainian LoRA from scratch:** follow the step-by-step playbook
> [`docs/ukrainian-lora-playbook.md`](docs/ukrainian-lora-playbook.md)
> (audio-intel → bilingual YouTube → HF import → full `combined` merge → `bg-train` → serve).

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (recommended)
- Fish Speech `s2-pro` checkpoints (downloaded by `./run.sh install all`)

## Quick start

```bash
./run.sh install all
cp .env.example .env
./run.sh vllm install
./run.sh stack          # vLLM + HTTP proxy in the background
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
| `speaker_wav` / `speaker_wav_b64` | yes      | —                    | Voice clone clip; **first file** is also the timing slot target |
| `speaker_text`                    | no       | —                    | Reference transcript (recommended) |
| `language`                        | no       | `INFERENCE_LANGUAGE` | Language code                      |
| `match_timing`                    | no       | `true`               | Fit the first ref slot: pauses, then rate-limited PSOLA tempo |

Sampling (`temperature`, `top_p`, `repetition_penalty`) is fixed at vLLM startup
in `configs/fish_speech_deploy.yaml`, not per request.

Up to 5 `speaker_wav` files are concatenated into one prompt in order. The model
reads that prompt as a single voice, so every clip after the first is scaled to
the first clip's speech loudness and a 0.25 s gap marks each seam — otherwise
the loudest clip pulls the cloned voice toward itself.

Each successful synthesize also dumps under `{DATA_ROOT}/logs/synthesis/` (see
`FISH_SPEECH_SYNTH_LOG`): `request.json`, `reference_*.wav`, `synth_raw.wav`
(before timing), `synth_final.wav` (returned audio). `LATEST` names the newest
request id. Keep count: `FISH_SPEECH_SYNTH_LOG_KEEP` (default 40).

Each chunk is retried up to `FISH_SPEECH_SYNTH_ATTEMPTS` times (default 5) if
the raw take is silence, a cutoff, or the ECAPA cosine vs the clone prompt is
below `FISH_SPEECH_VOICE_RETRY_BELOW` (default 0.3; short lines included).
`X-Voice-Similarity` is the weakest chunk score. A take still below the
threshold after every attempt is returned with `X-Synth-Warning`.

### Timing fit

The dub must land in the original line's wall-clock slot, so `match_timing`
spends the cheapest artifact first: trim the edges, edit phrase pauses toward
the reference pause budget, then Praat PSOLA, then shrink pauses once more for
the remainder. `praat-parselmouth` is required — the server refuses to load
without it.

Tempo is compared as **syllables per second of active speech**, each side
against its own text — raw seconds are meaningless across languages, since a
Ukrainian line carries more syllables than the English original and matching
absolute speech duration would compress it to an impossible rate. The slot says
how much speed-up is wanted, the band 4–5.6 syl/s says how much is allowed, and
Tempo stretch only speeds up, never slows down, and stops at 1.25×. A take that
drags is sped up to the band floor even when the slot has room — unless the
original is itself unhurried, in which case the floor yields and the delivery
is left alone.

Whatever still does not fit stays unfit: `request.json` carries a `timing` block
(`stretch_rate`, `syl_per_sec_*`, `overrun_sec`) and the server logs
`needs_shorter_line` — the signal that the translation, not the audio, is too
long for that slot.

Right after synthesis, the take goes through ffmpeg ``dynaudnorm`` so ragged
syllable loudness is evened *before* pause detection and tempo. After timing,
a single speech-gated LUFS gain matches the first ``speaker_wav`` — the same
linear match as before. ffmpeg ``loudnorm`` is not used here: a 1–6 s line
makes integrated ``I`` unstable, and dual-pass would recompress the range
``dynaudnorm`` just evened. Metrics land in ``request.json`` under
``speech_norm`` and ``loudness``.

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

**Do not overwrite `model.safetensors` while vLLM is running.** The server
memory-maps that file; an in-place rewrite can silently corrupt cloning (wrong
speaker/gender) until restart. Stop vLLM first, or export to a new directory
and point `FISH_SPEECH_MODEL` at it, then restart.

## Project scripts

Single entry point: **`./run.sh`**

| Command                     | Description                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| `./run.sh install [target]` | Create venv, install deps, download checkpoints                                             |
| `./run.sh server`           | Start Fish Speech TTS HTTP server                                                           |
| `./run.sh stack [cmd]`      | Whole stack in the background: `start` (default), `stop`, `restart`, `status`               |
| `./run.sh vllm <cmd>`       | vLLM-Omni server: `install`, `start`, `stop`, `restart`, `status`                           |
| `./run.sh dataset <cmd>`    | Dataset builder (`run`, `merge`, `all`, `sources`, `datasets`, `hf-import`, …)               |
| `./run.sh dataset-build`    | Run all enabled sources + merge (shortcut)                                                  |
| `./run.sh train <step>`     | Fish Speech LoRA: `export`, `vq`, `protos`, `train`, `merge`, `export-vllm`, `infer`, `all` |
| `./run.sh bg-train`         | LoRA training in background (`data/logs/training.log`)                                      |
| `./run.sh tensorboard`      | TensorBoard for training runs: `start`, `stop`, `status`                                    |
| `./run.sh server length-check` | Probe the served model for early termination (see the playbook)                          |
| `./run.sh server uk-eval`   | Score the served model: stress, «р», «и/і», pitch range, clone on held-out voices (see the playbook) |
| `./run.sh server start`     | Background server → `data/logs/server.log`                                                  |
| `./run.sh status`           | Datasets, checkpoints, server, GPU, audio-intel warning                                     |
| `./run.sh logs <name>`      | `server`, `training`, `pipeline`                                                            |
| `./run.sh analyze <kind>`   | Quality helpers: `transcripts` \| `clips`                                                   |
| `./run.sh synthesize`       | Quick HTTP synthesis test                                                                   |
| `./run.sh init`             | Create `.env` from `.env.example`                                                           |

`install` targets: `server` (default), `dataset`, `training`, `all`, `dev`

## Dataset builder

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
  --source patriotyk/filatov_24000=filatov \
  --streaming --max-wer 0.15
```

Source spec: `repo_id[:config][@split]=speaker`. Each source becomes its
own speaker, because Fish groups training prompts per speaker folder and samples
the reference clip from inside that group — flattening voices together teaches
the model to ignore the reference when cloning.

Many HF corpora pad clips with 1–2 s of digital silence on both sides, which
trains the acoustic decoder to fade out. `SEGMENTATION_TRIM_SILENCE=true`
(default) strips leading/trailing silence before `loudnorm` while leaving pauses
inside an utterance alone.

| Option             | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| `--probe`          | Report sample rates, durations and sample text, no import |
| `--streaming`      | Stream rows instead of downloading the whole repo         |
| `--max-wer`        | Drop rows whose dataset-provided `wer` is too high        |

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

# Serve and score it:
./run.sh train export-vllm --output data/training/vllm
# .env: FISH_SPEECH_MODEL=training/vllm
./run.sh stack restart
./run.sh server uk-eval --label vllm
```

| Command                      | Description                                          |
| ---------------------------- | ---------------------------------------------------- |
| `./run.sh train export`      | Dataset → Fish `.wav` (hard links) + stressed `.lab` under `training/raw/`, in `TRAINING_EXPORT_NUM_WORKERS` processes |
| `./run.sh train vq`          | Extract semantic tokens with the stock s2-pro codec  |
| `./run.sh train protos`      | Pack tokens into protobuf shards                     |
| `./run.sh train train`       | LoRA fine-tune LLAMA weights → `training/runs/`      |
| `./run.sh train merge`       | Fold LoRA, keep the `TRAINING_MERGE_SCALE_FOR` groups → `training/merged/` |
| `./run.sh train export-vllm` | Convert merged checkpoint for vLLM-Omni              |

Stock s2-pro weights under `checkpoints/fish-speech/` are never modified.

### Choosing LoRA targets

`TRAINING_LORA_TARGET_MODULES` decides which half of the Dual-AR model adapts:

| Target                                   | Trains                                             |
| ---------------------------------------- | -------------------------------------------------- |
| `attention`, `mlp`, `embeddings`         | Slow text→semantic stack — pronunciation, prosody. `embeddings` is the text table only; `codebook_embeddings` is separate |
| `mlp_w2`                                 | Only the `w2` projection of each slow MLP           |
| `fast_*` counterparts                    | Acoustic decoder — timbre and delivery              |

Default is `mlp_w2,embeddings`; the wider `attention,mlp,embeddings` pass
trains more and costs more clone (attention is the circuit that reads the
prompt). Semantic-id
rows of the text table stay frozen; the position gate keeps the system/ref
prefix stock during train. List `fast_*` explicitly to train the acoustic
decoder.

s2-pro ties the logit head to the embedding table, so the `embeddings` target
covers both: a training patch (`patch_tied_embedding_logits`) feeds the
adapter's delta into the tied logits during training, so the merged checkpoint
computes exactly the function that was trained.

`./run.sh train merge` regenerates a matching hydra LoRA config from these
settings, folds the adapter, then keeps only part of the fold. Each tensor
gets the dose of the first `TRAINING_MERGE_SCALE_FOR` group whose regex
matches its name; a tensor no group matches goes back to stock:

```
W = stock + scale × (ft − stock)
```

The default keeps the `w2` projections of slow layers 0–11 and the text table
at full dose and leaves everything else stock. That is where the pronunciation
of an `mlp_w2,embeddings` adapter lives; the late `w2` layers add none and cost
clone on voices the model never heard. Pass `--merge-scale-for PATTERN=SCALE`
(repeatable) to experiment, and measure the result with
`./run.sh server uk-eval` before serving it.

### Ukrainian stress marks

Ukrainian stress is lexical, so a model can only place it on word forms it
memorised. s2-pro honours a combining acute (U+0301) in its input and ignores the
spacing acute (U+00B4), which makes `ліхта́рик` a reliable way to correct words it
reads wrong — including domain vocabulary absent from any audiobook corpus.

Dataset export (`./run.sh dataset …` and `./run.sh train export`) marks
transcripts as it writes them: apostrophe normalisation → dictionary/Stanza
(heteronyms stay unmarked unless Stanza features pick a reading) → unambiguous
lexicon (`configs/stress_lexicon.txt`) → forced alignment of the clip WAV
(`STRESS_ACOUSTIC_FALLBACK`) for words still unmarked, written only where the
winning vowel beats the runner-up by `STRESS_ACOUSTIC_MARGIN`. Synthesis uses
the same text pipeline without the acoustic step (no aligned audio on the
request). Marked text sounds natural
only after fine-tuning on marked transcripts, so keep `STRESS_*` settings
identical between training and serving.

Changing stress settings (or the lexicon) means re-exporting — there is no
in-place backfill step:

```bash
./run.sh train export   # relinks .wav, rewrites .lab (drops existing VQ .npy next to them)
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

To serve the fine-tuned model, export it for vLLM and point `FISH_SPEECH_MODEL`
at the export (see *Fish Speech (vLLM-Omni)* above).

Training hyperparameters: `TRAINING_*` variables in `.env`.

## Code layout

Single Python package `fish_studio`:

```
src/fish_studio/
  config.py, paths.py, cli.py, synthesis.py   # shared config and types
  stress.py, stress_align.py                  # stress marking; CTC alignment and accent measures
  dataset/          # YouTube/local → transcribe → segment → export
  training/         # LoRA pipeline (export, vq, protos, train, merge, infer)
  runtime/          # inference checkpoint paths, vLLM deploy helpers
  server/           # FastAPI HTTP server + vLLM proxy; uk_eval.py scores a served model
scripts/            # ./run.sh entry points plus the uk-eval helpers (probe set,
                    # held-out voices, per-token vowel and clone diagnostics)
```

## Tests

```bash
./run.sh install dev          # pip install + protobuf override (see scripts/install.sh)
pytest
```

Use ``./run.sh install …``, not bare ``pip install -e .``. The install script
re-applies ``protobuf`` 4.x (audiotools wants 3.19; fish-speech protos need 4.x),
installs Stanza (HTTP stress disambiguation), and checks the imports. A raw
editable install drops those fixes and the server starts returning 500s.

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

This repository's original source is **MIT**, Copyright (c) 2026 Valerii Sydoruk.
See [LICENSE](LICENSE).

Fish Speech (`v2.0.0-beta`) and `fishaudio/s2-pro` are **Fish Audio Materials**
under the [Fish Audio Research License](licenses/FISH-AUDIO-RESEARCH-LICENSE.txt).
FARL allows research and non-commercial use only. LoRA adapters and merged
checkpoints this pipeline writes are derivative works of s2-pro and stay under
that license. Attribution required by FARL is in [NOTICE](licenses/NOTICE).

## Responsible use

- Clone only voices you have the right to use. Get consent for living speakers.
- YouTube / local sources in `SOURCES` must be legal for you to download and train on.
- Do not publish `.env`, tokens, or datasets that contain other people's speech without a license.

## Credits

- [Fish Speech](https://github.com/fishaudio/fish-speech) / [Fish Audio](https://fish.audio) — s2-pro model
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) — serving
- [audio-intel](https://github.com/thesydoruk/audio-intel) — transcription, alignment, diarization

## Contributing

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md). To report a vulnerability, use
[.github/SECURITY.md](.github/SECURITY.md).
