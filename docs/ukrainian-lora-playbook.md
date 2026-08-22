# Ukrainian LoRA playbook (end-to-end)

This is the full path from bilingual YouTube + Hugging Face speech data to a
Fish Speech s2-pro LoRA that speaks better Ukrainian (and keeps EN/UK pairs fo
cross-lingual cloning experiments).

All commands are run from the repo root after `./run.sh install all`.

## Why bilingual UK/EN data matters

Stock s2-pro already speaks English well and clones from an English reference
into English text. The weak spot for this project is the other direction:

**English reference → Ukrainian text** (cross-lingual clone), plus Ukrainian that
does not drift toward Russian.

A monolingual Ukrainian corpus (HF `uk-mix`, audiobooks, news) teaches
pronunciation and stress, but every training prompt is “UK audio ↔ UK text”.
The model never sees the *same voice* saying English and Ukrainian in matched
conditions, so at inference time an EN reference gives little usable speake
conditioning for UK synthesis.

Bilingual teaching channels fix that:

- One stable speaker per channel speaks **both** languages in the same room,
  mic, and style.
- Script-split turns mixed lessons into separate EN and UK clips that still
  share the same Fish `speaker_name`.
- LoRA can learn “this timbre in EN reference ↔ this timbre speaking UK”, which
  is exactly the cloning path you care about in production.

You still want a **UK-heavy** mix overall (HF imports dominate hours) so the
LoRA specializes Ukrainian. Keep bilingual YouTube as the minority that carries
same-speaker EN/UK pairs — quality of pairing beats raw EN hours.

## 0. One-time setup

```bash
./run.sh install all
cp .env.example .env
# Edit .env (see recommended blocks below)
```

You also need a running [audio-intel](https://github.com/thesydoruk/audio-intel)
ASR service (default `http://127.0.0.1:8081`) with:

| audio-intel `.env` | Why |
| --- | --- |
| `WORD_ALIGN_ENABLED=1` | Word timestamps for segmentation / script split |
| `SPEAKERS_ENABLED=1` + valid `HF_TOKEN` | Diarization → per-speaker Fish folders |
| `SOUND_EVENTS_ENABLED=1` | Non-speech timeline for junk filter |
| `ASR_PARALLEL_WORKERS=1` (or 2 max) | With PANNs, higher parallelism often OOMs on 24–32 GB GPUs |

Confirm:

```bash
curl -s http://127.0.0.1:8081/health
./run.sh status
```

## 1. Configure sources and bilingual segmentation

In project `.env`:

```bash
# Example YouTube teachers (one stable voice per channel works best)
SOURCES=[
  {"id":"channel-a","kind":"youtube","enabled":true,"url":"https://www.youtube.com/@CHANNEL_A/videos","speaker_name":"speaker_a"},
  {"id":"channel-b","kind":"youtube","enabled":true,"url":"https://www.youtube.com/@CHANNEL_B/videos","speaker_name":"speaker_b"}
]

AUDIO_INTEL_BASE_URL=http://127.0.0.1:8081
AUDIO_INTEL_LANGUAGE=auto          # bilingual lessons
AUDIO_INTEL_ALIGN=true
AUDIO_INTEL_DIARIZE=true
AUDIO_INTEL_SOUND_EVENTS=true      # required for junk filtering

SEGMENTATION_FILTER_NON_TARGET_LANGUAGE=false   # do not drop EN when language=auto
SEGMENTATION_SPLIT_BY_SCRIPT=true               # cut Latin vs Cyrillic into separate clips
SEGMENTATION_ALLOWED_LANGUAGES=uk,en            # drop RU and other LID tags
SEGMENTATION_SAMPLE_RATE=44100                  # match s2-pro codec

# Cross-video speaker merge within each SOURCES channel (embedding cosine).
SPEAKER_CLUSTER_THRESHOLD=0.75
SPEAKER_CLUSTER_MIN_CLIPS=100
SPEAKER_CLUSTER_MIN_SPEECH_SEC=300

TRAINING_DATASET_ID=combined
TRAINING_PROJECT_NAME=fish-uk-stress-v3
```

Notes:

- **Transcripts are full ASR JSON.** Filtering (junk / language / duration / chars)
  happens only at `segment` → `export`. You can re-segment without re-downloading.
- **Junk filter is not configurable:** any clip that overlaps a non-speech
  `sound_event` is dropped. Keep `SOUND_EVENTS_EXCLUDE_SPEECH=1` on audio-intel.
- **Speaker IDs are video-scoped**, then clustered: segment writes
  `{speaker}__{video_id}__{spk_N}`; the `cluster` step builds
  `work/<source>/speaker_map.json` via roster embeddings; export remaps to
  `{speaker}_s{k}`. Older transcripts without embeddings fall back to merging
  by diarization id suffix (same `spk_0` across videos → `_s0`) until you
  re-transcribe with diarize so audio-intel stores embeddings. Export then
  drops speakers below `SPEAKER_CLUSTER_MIN_CLIPS` **and**
  `SPEAKER_CLUSTER_MIN_SPEECH_SEC` (defaults 100 clips / 300 s).
- Cross-channel merges are **manual**: `./run.sh dataset speakers-remap -i map.json --dataset combined`.
- Sources in `SOURCES` are processed **sequentially** by `fish-dataset run`.

## 2. Build YouTube datasets

```bash
# All enabled SOURCES, then merge everything export-ready on disk
./run.sh dataset-build

# Or step by step:
./run.sh dataset run                          # download → transcribe → segment → cluster → export
./run.sh dataset sources                      # SOURCES + on-disk exports
./run.sh dataset datasets                     # only on-disk export-ready folders
```

Resume after a crash (skips completed transcripts unless `--force`):

```bash
./run.sh dataset run --source channel-a --step transcribe --step segment --step cluster --step export
```

Force a cleaner re-cut after changing junk / script-split / speaker naming (no new ASR):

```bash
./run.sh dataset run --source channel-a --step segment --step cluster --step export --force
```

Rebuild YouTube speakers after upgrading the video-scoped + cluster pipeline
(requires roster embeddings in transcripts):

```bash
./run.sh dataset run --source channel-a --step segment --step cluster --step export --force
./run.sh dataset merge -o combined
```

## 3. Import monolingual Ukrainian (Hugging Face)

HF imports are **not** listed in `SOURCES`. They still land unde
`data/datasets/<id>/` and are included by the default merge.

```bash
./run.sh dataset hf-import -o uk-mix \
  --source speech-uk/opentts-mykyta=mykyta \
  --source speech-uk/opentts-lada=lada \
  --source speech-uk/opentts-tetiana=tetiana \
  --source speech-uk/opentts-oleksa=oleksa \
  --source speech-uk/opentts-kateryna=kateryna \
  --source patriotyk/filatov_24000=filatov#40000 \
  --streaming --max-wer 0.15
```

Adjust speaker list / limits to what you have access to. Use `--probe` first if unsure.

## 4. Merge into one giant training set

```bash
# Default: every export-ready folder under data/datasets/ except -o
./run.sh dataset merge -o combined

# Legacy: only enabled SOURCES (skips uk-mix and other imports)
./run.sh dataset merge --from-sources -o combined
```

`dataset-build` / `dataset all` already call this full on-disk merge after the
YouTube pipeline (unless `--merge-from-sources`).

Expect disk use ≈ sum of input `datasets/*/wavs` (files are copied).

## 5. Fine-tune LoRA

**Stop audio-intel first** — it holds a large chunk of VRAM. Training and vLLM
need the GPU.

```bash
# on the audio-intel host
pkill -f 'python -m audio_intel.server' || true

# project .env
# TRAINING_DATASET_ID=combined
# TRAINING_PROJECT_NAME=fish-uk-stress-v3   # new name = new run directory
# TRAINING_CONTINUE_PATH=                  # empty = train from base s2-pro

./run.sh bg-train all          # export → vq → protos → train → merge
./run.sh logs training
./run.sh status
./run.sh tensorboard start     # optional :6006
```

Foreground alternative: `./run.sh train all`.

Steps inside `train all`:

| Step | Output |
| --- | --- |
| `export` | `training/raw/{speaker}/*.wav` + stressed `.lab` |
| `vq` | `.npy` semantic tokens beside each wav |
| `protos` | `training/protos/` shards |
| `train` | `training/runs/<TRAINING_PROJECT_NAME>/` |
| `merge` | `training/merged/` standalone weights |

Keep `STRESS_*` identical between training and later serving.

## 6. Smoke-test and serve

```bash
./run.sh train infer \
  --text "Доброго дня! Вартість квитка 150 грн." \
  --speaker-wav data/datasets/combined/reference.wav \
  --speaker-text "Доброго дня!" \
  --out /tmp/smoke.wav

./run.sh train export-vllm          # → data/training/vllm/
# .env: FISH_SPEECH_MODEL=training/vllm
#       FISH_SPEECH_USE_FINETUNED=true
./run.sh vllm restart
./run.sh serve
```

## 7. Practical expectations

- **UK-heavy `combined` is normal** when `uk-mix` is included (often ~80–90% UK
  by duration). That is desirable for Ukrainian LoRA. Bilingual YouTube channels
  still provide same-speaker EN+UK pairs for EN-reference cloning experiments.
- Latin tokens like `ACP-125` / `PTSD` are classified as **EN** by script split.
- Teaching channels produce repeated filler phrases; duration/char gates in
  `.env` control how short a clip may be. There is no template-dedupe filter.
- With sound events enabled, prefer **low ASR parallelism** to avoid CUDA OOM.

## 8. Useful commands cheatsheet

```bash
./run.sh status
./run.sh dataset sources
./run.sh dataset datasets
./run.sh dataset merge -o combined
./run.sh bg-train all
./run.sh logs training
./run.sh train infer --text "…" --speaker-wav ref.wav --out out.wav

# Inspect quality helpers (optional)
./run.sh analyze transcripts data/work/<source>/transcripts
./run.sh analyze clips data/work/<source>/segments
```

## 9. Data layout reminde

| Path | Role |
| --- | --- |
| `data/work/<source>/downloads` | Raw audio (keep / archive) |
| `data/work/<source>/transcripts` | Full ASR JSON (unfiltered) |
| `data/work/<source>/segments` | Filtered training clips |
| `data/datasets/<source>/` | Per-source export |
| `data/datasets/combined/` | Merged training set |
| `data/training/raw/` | Stressed wav+lab for Fish |
| `data/training/runs/` | LoRA checkpoints |
| `data/training/merged/` | Merged standalone weights |
| `data/training/vllm/` | HF layout for vLLM-Omni |
| `data/logs/training.log` | `bg-train` log |

## 10. Hardware and disk (ballpark)

| Stage | GPU | Disk |
| --- | --- | --- |
| audio-intel (align + diarize + PANNs) | ~10–20+ GB VRAM; keep `ASR_PARALLEL_WORKERS=1` | downloads + transcripts (tens–hundreds of GB for multi-channel YouTube) |
| `dataset merge` | CPU | ≈ sum of input `datasets/*/wavs` (copy) |
| `train export` / `vq` / LoRA | prefer free GPU (stop audio-intel) | `training/raw` ≈ dataset size; checkpoints ~100 MB each if `SAVE_TOP_K=-1` |

A bilingual multi-channel + HF `uk-mix` merge on the order of **~200k clips / ~200 h** is a realistic large run; smaller subsets still work for smoke tests.

## 11. Troubleshooting

| Symptom | What to do |
| --- | --- |
| CUDA OOM / hung VRAM during ASR | Set audio-intel `ASR_PARALLEL_WORKERS=1`, restart the server, resume `dataset run` (completed transcripts are skipped) |
| Music/intro still in clips | Confirm `SOUND_EVENTS_ENABLED=1` on audio-intel and `AUDIO_INTEL_SOUND_EVENTS=true`; re-run `--step segment --step export --force` (no re-ASR) |
| EN phrases glued to UK text | `SEGMENTATION_SPLIT_BY_SCRIPT=true`, `AUDIO_INTEL_LANGUAGE=auto`, re-segment with `--force` |
| Merge missing `uk-mix` | Do **not** pass `--from-sources`; default merge scans all export-ready folders |
| Training OOM | Lower `TRAINING_BATCH_SIZE`, raise `TRAINING_GRAD_ACCUM`; stop audio-intel / vLLM |
| New LoRA run overwrites old | Change `TRAINING_PROJECT_NAME` (each name → `training/runs/<name>/`) |
| `./run.sh status` shows audio-intel running | Stop it before `bg-train` / `vllm` so VRAM is free |

Optional quality spot-checks:

```bash
./run.sh analyze transcripts data/work/<source>/transcripts
./run.sh analyze clips data/work/<source>/segments
```

Archiving `downloads/` + `transcripts/` (zip to NAS, etc.) is optional ops hygiene — not required by the pipeline. Re-segment only needs transcripts + downloads on disk.
