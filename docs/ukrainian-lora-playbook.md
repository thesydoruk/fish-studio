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
TRAINING_PROJECT_NAME=fish-uk
TRAINING_LORA_TARGET_MODULES=attention,mlp,embeddings
TRAINING_MERGE_SCALE=0.5
```

Notes:

- **Transcripts are full ASR JSON.** Filtering (junk / language / duration / chars)
  happens only at `segment` → `export`. You can re-segment without re-downloading.
- **Junk filter is not configurable:** any clip that overlaps a non-speech
  `sound_event` is dropped. Keep `SOUND_EVENTS_EXCLUDE_SPEECH=1` on audio-intel.
- **Speaker IDs are video-scoped**, then clustered: segment writes
  `{speaker}__{video_id}__{spk_N}`; the `cluster` step builds
  `work/<source>/speaker_map.json` via roster embeddings; export remaps to
  `{speaker}_s{k}`. Export then drops speakers below
  `SPEAKER_CLUSTER_MIN_CLIPS` **and** `SPEAKER_CLUSTER_MIN_SPEECH_SEC`
  (defaults 100 clips / 300 s).
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
  --source patriotyk/filatov_24000=filatov \
  --streaming --max-wer 0.15
```

Adjust speaker list / limits to what you have access to. Use `--probe` first if unsure.

HF corpora commonly pad every clip with 1–2 s of digital silence on both sides
(`opentts-lada` clips measured ~50% silence). `SEGMENTATION_TRIM_SILENCE=true`
strips it before `loudnorm`, keeping `SEGMENTATION_TRIM_SILENCE_KEEP_SEC` of
margin; pauses inside an utterance survive. Without that trim the acoustic
decoder learns to fade out mid-utterance.

## 4. Merge into one giant training set

```bash
# Every export-ready folder under data/datasets/ except -o
./run.sh dataset merge -o combined

# Only enabled SOURCES (skips uk-mix and other imports)
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
# TRAINING_PROJECT_NAME=fish-uk   # new name = new run directory
# TRAINING_CONTINUE_PATH=                  # empty = train from base s2-pro
# TRAINING_LORA_TARGET_MODULES=attention,mlp,embeddings
# TRAINING_MERGE_SCALE=0.5

./run.sh bg-train all          # export → vq → protos → train → merge
./run.sh logs training
./run.sh status
./run.sh tensorboard start     # optional :6006
```

Foreground alternative: `./run.sh train all`.

One training pass covers both the slow linears and the text table. Semantic-id
rows of `embeddings` stay frozen; the position gate keeps the system/ref+VQ
prefix stock during train.

Steps inside `train all`:

| Step | Output |
| --- | --- |
| `export` | `training/raw/{speaker}/*.wav` + stressed `.lab` |
| `vq` | `.npy` semantic tokens beside each wav |
| `protos` | `training/protos/` shards |
| `train` | `training/runs/<TRAINING_PROJECT_NAME>/` |
| `merge` | `training/merged/` at `TRAINING_MERGE_SCALE` (default 0.5) |

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

### Never overwrite weights under a live vLLM

vLLM keeps the served `model.safetensors` mapped from disk. If you run
`export-vllm` (or otherwise overwrite that file) while vLLM is still running,
the process can silently start reading a mix of old and new pages. Cloning then
goes wrong in weird ways — wrong gender, wrong speaker — with no error in the
logs, and it stays broken until you restart vLLM.

Safe sequence when swapping a checkpoint:

1. **Stop** vLLM (`./run.sh vllm stop` or `docker compose stop vllm`).
2. Export / copy the new weights into the serve path.
3. **Start** vLLM again (`./run.sh vllm start` / `docker compose up -d vllm`).

Or export into a **new directory**, point `FISH_SPEECH_MODEL` at it, then
restart. Do not `cp` / `save` on top of the file a live server is reading —
especially dangerous during a long synthesis batch.

## 6b. Merge scale

Do **not** serve a raw fold. One slow pass (`attention,mlp,embeddings`,
position-gated, semantic ids frozen) at full scale gives strong Ukrainian and
destroys in-context clone. The same adapter at a lower dose keeps both sides.

`./run.sh train merge` already writes that dose:

```
W = stock + TRAINING_MERGE_SCALE × (ft − stock)
```

Default `TRAINING_MERGE_SCALE=0.5`. Scale axis, measured on-ear:

- **0.4** — more clone, weaker UA
- **0.5** — people clone well, UA clearly better than stock
- **0.6** — more UA; unusual voices start to slip toward the dataset

Do not raise the blend to force a harder accent — that is the same knob that
kills speaker identity.

To try another scale, re-merge the same LoRA:

```bash
./run.sh train merge --merge-scale 0.4
```

Then `export-vllm` into a **new** directory (stop vLLM first; never overwrite
a live `model.safetensors`):

```bash
./run.sh vllm stop
./run.sh train export-vllm --output data/training/vllm
# FISH_SPEECH_MODEL=training/vllm
./run.sh stack start
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
| `data/training/merged/` | Merged standalone weights (already at `TRAINING_MERGE_SCALE`) |
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
| Synthesis fades away and volume leveling does not rescue it | Measure it: `length-check` reports `quiet=NNdB` against the reference. Past ~20 dB the adapter has moved the codebook embeddings off the audio decoder's own vocabulary — see §14 |
| Synthesis trails off or the end is cut | Compare `SEGMENTATION_MAX_DURATION_SEC` against `FISH_SPEECH_CHUNK_LENGTH / 14`; a LoRA trained only on clips shorter than one request learns to stop before the line ends |
| Cloning "sounds better" but long lines are unusable | Screen every checkpoint with `length-check` before listening — the damage is invisible on short samples |
| LoRA has no audible effect | Multiply `TRAINING_MAX_STEPS` by `TRAINING_BATCH_SIZE` and compare with the clip count: at batch 2, 2000 steps is 1.8% of one pass over 220k clips. Then check `TRAINING_LR` — at `3e-6` even 36k steps barely move `val/loss` |
| Stress or pronunciation barely moves with more steps | Raise `TRAINING_LORA_R` / `TRAINING_LORA_ALPHA`; `r=8` is thin for reshaping the phonetics of a whole language |
| `./run.sh status` shows audio-intel running | Stop it before `bg-train` / `vllm` so VRAM is free |

Optional quality spot-checks:

```bash
./run.sh analyze transcripts data/work/<source>/transcripts
./run.sh analyze clips data/work/<source>/segments
```

## 12. Clip length is a training target, not a detail

A LoRA also learns *how long an utterance lasts*. Train it on clips whose median
is 3 s and whose longest is 12 s, then ask for 200 characters — about 14 s of
Ukrainian at ~14 chars/s — and the adapter will cut the line off somewhere in
the middle. The failure is stochastic, so it looks like "synthesis sometimes
breaks" rather than a length problem, and it gets worse the longer you train,
because every step sharpens the model's belief about where an utterance ends.

Keep the two ends of the pipeline in agreement:

```
SEGMENTATION_MAX_DURATION_SEC >= FISH_SPEECH_CHUNK_LENGTH / 14
SEGMENTATION_MAX_CHARS        >= SEGMENTATION_MAX_DURATION_SEC * 14
SEGMENTATION_MERGE_GAP_SEC    >= 0.5   # or a sentence pause blocks every merge
```

Segments are joined into longer clips by `merge_gap_sec` at segmentation time,
so widening these limits and re-running `--step segment` is enough — no
re-transcription. Delete `work/<source>/segments/` first: clip ids are
positional, and the exporter skips a clip whose WAV already exists, so a rerun
would otherwise pair new text with stale audio.

HF corpora (`uk-mix`) ship fixed short clips and cannot be lengthened; a mixture
is fine, as long as the long end of the distribution exists at all.

## 13. Screening a checkpoint before you listen

A checkpoint can clone better and still be unusable, so judge it by measurement,
not by ear:

```bash
./run.sh server length-check --label step250 \
  --ref data/training/raw/<speaker>/000002.wav \
  --ref-text-file data/training/raw/<speaker>/000002.lab
```

It synthesizes texts of growing length and reports two things, both graded on
the *worst* sample: how much of the expected duration came back, and how far
below the reference clip the output sits (`quiet=NNdB`). Both failures are
intermittent, so an average stays respectable long after the model is unusable.
Exit status is non-zero below `--fail-under` (default `0.70`) or above
`--max-level-drop` (default `12` dB). Use one frozen reference clip for every
checkpoint, otherwise the numbers are not comparable.

## 14. Why an adapter can wreck the audio decoder

`loralib` seeds its two wrappers as mirror images, and only one of them is
scaled:

```
Linear:     lora_A = kaiming_uniform  (‖A‖ ≈ 3.3 at r=32, d=2560),  lora_B = 0
Embedding:  lora_A = 0,                lora_B = normal(0, 1)  →  ‖B‖ ≈ 287
```

Both start at a zero update, so nothing looks wrong. But each factor's gradient
is proportional to the other, so the embedding's factor moves under a matrix
ninety times larger than the linear one's, and its update lands large from the
first step rather than growing into place. Against released s2-pro weights at
`lr=5e-5`, `r=32`, `alpha=64`, `codebook_embeddings` drifts an order of
magnitude faster than the transformer matrices:

| matrix | ‖ΔW‖ / ‖W‖ at step 500 | at step 12000 |
| --- | --- | --- |
| `codebook_embeddings` (audio decoder) | 17.7% | 48.8% |
| `fast_embeddings` | 16.9% | 22.1% |
| text `embeddings` | 4.7% | 14.8% |
| every transformer matrix, both stacks | 0.7–2.5% | 1.8–4.3% |

`codebook_embeddings` turns an acoustic code index into a vector for the audio
decoder. Move it far enough and the decoder no longer agrees with the tokens
the model emits — quiet, rate-scrambled audio that sounds like fading.

Three patches in `training/lora_patch.py` keep fine-tuning aligned with serve:

0. **VQ embedding scale.** s2-pro inference divides each VQ embedding by
   `sqrt(num_codebooks+1)` ≈ 3.3 (`scale_codebook_embeddings`). Upstream
   training skips that division, so a fine-tune would otherwise optimise
   acoustic inputs 3.3× larger than the served model sees.
   `patch_scaled_codebook_embed` applies the inference scale during training.

1. **Embedding LoRA scale.** `_rescale_embedding_lora` reseeds `lora_B` at the
   Kaiming scale so one learning rate means the same thing for every adapted
   matrix. Compare any checkpoint against the released weights and confirm no
   group is an order of magnitude out of line with the others.

2. **Tied logit head.** s2-pro ties text `embeddings` to the token logits
   (`tie_word_embeddings=True`, no separate `output` module):
   `F.linear(slow_out, embeddings.weight)`. That read bypasses LoRA, so without
   a patch the delta trains on the input lookup only and merging shifts every
   token logit — including semantic ids and `im_end` — in a way training never
   saw. `patch_tied_embedding_logits` adds the same delta to the logits during
   training so the trained function matches the merged checkpoint.

Default `TRAINING_LORA_TARGET_MODULES=attention,mlp,embeddings` is one pass
over the slow stack. `codebook_embeddings` and `fast_*` must be listed
separately to adapt the acoustic codebook / decoder. `output` matches nothing
on s2-pro: the head it would adapt does not exist as a separate matrix (it
*is* the embedding table).

Archiving `downloads/` + `transcripts/` off-box is optional. Re-segment only needs those two trees on disk.
