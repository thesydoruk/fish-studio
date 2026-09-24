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
TRAINING_LORA_TARGET_MODULES=mlp_w2,embeddings
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
# TRAINING_LORA_TARGET_MODULES=mlp_w2,embeddings

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
| `export` | `training/raw/{speaker}/*.wav` (hard links into the dataset) + stressed `.lab`; `TRAINING_EXPORT_NUM_WORKERS` processes, each with its own aligner — stop the TTS stack first when `STRESS_ACOUSTIC_DEVICE=cuda` |
| `vq` | `.npy` semantic tokens beside each wav |
| `protos` | `training/protos/` shards |
| `train` | `training/runs/<TRAINING_PROJECT_NAME>/` |
| `merge` | `training/merged/`: early `w2` + text table from the adapter, rest stock — see 6b |

Keep `STRESS_*` identical between training and later serving.

What is measured and what is not: the targets (`mlp_w2,embeddings`) and the
merge groups (section 6b) were chosen against numbers. The rank, alpha,
learning rate, step count and batch in `.env.example` are the values of the
one adapter those numbers were taken on; no sweep over them exists, so treat
them as a working point, not an optimum.

## 6. Smoke-test and serve

```bash
./run.sh train export-vllm          # → data/training/vllm/
# .env: FISH_SPEECH_MODEL=training/vllm
./run.sh stack restart
./run.sh synthesize -t "Доброго дня! Вартість квитка 150 грн." -w data/datasets/combined/reference.wav
./run.sh server uk-eval --label vllm   # section 13
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

## 6b. What the merge keeps

`./run.sh train merge` folds the adapter and then keeps only part of it. Each
tensor takes the dose of the first `TRAINING_MERGE_SCALE_FOR` group whose regex
matches its fish-native name; a tensor no group matches goes back to stock:

```
W = stock + scale × (ft − stock)
```

The default is the measured recipe for an `mlp_w2,embeddings` adapter: the
`w2` projections of slow layers 0–11 and the text table at full dose, every
other tensor stock. Pronunciation lives in those early layers; the late `w2`
layers add no pronunciation and cost clone on voices the model never heard.
Against a single 0.7 blend of the whole adapter this scores better on both
axes at once: on the frozen probe set stress placement 75.5% against 72.2%
(stock 59.0%, human recordings of the same lines 73.1%), and clone similarity
on six voices the model never heard 0.514 against 0.497 (stock 0.513). Dose on
the early layers does not matter: 0.7, 0.85 and 1.0 score the same, so the
default is 1.0.

To experiment, pass groups on the command line (they replace the default set):

```bash
./run.sh train merge \
  --merge-scale-for '^layers\.([0-9]|1[0-7])\.feed_forward\.w2\.=1.0' \
  --merge-scale-for '^embeddings\.=1.0'
```

Then `export-vllm` into a **new** directory (stop vLLM first; never overwrite
a live `model.safetensors`), and score it with `uk-eval` (section 13):

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
./run.sh server uk-eval --label <name>    # score the served model (section 13)

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
| `data/training/merged/` | Merged standalone weights (only the `TRAINING_MERGE_SCALE_FOR` groups of the fold) |
| `data/training/vllm/` | HF layout for vLLM-Omni |
| `data/logs/training.log` | `bg-train` log |

## 10. Hardware and disk (ballpark)

| Stage | GPU | Disk |
| --- | --- | --- |
| audio-intel (align + diarize + PANNs) | ~10–20+ GB VRAM; keep `ASR_PARALLEL_WORKERS=1` | downloads + transcripts (tens–hundreds of GB for multi-channel YouTube) |
| `dataset merge` | CPU | ≈ sum of input `datasets/*/wavs` (copy) |
| `train export` / `vq` / LoRA | free GPU (stop audio-intel and the TTS stack) | `training/raw` is hard-linked, ~0 extra; VQ `.npy` small; checkpoints ~100 MB each if `SAVE_TOP_K=-1` |

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

Once a checkpoint survives that, score its Ukrainian against the frozen probe
set (`configs/uk_probe.tsv`, 134 lines with human recordings of the same text):

```bash
HF_HUB_OFFLINE=1 ./run.sh server uk-eval --label <name> --json-out data/logs/uk_eval/<name>.json
```

One line per model, each axis with the human recording's own number beside it
as the ceiling (`h…`), never 100%:

| axis | what it measures |
| --- | --- |
| `stress` | stressed vowel against curated truth: CTC forced alignment (Ukrainian wav2vec2) gives each vowel its span, filled duration × energy picks the stressed one |
| `follow` | the same against the marks the server's own text pipeline put on the request |
| `voice` | ECAPA cosine to the reference clip (training speakers) |
| `--voices` | the same on speakers the model never saw: `data/eval/voices/index.tsv` from `scripts/extract_heldout_voices.py` |
| `gop`, `margin` | alignment confidence per letter; how far «г» beats «ґ» and «и» beats «і» |
| `palatal`, `trill` | «р»: F2 at the release into the vowel; trill duration, r/v level, closures, share of weak trills |
| `pitch` | F0 σ and 5–95 range in semitones |
| `vowel` | median F1/F2 of «и» and «і» and the distance between them |
| `rhotic` | share of «р» whose F3 drops below 0.8 of the following vowel's: the English approximant [ɹ] a listener hears as a soft «р» |

Two things this set cannot show: intonation is flatter than native speech for
every fine-tune (range ~10 st against 12.7) and does not move with the merge,
and a residual soft «р» or «и» read as «і» sits below what these features
resolve — listen for those. `scripts/vowel_tokens.py` pairs every «и» with the
human recording of the same word when the average hides a shift.

## 14. Why an adapter can wreck the audio decoder

`loralib` seeds its two wrappers as mirror images, and only one of them is
scaled:

```
Linear:     lora_A = kaiming_uniform  (‖A‖ ≈ 3.3 at r=32, d=2560),  lora_B = 0
Embedding:  lora_A = 0,                lora_B = normal(0, 1)  →  ‖B‖ ≈ 287
```

Both start at a zero update, so nothing looks wrong. But each factor's gradient
is proportional to the other, so an unpatched embedding adapter moves under a
matrix ninety times larger than a linear one's and lands large from the first
step instead of growing into place. `codebook_embeddings` turns an acoustic
code index into a vector for the audio decoder; move it far enough and the
decoder no longer agrees with the tokens the model emits — quiet,
rate-scrambled audio that sounds like fading. That is why the acoustic side is
never in the default targets and why the embedding adapter is rescaled.

Three patches in `training/lora_patch.py` keep fine-tuning aligned with serve:

0. **VQ embedding scale.** s2-pro inference divides each VQ embedding by
   `sqrt(num_codebooks+1)` ≈ 3.3 (`scale_codebook_embeddings`). Upstream
   training skips that division, so a fine-tune would otherwise optimise
   acoustic inputs 3.3× larger than the served model sees.
   `patch_scaled_codebook_embed` applies the inference scale during training.

1. **Embedding LoRA scale.** `_rescale_embedding_lora` reseeds `lora_B` at the
   Kaiming scale so one learning rate means the same thing for every adapted
   matrix.

2. **Tied logit head.** s2-pro ties text `embeddings` to the token logits
   (`tie_word_embeddings=True`, no separate `output` module):
   `F.linear(slow_out, embeddings.weight)`. That read bypasses LoRA, so without
   a patch the delta trains on the input lookup only and merging shifts every
   token logit — including semantic ids and `im_end` — in a way training never
   saw. `patch_tied_embedding_logits` adds the same delta to the logits during
   training so the trained function matches the merged checkpoint.

Default `TRAINING_LORA_TARGET_MODULES=mlp_w2,embeddings` adapts the `w2`
down-projection of each slow MLP and the text table; `attention,mlp,embeddings`
is the wider slow pass. `codebook_embeddings` and `fast_*` must be listed
separately to adapt the acoustic codebook / decoder. The logit head is the
embedding table itself, so `embeddings` covers it.

Archiving `downloads/` + `transcripts/` off-box is optional. Re-segment only needs those two trees on disk.
