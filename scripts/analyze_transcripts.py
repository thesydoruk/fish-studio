#!/usr/bin/env python3
"""Analyze transcript JSON quality on disk."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = round((p / 100) * (len(xs) - 1))
    return xs[i]


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/work/youtube-channel-1/transcripts")
    files = sorted(root.glob("*.json"))
    if not files:
        print("No transcripts found")
        return

    seg_count = 0
    aligned_files = 0
    logprobs: list[float] = []
    no_speech: list[float] = []
    compression: list[float] = []
    lang_seg: Counter[str] = Counter()
    lang_file: Counter[str] = Counter()
    word_scores: list[float] = []
    low_logprob: list[tuple[str, float, str]] = []
    high_no_speech: list[tuple[str, float, str]] = []
    high_compression: list[tuple[str, float, str]] = []
    non_uk_segments = 0
    empty_text = 0
    short_text = 0
    total_speech_sec = 0.0
    stats_agg: Counter[str] = Counter()

    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        if data.get("aligned_at"):
            aligned_files += 1
        lang_file[data.get("language", "?")] += 1
        st = data.get("stats") or {}
        for key in (
            "segments_kept",
            "rejected_quality",
            "rejected_non_target_language",
            "rejected_empty",
            "rejected_too_short",
        ):
            stats_agg[key] += int(st.get(key, 0) or 0)
        total_speech_sec += float(st.get("kept_speech_duration_sec", 0) or 0)
        for lang, count in (st.get("detected_languages") or {}).items():
            lang_seg[lang] += count

        for seg in data.get("segments", []):
            seg_count += 1
            lp = float(seg.get("avg_logprob", 0))
            nsp = float(seg.get("no_speech_prob", 0))
            cr = float(seg.get("compression_ratio", 0))
            logprobs.append(lp)
            no_speech.append(nsp)
            compression.append(cr)
            text = (seg.get("text") or "").strip()
            if not text:
                empty_text += 1
            elif len(text) < 10:
                short_text += 1
            sl = seg.get("language")
            if sl and sl != "uk":
                non_uk_segments += 1
            if lp < -1.0:
                low_logprob.append((fp.name, lp, text[:80]))
            if nsp > 0.6:
                high_no_speech.append((fp.name, nsp, text[:80]))
            if cr > 2.4:
                high_compression.append((fp.name, cr, text[:80]))
            for word in seg.get("words", []):
                score = word.get("score", word.get("probability"))
                if score is not None:
                    word_scores.append(float(score))

    print("=== OVERVIEW ===")
    print(f"transcripts: {len(files)}")
    print(f"aligned: {aligned_files} ({aligned_files / len(files) * 100:.1f}%)")
    print(f"segments total: {seg_count}")
    print(f"kept speech from stats: {total_speech_sec / 3600:.2f} h")
    print(
        "stats kept / rejected_quality / rejected_non_uk / rejected_empty / rejected_short: "
        f"{stats_agg['segments_kept']} / {stats_agg['rejected_quality']} / "
        f"{stats_agg['rejected_non_target_language']} / {stats_agg['rejected_empty']} / "
        f"{stats_agg['rejected_too_short']}"
    )
    print()
    print("=== FILE LANGUAGE ===")
    for key, value in lang_file.most_common():
        print(f"  {key}: {value} files")
    print("=== DETECTED LANGUAGES IN VAD REGIONS ===")
    for key, value in lang_seg.most_common(12):
        print(f"  {key}: {value}")
    print(f"non-uk segments in kept transcripts: {non_uk_segments}")
    print()
    print("=== QUALITY METRICS ===")
    for name, arr in [
        ("avg_logprob", logprobs),
        ("no_speech_prob", no_speech),
        ("compression_ratio", compression),
    ]:
        print(
            f"{name}: min={min(arr):.3f} p50={statistics.median(arr):.3f} "
            f"p90={pct(arr, 90):.3f} max={max(arr):.3f} mean={statistics.mean(arr):.3f}"
        )
    if word_scores:
        print(
            f"word score: n={len(word_scores)} p50={statistics.median(word_scores):.3f} "
            f"p10={pct(word_scores, 10):.3f} mean={statistics.mean(word_scores):.3f}"
        )
    else:
        print("word score: no aligned words yet")
    print(f"empty text segments: {empty_text}; short under 10 chars: {short_text}")
    print(f"below min_avg_logprob -1.0: {len(low_logprob)}")
    print(f"above max_no_speech_prob 0.6: {len(high_no_speech)}")
    print(f"above max_compression_ratio 2.4: {len(high_compression)}")
    print()
    print("=== WORST avg_logprob ===")
    for item in sorted(low_logprob, key=lambda x: x[1])[:5]:
        print(f"  {item[0]} logprob={item[1]:.3f} | {item[2]}")
    print("=== HIGHEST no_speech_prob ===")
    for item in sorted(high_no_speech, key=lambda x: -x[1])[:5]:
        print(f"  {item[0]} no_speech={item[1]:.3f} | {item[2]}")
    print("=== SAMPLE GOOD SEGMENTS ===")
    good = sorted(
        [
            (fp.name, seg)
            for fp in files
            for seg in json.loads(fp.read_text(encoding="utf-8")).get("segments", [])
        ],
        key=lambda x: float(x[1].get("avg_logprob", -99)),
        reverse=True,
    )[:3]
    for name, seg in good:
        print(
            f"  {name} logprob={float(seg.get('avg_logprob', 0)):.3f} | {seg.get('text', '')[:100]}"
        )


if __name__ == "__main__":
    main()
