#!/usr/bin/env python3
"""Compare WhisperX backtrack warnings in pipeline.log vs dropped segments."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> None:
    log_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/logs/pipeline.log")
    transcripts_dir = Path(
        sys.argv[2] if len(sys.argv) > 2 else "data/work/youtube-channel-1/transcripts"
    )

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(
        r'Failed to align segment \("((?:[^"\\]|\\.)*)"\): backtrack failed',
    )
    failed_quotes = [m.group(1).encode().decode("unicode_escape") for m in pat.finditer(log_text)]

    stats_dropped = 0
    flag_true = 0
    empty_words = 0
    total_segs = 0

    by_text: dict[str, dict] = {}
    for fp in sorted(transcripts_dir.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        stats_dropped += int((data.get("stats") or {}).get("rejected_alignment_failed", 0) or 0)
        for seg in data.get("segments", []):
            total_segs += 1
            text = (seg.get("text") or "").strip()
            if seg.get("alignment_failed"):
                flag_true += 1
            if text and not seg.get("words"):
                empty_words += 1
            if text:
                by_text[text] = {
                    "file": fp.name,
                    "words": len(seg.get("words") or []),
                    "avg_prob": _avg_prob(seg.get("words") or []),
                }

    matched = 0
    kept_with_words = 0
    kept_no_words = 0
    examples: list[tuple] = []

    for quote in failed_quotes:
        text = quote.strip()
        if not text:
            continue
        info = by_text.get(text)
        if info is None:
            # fuzzy: substring match
            for key, val in by_text.items():
                if text in key or key in text:
                    info = val
                    text = key
                    break
        if info is None:
            continue
        matched += 1
        if info["words"]:
            kept_with_words += 1
            if len(examples) < 10:
                examples.append((info["file"], text[:70], info["words"], info["avg_prob"]))
        else:
            kept_no_words += 1

    print("=== LOG vs JSON ===")
    print(f"backtrack warnings in log: {len(failed_quotes)}")
    print(f"stats.rejected_alignment_failed (sum): {stats_dropped}")
    print(f"segments in JSON: {total_segs}")
    print(f"segments alignment_failed=true: {flag_true}")
    print(f"segments text but no words: {empty_words}")
    print()
    print("=== BACKTRACK WARNINGS MATCHED TO KEPT SEGMENTS ===")
    print(f"matched to transcript segments: {matched}")
    print(f"still kept WITH words (resorted to original): {kept_with_words}")
    print(f"still kept WITHOUT words: {kept_no_words}")
    print()
    print("=== EXAMPLES STILL KEPT AFTER BACKTRACK FAIL ===")
    for ex in examples:
        print(f"  {ex[0]} | words={ex[2]} avg_prob={ex[3]:.3f} | {ex[1]}")


def _avg_prob(words: list[dict]) -> float:
    if not words:
        return 0.0
    return sum(float(w.get("probability", 0.0)) for w in words) / len(words)


if __name__ == "__main__":
    main()
