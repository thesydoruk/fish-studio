#!/usr/bin/env python3
"""Analyze clip manifests after segment."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from fish_studio.dataset.segment import AudioClip


def _load_clips_from_segments(segments_dir: Path) -> list[AudioClip]:
    clips: list[AudioClip] = []
    for manifest in sorted(segments_dir.glob("*/clips.jsonl")):
        with manifest.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                clips.append(AudioClip(**json.loads(line)))
    return clips


def _clip_rank_score(clip: AudioClip) -> float:
    if clip.quality_score is not None:
        return clip.quality_score
    if clip.avg_word_score is not None:
        return clip.avg_word_score
    if clip.avg_logprob is not None:
        return max(0.0, min(1.0, clip.avg_logprob + 1.0))
    return 0.0


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((p / 100.0) * (len(ordered) - 1))
    return ordered[idx]


def main() -> None:
    segments_dir = Path(
        sys.argv[1] if len(sys.argv) > 1 else "data/work/youtube-channel-1/segments"
    )

    clips = _load_clips_from_segments(segments_dir)
    if not clips:
        print(f"No clips.jsonl found under {segments_dir}")
        return

    total_hours = sum(c.duration for c in clips) / 3600.0
    scores = [_clip_rank_score(c) for c in clips]

    print("=== CLIPS OVERVIEW ===")
    print(f"segments dir: {segments_dir}")
    print(f"clips: {len(clips)} ({total_hours:.2f} h)")
    print(f"videos: {len({c.video_id for c in clips})}")
    print()
    print("=== QUALITY DISTRIBUTION ===")
    print(
        f"score: min={min(scores):.3f} p50={statistics.median(scores):.3f} "
        f"p90={_pct(scores, 90):.3f} max={max(scores):.3f} mean={statistics.mean(scores):.3f}"
    )
    durations = [c.duration for c in clips]
    print(
        f"duration sec: p50={statistics.median(durations):.2f} "
        f"p90={_pct(durations, 90):.2f} mean={statistics.mean(durations):.2f}"
    )
    word_scores = [c.avg_word_score for c in clips if c.avg_word_score is not None]
    if word_scores:
        print(
            f"word score: n={len(word_scores)} p50={statistics.median(word_scores):.3f} "
            f"mean={statistics.mean(word_scores):.3f}"
        )
    else:
        print("word score: not available yet (run align + segment)")
    print()
    print("=== TOP 5 CLIPS ===")
    ranked = sorted(clips, key=_clip_rank_score, reverse=True)[:5]
    for clip in ranked:
        score = _clip_rank_score(clip)
        print(
            f"  {clip.video_id}/{clip.clip_id} score={score:.3f} dur={clip.duration:.1f}s | {clip.text[:90]}"
        )


if __name__ == "__main__":
    main()
