"""Cross-video speaker clustering within a single SOURCES channel."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from fish_studio.dataset.speakers import global_speaker_name, resolve_speaker_name
from fish_studio.dataset.transcript import load_transcript


SPEAKER_MAP_FILENAME = "speaker_map.json"


@dataclass(frozen=True)
class SpeakerNode:
    """One diarized speaker observed in a single video transcript."""

    local_name: str
    video_id: str
    speaker_id: str
    speech_seconds: float
    embedding: list[float]


@dataclass
class ClusterResult:
    """Outcome of per-channel embedding clustering."""

    mapping: dict[str, str]
    clustered: int
    singletons: int
    missing_embeddings: int
    map_path: str


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity for embedding vectors (same idea as audio-intel)."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    denom = math.sqrt(norm_left) * math.sqrt(norm_right)
    if denom <= 0.0:
        return 0.0
    return float(dot / denom)


def collect_speaker_nodes(
    transcripts_dir: str | Path,
    *,
    base_speaker: str,
) -> tuple[list[SpeakerNode], int]:
    """Load roster embeddings from transcript JSON files.

    Returns nodes that have embeddings, plus a count of roster rows missing vectors.
    """
    root = Path(transcripts_dir)
    nodes: list[SpeakerNode] = []
    missing = 0
    for path in sorted(root.glob("*.json")):
        transcript = load_transcript(path)
        video_id = transcript.video_id or path.stem
        for speaker in transcript.speakers:
            local = resolve_speaker_name(
                base_speaker,
                speaker.id,
                video_id=video_id,
            )
            if not speaker.embedding:
                missing += 1
                continue
            nodes.append(
                SpeakerNode(
                    local_name=local,
                    video_id=video_id,
                    speaker_id=speaker.id,
                    speech_seconds=float(speaker.speech_seconds),
                    embedding=list(speaker.embedding),
                )
            )
    return nodes, missing


def cluster_local_speakers(
    nodes: list[SpeakerNode],
    *,
    base_speaker: str,
    threshold: float,
) -> dict[str, str]:
    """Merge video-scoped speakers whose embeddings exceed ``threshold``.

    Returns ``local_name → {base}_s{k}``. Cluster indices prefer larger speech
    totals (``s0`` = most speech among clusters).
    """
    if not nodes:
        return {}

    by_name = {node.local_name: node for node in nodes}
    names = sorted(by_name, key=lambda name: (-by_name[name].speech_seconds, name))
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        # Prefer the root with more speech (stable, useful canonical).
        if by_name[root_left].speech_seconds >= by_name[root_right].speech_seconds:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    for index, left in enumerate(names):
        left_emb = by_name[left].embedding
        for right in names[index + 1 :]:
            if cosine_similarity(left_emb, by_name[right].embedding) > threshold:
                union(left, right)

    clusters: dict[str, list[str]] = {}
    for name in names:
        clusters.setdefault(find(name), []).append(name)

    ranked_roots = sorted(
        clusters,
        key=lambda root: (
            -sum(by_name[member].speech_seconds for member in clusters[root]),
            root,
        ),
    )
    mapping: dict[str, str] = {}
    for cluster_index, root in enumerate(ranked_roots):
        global_name = global_speaker_name(base_speaker, cluster_index)
        for member in clusters[root]:
            mapping[member] = global_name
    return mapping


def write_speaker_map(
    work_dir: str | Path,
    mapping: dict[str, str],
    *,
    base_speaker: str,
    threshold: float,
    clustered: int,
    singletons: int,
    missing_embeddings: int,
) -> Path:
    """Persist ``speaker_map.json`` under the source work directory."""
    path = Path(work_dir) / SPEAKER_MAP_FILENAME
    payload = {
        "base_speaker": base_speaker,
        "threshold": threshold,
        "clustered_locals": clustered,
        "singleton_clusters": singletons,
        "missing_embeddings": missing_embeddings,
        "mapping": mapping,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_speaker_map(work_dir: str | Path) -> dict[str, str]:
    """Load local→global mapping; empty dict when the file is absent."""
    path = Path(work_dir) / SPEAKER_MAP_FILENAME
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("mapping") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if key and value}


def apply_speaker_map_to_name(speaker_name: str, mapping: dict[str, str]) -> str:
    """Remap one speaker label; identity when unmapped."""
    if not mapping:
        return speaker_name
    return mapping.get(speaker_name, speaker_name)


def fallback_map_by_diarization_id(
    local_names: list[str],
    *,
    base_speaker: str,
) -> dict[str, str]:
    """Map locals that share the same trailing diarization id onto ``{base}_s{k}``.

    Used when transcript JSON has no embeddings (older ASR runs). Same risk as the
    historical string collision for rare guest ``spk_N`` labels, but keeps the
    dominant host id aligned across videos until a re-transcribe stores vectors.
    """
    by_suffix: dict[str, list[str]] = {}
    for name in local_names:
        parts = name.split("__")
        suffix = parts[-1] if len(parts) >= 3 else name
        by_suffix.setdefault(suffix, []).append(name)

    # Prefer lower spk_N / SPEAKER_NN as earlier cluster indices.
    def suffix_key(suffix: str) -> tuple:
        digits = "".join(ch for ch in suffix if ch.isdigit())
        return (int(digits) if digits else 10**9, suffix)

    mapping: dict[str, str] = {}
    for index, suffix in enumerate(sorted(by_suffix, key=suffix_key)):
        global_name = global_speaker_name(base_speaker, index)
        for local in by_suffix[suffix]:
            mapping[local] = global_name
    return mapping


def collect_local_names_from_segments(segments_dir: str | Path) -> list[str]:
    """Unique speaker_name values from clips.jsonl manifests."""
    root = Path(segments_dir)
    names: set[str] = set()
    for manifest in root.glob("*/clips.jsonl"):
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                speaker = str(payload.get("speaker_name") or "").strip()
                if speaker:
                    names.add(speaker)
    return sorted(names)


def cluster_source_transcripts(
    work_dir: str | Path,
    transcripts_dir: str | Path,
    *,
    base_speaker: str,
    threshold: float = 0.75,
    segments_dir: str | Path | None = None,
    allow_label_fallback: bool = True,
) -> ClusterResult:
    """Cluster roster embeddings for one source and write ``speaker_map.json``."""
    nodes, missing = collect_speaker_nodes(transcripts_dir, base_speaker=base_speaker)
    mapping = cluster_local_speakers(nodes, base_speaker=base_speaker, threshold=threshold)
    used_fallback = False
    if not mapping and allow_label_fallback:
        local_names = collect_local_names_from_segments(segments_dir or Path(work_dir) / "segments")
        if not local_names:
            # Derive expected locals from transcript rosters (pre-segment).
            local_names = []
            for path in sorted(Path(transcripts_dir).glob("*.json")):
                transcript = load_transcript(path)
                video_id = transcript.video_id or path.stem
                if transcript.speakers:
                    for speaker in transcript.speakers:
                        local_names.append(
                            resolve_speaker_name(
                                base_speaker,
                                speaker.id,
                                video_id=video_id,
                            )
                        )
                else:
                    local_names.append(
                        resolve_speaker_name(base_speaker, None, video_id=video_id)
                    )
        mapping = fallback_map_by_diarization_id(local_names, base_speaker=base_speaker)
        used_fallback = bool(mapping)

    globals_used = set(mapping.values())
    singletons = sum(
        1
        for global_name in globals_used
        if sum(1 for value in mapping.values() if value == global_name) == 1
    )
    clustered = max(0, len(mapping) - singletons)
    path = write_speaker_map(
        work_dir,
        mapping,
        base_speaker=base_speaker,
        threshold=threshold,
        clustered=clustered,
        singletons=singletons,
        missing_embeddings=missing,
    )
    # Annotate fallback in the map file for operators.
    if used_fallback:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["fallback"] = "diarization_id"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ClusterResult(
        mapping=mapping,
        clustered=clustered,
        singletons=singletons,
        missing_embeddings=missing,
        map_path=str(path),
    )


def prune_tiny_speakers(
    clips: list,
    *,
    min_clips: int = 100,
    min_speech_sec: float = 300.0,
) -> tuple[list, dict[str, tuple[int, float]]]:
    """Drop clips whose speaker fails either clip-count or duration floor.

    ``clips`` items must expose ``speaker_name`` and ``duration`` attributes
    (e.g. :class:`~fish_studio.dataset.segment.AudioClip`).

    Returns ``(kept_clips, dropped_stats)`` where ``dropped_stats`` maps
    speaker → ``(clip_count, speech_sec)``.
    """
    if min_clips <= 0 and min_speech_sec <= 0:
        return list(clips), {}

    totals: dict[str, list[float]] = {}
    for clip in clips:
        speaker = str(getattr(clip, "speaker_name", "") or "speaker")
        duration = float(getattr(clip, "duration", 0.0) or 0.0)
        totals.setdefault(speaker, []).append(duration)

    keep: set[str] = set()
    dropped: dict[str, tuple[int, float]] = {}
    for speaker, durations in totals.items():
        count = len(durations)
        speech = sum(durations)
        if count >= min_clips and speech >= min_speech_sec:
            keep.add(speaker)
        else:
            dropped[speaker] = (count, speech)

    if not dropped:
        return list(clips), {}

    kept_clips = [
        clip
        for clip in clips
        if str(getattr(clip, "speaker_name", "") or "speaker") in keep
    ]
    return kept_clips, dropped


def remap_dataset_speakers(
    dataset_dir: str | Path,
    mapping: dict[str, str],
) -> dict[str, int]:
    """Rewrite ``speaker_name`` columns in metadata_train/eval.csv in-place.

    Returns counts of rewritten rows per metadata file basename.
    """
    root = Path(dataset_dir)
    if not mapping:
        return {}
    rewritten: dict[str, int] = {}
    for name in ("metadata_train.csv", "metadata_eval.csv"):
        path = root / name
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        header = lines[0]
        out_lines = [header]
        changed = 0
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                out_lines.append(line)
                continue
            audio, text, speaker = parts
            new_speaker = apply_speaker_map_to_name(speaker, mapping)
            if new_speaker != speaker:
                changed += 1
            out_lines.append(f"{audio}|{text}|{new_speaker}")
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        rewritten[name] = changed
    return rewritten
