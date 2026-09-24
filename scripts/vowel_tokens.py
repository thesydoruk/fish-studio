"""Pair every и/і token of the probe set: model take against the human recording.

The harness reports one distance per run. This script keeps every token, so a
shift that lives in one context (unstressed и, и after a given consonant, word
end) shows up where the average hides it. A model и counts as "reads as і" when
its F2 sits closer to the human's і than to the human's и of the same take.

    python scripts/vowel_tokens.py --label w2l0-11emb --takes data/eval/takes_w2l0-11emb
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

from fish_studio.server.uk_eval import load_probes, read_audio, synthesize
from fish_studio.stress_align import (
    CtcAligner,
    _formants_at,
    marked_ordinals,
    normalize_for_ctc,
)

VOWELS = "аеиіоуяюєї"


def tokens(text: str, samples: np.ndarray, rate: int, aligner: CtcAligner) -> dict:
    """{(word, ordinal): (char, F2)} for every aligned и/і with usable formants."""
    spans = aligner.align(text, samples, rate)
    if spans is None:
        return {}
    out = {}
    for span in spans:
        if span.word_index < 0 or span.char not in "иі":
            continue
        end = max(span.filled_end, span.end)
        if end - span.start < 0.04:
            continue
        pair = _formants_at(samples, rate, (span.start + end) / 2)
        if pair is not None:
            out[(span.word_index, span.vowel_ordinal)] = (span.char, pair[1])
    return out


def context(words: list[str], word: int, ordinal: int, char: str) -> dict:
    text = words[word]
    seen = -1
    for pos, letter in enumerate(text):
        if letter in VOWELS:
            seen += 1
            if seen == ordinal:
                before = text[pos - 1] if pos else "#"
                after = text[pos + 1] if pos + 1 < len(text) else "#"
                return {"word": text, "before": before, "after": after, "final": after == "#"}
    return {"word": text, "before": "?", "after": "?", "final": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=Path, default=Path("configs/uk_probe.tsv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--label", required=True)
    parser.add_argument("--takes", type=Path, help="reuse/keep synthesized WAVs here")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shift", type=float, default=250.0, help="Hz over the human и to flag")
    args = parser.parse_args(argv)

    probes = load_probes(args.probes, args.data_root, limit=args.limit)
    aligner = CtcAligner()
    rows = []
    if args.takes is not None:
        args.takes.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.base_url, timeout=300) as client:
        for probe in probes:
            take = args.takes / f"{probe.id}.wav" if args.takes is not None else None
            if take is not None and take.exists():
                payload = take.read_bytes()
            else:
                try:
                    payload = synthesize(
                        client,
                        text=probe.text,
                        ref_wav=probe.ref_wav,
                        ref_text=probe.ref_text,
                        language="uk",
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] {probe.id}: {exc}", file=sys.stderr)
                    continue
                if take is not None:
                    take.write_bytes(payload)
            model = tokens(probe.text, *read_audio(io.BytesIO(payload)), aligner)
            human = tokens(probe.text, *read_audio(probe.human_wav), aligner)
            stressed = marked_ordinals(probe.truth)
            words = normalize_for_ctc(probe.text).split()
            for key, (char, f2_model) in model.items():
                if key not in human or human[key][0] != char:
                    continue
                ctx = context(words, key[0], key[1], char)
                ctx.update(
                    probe=probe.id,
                    char=char,
                    stressed=stressed.get(key[0]) == key[1],
                    model=f2_model,
                    human=human[key][1],
                )
                rows.append(ctx)

    ii = [r for r in rows if r["char"] == "и"]
    flagged = [r for r in ii if r["model"] - r["human"] > args.shift]
    print(
        f"{args.label}: и tokens paired={len(ii)}  flagged(model F2 > human+{args.shift:.0f})="
        f"{len(flagged)} ({100 * len(flagged) / max(1, len(ii)):.0f}%)"
    )
    if not ii:
        print("  nothing paired: is the server up and the probe set on disk?", file=sys.stderr)
        return 1
    delta = np.array([r["model"] - r["human"] for r in ii])
    print(f"  ΔF2(model−human) median={np.median(delta):+.0f} p90={np.percentile(delta, 90):+.0f}")

    def by(name: str, key):
        groups: dict = defaultdict(list)
        for r in ii:
            groups[key(r)].append(r)
        print(f"  by {name}:")
        for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(rs) < 4:
                continue
            d = np.median([r["model"] - r["human"] for r in rs])
            f = sum(r["model"] - r["human"] > args.shift for r in rs)
            print(
                f"    {g!s:>8}  n={len(rs):3d}  ΔF2={d:+5.0f}  flagged={100 * f / len(rs):3.0f}%"
            )

    by("stress", lambda r: "stressed" if r["stressed"] else "unstressed")
    by("position", lambda r: "final" if r["final"] else "inner")
    by("before", lambda r: r["before"])
    by("after", lambda r: r["after"])
    print("  worst tokens:")
    for r in sorted(ii, key=lambda r: r["human"] - r["model"])[:15]:
        print(
            f"    {r['word']:<16} {'ˈ' if r['stressed'] else ' '} model={r['model']:.0f} "
            f"human={r['human']:.0f}  ({r['probe']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
