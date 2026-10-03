"""Rebuild dev/test with a per-(corpus, singer) cap and round-robin sampling over (technique, song) groups; ``train.txt`` is untouched.

Usage: PYTHONPATH=D:/BreathLab python tools/data/make_bal_splits.py --cap 250 [--splits dev]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from breathlab.common import load_config


def group_key(name: str, corpus: str) -> str:
    """Grouping key ``(technique, song)``, fine-grained so round-robin spreads coverage."""
    parts = name.split("/")
    if corpus == "c10" and len(parts) >= 5:
        return "/".join(parts[:5])          # alias/lang/singer/technique/song
    if len(parts) >= 3:
        return "/".join(parts[:3])
    return "/".join(parts[:2]) or name


def singer_key(name: str, corpus: str) -> str:
    """Singer identifier for a clip; single-singer corpora fall back to the corpus alias."""
    parts = name.split("/")
    if corpus == "c10" and len(parts) >= 3:
        return parts[2]                      # c10/<lang>/<singer>/...
    if corpus == "c13" and len(parts) >= 2:
        return parts[1].split("#")[0]        # c13/<singer>#<song>/NNNN
    if len(parts) >= 2:
        return parts[0]                      # single-singer corpora
    return "unknown"


def corpus_of(name: str) -> str:
    """Corpus alias of a clip (first path component)."""
    return name.split("/")[0]


def cap_group(names: list[str], cap: int, seed: int) -> list[str]:
    """Draw round-robin across (technique, song) groups until ``cap`` clips are collected."""
    buckets = defaultdict(list)
    for n in names:
        buckets[group_key(n, corpus_of(n))].append(n)
    rng = random.Random(seed)
    keys = sorted(buckets)
    for k in keys:
        buckets[k] = sorted(buckets[k])
        rng.shuffle(buckets[k])
    out: list[str] = []
    round_i = 0
    while len(out) < cap:
        added = False
        for k in keys:
            if len(out) >= cap:
                break
            if round_i < len(buckets[k]):
                out.append(buckets[k][round_i])
                added = True
        if not added:
            break
        round_i += 1
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="breathlab/configs/default.yaml")
    ap.add_argument("--cap", type=int, default=250,
                    help="maximum clips kept per (corpus, singer)")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--suffix", default="_bal")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sdir = Path(cfg["paths"]["splits_dir"])
    summary = {}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        src = sdir / f"{split}.txt"
        names = [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
        per_singer = defaultdict(list)
        for n in names:
            per_singer[(corpus_of(n), singer_key(n, corpus_of(n)))].append(n)

        kept: list[str] = []
        rows = []
        for (corpus, singer), cs in sorted(per_singer.items()):
            if len(cs) > args.cap:
                k = cap_group(cs, args.cap, seed=f"{split}:{corpus}:{singer}")
            else:
                k = sorted(cs)
            kept += k
            rows.append({"corpus": corpus, "singer": singer,
                         "before": len(cs), "after": len(k)})
        kept = sorted(kept)
        out = sdir / f"{split}{args.suffix}.txt"
        out.write_text("\n".join(kept) + "\n", encoding="utf-8")
        summary[split] = {"cap": args.cap, "before": len(names), "after": len(kept),
                          "out": str(out), "per_singer": rows}
        print(f"\n==== {split} -> {out.name}: {len(names)} -> {len(kept)} clips "
              f"(cap={args.cap}/singer) ====")
        print(f"{'corpus':10s} {'singer':16s} {'before':>7s} {'after':>6s}")
        for r in rows:
            print(f"{r['corpus']:10s} {r['singer'][:16]:16s} {r['before']:7d} {r['after']:6d}")
        print(f"singers: {len(rows)}")

    (sdir / "split_summary_bal.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary: {sdir / 'split_summary_bal.json'}")


if __name__ == "__main__":
    main()
