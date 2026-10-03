"""Build leakage-aware train/dev/test splits across corpora (per-corpus strategy from ``data.corpora[].split``); grouping by singer or song keeps related material from straddling train and test."""
from __future__ import annotations

import argparse
import itertools
import json
import random
from typing import Dict, List, Sequence

from breathlab.common import ensure_dir, load_config
from breathlab.data.parse_labels import Clip, load_all_clips

SplitDict = Dict[str, List[str]]


def _singer_stats(clips: List[Clip]) -> Dict[str, Dict[str, int]]:
    """Aggregate per-singer clip / breath / positive-clip counts."""
    stats: Dict[str, Dict[str, int]] = {}
    for c in clips:
        d = stats.setdefault(c.singer, {"clips": 0, "breaths": 0, "pos_clips": 0})
        d["clips"] += 1
        d["breaths"] += c.n_breath
        d["pos_clips"] += 1 if c.n_breath > 0 else 0
    return stats


def choose_dev_singers(
    stats: Dict[str, Dict[str, int]],
    candidates: Sequence[str],
    n_dev: int,
    target_frac: float = 0.15,
    seed: int = 1337,
) -> List[str]:
    """Pick the dev singers whose breath share is closest to ``target_frac`` (every chosen singer must contain at least one positive clip; ``seed`` is accepted only for signature stability)."""
    total_breaths = sum(stats[s]["breaths"] for s in candidates)
    ordered = sorted(candidates)
    best = None
    mean_clips = sum(stats[s]["clips"] for s in candidates) / max(1, len(candidates))
    for combo in itertools.combinations(ordered, n_dev):
        if any(stats[s]["pos_clips"] == 0 for s in combo):
            continue
        dev_breaths = sum(stats[s]["breaths"] for s in combo)
        frac = dev_breaths / max(total_breaths, 1)
        dev_clips = sum(stats[s]["clips"] for s in combo)
        key = (abs(frac - target_frac), abs(dev_clips - n_dev * mean_clips))
        if best is None or key < best[0]:
            best = (key, list(combo))
    return best[1] if best else ordered[:n_dev]


def _split_by_singer(clips: List[Clip], corpus_cfg: Dict, split_cfg: Dict, seed: int) -> SplitDict:
    """Split a multi-singer corpus by singer, with officially listed test singers and auto-selected dev singers."""
    stats = _singer_stats(clips)
    all_singers = sorted(stats.keys())
    test_singers = [s for s in split_cfg["test_singers"] if s in stats]
    remaining = [s for s in all_singers if s not in test_singers]
    dev_singers = list(split_cfg.get("dev_singers") or [])
    if not dev_singers:
        dev_singers = choose_dev_singers(stats, remaining, split_cfg["n_dev_singers"], 0.15, seed)
    train_singers = [s for s in remaining if s not in dev_singers]

    def pick(singers) -> List[str]:
        return sorted(c.name for c in clips if c.singer in singers)

    return {"train": pick(train_singers), "dev": pick(dev_singers), "test": pick(test_singers)}


def _split_by_song(clips: List[Clip], corpus_cfg: Dict, seed: int) -> SplitDict:
    """Split a single-singer corpus by song after a seeded shuffle (never per segment: same-recording excerpts must not straddle train and test)."""
    names = sorted(c.name for c in clips)
    rng = random.Random(f"{seed}:{corpus_cfg['alias']}")
    rng.shuffle(names)
    n = len(names)
    n_test = max(1, round(n * corpus_cfg.get("test_frac", 0.15)))
    n_dev = max(1, round(n * corpus_cfg.get("dev_frac", 0.15)))
    test = sorted(names[:n_test])
    dev = sorted(names[n_test:n_test + n_dev])
    train = sorted(names[n_test + n_dev:])
    return {"train": train, "dev": dev, "test": test}


def _split_official_test_by_song(clips: List[Clip], corpus_cfg: Dict, seed: int) -> SplitDict:
    """Official test segments go to test; the rest are grouped by song id so segments of one song never straddle train and dev."""
    test = sorted(c.name for c in clips if c.split_hint == "test")
    rest = [c for c in clips if c.split_hint != "test"]
    songs = sorted({c.name.split("/")[1][:4] for c in rest})
    rng = random.Random(f"{seed}:{corpus_cfg['alias']}")
    rng.shuffle(songs)
    n_dev = max(1, round(len(songs) * corpus_cfg.get("dev_frac", 0.08)))
    dev_songs = set(songs[:n_dev])
    dev = sorted(c.name for c in rest if c.name.split("/")[1][:4] in dev_songs)
    train = sorted(c.name for c in rest if c.name.split("/")[1][:4] not in dev_songs)
    return {"train": train, "dev": dev, "test": test}


def _split_by_singer_frac(clips: List[Clip], corpus_cfg: Dict, seed: int) -> SplitDict:
    """Assign whole singers to sets by fraction, so nothing about a voice (timbre, microphone, session) leaks across splits."""
    singers = sorted({c.singer for c in clips})
    rng = random.Random(f"{seed}:{corpus_cfg['alias']}")
    rng.shuffle(singers)
    n = len(singers)
    n_test = max(1, round(n * corpus_cfg.get("test_frac", 0.1)))
    n_dev = max(1, round(n * corpus_cfg.get("dev_frac", 0.1)))
    test_s = set(singers[:n_test])
    dev_s = set(singers[n_test:n_test + n_dev])

    def pick(pred) -> List[str]:
        return sorted(c.name for c in clips if pred(c.singer))

    return {
        "test": pick(lambda s: s in test_s),
        "dev": pick(lambda s: s in dev_s),
        "train": pick(lambda s: s not in test_s and s not in dev_s),
    }


def make_splits(cfg: Dict):
    """Build the splits for every corpus (each split independently) and write ``train/dev/test.txt`` plus ``split_summary.json``."""
    clips = load_all_clips(cfg)
    by_corpus: Dict[str, List[Clip]] = {}
    for c in clips:
        by_corpus.setdefault(c.corpus, []).append(c)

    corpora_cfg = {c["alias"]: c for c in (cfg.get("data") or {}).get("corpora", [])}
    splits = {"train": [], "dev": [], "test": []}
    per_corpus_summary = {}

    for corpus, cclips in by_corpus.items():
        ccfg = corpora_cfg.get(corpus, {"alias": corpus, "split": "by_song"})
        method = ccfg.get("split", "by_song")
        if method == "by_singer":
            part = _split_by_singer(cclips, ccfg, cfg["split"], cfg.get("seed", 1337))
        elif method == "official_test_by_song":
            part = _split_official_test_by_song(cclips, ccfg, cfg.get("seed", 1337))
        elif method == "by_singer_frac":
            part = _split_by_singer_frac(cclips, ccfg, cfg.get("seed", 1337))
        else:
            part = _split_by_song(cclips, ccfg, cfg.get("seed", 1337))
        for k in splits:
            splits[k].extend(part[k])
        bn = {c.name: c.n_breath for c in cclips}
        per_corpus_summary[corpus] = {
            k: {"clips": len(part[k]), "breaths": sum(bn[n] for n in part[k])} for k in splits
        }

    for k in splits:
        splits[k] = sorted(splits[k])

    out_dir = ensure_dir(cfg["paths"]["splits_dir"])
    for k, v in splits.items():
        (out_dir / f"{k}.txt").write_text("\n".join(v) + "\n", encoding="utf-8")

    n_breath = {c.name: c.n_breath for c in clips}
    totals = {}
    for k in splits:
        totals[k] = {"clips": len(splits[k]), "breaths": sum(n_breath[n] for n in splits[k])}
    summary = {"per_corpus": per_corpus_summary, "totals": totals}
    blob = json.dumps(summary, indent=2, ensure_ascii=False)
    (out_dir / "split_summary.json").write_text(blob, encoding="utf-8")
    return splits, summary


def main() -> None:
    """CLI entry point: build the splits and print a per-corpus summary."""
    ap = argparse.ArgumentParser(
        description="Build train/dev/test splits for every configured corpus.",
    )
    ap.add_argument("--config", default="breathlab/configs/default.yaml",
                    help="Path to the YAML configuration file.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    splits, summary = make_splits(cfg)
    print("Per-corpus split (clips / breaths):")
    for corpus, d in summary["per_corpus"].items():
        print(f"  [{corpus}]")
        for k in ["train", "dev", "test"]:
            print(f"    {k:5s} clips={d[k]['clips']:4d}  breaths={d[k]['breaths']:5d}")
    print("Totals:")
    for k in ["train", "dev", "test"]:
        t = summary["totals"][k]
        print(f"    {k:5s} clips={t['clips']:4d}  breaths={t['breaths']:5d}")
    print("Output directory:", cfg["paths"]["splits_dir"])


if __name__ == "__main__":
    main()
