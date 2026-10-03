"""Mine hard negatives: export spans where the model reports a breath with high confidence but there is none.

Safety valves against learning away missed labels: minimum confidence, exclusion collar around
GT breaths, skip GT silence/uncertain annotations, minimum duration and per-clip cap (see --help).

Usage: PYTHONPATH=D:/BreathLab python tools/data/mine_hard_negatives.py --ckpt runs/conformer_v2/best.pt --split train --corpora c10 --out data_cache/hardneg_v2.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from breathlab.common import load_config
from breathlab.data.dataset import ClipDataset, PreparedClips, read_split
from breathlab.engine import predict_clip_probs
from breathlab.evaluate import load_ckpt
from breathlab.postprocess import prefilter_probs, prob_to_events


def overlap(a0, a1, b0, b1) -> bool:
    """Whether two half-open intervals (in seconds) intersect."""
    return min(a1, b1) - max(a0, b0) > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="breathlab/configs/default.yaml")
    ap.add_argument("--ckpt", default="runs/conformer_v2/best.pt")
    ap.add_argument("--split", default="train")
    ap.add_argument("--corpora", default="c10", help="comma separated, empty = all")
    ap.add_argument("--techniques", default="",
                    help="only mine these techniques (4th path segment of c10 clips)")
    ap.add_argument("--limit", type=int, default=3000, help="max clips to scan (0 = all)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threshold", type=float, default=None,
                    help="event threshold (default: stored in ckpt)")
    ap.add_argument("--min-prob", type=float, default=0.8,
                    help="lower bound on max raw probability inside the span")
    ap.add_argument("--min-dur-ms", type=float, default=100.0, help="minimum span duration")
    ap.add_argument("--collar-ms", type=float, default=200.0,
                    help="exclusion collar around GT breaths")
    ap.add_argument("--n-per-clip", type=int, default=3, help="max spans mined per clip")
    ap.add_argument("--skip-sp", action=argparse.BooleanOptionalAction, default=True,
                    help="skip predictions landing in GT silence (SP) (default: skip; may be a "
                         "silent_breath, semantically ambiguous); --no-skip-sp treats them as "
                         "negatives too")
    ap.add_argument("--skip-uncertain", action=argparse.BooleanOptionalAction, default=True,
                    help="skip predictions landing in uncertain annotations (default: skip)")
    ap.add_argument("--out", default="data_cache/hardneg_mined.json")
    ap.add_argument("--dump", type=int, default=5, help="print a few samples for manual review")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = load_config(args.config)
    ckpt, cfg, model = load_ckpt(args.ckpt, device, data_cfg=data_cfg)
    thr = float(args.threshold if args.threshold is not None else ckpt.get("threshold", 0.4))
    fps = cfg["fps"]
    pp = cfg["postprocess"]

    names = read_split(cfg["paths"]["splits_dir"], args.split)
    corpora = [s for s in args.corpora.split(",") if s] or None
    techs = {s.strip() for s in args.techniques.split(",") if s.strip()}

    def corpus_of(n: str) -> str:
        parts = n.split("/")
        return parts[0]

    picked = []
    for n in names:
        parts = n.split("/")
        if corpora and corpus_of(n) not in corpora:
            continue
        if techs and not (parts[0] == "c10" and len(parts) >= 4 and parts[3] in techs):
            continue
        picked.append(n)
    random.Random(args.seed).shuffle(picked)
    if args.limit:
        picked = picked[:args.limit]
    print(f"[mine] split={args.split} matched {len(picked)} clips "
          f"(corpora={corpora} techniques={techs or 'any'}) threshold={thr}")

    prep = PreparedClips(cfg, picked, build_breath_bank=False, preload=True)
    ds = ClipDataset(prep, cfg, ckpt["mean"], ckpt["std"])
    _cb = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        probs, _sp, _extras = predict_clip_probs(model, prep, ds, device)
    finally:
        torch.backends.cudnn.benchmark = _cb
    med = prefilter_probs(probs, pp["median_filter_frames"])

    collar = args.collar_ms / 1000.0
    mined, missed, stats = {}, {}, Counter()
    durs = []
    for i, c in enumerate(tqdm(prep.clips, desc="mining", unit="clip")):
        preds = prob_to_events(med[i], fps, thr, pp["median_filter_frames"], pp["min_dur_ms"],
                               pp["merge_gap_ms"], pre_filtered=True)
        gt = [(e.start, e.end) for e in c.breath_events]
        sp = [(e.start, e.end) for e in c.sp_events] if args.skip_sp else []
        unc = [(e.start, e.end) for e in c.uncertain] if args.skip_uncertain else []
        out = []
        for s, e in preds:
            if (e - s) * 1000.0 < args.min_dur_ms:
                stats["skipped: too short"] += 1
                continue
            a, b = int(s * fps), int(np.ceil(e * fps))
            pk = float(np.max(probs[i][a:b]))
            # On top of a GT breath -> drop; adjacent but disjoint (±collar) -> suspected
            # missing label / boundary offset, keep high-confidence ones as a label-QA lead.
            if any(overlap(s, e, g0, g1) for g0, g1 in gt):
                stats["skipped: on top of a GT breath"] += 1
                continue
            if any(overlap(s, e, g0 - collar, g1 + collar) for g0, g1 in gt):
                stats["suspicious: next to a GT breath (missing label / offset)"] += 1
                if pk >= args.min_prob:
                    missed.setdefault(c.name, []).append(
                        [round(float(s), 3), round(float(e), 3), round(pk, 3)])
                continue
            if pk < args.min_prob:
                stats["skipped: confidence too low"] += 1
                continue
            if any(overlap(s, e, m0, m1) for m0, m1 in sp):
                stats["skipped: inside GT silence"] += 1
                continue
            if any(overlap(s, e, m0, m1) for m0, m1 in unc):
                stats["skipped: inside uncertain annotation"] += 1
                continue
            out.append([round(float(s), 3), round(float(e), 3)])
            if len(out) >= args.n_per_clip:
                break
        if out:
            mined[c.name] = out
            durs += [(b - a) for a, b in out]
            stats[f"accepted:{c.corpus}"] += len(out)

    print(f"\n[mine] result: {sum(len(v) for v in mined.values())} spans / {len(mined)} clips")
    for k, v in stats.most_common():
        print(f"    {k:28s} {v}")
    if durs:
        print(f"   accepted span duration: median {np.median(durs):.2f}s  "
              f"mean {np.mean(durs):.2f}s  total {sum(durs):.0f}s")
    if args.dump:
        print("\n[mine] samples (please verify manually that these are really not breaths):")
        for name, evs in list(mined.items())[:args.dump]:
            print(f"    {name}  ->  {evs}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mined, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[mine] wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    if missed:
        mp = out.with_name(out.stem + "_cand_missing_labels.json")
        mp.write_text(json.dumps(missed, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[mine] {sum(len(v) for v in missed.values())} high-confidence predictions also "
              f"conflict with GT breaths (suspected missing labels / {len(missed)} clips) -> {mp}")
        print("[mine] ^ this is a label-QA lead, not a negative; spot-check a few by hand")
    print(f"[mine] to enable: set data.hardneg_file: \"{out}\" in default.yaml (window "
          f"oversampling), optionally train.hardneg_loss_weight: 0.5~3.0 (per-frame weighting)")


if __name__ == "__main__":
    main()
