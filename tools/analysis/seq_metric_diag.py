"""Why are breaths missed / over-detected: threshold + postprocess sweeps and error attribution on the position metric (misses bucketed by in-event probability peak; extras by distance to the nearest GT breath). Probability curves are cached to .npz so sweeps are free.

Usage: PYTHONPATH=D:/BreathLab python tools/analysis/seq_metric_diag.py --ckpt runs/v5_24k/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_seq_eval import (  # noqa: E402
    BackboneCache,
    backbone,
    event_slots,
    lcs_pairs,
)

from breathlab.data.dataset import ClipDataset, PreparedClips, read_split  # noqa: E402
from breathlab.engine import predict_clip_probs  # noqa: E402
from breathlab.evaluate import load_ckpt  # noqa: E402
from breathlab.postprocess import prefilter_probs, prob_to_events  # noqa: E402


def match_tol(ref: list, hyp: list, tol: int) -> tuple:
    """Monotone matching allowing |slot difference| <= tol; returns (matched, missing, extra)."""
    pairs = lcs_pairs(ref, hyp) if tol == 0 else []
    if tol == 0:
        return len(pairs), len(ref) - len(pairs), len(hyp) - len(pairs)
    i = j = m = 0
    while i < len(ref) and j < len(hyp):
        d = hyp[j] - ref[i]
        if abs(d) <= tol:
            m += 1
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return m, len(ref) - m, len(hyp) - m


def prf(m: int, miss: int, extra: int) -> tuple:
    p = m / (m + extra) if (m + extra) else 0.0
    r = m / (m + miss) if (m + miss) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default="runs/v5_24k/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default="breathlab/configs/default.yaml")
    ap.add_argument("--cache", default=None, help="npz holding the cached probability curves")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from breathlab.common import load_config
    ckpt, cfg, model = load_ckpt(args.ckpt, device, data_cfg=load_config(args.config))
    mean, std = ckpt["mean"], ckpt["std"]
    thr0 = float(ckpt.get("threshold", 0.5))
    fps = cfg["fps"]
    pp = cfg["postprocess"]
    tag = Path(args.ckpt).parent.name
    cache_path = Path(args.cache) if args.cache else Path("runs/diag") / f"probs_{tag}_{args.split}.npz"

    print(f"[diag] ckpt={args.ckpt} split={args.split} thr={thr0:.2f}")
    names = read_split(cfg["paths"]["splits_dir"], args.split)
    prep = PreparedClips(cfg, names, build_breath_bank=False, preload=False)
    print(f"[diag] clips={len(prep)}")

    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        probs = [z[k] for k in sorted(z.files, key=lambda s: int(s.split("_")[1]))]
        print(f"[diag] loaded cached probs: {cache_path}")
    else:
        ds = ClipDataset(prep, cfg, mean, std)
        _cb = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        try:
            probs, _sp, _ex = predict_clip_probs(model, prep, ds, device)
        finally:
            torch.backends.cudnn.benchmark = _cb
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **{f"p_{i}": p for i, p in enumerate(probs)})
        print(f"[diag] saved probs: {cache_path}")

    cache = BackboneCache(cfg)
    starts_list = [backbone(c, cache) for c in prep.clips]
    gt = [[(e.start, e.end) for e in c.breath_events] for c in prep.clips]
    usable = [i for i, s in enumerate(starts_list) if s is not None]
    print(f"[diag] clips with phoneme backbone: {len(usable)}/{len(prep)}")

    def events_at(thr, med, min_dur, gap):
        return [prob_to_events(med[i], fps, thr, pp["median_filter_frames"],
                               min_dur, gap, pre_filtered=True) for i in usable]

    def score(events, tol=0):
        m = miss = extra = 0
        for k, i in enumerate(usable):
            ref = event_slots(gt[i], starts_list[i], fps)
            hyp = event_slots(events[k], starts_list[i], fps)
            a, b, c = match_tol(ref, hyp, tol)
            m += a
            miss += b
            extra += c
        return m, miss, extra

    med_cur = prefilter_probs(probs, pp["median_filter_frames"])

    print("\n==== threshold sweep (cfg postprocess: "
          f"min_dur={pp['min_dur_ms']}ms gap={pp['merge_gap_ms']}ms) ====")
    sweep = []
    for t in [round(x, 2) for x in np.arange(0.20, 0.86, 0.05)]:
        ev = events_at(t, med_cur, pp["min_dur_ms"], pp["merge_gap_ms"])
        m, miss, extra = score(ev)
        p, r, f = prf(m, miss, extra)
        m1, mi1, ex1 = score(ev, tol=1)
        _p1, _r1, f1 = prf(m1, mi1, ex1)
        sweep.append({"thr": t, "P": p, "R": r, "F1": f, "F1_tol1": f1,
                      "matched": m, "missing": miss, "extra": extra})
        print(f"  t={t:.2f}  P={p:.3f} R={r:.3f} F1={f:.3f} | 漏={miss:5d} 多={extra:5d} "
              f"| 放宽±1音位 F1={f1:.3f}")
    best = max(sweep, key=lambda d: d["F1"])
    base = [d for d in sweep if abs(d["thr"] - thr0) < 1e-9]
    base = base[0] if base else min(sweep, key=lambda d: abs(d["thr"] - thr0))
    print(f"  当前 ckpt 阈值 {thr0:.2f}: F1={base['F1']:.3f} | 最优 {best['thr']:.2f}: "
          f"F1={best['F1']:.3f}  -> 阈值可调空间 {best['F1'] - base['F1']:+.3f}")

    print("\n==== postprocess sweep (at ckpt threshold) ====")
    pp_rows = []
    for med_k in (1, 3, 5):
        med = prefilter_probs(probs, med_k)
        for md in (30, 60, 120, 200):
            for gap in (80, 150, 300):
                ev = events_at(thr0, med, md, gap)
                m, miss, extra = score(ev)
                p, r, f = prf(m, miss, extra)
                pp_rows.append({"median": med_k, "min_dur": md, "gap": gap,
                                "P": p, "R": r, "F1": f})
    top = sorted(pp_rows, key=lambda d: -d["F1"])[:6]
    for d in top:
        print(f"  median={d['median']} min_dur={d['min_dur']:3d}ms gap={d['gap']:3d}ms -> "
              f"F1={d['F1']:.3f} (P={d['P']:.3f} R={d['R']:.3f})")
    cur = [d for d in pp_rows if d["median"] == pp["median_filter_frames"]
           and d["min_dur"] == pp["min_dur_ms"] and d["gap"] == pp["merge_gap_ms"]][0]
    print(f"  当前配置 F1={cur['F1']:.3f} -> 后处理可调空间 {top[0]['F1'] - cur['F1']:+.3f}")

    ev = events_at(thr0, med_cur, pp["min_dur_ms"], pp["merge_gap_ms"])
    miss_buckets = {"<0.1 (完全没信号)": 0, "0.1-0.2": 0, "0.2-0.3": 0, "0.3-0.5": 0,
                    ">=0.5 (标了但被合并/时长规则吃掉)": 0}
    extra_buckets = {"紧贴 GT(<=0.3s, 边界/口径差)": 0, "0.3-1s": 0, ">1s": 0, "整条无 GT 呼吸": 0}
    extra_in_sp = 0
    for k, i in enumerate(usable):
        st = starts_list[i]
        ref_slots = event_slots(gt[i], st, fps)
        hyp_slots = event_slots(ev[k], st, fps)
        pairs = lcs_pairs(ref_slots, hyp_slots)
        mr = {a for a, _ in pairs}
        mh = {b for _, b in pairs}
        p = probs[i]
        for gi, (s, e) in enumerate(gt[i]):
            if gi in mr:
                continue
            a, b = max(0, int(s * fps)), min(len(p), int(e * fps) + 1)
            v = float(p[a:b].max()) if b > a else 0.0
            if v >= 0.5:
                miss_buckets[">=0.5 (标了但被合并/时长规则吃掉)"] += 1
            elif v >= 0.3:
                miss_buckets["0.3-0.5"] += 1
            elif v >= 0.2:
                miss_buckets["0.2-0.3"] += 1
            elif v >= 0.1:
                miss_buckets["0.1-0.2"] += 1
            else:
                miss_buckets["<0.1 (完全没信号)"] += 1
        sp_iv = [(x.start, x.end) for x in prep.clips[i].sp_events]
        for hj, (s, e) in enumerate(ev[k]):
            if hj in mh:
                continue
            if not gt[i]:
                extra_buckets["整条无 GT 呼吸"] += 1
                continue
            d = min(min(abs(s - gs), abs(s - ge)) for gs, ge in gt[i])
            key = ("紧贴 GT(<=0.3s, 边界/口径差)" if d <= 0.3
                   else "0.3-1s" if d <= 1.0 else ">1s")
            extra_buckets[key] += 1
            mid = 0.5 * (s + e)
            if any(a <= mid <= b for a, b in sp_iv):
                extra_in_sp += 1

    n_miss = sum(miss_buckets.values())
    n_extra = sum(extra_buckets.values())
    print(f"\n==== 漏报归因（{n_miss} 个）: GT 区间内概率峰值 ====")
    for k, v in miss_buckets.items():
        print(f"  {k:32s} {v:5d}  ({v / n_miss:.1%})" if n_miss else f"  {k:32s} {v:5d}")
    print(f"==== 多报归因（{n_extra} 个）: 距最近 GT 呼吸 ====")
    for k, v in extra_buckets.items():
        print(f"  {k:32s} {v:5d}  ({v / n_extra:.1%})" if n_extra else f"  {k:32s} {v:5d}")
    print(f"  其中落在标注静音(SP)区间内: {extra_in_sp}  ({extra_in_sp / n_extra:.1%})"
          if n_extra else "")

    out = Path(args.out) if args.out else Path("runs/diag") / f"seq_diag_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": args.ckpt, "split": args.split, "threshold": thr0,
        "n_clips": len(prep), "n_usable": len(usable),
        "current": cur, "best_threshold": best, "sweep": sweep,
        "postprocess_top": top, "miss_attribution": miss_buckets,
        "extra_attribution": extra_buckets, "extra_in_sp": extra_in_sp,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
