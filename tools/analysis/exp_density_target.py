"""Offline experiment: density-constrained decoding (per clip, top-K candidate events by peak probability, K = target_density x duration) vs a fixed threshold, on cached probability curves.

K flavours: ``oracle`` (true breath count), ``corpus`` (per-corpus GT density), ``global`` (one global density).

Usage: PYTHONPATH=D:/BreathLab python tools/analysis/exp_density_target.py --split test --probs runs/diag/probs_v5_24k_test.npz
"""
from __future__ import annotations

import argparse
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_seq_eval import (  # noqa: E402
    BackboneCache,
    backbone,
    event_slots,
    lcs_pairs,
)

from breathlab.common import load_config  # noqa: E402
from breathlab.data.dataset import PreparedClips, read_split  # noqa: E402
from breathlab.postprocess import prefilter_probs, prob_to_events  # noqa: E402

CAND_THR = 0.15          # low threshold: candidates only need to be *plausible*


def topk_events(prob: np.ndarray, fps: int, k: int, min_dur: int, gap: int,
                med: np.ndarray) -> list:
    """Keep the ``k`` highest-scoring candidate events of one clip."""
    cands = prob_to_events(med, fps, CAND_THR, 1, min_dur, gap, pre_filtered=True)
    if k <= 0:
        return []
    if len(cands) <= k:
        return cands
    scored = sorted(cands, key=lambda e: -float(prob[int(e[0] * fps):int(e[1] * fps) + 1].max()))
    return sorted(scored[:k], key=lambda e: e[0])


def score(refs, hyps) -> tuple:
    m = miss = extra = 0
    for ref, hyp in zip(refs, hyps):
        pairs = lcs_pairs(ref, hyp)
        m += len(pairs)
        miss += len(ref) - len(pairs)
        extra += len(hyp) - len(pairs)
    p = m / (m + extra) if (m + extra) else 0.0
    r = m / (m + miss) if (m + miss) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default="runs/v5_24k/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default="breathlab/configs/default.yaml")
    ap.add_argument("--probs", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from breathlab.evaluate import load_ckpt
    ckpt, cfg, _model = load_ckpt(args.ckpt, torch.device("cpu"),
                                  data_cfg=load_config(args.config))
    thr = float(ckpt.get("threshold", 0.5))
    fps = cfg["fps"]
    pp = cfg["postprocess"]
    names = read_split(cfg["paths"]["splits_dir"], args.split)
    if args.limit:
        names = names[:args.limit]
    prep = PreparedClips(cfg, names, build_breath_bank=False, preload=False)

    src = Path(args.probs) if args.probs else Path("runs/diag") / \
        f"probs_{Path(args.ckpt).parent.name}_{args.split}.npz"
    z = np.load(src, allow_pickle=True)
    probs_all = [z[k] for k in sorted(z.files, key=lambda s: int(s.split("_")[1]))]
    print(f"[exp] probs={src} ({len(probs_all)} curves)  split={args.split}  clips={len(prep)}")

    cache = BackboneCache(cfg)
    rows = []
    for i, clip in enumerate(prep.clips):
        st = backbone(clip, cache)
        if st is None or i >= len(probs_all):
            continue
        rows.append((i, clip, st, probs_all[i]))
    med = prefilter_probs([r[3] for r in rows], pp["median_filter_frames"])
    print(f"[exp] usable clips={len(rows)}")

    refs = [event_slots([(e.start, e.end) for e in r[1].breath_events], r[2], fps)
            for r in rows]
    gt_counts = [len(r) for r in refs]

    # --- baseline ---
    base_hyps = [event_slots(prob_to_events(m, fps, thr, pp["median_filter_frames"],
                                            pp["min_dur_ms"], pp["merge_gap_ms"],
                                            pre_filtered=True), rows[k][2], fps)
                 for k, m in enumerate(med)]
    p, r, f = score(refs, base_hyps)
    print(f"\nbaseline  thr={thr:.2f}                 P={p:.3f} R={r:.3f} F1={f:.3f}")

    # --- K flavours ---
    durations = [r[1].duration_sec for r in rows]
    corpora = [r[1].corpus for r in rows]
    corpus_density = {}
    for c in set(corpora):
        n = sum(gt_counts[k] for k in range(len(rows)) if corpora[k] == c)
        d = sum(durations[k] for k in range(len(rows)) if corpora[k] == c)
        corpus_density[c] = n / d if d else 0.0
    glob = sum(gt_counts) / sum(durations)

    def run(name, ks):
        hyps = [event_slots(topk_events(rows[k][3], fps, ks[k], pp["min_dur_ms"],
                                        pp["merge_gap_ms"], med[k]), rows[k][2], fps)
                for k in range(len(rows))]
        p, r, f = score(refs, hyps)
        print(f"{name:32s} P={p:.3f} R={r:.3f} F1={f:.3f}"
              + (f"   (vs baseline {f - base_f:+.3f})" if base_f else ""))
        return f

    base_f = f
    print(f"\n-- 候选阈值 {CAND_THR:.2f}，按事件峰值概率排序取 top-K --")
    run("top-K  K=GT 数（oracle 天花板）", gt_counts)
    run("top-K  K=逐语料密度×时长",
        [int(round(corpus_density[corpora[k]] * durations[k])) for k in range(len(rows))])
    run(f"top-K  K=全局密度 {glob:.3f}/s ×时长",
        [int(round(glob * durations[k])) for k in range(len(rows))])
    for g in (0.10, 0.15, 0.20, 0.25):
        run(f"top-K  K=固定 {g:.2f}/s ×时长", [int(round(g * durations[k]))
                                                for k in range(len(rows))])

    print("\n注：逐语料密度是乐观估计——真实部署只能从 dev/校准集估，未知语料只能取全局值。")


if __name__ == "__main__":
    main()
