"""Evaluate a trained breath detector: dev threshold sweep picks two operating points
(``balanced`` = best event F1; ``precision_first`` = highest precision among thresholds
with F1 >= 0.95 * best), then each requested split is evaluated at both, including a
hard-negative misfire analysis; the report goes to ``<ckpt dir>/eval_report.json``."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from breathlab.common import load_config
from breathlab.data.dataset import ClipDataset, PreparedClips, read_split
from breathlab.engine import evaluate_probs, predict_clip_probs
from breathlab.models.crnn import build_model
from breathlab.speed import setup_speed
from breathlab.postprocess import density_constrained_events, prob_to_events


def overlap_frac(ev: tuple[float, float], preds: Iterable[tuple[float, float]]) -> float:
    """Compute the largest fraction of an event covered by any prediction."""
    s, e = ev
    best = 0.0
    for ps, pe in preds:
        ov = min(e, pe) - max(s, ps)
        if ov > 0:
            best = max(best, ov / max(e - s, 1e-9))
    return best


def negative_misfire(
    prep: PreparedClips,
    preds_per_clip: Sequence[Sequence[tuple[float, float]]],
    frac_thr: float = 0.5,
) -> dict[str, float | int]:
    """Measure the event-level rate at which hard-negative / silent-breath events
    are misfired as AP (overlap with a prediction of at least ``frac_thr``)."""
    hard_tot = hard_fire = sil_tot = sil_fire = 0
    for i, c in enumerate(prep.clips):
        preds = preds_per_clip[i]
        for ev in c.hard_negatives:
            hard_tot += 1
            hard_fire += overlap_frac((ev.start, ev.end), preds) >= frac_thr
        for ev in c.silent_breaths:
            sil_tot += 1
            sil_fire += overlap_frac((ev.start, ev.end), preds) >= frac_thr
    return {
        "hard_negative_total": hard_tot,
        "hard_negative_fired": hard_fire,
        "hard_negative_misfire_rate": round(hard_fire / max(hard_tot, 1), 4),
        "silent_breath_total": sil_tot,
        "silent_breath_fired": sil_fire,
        "silent_breath_misfire_rate": round(sil_fire / max(sil_tot, 1), 4),
    }


def load_ckpt(
    ckpt_path: str | Path,
    device: torch.device,
    data_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module]:
    """Load a checkpoint, rebuild the model, and move it to the target device."""
    last = None
    for _ in range(3):  # training may be overwriting the file, so retry on failure
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            break
        except Exception as e:
            last = e
    else:
        raise last
    cfg = ckpt["cfg"]
    if data_cfg is not None:
        # Override only the sections deciding which clips are loaded; model/feature/
        # postprocess/eval stay from the checkpoint, so an old checkpoint can still see
        # newly added corpora or splits instead of silently dropping those clips.
        for sec in ("paths", "data", "labels"):
            if sec in data_cfg:
                cfg[sec] = data_cfg[sec]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    # No torch.compile for evaluation: shapes vary a lot and the first compile is slow.
    model = setup_speed(model.to(device), cfg, device).eval()
    return ckpt, cfg, model


def predict_split(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    mean: Any,
    std: Any,
    device: torch.device,
    split: str,
    cache: dict[str, tuple[Any, Any, Any, Any, Any]],
) -> tuple[Any, Any, Any, Any, Any]:
    """Run inference on one split, memoising the result in ``cache``."""
    if split not in cache:
        names = read_split(cfg["paths"]["splits_dir"], split)
        prep = PreparedClips(cfg, names, build_breath_bank=False)
        ds = ClipDataset(prep, cfg, mean, std)
        # cudnn.benchmark off: clip lengths vary wildly and every new shape would
        # trigger an algorithm re-search.
        _cb = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        try:
            probs, sp_probs, extras = predict_clip_probs(model, prep, ds, device)
        finally:
            torch.backends.cudnn.benchmark = _cb
        cache[split] = (prep, ds, probs, sp_probs, extras)
    return cache[split]


def eval_at(
    cfg: dict[str, Any],
    prep: Any,
    ds: Any,
    probs: Any,
    threshold: float,
    sp_probs: Any = None,
    density_target: float = 0.0,
    extras: Any = None,
) -> dict[str, Any]:
    """Evaluate a split at one fixed threshold."""
    m = evaluate_probs(probs, prep, ds, cfg, threshold, sp_probs=sp_probs,
                       extras_probs=extras, density_target=density_target)
    pp = cfg["postprocess"]
    if density_target > 0:
        preds = [density_constrained_events(p, cfg["fps"], float(c.duration_sec),
                                            density_target, pp,
                                            float(pp.get("density_cand_threshold", 0.15)))
                 for p, c in zip(probs, prep.clips)]
    else:
        preds = [prob_to_events(p, cfg["fps"], threshold,
                                pp["median_filter_frames"], pp["min_dur_ms"], pp["merge_gap_ms"])
                 for p in probs]
    neg = negative_misfire(prep, preds)
    out = {"threshold": round(float(threshold), 3), "event": m["event"], "frame": m["frame"],
           "sp_frame": m.get("sp_frame"), "negatives": neg, "n_clips": len(prep)}
    for h in ("ep", "gs"):
        if f"{h}_frame" in m:
            out[f"{h}_frame"] = m[f"{h}_frame"]
    return out


def main() -> None:
    """Select dev operating points and evaluate every requested split."""
    ap = argparse.ArgumentParser(
        description="Evaluate a checkpoint: dev threshold sweep + final test evaluation.")
    ap.add_argument("--ckpt", default="runs/crnn_v2/best.pt")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--config", default=None,
                    help="Override the checkpoint's stale config with the current one's "
                         "paths/data (corpus list), so an old checkpoint can evaluate new "
                         "corpora or a new split (default None = use the checkpoint's "
                         "config as-is).")
    ap.add_argument("--density-target", type=float, default=0.0,
                    help="Breaths per second; > 0 switches from thresholding to per-clip "
                         "top-K density-constrained decoding (experimental: measured "
                         "WORSE than the fixed threshold on v5_24k/test, see "
                         "postprocess.density_constrained_events). 0 = off.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = load_config(args.config) if args.config else None
    ckpt, cfg, model = load_ckpt(args.ckpt, device, data_cfg=data_cfg)
    mean, std = ckpt["mean"], ckpt["std"]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"[evaluate] ckpt={args.ckpt} (epoch {ckpt.get('epoch')})  splits={splits}")

    report: dict[str, Any] = {"ckpt": str(args.ckpt), "epoch": ckpt.get("epoch"),
                              "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    cache: dict[str, tuple[Any, Any, Any, Any, Any]] = {}

    if "dev" in splits:
        prep, ds, probs, sp_probs, extras = predict_split(model, cfg, mean, std, device,
                                                          "dev", cache)
        lo, hi, step = cfg["eval"]["threshold_sweep"]
        sweep = []
        print("\n---- dev threshold sweep ----")
        for t in np.arange(lo, hi + 1e-9, step):
            m = evaluate_probs(probs, prep, ds, cfg, float(t), sp_probs=sp_probs,
                               extras_probs=extras)
            e, f = m["event"], m["frame"]
            sweep.append({"threshold": round(float(t), 3), "event": e, "frame_f1": f["f1"]})
            print(f"  t={t:.2f}  event F1={e['f1']:.3f} P={e['precision']:.3f} R={e['recall']:.3f}"
                  f" | frame F1={f['f1']:.3f}")
        best = max(sweep, key=lambda r: (r["event"]["f1"], r["event"]["precision"]))
        cand = [r for r in sweep if r["event"]["f1"] >= 0.95 * best["event"]["f1"]]
        pfirst = max(cand, key=lambda r: (r["event"]["precision"], r["event"]["f1"]))
        t_bal, t_prec = best["threshold"], pfirst["threshold"]
        report["dev_sweep"] = sweep
        report["t_balanced"], report["t_precision_first"] = t_bal, t_prec
        print(f"Selected: balanced t={t_bal:.2f} (F1={best['event']['f1']:.3f})  |  "
              f"precision-first t={t_prec:.2f} (P={pfirst['event']['precision']:.3f})")
        for name, t in [("balanced", t_bal), ("precision_first", t_prec)]:
            report[f"dev@{name}"] = eval_at(cfg, prep, ds, probs, t, sp_probs=sp_probs,
                                      density_target=args.density_target, extras=extras)

    for split in [s for s in splits if s != "dev"]:
        prep, ds, probs, sp_probs, extras = predict_split(model, cfg, mean, std, device,
                                                          split, cache)
        points = [("balanced", report.get("t_balanced", 0.5)),
                  ("precision_first", report.get("t_precision_first", 0.5))]
        for name, t in points:
            report[f"{split}@{name}"] = eval_at(cfg, prep, ds, probs, t, sp_probs=sp_probs,
                                          density_target=args.density_target, extras=extras)

    print("\n==== summary (event-level @200ms tolerance) ====")
    for split in splits:
        for name in ["balanced", "precision_first"]:
            r = report.get(f"{split}@{name}")
            if not r:
                continue
            e, f, n = r["event"], r["frame"], r["negatives"]
            counts = f"{e['tp']}/{e['tp'] + e['fp']}/{e['fn']}"
            opt = "".join(f" | {h.upper()} frame F1={r[f'{h}_frame']['f1']:.3f}"
                          for h in ("ep", "gs") if r.get(f"{h}_frame"))
            print(f"{split:5s}@{name:15s} t={r['threshold']:.2f}  event F1={e['f1']:.3f} "
                  f"(P={e['precision']:.3f} R={e['recall']:.3f}, {counts})"
                  f" | frame F1={f['f1']:.3f}{opt}"
                  f" | hardneg misfire {n['hard_negative_fired']}/{n['hard_negative_total']}"
                  f" ({n['hard_negative_misfire_rate']*100:.1f}%)"
                  f" | silent misfire {n['silent_breath_fired']}/{n['silent_breath_total']}")

    out = Path(args.ckpt).parent / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to: {out}")


if __name__ == "__main__":
    main()
