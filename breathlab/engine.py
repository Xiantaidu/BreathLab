"""Inference and evaluation engine: windowed whole-clip probabilities plus
threshold sweeps and event/frame-level metrics (with per-group breakdowns)."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from breathlab.data.dataset import ClipDataset, PreparedClips
from breathlab.metrics import aggregate_event_prf, frame_prf
from breathlab.postprocess import (density_constrained_events, prefilter_probs,
                                  prob_to_events)


@torch.no_grad()
def predict_probs_windowed(
    model,
    lm: torch.Tensor,
    cfg: Dict,
    device: torch.device | str,
    batch: int = 8,
) -> Tuple[np.ndarray, np.ndarray | None, Dict[str, np.ndarray]]:
    """Run windowed sliding-window inference over a normalized log-mel spectrogram,
    averaging overlaps (windows bound attention's O(T^2) cost while still containing
    complete breaths; batched windows do not interact, so results are identical)."""
    ev = cfg.get("eval", {})
    fps = cfg["fps"]
    win = int(round(ev.get("infer_window_sec", 12.0) * fps))
    hop = int(round(ev.get("infer_hop_sec", 6.0) * fps))
    amp = device.type == "cuda"
    T = lm.shape[1]
    if T <= win:
        starts = [0]
    else:
        starts = list(range(0, T - win + 1, max(1, hop)))
        if starts[-1] != T - win:
            starts.append(T - win)
    ap_acc = np.zeros(T, dtype=np.float64)
    sp_acc = np.zeros(T, dtype=np.float64)
    hi = getattr(model, "head_index", None) or {"ap": 0, "sp": 1}
    MANDATORY = {"ap", "sp", "start", "end"}
    opt_heads = {k: int(v) for k, v in hi.items() if k not in MANDATORY}
    opt_acc = {k: np.zeros(T, dtype=np.float64) for k in opt_heads}
    wsum = np.zeros(T, dtype=np.float64)
    ap_i = int(hi.get("ap", 0))
    sp_i = int(hi["sp"]) if "sp" in hi else None
    sp_ok = False
    for i0 in range(0, len(starts), batch):
        cs = starts[i0:i0 + batch]
        w = min(win, T - cs[0])   # equal length within a batch: win, or T for a short clip
        xs = torch.stack([lm[:, s:s + w] for s in cs]).to(device)
        with torch.autocast(
            device_type="cuda" if amp else "cpu", dtype=torch.bfloat16, enabled=amp
        ):
            out = model(xs)                      # [n, C, w]
        p = torch.sigmoid(out.float().cpu())     # [n, C, w]
        for j, s in enumerate(cs):
            ap_acc[s:s + w] += p[j, ap_i].numpy()
            if sp_i is not None and out.shape[1] > sp_i:
                sp_acc[s:s + w] += p[j, sp_i].numpy()
                sp_ok = True
            for k, ci in opt_heads.items():
                if out.shape[1] > ci:
                    opt_acc[k][s:s + w] += p[j, ci].numpy()
            wsum[s:s + w] += 1.0
    w = np.maximum(wsum, 1e-8)
    ap = (ap_acc / w).astype(np.float32)
    sp = (sp_acc / w).astype(np.float32) if sp_ok else None
    extras = {k: (v / w).astype(np.float32) for k, v in opt_acc.items()}
    return ap, sp, extras


@torch.no_grad()
def predict_clip_probs(
    model,
    prepared: PreparedClips,
    clip_ds: ClipDataset,
    device: torch.device | str,
    progress: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray | None], List[Dict[str, np.ndarray]]]:
    """Run inference over every clip of a :class:`ClipDataset`, returning per-clip
    ``(ap_probs, sp_probs, extras_probs)`` probability lists."""
    model.eval()
    ap_list, sp_list, extras_rows = [], [], []
    it = range(len(clip_ds))
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc="dev inference", unit="clip", leave=False)
    for i in it:
        lm, _ap, _sp, _mask, _ep, _gs, _idx = clip_ds[i]   # [F, T], already normalized
        ap, sp, extras = predict_probs_windowed(model, lm, clip_ds.cfg, device)
        ap_list.append(ap)
        sp_list.append(sp)
        extras_rows.append(extras)
    # transpose list-of-per-clip-dicts -> per-head lists (evaluate_probs' layout)
    if extras_rows and extras_rows[0]:
        extras_probs = {h: [row[h] for row in extras_rows] for h in extras_rows[0]}
    else:
        extras_probs = {}
    return ap_list, sp_list, extras_probs


def gt_events(prepared: PreparedClips) -> List[List[Tuple[float, float]]]:
    """Collect ground-truth breath events for every clip."""
    return [[(ev.start, ev.end) for ev in c.breath_events] for c in prepared.clips]


def evaluate_probs(
    probs: List[np.ndarray],
    prepared: PreparedClips,
    clip_ds: ClipDataset,
    cfg: Dict,
    threshold: float,
    sp_probs: List[np.ndarray] | None = None,
    med_probs: List[np.ndarray] | None = None,
    extras_probs: Dict[str, List[np.ndarray]] | None = None,
    density_target: float = 0.0,
) -> Dict:
    """Evaluate one threshold at frame and event level (``med_probs`` skips re-filtering
    during sweeps; ``extras_probs`` adds an ``<name>_frame`` entry per optional head)."""
    fps = cfg["fps"]
    pp = cfg["postprocess"]
    collar = cfg["eval"]["event_collar_ms"] * 0.001

    gts = gt_events(prepared)
    src = med_probs if med_probs is not None else probs
    if density_target > 0:
        # Optional density-constrained decoding, measured worse than plain thresholding
        # (see postprocess.density_constrained_events).
        clips = clip_ds.p.clips
        preds = [
            density_constrained_events(p, fps, float(clips[i].duration_sec), density_target,
                                       pp, float(pp.get("density_cand_threshold", 0.15)),
                                       pre_filtered=med_probs is not None)
            for i, p in enumerate(src)
        ]
    else:
        preds = [
            prob_to_events(p, fps, threshold, pp["median_filter_frames"], pp["min_dur_ms"],
                           pp["merge_gap_ms"], pre_filtered=med_probs is not None)
            for p in src
        ]
    ev = aggregate_event_prf(preds, gts, collar=collar, require_offset=False)

    all_p = np.concatenate(probs)
    all_l = np.concatenate([clip_ds.p.labels[i] for i in range(len(clip_ds))])
    all_m = np.concatenate([clip_ds.p.masks[i] for i in range(len(clip_ds))])
    fr = frame_prf(all_p, all_l, all_m, threshold)
    out = {"event": ev, "frame": fr, "threshold": threshold}

    if sp_probs is not None and any(s is not None for s in sp_probs):
        all_sp = np.concatenate([s for s in sp_probs if s is not None])
        all_sl = np.concatenate([clip_ds.p.sp_labels[i] for i in range(len(clip_ds))])
        out["sp_frame"] = frame_prf(all_sp, all_sl, all_m, float(pp.get("sp_threshold", 0.5)))
    if extras_probs:
        for name, per_clip in extras_probs.items():
            if not per_clip or all(p is None for p in per_clip):
                continue
            all_o = np.concatenate([p for p in per_clip if p is not None])
            lab_attr = {"ep": "ep_labels", "gs": "gs_labels"}.get(name, f"{name}_labels")
            all_ol = np.concatenate([
                getattr(clip_ds.p, lab_attr)[i] for i in range(len(clip_ds))])
            out[f"{name}_frame"] = frame_prf(
                all_o, all_ol, all_m, float(pp.get(f"{name}_threshold", 0.5)))
    return out


class SubsetView:
    """Minimal index-subset view exposing the fields ``evaluate_probs`` needs, so
    per-group metrics avoid re-running inference or re-decoding audio."""

    def __init__(self, prep, idx: List[int]):
        """Build the subset view from ``prep`` and the clip indices ``idx``."""
        self.p = self
        self.clips = [prep.clips[i] for i in idx]
        self.labels = [prep.labels[i] for i in idx]
        self.masks = [prep.masks[i] for i in idx]
        self.sp_labels = [prep.sp_labels[i] for i in idx]
        self.ep_labels = [prep.ep_labels[i] for i in idx]
        self.gs_labels = [prep.gs_labels[i] for i in idx]

    def __len__(self) -> int:
        """Return the number of clips in the subset."""
        return len(self.clips)


def group_indices(prep, key_fn: Callable) -> Dict[str, List[int]]:
    """Group clip indices by ``key_fn(clip)``, sorted by descending group size."""
    g: Dict[str, List[int]] = {}
    for i, c in enumerate(prep.clips):
        g.setdefault(key_fn(c), []).append(i)
    return dict(sorted(g.items(), key=lambda kv: -len(kv[1])))


def group_metrics(
    probs: List[np.ndarray],
    prepared: PreparedClips,
    clip_ds: ClipDataset,
    cfg: Dict,
    threshold: float,
    key_fn: Callable,
    sp_probs: List[np.ndarray] | None = None,
    med_probs: List[np.ndarray] | None = None,
    extras_probs: Dict[str, List[np.ndarray]] | None = None,
) -> Dict[str, Dict]:
    """Compute metrics per group via subset views (no re-inference): a single aggregate
    F1 can be dominated by one singer/corpus and hide where the model actually fails."""
    out: Dict[str, Dict] = {}
    for name, idx in group_indices(prepared, key_fn).items():
        view = SubsetView(prepared, idx)
        out[name] = evaluate_probs(
            [probs[i] for i in idx], view, view, cfg, threshold,
            sp_probs=None if sp_probs is None else [sp_probs[i] for i in idx],
            med_probs=None if med_probs is None else [med_probs[i] for i in idx],
            extras_probs=None if extras_probs is None else
            {k: [v[i] for i in idx] for k, v in extras_probs.items()},
        )
    return out


def sweep_threshold(
    probs: List[np.ndarray],
    prepared: PreparedClips,
    clip_ds: ClipDataset,
    cfg: Dict,
) -> Tuple[float, List[Dict]]:
    """Sweep the decision threshold on dev and pick the best one by event F1
    (ties broken by precision)."""
    lo, hi, step = cfg["eval"]["threshold_sweep"]
    results = []
    ts = np.arange(lo, hi + 1e-9, step)
    for t in ts:
        r = evaluate_probs(probs, prepared, clip_ds, cfg, float(t))
        results.append(r)
    best = max(results, key=lambda r: (r["event"]["f1"], r["event"]["precision"]))
    return best["threshold"], results
