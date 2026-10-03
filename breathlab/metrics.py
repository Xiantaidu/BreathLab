"""Evaluation metrics: frame-level and event-level precision / recall / F1."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def frame_prf(
    prob: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Frame-level precision / recall / F1 over frames with ``mask == 1`` only."""
    pred = (prob >= threshold).astype(np.float32)
    m = mask.astype(bool)
    p, l = pred[m], label[m]
    tp = float(np.sum((p == 1) & (l == 1)))
    fp = float(np.sum((p == 1) & (l == 0)))
    fn = float(np.sum((p == 0) & (l == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def match_events(
    pred: Sequence[Tuple[float, float]],
    gt: Sequence[Tuple[float, float]],
    collar: float = 0.2,
    require_offset: bool = False,
) -> Tuple[int, int, int]:
    """Greedily match predicted events to ground truth one-to-one by onset proximity (within ``collar``, optionally also offset)."""
    gt_used = [False] * len(gt)
    tp = 0
    pred_sorted = sorted(pred, key=lambda e: e[0])
    for ps, pe in pred_sorted:
        best_j, best_d = -1, None
        for j, (gs, ge) in enumerate(gt):
            if gt_used[j]:
                continue
            if abs(ps - gs) <= collar and (not require_offset or abs(pe - ge) <= collar):
                d = abs(ps - gs)
                if best_d is None or d < best_d:
                    best_d, best_j = d, j
        if best_j >= 0:
            gt_used[best_j] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def event_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Turn event counts ``(tp, fp, fn)`` into precision / recall / F1."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def aggregate_event_prf(
    preds_per_clip: List[Sequence[Tuple[float, float]]],
    gts_per_clip: List[Sequence[Tuple[float, float]]],
    collar: float = 0.2,
    require_offset: bool = False,
) -> Dict[str, float]:
    """Pool event counts across clips, then compute P/R/F1 once (weights clips by event count instead of averaging per-clip F1)."""
    TP = FP = FN = 0
    for pred, gt in zip(preds_per_clip, gts_per_clip):
        tp, fp, fn = match_events(pred, gt, collar, require_offset)
        TP += tp
        FP += fp
        FN += fn
    return event_prf(TP, FP, FN)
