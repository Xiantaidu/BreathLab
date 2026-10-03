"""Post-processing: breath probabilities -> AP events, and full AP/SP/V segmentation."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import medfilt


def _runs(binary: np.ndarray) -> List[Tuple[int, int]]:
    """Maximal runs of non-zero entries as half-open ``(start, end)`` frame spans."""
    out = []
    n = len(binary)
    i = 0
    while i < n:
        if binary[i] > 0:
            j = i
            while j < n and binary[j] > 0:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def prob_to_events(
    prob: np.ndarray,
    fps: int,
    threshold: float,
    median_filter_frames: int = 5,
    min_dur_ms: int = 120,
    merge_gap_ms: int = 80,
    pre_filtered: bool = False,
) -> List[Tuple[float, float]]:
    """Per-frame breath probability -> events: optional median filter -> threshold -> merge gaps -> drop runs shorter than ``min_dur_ms``."""
    p = prob.astype(np.float32)
    k = median_filter_frames
    if not pre_filtered and k and k >= 3 and k % 2 == 1:
        p = medfilt(p, kernel_size=k)
    binary = (p >= threshold).astype(np.int32)
    runs = _runs(binary)

    merge_gap = int(round(merge_gap_ms * 0.001 * fps))
    merged: List[Tuple[int, int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    min_dur = int(round(min_dur_ms * 0.001 * fps))
    events = [(a / fps, b / fps) for a, b in merged if (b - a) >= min_dur]
    return events


def density_constrained_events(
    prob: np.ndarray,
    fps: int,
    duration_sec: float,
    target_density: float,
    pp: Dict,
    cand_threshold: float = 0.15,
    pre_filtered: bool = False,
) -> List[Tuple[float, float]]:
    """Decode with an imposed breath count ``K = round(target_density * duration)``; measured to not beat a fixed threshold (F1 0.717 -> 0.701 even with the oracle per-clip count), kept as an off-by-default experimental switch."""
    k = int(round(float(target_density) * max(float(duration_sec), 0.0)))
    if k <= 0:
        return []
    cands = prob_to_events(prob, fps, float(cand_threshold), 1, pp["min_dur_ms"],
                           pp["merge_gap_ms"], pre_filtered=pre_filtered)
    if len(cands) <= k:
        return cands
    def peak(e: Tuple[float, float]) -> float:
        a, b = int(e[0] * fps), max(int(e[1] * fps) + 1, int(e[0] * fps) + 1)
        return float(np.asarray(prob)[a:b].max())
    keep = sorted(cands, key=peak, reverse=True)[:k]
    return sorted(keep, key=lambda e: e[0])


def prefilter_probs(probs: List[np.ndarray], median_filter_frames: int) -> List[np.ndarray]:
    """Apply the configured median filter once, so a threshold sweep can reuse it."""
    k = median_filter_frames
    if k and k >= 3 and k % 2 == 1:
        return [medfilt(p.astype(np.float32), kernel_size=k) for p in probs]
    return [p.astype(np.float32) for p in probs]


def frames_to_ap_mask(events: List[Tuple[float, float]], n_frames: int, fps: int) -> np.ndarray:
    """Rasterise breath events back into a per-frame binary mask."""
    m = np.zeros(n_frames, dtype=np.int32)
    for s, e in events:
        a = max(0, int(round(s * fps)))
        b = min(n_frames, int(round(e * fps)))
        m[a:b] = 1
    return m


def _segments_from_classes(cls: np.ndarray, fps: int) -> List[Tuple[str, float, float]]:
    """Collapse a per-frame class array (0=V, 1=AP, 2=SP) into contiguous ``(label, start_sec, end_sec)`` segments."""
    names = {0: "V", 1: "AP", 2: "SP"}
    out = []
    n = len(cls)
    i = 0
    while i < n:
        j = i
        while j < n and cls[j] == cls[i]:
            j += 1
        out.append((names[int(cls[i])], i / fps, j / fps))
        i = j
    return out


def full_segmentation(
    prob: np.ndarray,
    energy_db: np.ndarray,
    cfg: Dict,
    threshold: float,
    hf_db: np.ndarray | None = None,
    sp_prob: np.ndarray | None = None,
    ap_events: List[Tuple[float, float]] | None = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[str, float, float]]]:
    """Breath events plus a full AP/SP/V segmentation; SP is decided by a supervised SP head when available, else by an energy rule with an optional high-band guard that excludes fricatives."""
    fps = cfg["fps"]
    pp = cfg["postprocess"]
    # SP veto: a confidently-silent frame (supervised SP head) must not become AP; disable with postprocess.sp_veto_ap: false
    if sp_prob is not None and pp.get("sp_veto_ap", True):
        veto = np.asarray(sp_prob) >= float(pp.get("sp_threshold", 0.5))
        if veto.shape == prob.shape and veto.any():
            prob = np.where(veto, 0.0, prob)
    if ap_events is None:
        ap_events = prob_to_events(
            prob, fps, threshold,
            pp["median_filter_frames"], pp["min_dur_ms"], pp["merge_gap_ms"],
        )
    n = len(prob)
    cls = np.zeros(n, dtype=np.int32)
    ap_mask = frames_to_ap_mask(ap_events, n, fps)
    cls[ap_mask == 1] = 1

    non_ap = ap_mask == 0
    if np.any(non_ap):
        if sp_prob is not None:
            sp_cand = non_ap & (sp_prob >= float(pp.get("sp_threshold", 0.5)))
        else:
            floor_pct = float(pp.get("sp_floor_percentile", pp.get("sp_energy_percentile", 5)))
            floor_e = np.percentile(energy_db[non_ap], floor_pct)
            sp_cand = non_ap & (energy_db <= floor_e + float(pp.get("sp_floor_margin_db", 8.0)))
            if hf_db is not None and len(hf_db) >= n:
                floor_h = np.percentile(hf_db[non_ap], floor_pct)
                sp_cand &= hf_db <= floor_h + float(pp.get("sp_hf_guard_db", 6.0))
        cls[sp_cand] = 2

    sp_min = int(round(pp["sp_min_dur_ms"] * 0.001 * fps))
    binary_sp = (cls == 2).astype(np.int32)
    for a, b in _runs(binary_sp):
        if (b - a) < sp_min:
            cls[a:b] = 0

    segments = _segments_from_classes(cls, fps)
    return ap_events, segments
