# -*- coding: utf-8 -*-
"""Re-evaluate HFA nonlexicon_labeler on test_cut for BOTH output classes (AP + EP), saving per-chunk probabilities so future charts never need the ONNX again.

EP ground truth: chunk-local slices of the corpus exhale events; AP wins on overlap, and
uncertain masks non-AP frames only (same convention as ``evaluate_probs`` ep_frame).

Run: python tools/analysis/eval_hfa_ep_testcut.py
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("D:/BreathLab")
HFA_DIR = Path("Z:/HubertFA-main/1218_hfa_model_new_dict")
CUT = Path("C:/BreathLab_data/test_cut")
OUT = ROOT / "runs" / "hfa_compare"
FPS = 100
THRESHOLD = 0.5


def softmax(x, axis=1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def hfa_events(prob, thr=0.5, max_gap=5, min_frames=10, frame_len=0.01):
    """Original HFA nll_decoder event rules."""
    evs, start, gap = [], None, 0
    for i in range(len(prob)):
        if prob[i] >= thr:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            if gap < max_gap:
                gap += 1
            else:
                end = i - gap - 1
                if end > start and (end - start) >= min_frames:
                    evs.append((start * frame_len, end * frame_len))
                start, gap = None, 0
    if start is not None and (len(prob) - start) >= min_frames:
        evs.append((start * frame_len, (len(prob) - 1) * frame_len))
    return evs


def event_prf(preds, gts, collar=0.2):
    used = set()
    tp = 0
    for ps, _pe in preds:
        for k, (gs, _ge) in enumerate(gts):
            if k not in used and abs(ps - gs) <= collar:
                used.add(k)
                tp += 1
                break
    fp, fn = len(preds) - tp, len(gts) - len(used)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return tp, fp, fn, p, r, 2 * p * r / max(p + r, 1e-9)


def slice_events(events, a, b):
    """Clip corpus events to the chunk window [a, b] and shift to chunk-local seconds."""
    out = []
    for ev in events:
        s, e = max(ev.start - a, 0.0), min(ev.end - a, b - a)
        if e - s > 0.01:
            out.append((s, e))
    return out


def main():
    import onnxruntime as ort
    import sys
    sys.path.insert(0, str(ROOT))
    from breathlab.common import load_config
    from breathlab.data.parse_labels import load_all_clips

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "breathlab/configs/default.yaml")
    clips = {c.name: c for c in load_all_clips(cfg)}
    print(f"[corpus] {len(clips)} clips loaded")

    cfg_hfa = json.loads((HFA_DIR / "config.json").read_text(encoding="utf-8"))
    mel = cfg_hfa["mel_spec_config"]
    sr, hop = mel["sample_rate"], mel["hop_size"]

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(HFA_DIR / "nonlexicon_labeler.onnx"), so,
                                providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    print(f"[hfa] provider: {sess.get_providers()[0]}  sr={sr}  hop={hop}")

    rows = list(csv.DictReader(open(CUT / "manifest.csv", encoding="utf-8-sig")))
    print(f"chunks: {len(rows)}")

    cnt = {k: {"ftp": 0, "ffp": 0, "ffn": 0, "etp": 0, "efp": 0, "efn": 0}
           for k in ("ap", "ep")}
    skipped = missing_gt = 0
    per_chunk = []
    probs_bag = {}
    t0 = time.time()
    for i, r in enumerate(rows):
        cid = r["chunk_id"]
        wav_path = CUT / "wavs" / r["wav_file"]
        ann = json.loads((CUT / "annotations" / f"{cid}.json").read_text(encoding="utf-8"))
        gts_ap = [(e["start_sec"], e["end_sec"]) for e in ann["breath_events"]]

        src = clips.get(r["source_name"])
        if src is None:
            missing_gt += 1
        a = float(r["chunk_start_sec"])
        b = float(r["chunk_end_sec"])
        gts_ep = slice_events(src.exhales, a, b) if src is not None else []
        unc = slice_events(src.uncertain, a, b) if src is not None else []

        wav, srr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if srr != sr:
            from scipy.signal import resample_poly
            g = np.gcd(srr, sr)
            wav = resample_poly(wav, sr // g, srr // g)
        dur = len(wav) / sr
        try:
            logits = sess.run(["cvnt_logits"], {"waveform": wav[None, :]})[0]
        except Exception as e:
            skipped += 1
            print(f"  [skip] {cid}: {type(e).__name__}")
            continue
        probs = softmax(logits)[0]                     # [T, 3] (None, AP, EP)
        T = min(len(probs), int(round(dur * FPS)))
        ap_p, ep_p = probs[:T, 1].astype(np.float32), probs[:T, 2].astype(np.float32)
        probs_bag[cid] = np.stack([ap_p, ep_p], axis=1).astype(np.float16)

        # ---- frame ground truth at 100 fps (same ordering as build_frame_targets) ----
        ap_lab = np.zeros(T, dtype=np.int8)
        ep_lab = np.zeros(T, dtype=np.int8)
        for s, e in gts_ap:
            ap_lab[max(0, int(round(s * FPS))):min(T, int(round(e * FPS)))] = 1
        for s, e in gts_ep:
            ep_lab[max(0, int(round(s * FPS))):min(T, int(round(e * FPS)))] = 1
        ep_lab[ap_lab == 1] = 0                        # AP wins on overlap
        mask = np.ones(T, dtype=bool)
        for s, e in unc:                               # uncertain masks non-AP frames only
            mask[max(0, int(round(s * FPS))):min(T, int(round(e * FPS)))] &= \
                (ap_lab[max(0, int(round(s * FPS))):min(T, int(round(e * FPS)))] == 0)

        pred_ap = (ap_p >= THRESHOLD)
        pred_ep = (ep_p >= THRESHOLD)
        for key, pred, lab in (("ap", pred_ap, ap_lab), ("ep", pred_ep, ep_lab)):
            c = cnt[key]
            c["ftp"] += int(((pred == 1) & (lab == 1) & mask).sum())
            c["ffp"] += int(((pred == 1) & (lab == 0) & mask).sum())
            c["ffn"] += int(((pred == 0) & (lab == 1)).sum())

        ev_ap = event_prf(hfa_events(ap_p), gts_ap)
        ev_ep = event_prf(hfa_events(ep_p), gts_ep)
        cnt["ap"]["etp"] += ev_ap[0]; cnt["ap"]["efp"] += ev_ap[1]; cnt["ap"]["efn"] += ev_ap[2]
        cnt["ep"]["etp"] += ev_ep[0]; cnt["ep"]["efp"] += ev_ep[1]; cnt["ep"]["efn"] += ev_ep[2]
        per_chunk.append((cid, r["corpus"], len(gts_ap), len(gts_ep),
                          *ev_ap[:3], round(ev_ap[5], 4), *ev_ep[:3], round(ev_ep[5], 4)))
        if (i + 1) % 500 == 0:
            ap_f1 = 2 * cnt["ap"]["ftp"] / max(2 * cnt["ap"]["ftp"] + cnt["ap"]["ffp"]
                                               + cnt["ap"]["ffn"], 1)
            print(f"  {i + 1}/{len(rows)} | AP frame F1 so far {ap_f1:.3f} "
                  f"| {time.time() - t0:.0f}s")

    def block(key):
        c = cnt[key]
        fp_ = c["ftp"] / max(c["ftp"] + c["ffp"], 1)
        fr_ = c["ftp"] / max(c["ftp"] + c["ffn"], 1)
        ff1 = 2 * fp_ * fr_ / max(fp_ + fr_, 1e-9)
        ep_ = c["etp"] / max(c["etp"] + c["efp"], 1)
        er_ = c["etp"] / max(c["etp"] + c["efn"], 1)
        ef1 = 2 * ep_ * er_ / max(ep_ + er_, 1e-9)
        return {"frame": {"P": round(fp_, 4), "R": round(fr_, 4), "F1": round(ff1, 4),
                          "tp": c["ftp"], "fp": c["ffp"], "fn": c["ffn"]},
                "event": {"P": round(ep_, 4), "R": round(er_, 4), "F1": round(ef1, 4),
                          "tp": c["etp"], "fp": c["efp"], "fn": c["efn"]}}

    summary = {"model": "HFA nonlexicon_labeler", "data": "test_cut (<=30s chunks)",
               "threshold": THRESHOLD, "n_chunks": len(rows) - skipped,
               "missing_gt_clips": missing_gt, "AP": block("ap"), "EP": block("ep")}
    print("\nHFA on test_cut (AP=class1, EP=class2):")
    for k in ("AP", "EP"):
        s = summary[k]
        print(f"  {k} frame  P={s['frame']['P']:.3f} R={s['frame']['R']:.3f} "
              f"F1={s['frame']['F1']:.3f}")
        print(f"  {k} event  P={s['event']['P']:.3f} R={s['event']['R']:.3f} "
              f"F1={s['event']['F1']:.3f} (tp {s['event']['tp']} fp {s['event']['fp']} "
              f"fn {s['event']['fn']})")

    with open(OUT / "hfa_ap_ep_per_chunk.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "corpus", "gt_ap", "gt_ep",
                    "ap_tp", "ap_fp", "ap_fn", "ap_event_F1",
                    "ep_tp", "ep_fp", "ep_fn", "ep_event_F1"])
        w.writerows(per_chunk)
    (OUT / "hfa_ap_ep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(CUT / "hfa_probs.npz", **probs_bag)
    print(f"\nsaved: {OUT / 'hfa_ap_ep_summary.json'}")
    print(f"saved: {OUT / 'hfa_ap_ep_per_chunk.csv'}")
    print(f"saved: {CUT / 'hfa_probs.npz'} ({len(probs_bag)} chunks)")


if __name__ == "__main__":
    main()
