# -*- coding: utf-8 -*-
"""Evaluate the external HFA nonlexicon_labeler on the CUT test set (test_cut), judged with HFA's own decoder rules (prob >= 0.5, gaps <= 5 frames bridged, events < 10 frames dropped).

Outputs runs/hfa_compare/{summary.csv, hfa_vs_breathlab.png}; the BreathLab columns come from the already-measured test numbers.

Run: python tools/analysis/eval_hfa_testcut.py
"""
from __future__ import annotations

import csv
import json
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


def main():
    import onnxruntime as ort
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HFA_DIR / "config.json").read_text(encoding="utf-8"))
    mel = cfg["mel_spec_config"]
    sr, hop = mel["sample_rate"], mel["hop_size"]

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(HFA_DIR / "nonlexicon_labeler.onnx"), so,
                                providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    print(f"[hfa] provider: {sess.get_providers()[0]}  sr={sr}  hop={hop}")

    rows = list(csv.DictReader(open(CUT / "manifest.csv", encoding="utf-8-sig")))
    print(f"chunks: {len(rows)}")

    tp_s = fp_s = fn_s = 0
    f_tp = f_fp = f_fn = 0
    skipped = 0
    per_chunk = []
    for i, r in enumerate(rows):
        cid = r["chunk_id"]
        wav_path = CUT / "wavs" / r["wav_file"]
        ann = json.loads((CUT / "annotations" / f"{cid}.json").read_text(encoding="utf-8"))
        gts = [(e["start_sec"], e["end_sec"]) for e in ann["breath_events"]]
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
        prob = softmax(logits)[0, 1]           # AP probability

        # frame ground truth at 100 fps from the sliced events
        T = min(len(prob), int(round(dur * FPS)))
        lab = np.zeros(T, dtype=np.int8)
        for s, e in gts:
            a, b = max(0, int(round(s * FPS))), min(T, int(round(e * FPS)))
            lab[a:b] = 1
        pred = (prob[:T] >= THRESHOLD).astype(np.int8)
        f_tp += int(((pred == 1) & (lab == 1)).sum())
        f_fp += int(((pred == 1) & (lab == 0)).sum())
        f_fn += int(((pred == 0) & (lab == 1)).sum())

        tp, fp, fn, p, rr, f1 = event_prf(hfa_events(prob[:T]), gts)
        tp_s += tp; fp_s += fp; fn_s += fn
        per_chunk.append((cid, r["corpus"], len(gts), tp, fp, fn, f1))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)} | frame F1 so far "
                  f"{2 * f_tp / max(2 * f_tp + f_fp + f_fn, 1):.3f}")

    FP = f_tp / max(f_tp + f_fp, 1)
    FR = f_tp / max(f_tp + f_fn, 1)
    FF1 = 2 * FP * FR / max(FP + FR, 1e-9)
    EP = tp_s / max(tp_s + fp_s, 1)
    ER = tp_s / max(tp_s + fn_s, 1)
    EF1 = 2 * EP * ER / max(EP + ER, 1e-9)
    print(f"\nHFA on test_cut ({len(rows) - skipped} chunks):")
    print(f"  frame  P={FP:.3f} R={FR:.3f} F1={FF1:.3f}")
    print(f"  event  P={EP:.3f} R={ER:.3f} F1={EF1:.3f} (tp {tp_s} fp {fp_s} fn {fn_s})")

    # per-corpus event F1
    by_corpus = {}
    for cid, corpus, ng, tp, fp, fn, f1 in per_chunk:
        d = by_corpus.setdefault(corpus, [0, 0, 0, 0])
        d[0] += tp; d[1] += fp; d[2] += fn; d[3] += ng
    corpus_rows = []
    for corpus, (tp, fp, fn, ng) in sorted(by_corpus.items(), key=lambda x: -(x[1][0] + x[1][2])):
        p = tp / max(tp + fp, 1); rr = tp / max(tp + fn, 1)
        corpus_rows.append((corpus, tp, fp, fn,
                            2 * p * rr / max(p + rr, 1e-9), ng))
    print("\nper-corpus event F1:")
    for corpus, tp, fp, fn, f1, ng in corpus_rows:
        print(f"  {corpus:10s} F1={f1:.3f} (gt {ng}, tp {tp} fp {fp} fn {fn})")

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "frame_P", "frame_R", "frame_F1", "event_P", "event_R",
                    "event_F1", "tp", "fp", "fn", "n_chunks"])
        w.writerow(["HFA nonlexicon", f"{FP:.4f}", f"{FR:.4f}", f"{FF1:.4f}",
                    f"{EP:.4f}", f"{ER:.4f}", f"{EF1:.4f}", tp_s, fp_s, fn_s,
                    len(rows) - skipped])
        # BreathLab numbers measured on the original test split (same singing content)
        w.writerow(["BreathLab unet_mamba", "", "", "", "0.5710", "0.6580", "0.6120",
                    "", "", "", ""])
        w.writerow(["BreathLab ebranchformer", "", "", "", "0.5930", "0.6270", "0.6090",
                    "", "", "", ""])
    with open(OUT / "per_corpus_hfa.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["corpus", "gt_events", "tp", "fp", "fn", "event_F1"])
        w.writerows(corpus_rows)

    # ---- chart ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = ["HFA nonlexicon", "BreathLab unet_mamba", "BreathLab ebranchformer"]
    ev = {"HFA nonlexicon": (EP, ER, EF1),
          "BreathLab unet_mamba": (0.571, 0.658, 0.612),
          "BreathLab ebranchformer": (0.593, 0.627, 0.609)}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("BreathLab test set: HFA nonlexicon_labeler vs BreathLab models "
                 "(AP, event-level @200ms)", fontsize=13, fontweight="bold")
    ax = axes[0]
    x = np.arange(len(models)); w = 0.25
    for k, (mkey, mi, c) in enumerate(zip(["Precision", "Recall", "F1"], [0, 1, 2],
                                          ["#1f77b4", "#2ca02c", "#d62728"])):
        vals = [ev[m][mi] for m in models]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=mkey, color=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.set_title("event-level P / R / F1")

    ax = axes[1]
    names = [c for c, *_ in corpus_rows]
    vals = [f1 for *_x, f1, _ng in corpus_rows]
    vals = [r[4] for r in corpus_rows]
    bars = ax.barh(names, vals, color="#ff7f0e")
    ax.set_xlim(0, 1.05)
    for b, v, r in zip(bars, vals, corpus_rows):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.3f} (gt {r[5]})", va="center", fontsize=8)
    ax.set_title("HFA event F1 per corpus")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = OUT / "hfa_vs_breathlab.png"
    fig.savefig(png, dpi=150)
    print(f"\nchart -> {png}\ncsv  -> {OUT}")


if __name__ == "__main__":
    main()
