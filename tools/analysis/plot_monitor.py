"""Plot a run's dev monitor curve from a training log, ``history.json`` or CSV.

Usage: PYTHONPATH=D:/BreathLab python tools/analysis/plot_monitor.py --log train.log [--history runs/x/history.json | --csv epochs.csv]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

FONT = "Microsoft YaHei"
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

C_LINE, C_TREND, C_BEST = "#534AB7", "#888780", "#1D9E75"


def parse_log(text: str) -> tuple[list[int], list[float]]:
    """Extract ``(epoch, monitor)`` pairs from training-log text."""
    ep = re.compile(r"epoch\s+(\d+)\s+loss=")
    mo = re.compile(r"monitor=([\d.]+)")
    ap = re.compile(r"dev event F1=([\d.]+)")
    epochs, mons, aps = [], [], []
    for ln in text.splitlines():
        m = ep.search(ln)
        if not m:
            continue
        mm = mo.search(ln) or ap.search(ln)
        if not mm:
            continue
        epochs.append(int(m.group(1)))
        mons.append(float(mm.group(1)))
    return epochs, mons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", default=None, help="training log file (default: stdin)")
    ap.add_argument("--history", default=None, help="run history.json instead of a log")
    ap.add_argument("--csv", default=None, help="CSV with epoch,monitor columns")
    ap.add_argument("--out", default="runs/diag/monitor_curve.png")
    args = ap.parse_args()

    if args.csv:
        import csv as _csv
        with open(args.csv, newline="", encoding="utf-8-sig") as f:
            rows = list(_csv.DictReader(f))
        epochs = [int(r["epoch"]) for r in rows]
        mons = [float(r["monitor"]) for r in rows]
    elif args.history:
        import json
        d = json.loads(Path(args.history).read_text(encoding="utf-8"))
        rows = d if isinstance(d, list) else list(d.values())
        epochs, mons = [], []
        for r in rows:
            ev = r.get("dev_event") or {}
            epf = (r.get("dev_ep_frame") or {})
            if "f1" not in ev:
                continue
            w_ap = 0.65
            w_ep = 0.35 if "f1" in epf else 0.0
            epochs.append(int(r.get("epoch", len(epochs) + 1)))
            mons.append(w_ap * ev["f1"] + (w_ep * epf["f1"] if w_ep else 0.0))
    else:
        text = Path(args.log).read_text(encoding="utf-8", errors="ignore") \
            if args.log else sys.stdin.read()
        epochs, mons = parse_log(text)

    if not epochs:
        raise SystemExit("no epoch/monitor rows found")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(epochs, mons, "-o", color=C_LINE, lw=1.8, ms=5, label="monitor")
    if len(epochs) >= 2:
        k, b = np.polyfit(epochs, mons, 1)
        xs = np.array([epochs[0], epochs[-1]], float)
        ax.plot(xs, k * xs + b, "--", color=C_TREND, lw=1.2,
                label=f"线性趋势 (slope={k:+.4f}/epoch)")
    best = max(range(len(mons)), key=lambda i: mons[i])
    ax.annotate(f"最好 epoch {epochs[best]}\nmonitor={mons[best]:.3f}",
                (epochs[best], mons[best]), textcoords="offset points",
                xytext=(8, -34), fontsize=10, color=C_BEST,
                arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.0))
    ax.set_xlabel("epoch")
    ax.set_ylabel("monitor = 0.65·AP event F1 + 0.35·EP frame F1")
    ax.set_title("dev monitor 曲线（只看 monitor）", fontsize=13)
    ax.grid(color="#D3D1C7", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}  epochs={len(epochs)}  best=epoch {epochs[best]} "
          f"({mons[best]:.3f})")
    print("monitor 序列:", ", ".join(f"{e}:{m}" for e, m in zip(epochs, mons)))


if __name__ == "__main__":
    main()
