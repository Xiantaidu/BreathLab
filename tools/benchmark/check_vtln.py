"""Verify the optional VTLN front end against the stock log-mel path: alpha=1 bit-exact, band centres match the slaney bank, warped banks stay monotone.

Usage: PYTHONPATH=D:/BreathLab python tools/benchmark/check_vtln.py
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from breathlab.common import load_config
from breathlab.data.features import LogMel, hz_to_mel_slaney, mel_to_hz_slaney


def tri_fb_from_f_pts(f_pts: np.ndarray, all_freqs: np.ndarray) -> np.ndarray:
    """Slaney-normalised triangular bank, mirroring ``torchaudio.functional.melscale_fbanks``."""
    n_mels = len(f_pts) - 2
    fb = np.zeros((len(all_freqs), n_mels), dtype=np.float64)
    for i in range(n_mels):
        left, centre, right = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        down = (all_freqs - left) / max(centre - left, 1e-12)
        up = (right - all_freqs) / max(right - centre, 1e-12)
        fb[:, i] = np.clip(np.minimum(down, up), 0.0, None)
    enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
    return (fb * enorm).T                      # [n_mels, n_bins]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="breathlab/configs/default.yaml")
    ap.add_argument("--sr", type=int, default=24000, help="sample rate to check at")
    ap.add_argument("--alpha-range", type=float, default=0.10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    f = cfg["feature"]
    cfg["audio"]["sample_rate"] = args.sr
    f["n_fft"] = int(round(0.025 * args.sr))
    f["win_length"] = f["n_fft"]
    f["hop_length"] = int(round(0.010 * args.sr))
    f["fmax"] = min(f["fmax"], args.sr // 2)
    lm = LogMel(cfg)

    rng = np.random.default_rng(0)
    y = (rng.standard_normal(args.sr * 3) * 0.02).astype(np.float32)

    got = lm(y)
    ref = torch.log(lm._mel(torch.from_numpy(y).unsqueeze(0)).squeeze(0)
                    + lm.log_offset).numpy().astype(np.float32)
    d1 = float(np.abs(got - ref).max())
    print(f"[1] alpha=1 vs stock torchaudio : max|diff| = {d1:.3e}  "
          f"{'OK' if d1 < 2e-6 else 'FAIL'}")

    n_bins = 1 + f["n_fft"] // 2
    all_freqs = np.linspace(0, args.sr // 2, n_bins)
    mine = tri_fb_from_f_pts(lm._f_pts, all_freqs)
    theirs = lm._stock_fb().numpy().astype(np.float64)
    d2 = float(np.abs(mine - theirs).max())
    print(f"[2] rebuilt fb vs torchaudio fb   : max|diff| = {d2:.3e}  "
          f"{'OK' if d2 < 1e-5 else 'FAIL'}")

    ok3, moves = True, []
    for alpha in np.linspace(1 - args.alpha_range, 1 + args.alpha_range, 9):
        fb = lm.fb(float(alpha)).numpy()
        centres = (fb * all_freqs[None, :]).sum(axis=1) / np.maximum(fb.sum(axis=1), 1e-12)
        if np.any(np.diff(centres) <= 0):
            ok3 = False
        moves.append(float(centres[len(centres) // 2]))
    span = max(moves) - min(moves)
    print(f"[3] monotone over alpha +/-{args.alpha_range:.2f}: {'OK' if ok3 else 'FAIL'}  "
          f"(mid-band centre moves {span:.0f} Hz across the range)")
    print(f"    mel spacing = {lm._m_spacing:.2f} mel/band, "
          f"fmax={f['fmax']} Hz, nyquist={args.sr // 2} Hz")

    d4 = float(np.abs(lm(y, alpha=1.0 + args.alpha_range) - got).mean())
    print(f"[4] mean|diff| alpha=1 vs 1+{args.alpha_range:.2f}: {d4:.4f} "
          f"({'warp has an effect' if d4 > 0.01 else 'NO EFFECT - suspicious'})")


if __name__ == "__main__":
    main()
