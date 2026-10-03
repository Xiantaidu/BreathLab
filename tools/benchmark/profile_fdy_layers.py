"""Per-layer timing of the six FDY dynamic convolutions in the CNN front end: triton vs mix vs a plain conv of the same shape.

Usage: PYTHONPATH=D:/BreathLab python tools/benchmark/profile_fdy_layers.py [--batch 32]
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn

from breathlab.common import load_config
from breathlab.models.conformer import ConvFront
from breathlab.models.fdy import FDYConv2d


def time_fn(fn, warm: int = 3, n: int = 10) -> float:
    """Median wall-clock time of ``fn`` in milliseconds, CUDA-synchronized."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--frames", type=int, default=600)
    args = ap.parse_args()

    cfg = load_config("breathlab/configs/default.yaml")
    m = cfg["model"]
    rows_in = int(cfg["feature"].get("input_rows") or cfg["feature"]["n_mels"])
    dev = torch.device("cuda")
    front = ConvFront(m["cnn_channels"], m["cnn_freq_pool"], rows_in,
                      m.get("dropout", 0.1), fdy_m=m["fdy_m"],
                      fdy_mode=m.get("fdy_mode", "triton"), fdy_ckpt=False,
                      ch_last=True).to(dev).train()
    layers = [x for x in front.net if isinstance(x, FDYConv2d)]

    # Per-layer input shapes: convolutions keep F, pooling halves it.
    shapes, cin, f = [], 1, rows_in
    for c, fp in zip(m["cnn_channels"], m["cnn_freq_pool"]):
        shapes += [(cin, c, f), (c, c, f)]
        cin, f = c, f // fp

    B, T = args.batch, args.frames
    print(f"FDY M={m['fdy_m']} mode={m.get('fdy_mode')} batch={B} frames={T} "
          f"(forward+backward, including input gradients, channels_last as in training)")
    print(f"{'layer':>5s} {'Ci->Co':>9s} {'F':>4s} {'GFLOP':>7s} "
          f"{'FDY(ms)':>9s} {'mix(ms)':>9s} {'plain conv':>12s} {'FDY/conv':>9s}")
    total_fdy = total_plain = 0.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, (ci, co, ff) in enumerate(shapes):
            x = torch.randn(B, ci, ff, T, device=dev, requires_grad=True)
            xc = x.detach().contiguous(memory_format=torch.channels_last).requires_grad_(True)
            plain = nn.Conv2d(ci, co, 3, padding=1, bias=False).to(dev)
            plain_w = plain.weight.detach().contiguous(memory_format=torch.channels_last)

            t_fdy = time_fn(lambda: layers[i](xc).float().pow(2).mean().backward())
            layers[i].mode = "mix"
            t_mix = time_fn(lambda: layers[i](xc).float().pow(2).mean().backward())
            layers[i].mode = m.get("fdy_mode", "triton")
            t_pl = time_fn(lambda: nn.functional.conv2d(
                xc, plain_w, padding=1).float().pow(2).mean().backward())

            gflop = B * ff * T * 9 * ci * co * 2 / 1e9
            total_fdy += t_fdy
            total_plain += t_pl
            print(f"L{i + 1:d}  {ci:4d}->{co:<4d} {ff:4d} {gflop:7.2f} "
                  f"{t_fdy:9.1f} {t_mix:9.1f} {t_pl:12.1f} {t_fdy / t_pl:8.2f}x")
    print(f"total: FDY {total_fdy:.1f} ms | if all layers were plain convs ~{total_plain:.1f} ms "
          f"(saves {total_fdy - total_plain:.1f} ms, but loses per-frequency adaptivity)")


if __name__ == "__main__":
    main()
