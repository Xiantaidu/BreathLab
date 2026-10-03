"""Numerical cross-check of a single FDY layer: mix / triton paths vs the reference loop, in fp32 and bf16.

Usage: PYTHONPATH=D:/BreathLab python tools/benchmark/check_fdy_kernel.py [--ci 64 --co 128 --f 16 --t 600]
"""
from __future__ import annotations

import argparse

import torch

from breathlab.models.fdy import FDYConv2d


def rel(p: torch.Tensor, q: torch.Tensor) -> float:
    """Relative max-abs difference: ``max|p - q| / max|p|``."""
    return (p - q).abs().max().item() / max(p.abs().max().item(), 1e-8)


def run(mode: str, sd, x0: torch.Tensor, dy0: torch.Tensor, dtype, co: int, m: int):
    """Run one FDY layer in ``mode`` (shared weights via ``sd``) and return ``(y, dx, dW, dA)``."""
    layer = FDYConv2d(x0.shape[1], co, 3, M=m, mode=mode, ckpt=False).to(x0.device)
    layer.load_state_dict(sd)
    x = x0.clone().to(dtype).requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16)):
        y = layer(x)
    (y.float() * dy0).sum().backward()
    return (y.detach().float(), x.grad.detach().float(),
            layer.convs[0].weight.grad.float(), layer.attn.weight.grad.float())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=3)
    ap.add_argument("--ci", type=int, default=8)
    ap.add_argument("--co", type=int, default=16)
    ap.add_argument("--f", type=int, default=12)
    ap.add_argument("--t", type=int, default=40)
    ap.add_argument("--m", type=int, default=4)
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    ref = FDYConv2d(a.ci, a.co, 3, M=a.m, mode="loop", ckpt=False).to(dev)
    sd = {k: v.clone() for k, v in ref.state_dict().items()}
    x0 = torch.randn(a.b, a.ci, a.f, a.t, device=dev, dtype=torch.float32)
    dy0 = torch.randn(a.b, a.co, a.f, a.t, device=dev, dtype=torch.float32)

    ok = True
    for dtype in (torch.float32, torch.bfloat16):
        tag = "fp32" if dtype == torch.float32 else "bf16"
        tol = 1e-3 if dtype == torch.float32 else 0.02     # bf16 only needs bf16-level agreement
        y1, g1, w1, a1 = run("loop", sd, x0, dy0, dtype, a.co, a.m)
        for mode in ("mix", "triton"):
            y2, g2, w2, a2 = run(mode, sd, x0, dy0, dtype, a.co, a.m)
            ds = {"y": rel(y1, y2), "dx": rel(g1, g2), "dW": rel(w1, w2), "dA": rel(a1, a2)}
            good = all(v < tol for v in ds.values())
            ok &= good
            print(f"[{tag}/{mode:6s}] " + " | ".join(f"{k} {v:.3e}" for k, v in ds.items())
                  + f"  -> {'PASS' if good else 'FAIL'}")
    msg = "PASS" if ok else "FAIL: custom kernel disagrees with the reference loop"
    print("[FDY kernel] " + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
