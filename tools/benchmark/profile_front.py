"""Kernel-level profiling: which kernels the CNN front end / whole model actually spend time on.

Usage: PYTHONPATH=D:/BreathLab python tools/benchmark/profile_front.py [--steps 3 | --front-only | --mode fuse]
"""
from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from breathlab.common import load_config
from breathlab.models import fdy as fdy_mod
from breathlab.models.crnn import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="breathlab/configs/default.yaml")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--mode", default=None, help="override fdy_mode: loop|fuse|mix")
    ap.add_argument("--front-only", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["model"]["name"] = "conformer"
    if args.mode:
        cfg["model"]["fdy_mode"] = args.mode
    device = torch.device("cuda")
    model = build_model(cfg).to(device).train()
    for m in model.modules():
        if isinstance(m, fdy_mod.FDYConv2d):
            m.ckpt = False                      # no recomputation while profiling: see one fwd/bwd
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    rows = int(cfg["feature"].get("input_rows") or cfg["feature"]["n_mels"])
    x = torch.randn(args.batch, rows, args.frames, device=device)
    label = (torch.rand(args.batch, args.frames, device=device) > 0.9).float()
    mask = torch.ones(args.batch, args.frames, device=device)

    def step() -> None:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if args.front_only:
                out = model.front(x)
                loss = out.float().pow(2).mean()
            else:
                out = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    out[:, 0].float(), label)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(2):
        step()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(args.steps):
            step()
    print(f"\n=== {'(front end only) ' if args.front_only else ''}fdy_mode="
          f"{cfg['model'].get('fdy_mode')} ===  sorted by total CUDA time")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.top))


if __name__ == "__main__":
    main()
