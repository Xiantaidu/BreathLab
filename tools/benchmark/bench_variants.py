"""Cost benchmark for architecture variants: params / front end / full step / peak VRAM.

Usage: PYTHONPATH=D:/BreathLab python tools/benchmark/bench_variants.py --batch 24 --frames 600 [--only d160] [--variants "name:k=v,...;name:..."]
"""
from __future__ import annotations

import argparse
import time

import torch

from breathlab.common import load_config
from breathlab.models.crnn import build_model, count_params
from breathlab.speed import setup_speed

# name -> overrides applied to cfg["model"]
BUILTIN = [
    ("current", {}),
    ("no-FDY (fdy_m=0)", dict(fdy_m=0)),
    ("all-FDY", dict(fdy_layers=None)),
    ("EBranch (conv)", dict(block="ebranchformer")),
    ("EBranch (cgmlp)", dict(block="ebranchformer", branch="cgmlp")),
    ("d160", dict(d_model=160)),
    ("d160+EBranch", dict(d_model=160, block="ebranchformer")),
    ("d192+8 layers", dict(n_layers=8)),
]


def parse_ov(text: str) -> dict:
    """Parse a ``k=v,...`` override string into typed values (bool / None / int, else str)."""
    ov = {}
    for kv in [s for s in text.split(",") if s]:
        k, _, v = kv.partition("=")
        v = v.strip()
        if v.lower() in ("true", "false"):
            ov[k] = v.lower() == "true"
        elif v.lower() in ("none", "null"):
            ov[k] = None
        else:
            try:
                ov[k] = int(v)
            except ValueError:
                ov[k] = v
    return ov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="breathlab/configs/default.yaml")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--only", default=None, help="only run variants whose name contains this text")
    ap.add_argument("--variants", default=None,
                    help="custom list: 'name:key=value,...;name:...' (variants separated by ';')")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    free, total = torch.cuda.mem_get_info() if dev.type == "cuda" else (0, 0)
    if args.variants:
        items = []
        for chunk in args.variants.split(";"):
            if chunk.strip():
                name, _, ovs = chunk.partition(":")
                items.append((name.strip(), parse_ov(ovs)))
    else:
        items = BUILTIN
    items = [(n, o) for n, o in items if not args.only or args.only in n]

    base = load_config(args.config)
    rows = int(base["feature"].get("input_rows") or base["feature"]["n_mels"])
    x = torch.randn(args.batch, rows, args.frames, device=dev)
    print(f"[var] device={dev} batch={args.batch} frames={args.frames} input rows {rows} | "
          f"free VRAM {free/2**30:.2f}/{total/2**30:.2f}GB")
    print(f"\n{'config':26s} {'params':>10s} {'front':>8s} {'step':>8s} {'peak VRAM':>9s}")
    for name, ov in items:
        cfg = load_config(args.config)
        cfg["model"]["name"] = "conformer"
        cfg["model"].update(ov)
        net = setup_speed(build_model(cfg).to(dev), cfg, dev).train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        amp = dev.type == "cuda"
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                net(x).mean().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            net(x)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                net(x)
            torch.cuda.synchronize()
            front = (time.time() - t0) / 3 * 1000
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.steps):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                net(x).mean().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step = (time.time() - t0) / args.steps * 1000
        print(f"{name:26s} {count_params(net):10,d} {front:7.1f}ms {step:7.1f}ms "
              f"{torch.cuda.max_memory_allocated()/2**30:8.2f}GB")
        del net, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
