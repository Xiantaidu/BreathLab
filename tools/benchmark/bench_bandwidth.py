"""Measured effect of bandwidth (sample rate / fmax / mel bins) on cost: VRAM and step time depend on hop x rows, not sample rate; only raising fmax AND n_mels adds high-frequency evidence, at a cost proportional to n_mels.

Usage: PYTHONPATH=D:/BreathLab python tools/benchmark/bench_bandwidth.py --batch 24 --frames 600
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from breathlab.common import load_config
from breathlab.data.features import LogMel
from breathlab.models.crnn import build_model, count_params
from breathlab.speed import setup_speed

# (label, sample_rate, fmax, n_mels) -- hop is fixed at 10ms, so T never changes
CANDS = [
    ("current 16k/8k/64", 16000, 8000, 64),
    ("24k/12k/64 (bandwidth only)", 24000, 12000, 64),
    ("24k/12k/72", 24000, 12000, 72),
    ("24k/12k/96", 24000, 12000, 96),
    ("24k/11k/80", 24000, 11000, 80),
    ("32k/16k/128", 32000, 16000, 128),
    ("44.1k/16k/128", 44100, 16000, 128),
    ("44.1k/8k/64 (sample rate only)", 44100, 8000, 64),
]


def mel_cost_ms(sr: int, fmax: int, n_mels: int, seconds: float = 6.0) -> float:
    """CPU cost (ms) of one log-mel computation under this setting (mean of 10 runs, 3 warm-up)."""
    cfg = load_config("breathlab/configs/default.yaml")
    cfg["audio"]["sample_rate"] = sr
    cfg["feature"].update({"n_mels": n_mels, "fmax": fmax,
                           "n_fft": int(0.025 * sr), "win_length": int(0.025 * sr),
                           "hop_length": int(0.010 * sr)})
    lm = LogMel(cfg)
    y = (np.random.randn(int(seconds * sr)) * 0.05).astype(np.float32)
    for _ in range(3):
        lm(y)
    t0 = time.time()
    for _ in range(10):
        lm(y)
    return (time.time() - t0) / 10 * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="breathlab/configs/default.yaml")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--frames", type=int, default=600, help="6s@100fps")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--blocks", default="conformer",
                    help="comma separated: conformer,ebranchformer")
    ap.add_argument("--only", default=None, help="only run configs whose label contains this text")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="bf16 mixed precision (default on, as in training); --no-amp uses fp32")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    free, total = torch.cuda.mem_get_info() if dev.type == "cuda" else (0, 0)
    print(f"[bw] device={dev} batch={args.batch} frames={args.frames} "
          f"(hop fixed at 10ms -> T unchanged)")
    print(f"[bw] free GPU memory {free/2**30:.2f}/{total/2**30:.2f}GB "
          f"(if another process holds the GPU, peaks above the free amount spill into host "
          f"memory and timings become meaningless)")
    print(f"\n{'config':30s} {'rows':>6s} {'params':>10s} {'mel(CPU)':>9s} "
          f"{'front':>8s} {'step':>8s} {'peak VRAM':>9s}")
    for label, sr, fmax, n_mels in CANDS:
        if args.only and args.only not in label:
            continue
        for blk in [s for s in args.blocks.split(",") if s]:
            cfg = load_config(args.config)
            cfg["model"]["name"] = "conformer"
            cfg["model"]["block"] = blk
            cfg["audio"]["sample_rate"] = sr
            cfg["feature"].update({"n_mels": n_mels, "fmax": fmax, "input_rows": n_mels + 2})
            rows = n_mels + 2
            x = torch.randn(args.batch, rows, args.frames, device=dev)
            net = setup_speed(build_model(cfg).to(dev), cfg, dev).train()
            opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
            torch.cuda.reset_peak_memory_stats()
            amp = args.amp and dev.type == "cuda"
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
                front_ms = (time.time() - t0) / 3 * 1000
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.steps):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    net(x).mean().backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_ms = (time.time() - t0) / args.steps * 1000
            vram = torch.cuda.max_memory_allocated() / 2 ** 30
            mc = mel_cost_ms(sr, fmax, n_mels)
            tag = f"{label}{'' if blk == 'conformer' else ' [' + blk + ']'}"
            print(f"{tag:30s} {rows:6d} {count_params(net):10,d} {mc:8.1f}ms "
                  f"{front_ms:7.1f}ms {step_ms:7.1f}ms {vram:8.2f}GB")
            del net, x, opt
            torch.cuda.empty_cache()
    print("\nNote: the GPU is shared with the training process, so absolute values are high; "
          "relative comparisons between columns are reliable.")


if __name__ == "__main__":
    main()
