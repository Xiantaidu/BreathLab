"""Sweep training batch sizes for unet_mamba: ms/step, windows/s, peak VRAM, implied epoch time; OOM-safe.

Run: python tools/benchmark/bench_unet_mamba_batch.py
"""
import time

import torch

from breathlab.common import load_config
from breathlab.models.crnn import build_model, count_params
from breathlab.train import focal_loss, bce_loss

dev = torch.device('cuda')
SIZES = [16, 24, 32, 48, 64, 96]
WINDOWS_PER_EPOCH = 88_000   # ~3674 steps x 24 at the current 11-corpus split

cfg = load_config('breathlab/configs/default.yaml')
cfg['model']['name'] = 'unet_mamba'
torch.backends.cudnn.benchmark = True
net = build_model(cfg).to(dev)
net.train()
opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
hi = net.head_index
print(f'params {count_params(net):,}, compile+autotune warmup per shape\n')

for B in SIZES:
    free_b, total_b = torch.cuda.mem_get_info()
    need = 0.13 * B                              # ~0.13 GiB per sample observed at batch24
    if free_b / 2**30 < need + 1.0:
        print(f'batch{B}: not enough free VRAM ({free_b / 2**30:.1f} GiB), stopping')
        break
    try:
        x = torch.randn(B, cfg['feature']['input_rows'], 600, device=dev)
        lab = (torch.rand(B, 600, device=dev) < 0.05).float()
        spl = (torch.rand(B, 600, device=dev) < 0.15).float()
        mask = torch.ones(B, 600, device=dev)
        holder = {}

        def step():
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = net(x)
                loss = focal_loss(out[:, hi['ap']], lab, mask, 0.25, 2.0) \
                    + 0.5 * bce_loss(out[:, hi['sp']], spl, mask, 1.0)
            loss.backward()
            opt.step()
            holder['out'], holder['loss'] = out, loss

        torch.cuda.reset_peak_memory_stats()
        for _ in range(6):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        n = 20
        for _ in range(n):
            step()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / n * 1000
        peak = torch.cuda.max_memory_allocated() / 2**30
        thr = B / (ms / 1000)
        epoch_min = WINDOWS_PER_EPOCH / thr / 60
        print(f'batch{B:3d}: {ms:6.0f} ms/step  {thr:6.0f} windows/s  '
              f'peak {peak:.2f} GiB  -> epoch train ~{epoch_min:.0f} min', flush=True)
        del x, lab, spl, mask, holder
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f'batch{B}: OOM, stopping sweep')
        break
    torch.cuda.empty_cache()
