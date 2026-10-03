import time
import torch

from breathlab.common import load_config
from breathlab.models.crnn import build_model, count_params
from breathlab.train import focal_loss, bce_loss

dev = torch.device('cuda')
B = 8
VARIANT = __import__('sys').argv[1] if len(__import__('sys').argv) > 1 else 'base'

if VARIANT == 'compile_scan':
    import breathlab.models.mamba_scan as ms
    ms.associative_scan = torch.compile(ms.associative_scan, dynamic=False)
    print('[bench] associative_scan compiled')
elif VARIANT == 'compile_block':
    import breathlab.models.mamba_scan as ms
    ms.MambaBlock.forward = torch.compile(ms.MambaBlock.forward, dynamic=False)
    print('[bench] MambaBlock.forward compiled')


def bench(name, extra_cfg=None):
    cfg = load_config('breathlab/configs/default.yaml')
    cfg['model']['name'] = 'unet_mamba'
    if extra_cfg:
        cfg['model'].update(extra_cfg)
    net = build_model(cfg).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    x = torch.randn(B, cfg['feature']['input_rows'], 600, device=dev)
    lab = (torch.rand(B, 600, device=dev) < 0.05).float()
    spl = (torch.rand(B, 600, device=dev) < 0.15).float()
    mask = torch.ones(B, 600, device=dev)
    hi = net.head_index
    holder = {}

    def step():
        opt.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(x)
            loss = focal_loss(out[:, hi['ap']], lab, mask, 0.25, 2.0) \
                + 0.5 * bce_loss(out[:, hi['sp']], spl, mask, 1.0)
        loss.backward()
        opt.step()
        holder['out'] = out
        holder['loss'] = loss

    torch.cuda.reset_peak_memory_stats()
    for _ in range(6):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    n = 20
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    ms_ = (time.time() - t0) / n * 1000
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f'[{VARIANT}] {name:28s} step={ms_:7.1f}ms  peak={peak:.2f}GiB  (batch {B}, '
          f'params {count_params(net):,})', flush=True)
    del net, opt, x, holder
    torch.cuda.empty_cache()


if VARIANT in ('base', 'compile_scan', 'compile_block'):
    bench('unet_mamba')
elif VARIANT == 'expand1':
    bench('unet_mamba', {'um_expand': 1})
elif VARIANT == 'expand1_compile':
    import breathlab.models.mamba_scan as ms
    ms.MambaBlock.forward = torch.compile(ms.MambaBlock.forward, dynamic=False)
    bench('unet_mamba', {'um_expand': 1})
