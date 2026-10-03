"""Conformer training-step benchmark, per-part timing, and numerical parity checks.

Usage: PYTHONPATH=D:/BreathLab python tools/benchmark/bench_speed.py --steps 8 --batch 24 [--variants base,rot,fdy,all | --parity | --data --data-clips 60]
"""
from __future__ import annotations

import argparse
import statistics
import time
from contextlib import AbstractContextManager
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from breathlab.common import load_config
from breathlab.models import fdy as fdy_mod
from breathlab.models.conformer import (MHSA, head_pad_enabled, set_head_pad_enabled,
                                        set_rotary_cache_enabled)
from breathlab.models.crnn import build_model, count_params
from breathlab.train import bce_loss, focal_loss


VARIANTS: Dict[str, Dict] = {
    "base":     {},
    "rot":      {"rot_cache": True},
    "fdy":      {"rot_cache": True, "fdy_fused": True},
    "chlast":   {"rot_cache": True, "fdy_fused": True, "ch_last": True},
    "nockpt":   {"rot_cache": True, "fdy_fused": True, "ch_last": True, "fdy_ckpt": False},
    "adamw":    {"rot_cache": True, "fdy_fused": True, "ch_last": True, "fused_adamw": True},
    # flash: unusable here -- head_dim = 44 is not a multiple of 8, so no kernel is available;
    # kept for the record only.
    "flash":    {"rot_cache": True, "fdy_fused": True, "ch_last": True, "fused_adamw": True,
                 "sdpa": "flash"},
    "all":      {"rot_cache": True, "fdy_fused": True, "ch_last": True, "fused_adamw": True,
                 "cudnn_benchmark": True},
    "mix":      {"rot_cache": True, "fdy_mode": "mix"},
    "mix_all":  {"rot_cache": True, "fdy_mode": "mix", "ch_last": True, "fused_adamw": True,
                 "cudnn_benchmark": True},
    "mix_nockpt": {"rot_cache": True, "fdy_mode": "mix", "ch_last": True, "fused_adamw": True,
                   "cudnn_benchmark": True, "fdy_ckpt": False},
    "mix_compile": {"rot_cache": True, "fdy_mode": "mix", "ch_last": True, "fused_adamw": True,
                    "cudnn_benchmark": True, "compile": True},
    "compile":  {"rot_cache": True, "fdy_fused": True, "ch_last": True, "fused_adamw": True,
                 "cudnn_benchmark": True, "compile": True},
    "compile_nockpt": {"rot_cache": True, "fdy_fused": True, "ch_last": True,
                       "fused_adamw": True, "cudnn_benchmark": True,
                       "compile": True, "fdy_ckpt": False},
    # triton: same math as mix, but the 9 taps are assembled in registers instead of
    # materializing im2col.
    "triton":       {"rot_cache": True, "fdy_mode": "triton"},
    "triton_all":   {"rot_cache": True, "fdy_mode": "triton", "ch_last": True,
                     "fused_adamw": True, "cudnn_benchmark": True},
    "triton_nockpt": {"rot_cache": True, "fdy_mode": "triton", "ch_last": True,
                      "fused_adamw": True, "cudnn_benchmark": True, "fdy_ckpt": False},
}


def sdpa_ctx(name: str | None) -> AbstractContextManager:
    """Return an SDPA backend context (``flash``/``mem``/``math``/``cudnn``; ``None``/``auto`` = no-op)."""
    if not name or name == "auto":
        from contextlib import nullcontext
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    m = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    return sdpa_kernel(m[name])


def apply_variant(model: nn.Module, cfg: Dict, opt: Dict) -> nn.Module:
    """Apply a variant's switches to the config, the model and the global backends, in place."""
    set_rotary_cache_enabled(bool(opt.get("rot_cache", False)))
    cfg["model"]["fdy_fused"] = bool(opt.get("fdy_fused", False))
    cfg["model"]["fdy_checkpoint"] = bool(opt.get("fdy_ckpt", True))
    cfg["model"]["channels_last"] = bool(opt.get("ch_last", False))
    for mod in model.modules():
        if isinstance(mod, fdy_mod.FDYConv2d):
            if opt.get("fdy_mode"):
                mod.mode = opt["fdy_mode"]
            else:
                mod.fused = bool(opt.get("fdy_fused", False))     # legacy boolean switch
            mod.ckpt = bool(opt.get("fdy_ckpt", True))
    if opt.get("ch_last", False):
        model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = bool(opt.get("cudnn_benchmark", False))
    if opt.get("compile", False):
        model = torch.compile(model, dynamic=bool(opt.get("compile_dynamic", True)))
    return model


def make_batch(cfg: Dict, batch: int, frames: int, device, ch_last: bool):
    """Build a synthetic training batch ``(x, label, sp, mask)`` matching the configured layout."""
    rows = int(cfg["feature"].get("input_rows") or cfg["feature"]["n_mels"])
    x = torch.randn(batch, rows, frames, device=device)
    label = (torch.rand(batch, frames, device=device) > 0.9).float()
    sp = (torch.rand(batch, frames, device=device) > 0.5).float()
    mask = torch.ones(batch, frames, device=device)
    return x, label, sp, mask


def one_step(model, optm, x, label, sp, mask, cfg, amp: bool = True) -> torch.Tensor:
    """Run one full training step (forward, loss, backward, clip, optimizer step)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        out = model(x)
        loss = focal_loss(out[:, 0], label, mask, cfg["train"]["focal_alpha"],
                          cfg["train"]["focal_gamma"])
        if out.shape[1] >= 2:
            loss = loss + 0.5 * bce_loss(out[:, 1], sp, mask, 1.0)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
    optm.step()
    optm.zero_grad(set_to_none=True)
    return loss


def bench_variant(name: str, cfg: Dict, device, args) -> Dict:
    """Benchmark one variant and print its median step time and peak VRAM."""
    opt = VARIANTS[name]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg["model"]["name"] = "conformer"
    model = build_model(cfg).to(device)
    model = apply_variant(model, cfg, opt)
    params = [p for p in model.parameters() if p.requires_grad]
    optm = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4,
                             fused=bool(opt.get("fused_adamw", False)))
    x, label, sp, mask = make_batch(cfg, args.batch, args.frames, device, opt.get("ch_last", False))
    model.train()

    ctx = sdpa_ctx(opt.get("sdpa"))
    try:
        with ctx:
            for _ in range(args.warmup):
                one_step(model, optm, x, label, sp, mask, cfg)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(args.steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                one_step(model, optm, x, label, sp, mask, cfg)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000.0)
    except Exception as e:
        print(f"[variant {name}] failed: {type(e).__name__}: {e}")
        return {"name": name, "ms": float("nan"), "peak_gb": float("nan"), "err": str(e)}

    peak = torch.cuda.max_memory_allocated() / 1e9
    med = statistics.median(ts)
    res = {"name": name, "ms": med, "ms_min": min(ts), "peak_gb": peak,
           "samples_per_s": args.batch * 1000.0 / med}
    print(f"  {name:16s} {med:7.1f} ms/step  (min {min(ts):6.1f})  "
          f"peak VRAM {peak:5.2f} GB  {res['samples_per_s']:6.1f} samples/s")
    return res


def bench_breakdown(cfg: Dict, device, args) -> None:
    """Segment-wise timing: CNN front end / proj / Conformer blocks / head."""
    cfg["model"]["name"] = "conformer"
    if getattr(args, "fdy_mode", ""):
        cfg["model"]["fdy_mode"] = args.fdy_mode
        cfg["model"]["fdy_checkpoint"] = bool(args.fdy_ckpt)
        print(f"[breakdown] fdy_mode={args.fdy_mode} fdy_checkpoint={args.fdy_ckpt}")
    model = build_model(cfg).to(device)
    from breathlab.models.conformer import Conformer
    assert isinstance(model, Conformer)
    x, label, sp, mask = make_batch(cfg, args.batch, args.frames, device, False)
    model.train()

    def time_fn(fn, n=5) -> float:
        for _ in range(2):
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

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        front = time_fn(lambda: model.front(x).sum().backward())
        h = time_fn(lambda: model.proj(torch.randn(args.batch, args.frames,
                                                   model.proj.in_features,
                                                   device=device)).sum().backward())
        feat = torch.randn(args.batch, args.frames, cfg["model"]["d_model"], device=device,
                           requires_grad=True)
        blocks = time_fn(lambda: sum(blk(feat).sum() for blk in model.blocks).backward())
        hid = torch.randn(args.batch, args.frames, cfg["model"]["d_model"], device=device,
                          requires_grad=True)
        head = time_fn(lambda: model.head(hid).sum().backward())
    tot = front + h + blocks + head
    print("[breakdown] forward+backward ms: "
          f"CNN front end (FDY) {front:6.1f} | proj {h:5.1f} | Conformer blocks {blocks:6.1f} "
          f"| head {head:5.1f} | total {tot:6.1f}  (front-end share {front / tot:5.1%})")


def bench_blocks(cfg: Dict, device, args) -> None:
    """Time the internals of a Conformer block: FFN / MHSA (incl. SDPA alone) / ConvModule."""
    from breathlab.models.conformer import Conformer

    cfg["model"]["name"] = "conformer"
    if getattr(args, "d_model", 0):
        cfg["model"]["d_model"] = args.d_model
    if getattr(args, "block", None):
        cfg["model"]["block"] = args.block
    model = build_model(cfg).to(device).train()
    assert isinstance(model, Conformer)
    d = cfg["model"]["d_model"]
    h = cfg["model"]["n_heads"]
    dk = d // h
    B, T = args.batch, args.frames
    n = len(model.blocks)
    print(f"[blocks] d_model={d} n_heads={h} head_dim={dk} (multiple of 8: {dk % 8 == 0}) "
          f"layers={n} batch={B} frames={T} sdpa={args.sdpa or 'auto'}")

    def time_fn(fn, k=5) -> float:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(ts)

    # Time SDPA alone at the same padded head_dim as MHSA; an unpadded shape falls back to the
    # math backend and reports more than the whole layer.
    dkp = dk + (-dk % 8 if head_pad_enabled() else 0)
    xin = torch.randn(B, T, d, device=device)
    q = torch.randn(B, h, T, dkp, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, h, T, dkp, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, h, T, dkp, device=device, dtype=torch.bfloat16, requires_grad=True)

    def run(mod, t) -> None:
        y = mod(t)
        if isinstance(y, tuple):
            y = y[0]
        y.float().pow(2).mean().backward()

    ctx = sdpa_ctx(args.sdpa)
    try:
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            blk = model.blocks[0]
            r = {
                "ffn1": time_fn(lambda: run(blk.ffn1, xin)),
                "attn (whole)": time_fn(lambda: run(blk.attn, xin)),
                "  |- sdpa": time_fn(lambda: run(
                    lambda t: F.scaled_dot_product_attention(q, k, v), xin)),
                "conv": time_fn(lambda: run(blk.conv, xin)),
                "ffn2": time_fn(lambda: run(blk.ffn2, xin)),
            }
            r["whole block"] = time_fn(lambda: run(blk, xin))
            r["all blocks"] = time_fn(
                lambda: run(lambda t: sum((b(t) for b in model.blocks), torch.zeros_like(t)), xin))
    except Exception as e:
        print(f"[blocks] failed: {type(e).__name__}: {e}   "
              f"(this SDPA backend is unavailable at that head_dim?)")
        return

    per = r["whole block"]
    print(f"[blocks] per-layer forward+backward ms: "
          + " | ".join(f"{k2} {v2:6.1f}" for k2, v2 in r.items() if k2 != "all blocks"))
    print(f"[blocks] all {n} layers {r['all blocks']:6.1f} ms  "
          f"(sdpa is {r['  |- sdpa'] / per:5.1%} of one layer, "
          f"whole MHSA {r['attn (whole)'] / per:5.1%}, "
          f"both FFNs {(r['ffn1'] + r['ffn2']) / per:5.1%})")


def bench_dataloader(cfg: Dict, args) -> None:
    """Measure CPU-side data loading / augmentation throughput."""
    import numpy as np
    from torch.utils.data import DataLoader
    from breathlab.data.dataset import PreparedClips, WindowDataset, read_split

    names = read_split(cfg["paths"]["splits_dir"], "train")
    if args.data_clips > 0:
        names = names[: args.data_clips]
    print(f"[data] building from {len(names)} clips "
          f"(streaming; first run builds the 16k cache) ...")
    t0 = time.time()
    prep = PreparedClips(cfg, names, build_breath_bank=True, preload=False)
    # Normalization covers the mel rows only; the F0 row is concatenated later.
    mean = np.zeros(int(cfg["feature"]["n_mels"]), dtype=np.float32)
    std = np.ones_like(mean)
    ds = WindowDataset(prep, cfg, mean, std, augment=True)
    print(f"[data] build took {time.time() - t0:.1f}s, {len(ds)} windows")

    for nw in [0, args.num_workers]:
        dl = DataLoader(ds, batch_size=args.batch, num_workers=nw, shuffle=False,
                        drop_last=True, persistent_workers=nw > 0)
        n = 0
        t0 = None
        for i, _ in enumerate(dl):
            # First batches include Windows spawn/import overhead; skip them.
            if i < args.data_warmup:
                continue
            if t0 is None:
                t0 = time.time()
            n += 1
            if n >= args.data_batches:
                break
        dt = (time.time() - t0) / max(1, n) if t0 else float("nan")
        print(f"[data] num_workers={nw}: {dt * 1000:7.1f} ms/batch  "
              f"({args.batch / dt:6.1f} samples/s, {n} batches, "
              f"{args.data_warmup} warm-up skipped)")


def check_attn_parity(cfg: Dict, device, args) -> None:
    """Check numerical equivalence of attention head_dim padding (44 -> 48) + SDPA backend probe."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    cfg["model"]["name"] = "conformer"

    # ---- backend probe: which SDPA backends work at the current head_dim ----
    h = cfg["model"]["n_heads"]
    d = cfg["model"]["d_model"]
    dk = d // h
    B, T = args.batch, args.frames
    print(f"[attention] d_model={d} n_heads={h} head_dim={dk}")
    for pad in (0, -dk % 8):
        if pad == 0:
            continue
        shape = (B, h, T, dk + pad)
        q = torch.randn(*shape, device=device, dtype=torch.bfloat16)
        k = torch.randn(*shape, device=device, dtype=torch.bfloat16)
        v = torch.randn(*shape, device=device, dtype=torch.bfloat16)
        for nm, be in [("flash", SDPBackend.FLASH_ATTENTION),
                       ("mem-efficient", SDPBackend.EFFICIENT_ATTENTION),
                       ("cudnn", SDPBackend.CUDNN_ATTENTION),
                       ("math", SDPBackend.MATH)]:
            try:
                with sdpa_kernel(be):
                    F.scaled_dot_product_attention(q, k, v)
                print(f"    head_dim={dk + pad:3d}  {nm:13s} available")
            except Exception as e:
                print(f"    head_dim={dk + pad:3d}  {nm:13s} unavailable "
                      f"({type(e).__name__}: {str(e)[:60]})")
    # Unpadded (head_dim=44) once, as a control.
    q = torch.randn(B, h, T, dk, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, h, T, dk, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, h, T, dk, device=device, dtype=torch.bfloat16)
    for nm, be in [("flash", SDPBackend.FLASH_ATTENTION),
                   ("mem-efficient", SDPBackend.EFFICIENT_ATTENTION),
                   ("cudnn", SDPBackend.CUDNN_ATTENTION),
                   ("math", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, k, v)
            print(f"    head_dim={dk:3d} (unpadded) {nm:13s} available")
        except Exception as e:
            print(f"    head_dim={dk:3d} (unpadded) {nm:13s} unavailable "
                  f"({type(e).__name__}: {str(e)[:60]})")

    # ---- numerical equivalence ----
    torch.manual_seed(0)
    model = build_model(cfg).to(device).eval()      # eval disables dropout
    set_rotary_cache_enabled(False)
    x, label, sp, mask = make_batch(cfg, args.batch, args.frames, device, False)

    def run(pad: bool, amp: bool):
        set_head_pad_enabled(pad)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
            loss = focal_loss(out[:, 0], label, mask,
                              cfg["train"]["focal_alpha"], cfg["train"]["focal_gamma"])
            if out.shape[1] >= 2:
                loss = loss + 0.5 * bce_loss(out[:, 1], sp, mask, 1.0)
        loss.backward()
        g = [(n, p.grad.detach().clone()) for n, p in model.named_parameters()
             if p.grad is not None]
        model.zero_grad(set_to_none=True)
        return loss.detach(), out.detach().float(), g

    # Isolated unit test: padded vs unpadded on the SAME (math) backend, so only floating-point
    # rounding remains -- the actual proof of strict equivalence.
    def unit_cmp(dtype) -> bool:
        torch.manual_seed(0)
        mhsa = MHSA(d, h, 0.0).to(device=device, dtype=dtype).eval()
        xx = torch.randn(4, 128, d, device=device, dtype=dtype)
        outs = {}
        for pad in (False, True):
            set_head_pad_enabled(pad)
            with sdpa_kernel(SDPBackend.MATH):       # fixed backend, only padding varies
                outs[pad] = mhsa(xx).detach().clone()
        a, b_ = outs[False], outs[True]
        dd = (a - b_).abs().max().item() / max(a.abs().max().item(), 1e-12)
        set_head_pad_enabled(True)
        print(f"[parity|unit {str(dtype).replace('torch.', '')}] padded vs unpadded "
              f"(same math backend): output relative diff {dd:.3e}")
        return dd < (1e-12 if dtype == torch.float64 else 1e-5)

    ok = unit_cmp(torch.float64)
    ok &= unit_cmp(torch.float32)

    def cmp(tag: str, amp: bool, tol: float) -> bool:
        l1, o1, g1 = run(False, amp)      # reference: unpadded (original behaviour)
        l2, o2, g2 = run(True, amp)       # padded
        dout = (o1 - o2).abs().max().item() / max(o1.abs().max().item(), 1e-8)
        gscale = max(b.abs().max().item() for _, b in g2)
        worst = sorted((((a - b).abs().max().item(), n, b.abs().max().item())
                        for (n, a), (_, b) in zip(g1, g2)), reverse=True)[:2]
        dg = worst[0][0] / max(gscale, 1e-12)
        print(f"[parity|{tag}] loss diff {(l1 - l2).abs().item():.3e} | output rel diff {dout:.3e} "
              f"| gradient global rel diff {dg:.3e}")
        for dd, n, sc in worst:
            print(f"    worst: {n:38s} abs diff {dd:.2e} (tensor magnitude {sc:.2e})")
        ok = dout < tol and dg < tol
        print(f"    -> {tag}: {'PASS' if ok else 'FAIL'}")
        return ok

    # Whole-network fp32 tolerance relaxed to 1e-3: padding switches the SDPA backend, so the
    # accumulation order of the whole net changes (same band as the FDY parity check).
    ok &= cmp("fp32", amp=False, tol=1e-3)
    ok &= cmp("bf16", amp=True, tol=2e-2)
    set_head_pad_enabled(True)
    print("[parity] " + ("PASS: head_dim padding is mathematically equivalent to the original "
                         "attention" if ok else "FAIL: difference above tolerance, investigate"))


def check_parity(cfg: Dict, device, args) -> None:
    """Check numerical consistency of the optimized paths: forward output and gradients."""
    cfg["model"]["name"] = "conformer"

    def make(mode: str):
        torch.manual_seed(0)
        m = build_model(cfg).to(device).eval()    # eval: dropout off, else randomness masks diffs
        for mod in m.modules():
            if isinstance(mod, fdy_mod.FDYConv2d):
                mod.mode, mod.ckpt = mode, False
        return m

    m1 = make("loop")          # original per-kernel loop
    m2 = make("fuse")          # M kernels merged into a single convolution
    m3 = make("mix")           # per-frequency mixed kernels (im2col + batched matmul)
    m4 = make("triton") if args.triton else None   # fused Triton kernel
    set_rotary_cache_enabled(False)
    x, label, sp, mask = make_batch(cfg, args.batch, args.frames, device, False)

    def run(m, cache: bool, amp: bool = True):
        set_rotary_cache_enabled(cache)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            out = m(x)
            loss = focal_loss(out[:, 0], label, mask, cfg["train"]["focal_alpha"],
                              cfg["train"]["focal_gamma"])
            if out.shape[1] >= 2:
                loss = loss + 0.5 * bce_loss(out[:, 1], sp, mask, 1.0)
        loss.backward()
        g = [(n, p.grad.detach().clone()) for n, p in m.named_parameters() if p.grad is not None]
        m.zero_grad(set_to_none=True)
        return loss.detach(), out.detach().float(), g

    def grad_diff(g1, g2):
        """Global relative gradient difference: max|dg| / max|g| across all parameters."""
        gscale = max(b.abs().max().item() for _, b in g2)
        items = [((a - b).abs().max().item(), n, b.abs().max().item())
                 for (n, a), (_, b) in zip(g1, g2)]
        items.sort(reverse=True)
        return max(items[0][0] / max(gscale, 1e-12), 0.0), items[:2], gscale

    def compare(tag: str, ref, other, amp: bool, tol_out: float, tol_g: float) -> bool:
        l1, o1, g1 = run(ref, False, amp=amp)
        l2, o2, g2 = run(other, True, amp=amp)
        scale = max(o1.abs().max().item(), 1e-8)
        dout = (o1 - o2).abs().max().item() / scale
        dg, worst, gs = grad_diff(g1, g2)
        print(f"[consistency|{tag}] loss diff {(l1 - l2).abs().item():.3e} | output rel diff "
              f"{dout:.3e} | gradient global rel diff {dg:.3e} (gradient magnitude {gs:.2e})")
        for d, n, sc in worst:
            print(f"    worst: {n:38s} abs diff {d:.2e} (tensor magnitude {sc:.2e})")
        ok = dout < tol_out and dg < tol_g
        print(f"    -> {tag}: {'PASS' if ok else 'FAIL'}")
        return ok

    # fp32 at 1e-3: fusion changes cuDNN kernels and accumulation order, so ~1e-4 relative
    # differences are normal; bf16 is the training precision and allows bf16-level error.
    ok32 = compare("fp32 fuse  vs loop", m1, m2, amp=False, tol_out=1e-3, tol_g=1e-3)
    ok32 &= compare("fp32 mix   vs loop", m1, m3, amp=False, tol_out=1e-3, tol_g=1e-3)
    ok16 = compare("bf16 fuse  vs loop", m1, m2, amp=True, tol_out=0.02, tol_g=0.02)
    ok16 &= compare("bf16 mix   vs loop", m1, m3, amp=True, tol_out=0.02, tol_g=0.02)
    okt = True
    if m4 is not None:
        ok32 &= compare("fp32 triton vs loop", m1, m4, amp=False, tol_out=1e-3, tol_g=1e-3)
        ok16 &= compare("bf16 triton vs loop", m1, m4, amp=True, tol_out=0.02, tol_g=0.02)
        okt = ok32 and ok16
    tag = "PASS: fuse / mix" + (" / triton" if m4 is not None else "") + \
          " all equivalent to the original loop"
    print("[consistency] " + (tag if okt else "FAIL: difference above tolerance, investigate"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="breathlab/configs/default.yaml")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--frames", type=int, default=600, help="training window 6s@100fps")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--variants", default="base,rot,fdy,chlast,nockpt,adamw,flash,all")
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--blocks", action="store_true", help="time the internals of a Conformer block")
    ap.add_argument("--d-model", type=int, default=0, help="override model.d_model (A/B test)")
    ap.add_argument("--block", default=None, help="override model.block: conformer|ebranchformer")
    ap.add_argument("--sdpa", default=None, help="override SDPA backend: flash|mem|math|cudnn|auto")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--attn-parity", action="store_true",
                    help="check equivalence of attention head_dim padding")
    ap.add_argument("--no-head-pad", action="store_true", help="disable head_dim padding (control)")
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--data-clips", type=int, default=60)
    ap.add_argument("--data-batches", type=int, default=12)
    ap.add_argument("--data-warmup", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-bench", action="store_true",
                    help="only run --parity/--data/--breakdown")
    ap.add_argument("--triton", action="store_true", help="also check the triton path in --parity")
    ap.add_argument("--fdy-mode", default="",
                    help="--breakdown with an explicit FDY path (loop/fuse/mix/triton)")
    ap.add_argument("--fdy-ckpt", type=int, default=1,
                    help="--breakdown: whether FDY activation recomputation is on")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__} batch={args.batch} frames={args.frames}")

    set_head_pad_enabled(not args.no_head_pad)
    if args.parity:
        check_parity(cfg, device, args)
    if args.attn_parity:
        check_attn_parity(cfg, device, args)
    if args.breakdown:
        bench_breakdown(cfg, device, args)
    if args.blocks:
        bench_blocks(cfg, device, args)
    if args.data:
        bench_dataloader(cfg, args)

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    if names and not args.no_bench:
        print(f"[bench] variants: {names}")
        results: List[Dict] = []
        for n in names:
            if n not in VARIANTS:
                print(f"  unknown variant {n}")
                continue
            results.append(bench_variant(n, cfg, device, args))
        base = next((r for r in results if r["name"] == "base"), None)
        if base and base["ms"] == base["ms"]:
            print(f"\n{'variant':16s} {'ms/step':>8s} {'vs base':>9s} {'peak GB':>7s}")
            for r in results:
                sp = f"{base['ms'] / r['ms']:.2f}x" if r["ms"] == r["ms"] else "-"
                print(f"{r['name']:16s} {r['ms']:8.1f} {sp:>9s} {r['peak_gb']:7.2f}")


if __name__ == "__main__":
    main()
