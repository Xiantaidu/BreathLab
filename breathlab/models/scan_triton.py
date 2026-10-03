"""Triton fused selective-scan kernel (the Mamba S6 recurrence) over the [B, D, T, N] layout (N contiguous); fp32 throughout, math identical to the reference torch path."""
from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:               # no triton -> caller falls back to the reference scan
    triton = None
    tl = None
    _TRITON_OK = False


if _TRITON_OK:

    def _cfg() -> list:
        """num_warps / num_stages candidates; BLOCK_N is derived from N (power of two)."""
        return [
            triton.Config({}, num_warps=1, num_stages=4),
            triton.Config({}, num_warps=2, num_stages=2),
            triton.Config({}, num_warps=1, num_stages=1),
        ]

    # autotune key contains only N -- never T or B: with T in the key every clip length
    # re-benchmarks all configs and the GPU appears to hang.
    @triton.autotune(configs=_cfg(), key=["N"])
    @triton.jit
    def _scan_fwd_kernel(A, Bp, C, Y, T, N, D,
                         BLOCK_N: tl.constexpr):
        """y[b,d,t] = sum_n c[b,t,n] * h[b,d,t,n];  h_t = a_t h_{t-1} + b_t (h never stored)."""
        pid = tl.program_id(0)                       # over B*D
        b = pid // D
        n_offs = tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        base = pid.to(tl.int64) * T * N
        cbase = b.to(tl.int64) * T * N
        carry = tl.zeros([BLOCK_N], dtype=tl.float32)
        for t in range(0, T):
            a = tl.load(A + base + t * N + n_offs, mask=n_mask, other=1.0)
            bv = tl.load(Bp + base + t * N + n_offs, mask=n_mask, other=0.0)
            cv = tl.load(C + cbase + t * N + n_offs, mask=n_mask, other=0.0)
            carry = a * carry + bv
            yv = tl.sum(cv * carry, axis=0)
            tl.store(Y + pid.to(tl.int64) * T + t, yv)

    @triton.autotune(configs=_cfg(), key=["N"])
    @triton.jit
    def _scan_bwd_kernel(A, Bp, C, DY, H, GA, GB, T, N, D,
                         BLOCK_N: tl.constexpr):
        """Pass 1 recomputes/stores h ascending (grad_a needs h_{t-1}); pass 2 runs the adjoint recurrence descending, writing grad_a / grad_b (grad_c is a torch-side reduction)."""
        pid = tl.program_id(0)
        b = pid // D
        n_offs = tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        base = pid.to(tl.int64) * T * N
        cbase = b.to(tl.int64) * T * N
        # ---- pass 1: h_t = a_t h_{t-1} + b_t ----
        carry = tl.zeros([BLOCK_N], dtype=tl.float32)
        for t in range(0, T):
            a = tl.load(A + base + t * N + n_offs, mask=n_mask, other=1.0)
            bv = tl.load(Bp + base + t * N + n_offs, mask=n_mask, other=0.0)
            carry = a * carry + bv
            tl.store(H + base + t * N + n_offs, carry, mask=n_mask)
        # ---- pass 2: g_t = e_t + a_{t+1} g_{t+1};  ga_t = g_t * h_{t-1};  gb_t = g_t ----
        gcarry = tl.zeros([BLOCK_N], dtype=tl.float32)
        a_next = tl.zeros([BLOCK_N], dtype=tl.float32)
        for i in range(0, T):
            t = T - 1 - i
            dy = tl.load(DY + pid.to(tl.int64) * T + t)
            cv = tl.load(C + cbase + t * N + n_offs, mask=n_mask, other=0.0)
            e = dy * cv
            g = e + a_next * gcarry
            gcarry = g
            hp = tl.load(H + base + (t - 1) * N + n_offs,
                         mask=n_mask & (t > 0), other=0.0)
            tl.store(GA + base + t * N + n_offs, g * hp, mask=n_mask)
            tl.store(GB + base + t * N + n_offs, g, mask=n_mask)
            a_next = tl.load(A + base + t * N + n_offs, mask=n_mask & (t > 0), other=0.0)

    class _FusedScanFn(torch.autograd.Function):
        """a, b: [B, D, T, N] fp32 contiguous | c: [B, T, N] fp32 -> y: [B, D, T] fp32."""

        @staticmethod
        def forward(ctx, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            B, D, T, N = a.shape
            y = torch.empty((B, D, T), device=a.device, dtype=torch.float32)
            _scan_fwd_kernel[(B * D,)](a, b, c, y, T, N, D, BLOCK_N=triton.next_power_of_2(N))
            ctx.save_for_backward(a, b, c)
            return y

        @staticmethod
        def backward(ctx, dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            a, b, c = ctx.saved_tensors
            B, D, T, N = a.shape
            dy = dy.contiguous()
            h = torch.empty_like(a)
            ga = torch.empty_like(a)
            gb = torch.empty_like(a)
            _scan_bwd_kernel[(B * D,)](a, b, c, dy, h, ga, gb, T, N, D,
                                       BLOCK_N=triton.next_power_of_2(N))
            # grad_c[b,t,n] = sum_d dy[b,d,t] * h[b,d,t,n]
            grad_c = (dy.unsqueeze(-1) * h).sum(dim=1)
            return ga, gb, grad_c


def triton_scan_available() -> bool:
    """Whether the fused scan kernel can run on this machine."""
    if not _TRITON_OK:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def fused_scan_y(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Triton fused scan: (a [B,D,T,N] fp32 contiguous, b [B,D,T,N] fp32 contiguous, c [B,T,N] fp32 contiguous) -> y [B,D,T] fp32."""
    return _FusedScanFn.apply(a, b, c)
