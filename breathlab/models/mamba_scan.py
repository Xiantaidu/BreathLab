"""Pure-PyTorch selective state-space (Mamba-style) block with a Hillis-Steele parallel scan (no native mamba-ssm dependency); the scan always runs in fp32 regardless of the outer autocast dtype."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def associative_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cumulative first-order recurrence h_t = a_t h_{t-1} + b_t (h_0 = b_0) via Hillis-Steele doubling: log2(T) batched passes."""
    T = a.shape[-1]
    step = 1
    while step < T:
        a_prev = torch.cat([torch.ones_like(a[..., :step]), a[..., :-step]], dim=-1)
        b_prev = torch.cat([torch.zeros_like(b[..., :step]), b[..., :-step]], dim=-1)
        b = a * b_prev + b
        a = a * a_prev
        step *= 2
    return b


class MambaBlock(nn.Module):
    """One direction of a selective SSM (Mamba-1 / S6, real diagonal A): [B, T, d] -> [B, T, d], causal in time."""

    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_rank: int | None = None, dt_min: float = 0.001,
                 dt_max: float = 0.1) -> None:
        super().__init__()
        self.d = d
        self.d_inner = expand * d
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d // 16)
        self.dt_min, self.dt_max = dt_min, dt_max

        self.in_proj = nn.Linear(d, 2 * self.d_inner, bias=False)   # x and gate (z)
        # left padding (k-1) makes the depthwise conv causal; the tail is sliced off
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1,
                                groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # S4D-Lin initialisation: A = -[1..N] for every channel, stored in log space.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))                     # [d_inner, N]
        self.D = nn.Parameter(torch.ones(self.d_inner))             # feedthrough skip
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)
        self._init_dt()

    def _init_dt(self) -> None:
        """dt_proj bias so that softplus(bias) is log-uniform in [dt_min, dt_max]."""
        dt = torch.exp(torch.rand(self.d_inner) *
                       (math.log(self.dt_max) - math.log(self.dt_min)) +
                       math.log(self.dt_min)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))                  # softplus^-1(dt)
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        std = self.dt_rank ** -0.5
        with torch.no_grad():
            self.dt_proj.weight.uniform_(-std, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the causal selective SSM over ``x`` of shape ``[B, T, d]``."""
        b, t, _ = x.shape
        xz = self.in_proj(x)                                        # [B, T, 2*di]
        xv, gate = xz.chunk(2, dim=-1)
        xv = self.conv1d(xv.transpose(1, 2))[..., :t].transpose(1, 2)
        xv = F.silu(xv)

        xdb = self.x_proj(xv)                                       # [B, T, rank + 2N]
        dt_in, b_in, c_in = xdb.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_in))                        # [B, T, di]
        A = -torch.exp(self.A_log)                                  # [di, N] (negative)

        # State tensors in [B, di, T, N] layout (N contiguous): what the broadcast produces
        # natively, and fp32 before the scan (the recurrence multiplies up to T decay
        # factors, bf16 would accumulate error across them).
        dtT = dt.float().transpose(1, 2)                            # [B, di, T]
        xvT = xv.float().transpose(1, 2)                            # [B, di, T]
        # A must align as [1, di, 1, N] against [B, di, T, 1] (right-aligned broadcast
        # would otherwise collide T with di)
        a = torch.exp(dtT.unsqueeze(-1) * A[None, :, None, :])           # [B, di, T, N]
        bdrv = dtT.unsqueeze(-1) * b_in.float().unsqueeze(1) * xvT.unsqueeze(-1)
        y = scan_y(a, bdrv, c_in)                                   # [B, di, T] fp32
        y = y.transpose(1, 2).to(xv.dtype)                          # [B, T, di]
        y = y + xv * self.D                                         # feedthrough
        y = y * F.silu(gate)
        return self.out_proj(y)


# Scan dispatch: Triton fused kernel (scan_triton.py) with a one-time permanent fallback
# to the torch reference. "auto" (default) uses Triton when available; "torch" forces the
# reference path (correctness baseline / debugging).

_SCAN_STATE = {"mode": "auto", "broken": False}


def set_scan_impl(mode: str) -> None:
    """Select the scan implementation: ``auto`` | ``triton`` | ``torch``."""
    _SCAN_STATE["mode"] = str(mode).lower()
    _SCAN_STATE["broken"] = False


def scan_y_torch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Reference scan on the ``[B, D, T, N]`` layout (Hillis-Steele along T)."""
    at = a.transpose(2, 3).contiguous()                             # [B, D, N, T]
    bt = b.transpose(2, 3).contiguous()
    h = associative_scan(at, bt)                                    # [B, D, N, T]
    cT = c.float().transpose(1, 2).unsqueeze(1)                     # [B, 1, N, T]
    return (h * cT).sum(dim=2)                                      # [B, D, T]


def scan_y(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """(a [B,D,T,N] fp32, b [B,D,T,N] fp32, c [B,T,N]) -> y [B,D,T] fp32; Triton fused path with permanent torch fallback."""
    mode = _SCAN_STATE["mode"]
    if not _SCAN_STATE["broken"] and mode in ("auto", "triton"):
        from breathlab.models.scan_triton import triton_scan_available, fused_scan_y
        if triton_scan_available():
            try:
                # the kernel reads contiguous [B, D, T, N] / [B, T, N] layouts; c comes
                # from a split() view whose row stride is rank+2N, so contiguous() is
                # load-bearing here (a/b are broadcast outputs and already contiguous)
                return fused_scan_y(a.float().contiguous(), b.float().contiguous(),
                                    c.float().contiguous())
            except Exception as e:                                  # pragma: no cover
                _SCAN_STATE["broken"] = True
                print(f"[mamba_scan] Triton fused scan failed once "
                      f"({type(e).__name__}: {e}); permanently falling back to the "
                      f"torch scan for the rest of this process")
    return scan_y_torch(a, b, c)


class BiMamba(nn.Module):
    """Bidirectional selective SSM: forward + time-flipped backward + learned fusion (``concat`` | ``sum``). [B, T, d] -> [B, T, d]."""

    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 fusion: str = "concat") -> None:
        super().__init__()
        self.fwd = MambaBlock(d, d_state, d_conv, expand)
        self.bwd = MambaBlock(d, d_state, d_conv, expand)
        self.fusion = fusion
        if fusion == "concat":
            self.mix = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse the two directional scans over ``x`` of shape ``[B, T, d]``."""
        f = self.fwd(x)
        r = self.bwd(x.flip(1)).flip(1)
        if self.fusion == "sum":
            return f + r
        return self.mix(torch.cat([f, r], dim=-1))
