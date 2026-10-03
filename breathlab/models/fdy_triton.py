"""Triton implementation of the FDY per-frequency dynamic convolution (forward, dx, and the correlation matrix G shared by dW/dA); never materializes im2col, math exactly equivalent to the loop / fuse / mix paths."""
from __future__ import annotations

from typing import List, Tuple

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:               # no triton installed -> the caller falls back to mix
    triton = None
    tl = None
    _TRITON_OK = False


def _cfg_fwd() -> List["triton.Config"]:
    """Forward-kernel candidates; the BCI = 16 entries win in practice (9 unrolled dots each take a shared-memory copy)."""
    return [
        triton.Config({"BCO": 64, "BCI": 16, "BT": 128}, num_warps=8, num_stages=1),
        triton.Config({"BCO": 32, "BCI": 16, "BT": 64}, num_warps=4, num_stages=2),
        triton.Config({"BCO": 32, "BCI": 32, "BT": 64}, num_warps=4, num_stages=1),
    ]


def _cfg_dx() -> List["triton.Config"]:
    """dx-kernel candidates; BCO = 32 is the safe choice (the BCO*BT shared-memory term dominates)."""
    return [
        triton.Config({"BCI": 16, "BCO": 32, "BT": 64}, num_warps=4, num_stages=2),
        triton.Config({"BCI": 16, "BCO": 32, "BT": 128}, num_warps=8, num_stages=1),
        triton.Config({"BCI": 16, "BCO": 64, "BT": 32}, num_warps=4, num_stages=1),
    ]


def _cfg_g() -> List["triton.Config"]:
    """Correlation-matrix kernel candidates (G is written fp32, so accumulators also cost registers)."""
    return [
        triton.Config({"BCO": 32, "BCI": 32, "BT": 32}, num_warps=4, num_stages=2),
        triton.Config({"BCO": 32, "BCI": 16, "BT": 64}, num_warps=4, num_stages=2),
        triton.Config({"BCO": 32, "BCI": 16, "BT": 128}, num_warps=8, num_stages=1),
    ]


# The autotune key contains only F / Ci / Co -- never T (or B): with T in the key every
# clip length re-benchmarks all configs and the GPU appears to hang. T is absorbed by the
# grid dimension, so changing T never requires a new config.
@triton.autotune(configs=_cfg_fwd(), key=["F", "Ci", "Co"])
@triton.jit
def _fwd_kernel(X, WE, Y, B, F, T, Ci, Co,
                sbx, scx, sfx, stx, sbw, sfw, scw, sby, scy, sfy, sty,
                BCO: tl.constexpr, BCI: tl.constexpr, BT: tl.constexpr,
                FP32: tl.constexpr):
    """y[b, co, f, t] = sum_{ci, tap} WE[b, f, co, ci*9+tap] * x[b, ci, f+di-1, t+dj-1]"""
    pid_bf = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_co = tl.program_id(2)
    b = pid_bf // F
    f = pid_bf % F
    offs_co = pid_co * BCO + tl.arange(0, BCO)
    offs_t = pid_t * BT + tl.arange(0, BT)
    co_mask = offs_co < Co
    t_mask = offs_t < T

    acc = tl.zeros((BCO, BT), dtype=tl.float32)
    for ci0 in range(0, Ci, BCI):
        offs_ci = ci0 + tl.arange(0, BCI)
        ci_mask = offs_ci < Ci
        for di in tl.static_range(3):
            fin = f + di - 1                       # source frequency row (scalar)
            kb = offs_ci * 9 + di * 3              # k base offset into WE
            for dj in tl.static_range(3):
                tin = offs_t + dj - 1              # source time column
                # [BCI, BT]: read x directly by index; out of bounds -> 0 (same as F.pad)
                xp = X + b * sbx + offs_ci[:, None] * scx + fin * sfx + tin[None, :] * stx
                xm = ci_mask[:, None] & (fin >= 0) & (fin < F) & \
                     (tin[None, :] >= 0) & (tin[None, :] < T)
                xt = tl.load(xp, mask=xm, other=0.0)
                # [BCO, BCI]: per-frequency mixed kernel for this tap
                wp = WE + b * sbw + f * sfw + offs_co[:, None] * scw + (kb + dj)[None, :]
                wm = co_mask[:, None] & ci_mask[None, :]
                wt = tl.load(wp, mask=wm, other=0.0)
                if FP32:
                    acc += tl.dot(wt, xt, input_precision="ieee")
                else:
                    acc += tl.dot(wt, xt)

    yp = Y + b * sby + offs_co[:, None] * scy + f * sfy + offs_t[None, :] * sty
    tl.store(yp, acc.to(Y.dtype.element_ty), mask=co_mask[:, None] & t_mask[None, :])


@triton.autotune(configs=_cfg_dx(), key=["F", "Ci", "Co"])
@triton.jit
def _dx_kernel(DY, WE, DX, B, F, T, Ci, Co,
               sbd, scd, sfd, std, sbw, sfw, scw, sbx, scx, sfx, stx,
               BCO: tl.constexpr, BCI: tl.constexpr, BT: tl.constexpr,
               FP32: tl.constexpr):
    """dx[b, ci, f, t] = sum_{di, dj, co} WE[b, f-di+1, co, ci*9+d] * dy[b, co, f-di+1, t-dj+1]"""
    pid_bf = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_ci = tl.program_id(2)
    b = pid_bf // F
    f = pid_bf % F
    offs_ci = pid_ci * BCI + tl.arange(0, BCI)
    offs_t = pid_t * BT + tl.arange(0, BT)
    ci_mask = offs_ci < Ci
    t_mask = offs_t < T

    acc = tl.zeros((BCI, BT), dtype=tl.float32)
    for co0 in range(0, Co, BCO):
        offs_co = co0 + tl.arange(0, BCO)
        co_mask = offs_co < Co
        for di in tl.static_range(3):
            fs = f - di + 1
            fsok = (fs >= 0) & (fs < F)
            kb = offs_ci * 9 + di * 3
            for dj in tl.static_range(3):
                ts = offs_t - dj + 1
                wp = WE + b * sbw + fs * sfw + offs_co[None, :] * scw + (kb + dj)[:, None]
                wm = co_mask[None, :] & ci_mask[:, None] & fsok
                wt = tl.load(wp, mask=wm, other=0.0)                 # [BCI, BCO]
                dyp = DY + b * sbd + offs_co[:, None] * scd + fs * sfd + ts[None, :] * std
                dm = co_mask[:, None] & fsok & (ts[None, :] >= 0) & (ts[None, :] < T)
                dyt = tl.load(dyp, mask=dm, other=0.0)               # [BCO, BT]
                if FP32:
                    acc = tl.dot(wt, dyt, acc, input_precision="ieee")
                else:
                    acc = tl.dot(wt, dyt, acc)

    xp = DX + b * sbx + offs_ci[:, None] * scx + f * sfx + offs_t[None, :] * stx
    tl.store(xp, acc.to(DX.dtype.element_ty), mask=ci_mask[:, None] & t_mask[None, :])


@triton.autotune(configs=_cfg_g(), key=["F", "Ci", "Co"])
@triton.jit
def _g_kernel(DY, X, G, B, F, T, Ci, Co,
              sbd, scd, sfd, std, sbx, scx, sfx, stx, sbg, sfg, scg,
              BCO: tl.constexpr, BCI: tl.constexpr, BT: tl.constexpr,
              FP32: tl.constexpr):
    """G[b,f,co,ci*9+d] = sum_t dy[b,co,f,t] * x[b,ci,f+di-1,t+dj-1]  (shared by dW/dA)"""
    pid_bf = tl.program_id(0)
    pid_blk = tl.program_id(1)
    nci = tl.cdiv(Ci, BCI)
    pid_co = pid_blk // nci
    pid_ci = pid_blk % nci
    b = pid_bf // F
    f = pid_bf % F
    offs_co = pid_co * BCO + tl.arange(0, BCO)
    offs_ci = pid_ci * BCI + tl.arange(0, BCI)
    co_mask = offs_co < Co
    ci_mask = offs_ci < Ci

    # Accumulate per tap and write back immediately: one [BCO, BCI] register block,
    # and the stores never conflict.
    for di in tl.static_range(3):
        fin = f + di - 1
        fok = (fin >= 0) & (fin < F)
        for dj in tl.static_range(3):
            acc = tl.zeros((BCO, BCI), dtype=tl.float32)
            for t0 in range(0, T, BT):
                offs_t = t0 + tl.arange(0, BT)
                tm = offs_t < T
                tin = offs_t + dj - 1
                dyp = DY + b * sbd + offs_co[:, None] * scd + f * sfd + offs_t[None, :] * std
                dyt = tl.load(dyp, mask=co_mask[:, None] & tm[None, :], other=0.0)  # [BCO, BT]
                xp = X + b * sbx + offs_ci[:, None] * scx + fin * sfx + tin[None, :] * stx
                xm = ci_mask[:, None] & fok & (tin[None, :] >= 0) & (tin[None, :] < T)
                xt = tl.load(xp, mask=xm, other=0.0)                                # [BCI, BT]
                if FP32:
                    acc = tl.dot(dyt, tl.trans(xt), acc, input_precision="ieee")
                else:
                    acc = tl.dot(dyt, tl.trans(xt), acc)
            gp = G + b * sbg + f * sfg + offs_co[:, None] * scg + \
                 (offs_ci * 9 + di * 3 + dj)[None, :]
            tl.store(gp, acc.to(G.dtype.element_ty), mask=co_mask[:, None] & ci_mask[None, :])


# ------------------------------ autograd ------------------------------

def _cdiv(a: int, b: int) -> int:
    """Integer ceiling division."""
    return (a + b - 1) // b


class _FDYDynConvFn(torch.autograd.Function):
    """x: [B, Ci, F, T] | w: [M, Co, Ci, 3, 3] | a: [B, M, F] -> y: [B, Co, F, T]"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B, Ci, F, T = x.shape
        M, Co, _, _, _ = w.shape
        src_dt = x.dtype
        # Match conv2d's autocast behaviour (fp32 -> bf16 before convolving), otherwise
        # the mixed-precision work is wasted and slower.
        dt = torch.bfloat16 if (src_dt == torch.float32 and torch.is_autocast_enabled()) \
            else src_dt
        xc = x if x.dtype == dt else x.to(dt)
        # Per-frequency mixed kernels (same matmul as the mix path); the trailing .to(dt)
        # is required: matmul itself silently downcasts to bf16 under autocast.
        w2 = w.reshape(M, Co * Ci * 9).to(dt)
        # NOTE: must be viewed as 4D [B, F, Co, Ci*9] -- the kernel uses we.stride(2) as
        # the "output channel stride"; 3D [B, F, Co*Ci*9] would make stride(2) == 1 and
        # only co == 0 would be correct.
        we = torch.matmul(a.transpose(1, 2).to(dt), w2).to(dt).reshape(B, F, Co, Ci * 9)
        y = torch.empty((B, Co, F, T), device=x.device, dtype=dt)

        _fwd_kernel[lambda meta: (B * F, _cdiv(T, meta["BT"]), _cdiv(Co, meta["BCO"]))](
            xc, we, y, B, F, T, Ci, Co,
            xc.stride(0), xc.stride(1), xc.stride(2), xc.stride(3),
            we.stride(0), we.stride(1), we.stride(2),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            FP32=(dt == torch.float32),
        )
        ctx.save_for_backward(xc, we, w, a)
        ctx.meta = (B, Ci, Co, F, T, M, dt, src_dt)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, we, w, a = ctx.saved_tensors
        B, Ci, Co, F, T, M, dt, src_dt = ctx.meta
        dy = dy.contiguous()
        fp32 = (dt == torch.float32)

        # ---- dx ----
        dx = torch.empty_like(x)
        _dx_kernel[lambda meta: (B * F, _cdiv(T, meta["BT"]), _cdiv(Ci, meta["BCI"]))](
            dy, we, dx, B, F, T, Ci, Co,
            dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
            we.stride(0), we.stride(1), we.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            FP32=fp32,
        )

        # ---- G: shared by dW and dA, computed only once ----
        g = torch.empty((B, F, Co, Ci * 9), device=x.device, dtype=torch.float32)
        _g_kernel[lambda meta: (B * F, _cdiv(Co, meta["BCO"]) * _cdiv(Ci, meta["BCI"]))](
            dy, x, g, B, F, T, Ci, Co,
            dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            g.stride(0), g.stride(1), g.stride(2),
            FP32=fp32,
        )
        # dW[m, co, k] = sum_{b, f} A[b, m, f] * G[b, f, co, k]
        dw = torch.einsum("mbf,bfok->mok", a.float().permute(1, 0, 2), g).view(M, Co, Ci, 3, 3)
        # dA[b, m, f] = sum_{co, k} W[m, co, k] * G[b, f, co, k]
        da = torch.einsum("mok,bfok->mbf", w.float().reshape(M, Co, Ci * 9), g)
        da = da.permute(1, 0, 2).contiguous()
        # autograd requires gradients to match the input dtype: cast back up here.
        return dx.to(src_dt), dw.to(w.dtype), da.to(a.dtype)


def fdy_dynconv_triton(x: torch.Tensor, conv_weights: List[torch.Tensor],
                       a: torch.Tensor) -> torch.Tensor:
    """Run the Triton FDY convolution: x [B,Ci,F,T], M kernels [Co,Ci,3,3], attention a [B,M,F] -> y [B,Co,F,T]."""
    w = torch.stack(conv_weights, dim=0)
    return _FDYDynConvFn.apply(x, w, a)


def triton_available() -> bool:
    """Return whether Triton is importable and a CUDA device is present."""
    if not _TRITON_OK:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False
