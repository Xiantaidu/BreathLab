"""Frequency-Dynamic Convolution (FDY): per-frequency attention blends M candidate kernels, out = sum_m A_m(f) * Conv_m(x). Every kernel starts as a copy of the same weights with zero-initialized attention (uniform after softmax), so the layer begins mathematically identical to a plain convolution -- old checkpoints can be warm-started."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class FDYConv2d(nn.Module):
    """Frequency-dynamic ``nn.Conv2d`` (k = 3), same output shape; modes ``loop`` | ``fuse`` | ``mix`` | ``triton`` are all mathematically equivalent. Training recomputes activations (``ckpt``) so the M kernel outputs need not stay alive."""

    def __init__(self, cin: int, cout: int, kernel_size: int = 3, M: int = 4,
                 fused: bool = True, ckpt: bool = True, mode: str | None = None) -> None:
        super().__init__()
        assert M >= 2
        self.M = M
        self.mode = mode or ("fuse" if fused else "loop")
        self.ckpt = ckpt          # activation recomputation switch
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.convs = nn.ModuleList([
            nn.Conv2d(cin, cout, kernel_size, padding=self.pad, bias=False) for _ in range(M)
        ])
        # Per-frequency attention: [B, C, F] --1x1 conv--> [B, M, F] --softmax over M-->
        self.attn = nn.Conv1d(cin, M, kernel_size=1)
        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)
        self._same_init()

    @property
    def fused(self) -> bool:
        """Legacy boolean switch (benchmarks / fallback): ``False`` for ``loop``."""
        return self.mode != "loop"

    @fused.setter
    def fused(self, on: bool) -> None:
        """Legacy switch: ``False`` -> ``loop``, ``True`` -> ``fuse`` if looping."""
        if not on:
            self.mode = "loop"
        elif self.mode == "loop":
            self.mode = "fuse"

    def _same_init(self) -> None:
        """Copy the first kernel's weights into every other candidate kernel."""
        with torch.no_grad():
            for c in self.convs[1:]:
                c.weight.copy_(self.convs[0].weight)

    def _compute_mix(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frequency kernel mixing: Conv(x; W_eff(f)) with W_eff(f) = sum_m A_m(f) * W_m (exact; one conv's worth of compute)."""
        b, cin, nf, nt = x.shape
        k = self.kernel_size
        a = torch.softmax(self.attn(x.mean(dim=3)), dim=1)              # [B, M, F]
        # Per-frequency mixed kernels: W_eff[b, f] = sum_m A[b, m, f] * W_m -> [B*F, cout, cin*k*k]
        w2 = torch.cat([c.weight.reshape(1, -1) for c in self.convs], dim=0)   # [M, cout*cin*k*k]
        we = torch.matmul(a.transpose(1, 2), w2)                        # [B, F, cout*cin*k*k]
        cout = w2.shape[1] // (cin * k * k)
        we = we.reshape(b * nf, cout, cin * k * k)
        # im2col: [B, cin*k*k, F*T] -> [B*F, cin*k*k, T]
        p = F.unfold(x, kernel_size=k, padding=self.pad)                # [B, cin*k*k, F*T]
        p = p.view(b, cin, k * k, nf, nt).permute(0, 3, 1, 2, 4).reshape(b * nf, cin * k * k, nt)
        y = torch.bmm(we, p)                                            # [B*F, cout, T]
        return y.view(b, nf, cout, nt).permute(0, 2, 1, 3).contiguous()  # [B, cout, F, T]

    def _compute_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Triton path: same math as ``mix`` without the im2col traffic; permanent fallback to ``mix`` if the kernel fails."""
        from breathlab.models.fdy_triton import fdy_dynconv_triton, triton_available
        if not getattr(self, "_triton_ok", True) or not triton_available():
            return self._compute_mix(x)
        try:
            a = torch.softmax(self.attn(x.mean(dim=3)), dim=1)
            return fdy_dynconv_triton(x, [c.weight for c in self.convs], a)
        except Exception as e:                      # compile/run failure -> permanent fallback
            # checkpoint(use_reentrant=False) terminates recomputation early by raising
            # _StopRecomputationError; re-raise it as-is, it is not a kernel failure.
            if type(e).__name__ == "_StopRecomputationError":
                raise
            self._triton_ok = False
            print(f"[fdy] Triton kernel unavailable ({type(e).__name__}: {e}); "
                  f"this layer permanently falls back to the mix path")
            return self._compute_mix(x)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch to the configured compute path."""
        if self.mode == "triton":
            return self._compute_triton(x)
        if self.mode == "mix":
            return self._compute_mix(x)
        a = torch.softmax(self.attn(x.mean(dim=3)), dim=1)   # [B, M, F]
        if not self.fused:
            out = None
            for m, conv in enumerate(self.convs):
                term = conv(x) * a[:, m].unsqueeze(1).unsqueeze(3)   # [B, cout, F, T]
                out = term if out is None else out + term
            return out
        w = torch.cat([c.weight for c in self.convs], dim=0)          # [M*cout, cin, k, k]
        y = F.conv2d(x, w, bias=None, stride=1, padding=self.pad)     # [B, M*cout, F, T]
        y = y.view(y.shape[0], self.M, -1, *y.shape[2:])              # [B, M, cout, F, T]
        return (y * a[:, :, None, :, None]).sum(dim=1)                # [B, cout, F, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, cin, F, T] -> [B, cout, F, T]; activation recomputation in training when ``ckpt``."""
        if self.ckpt and self.training and torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(self._compute, x, use_reentrant=False)
        return self._compute(x)
