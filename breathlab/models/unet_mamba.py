"""U-Net encoder + BiMamba bottleneck decoder for breath detection; topology is not checkpoint-compatible with the other backbones (CRNN/Conformer) -- train from scratch."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from breathlab.models.mamba_scan import BiMamba


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a 1x1 residual projection (no pooling inside)."""

    def __init__(self, cin: int, cout: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block on ``[B, C, F, T]``."""
        return self.drop(self.act(self.conv(x) + self.skip(x)))


class UNetMamba(nn.Module):
    """CNN U-Net encoder -> BiMamba bottleneck -> U-Net decoder -> per-frame heads (channel order in ``head_index``)."""

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        m = cfg["model"]
        from breathlab.models.mamba_scan import set_scan_impl
        set_scan_impl(str(m.get("um_scan", "auto")))
        n_rows = int(cfg["feature"].get("input_rows") or cfg["feature"]["n_mels"])
        dropout = float(m.get("dropout", 0.1))
        d_state = int(m.get("um_d_state", 16))
        d_conv = int(m.get("um_d_conv", 4))
        expand = int(m.get("um_expand", 2))
        n_layers = int(m.get("um_layers", 4))
        self.checkpoint = bool(m.get("um_checkpoint", True))
        self.ch_last = bool(cfg.get("train", {}).get("channels_last", False))

        # ---- encoder ----
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.eb1 = nn.Sequential(ResBlock(32, 64, dropout), nn.MaxPool2d((2, 2)))
        self.eb2 = nn.Sequential(ResBlock(64, 128, dropout), nn.MaxPool2d((2, 2)))
        # Default um_neck_stride_t=1 keeps the bottleneck at T/4, so a ~130 ms breath
        # (13 frames @100 fps) keeps 3-4 frames of resolution; set 2 to restore the
        # literal-diagram T/8 bottleneck.
        stride_t = int(m.get("um_neck_stride_t", 1))
        self.neck = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=(1, stride_t), padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))   # F -> 1, keep T

        # ---- BiMamba bottleneck (pre-norm residual stack) ----
        d = 256
        self.um_compile = bool(m.get("um_compile", True))
        self.norm_in = nn.LayerNorm(d)
        self.mamba_layers = nn.ModuleList([
            BiMamba(d, d_state=d_state, d_conv=d_conv, expand=expand,
                    fusion=str(m.get("um_fusion", "concat")))
            for _ in range(n_layers)
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d)
        # Compiled wrappers are created lazily and kept OUT of the module tree so
        # state_dict keys stay stable (checkpoints never see "_orig_mod" prefixes).
        self._compiled: List | None = None

        # ---- decoder: each stage is Up x2 -> conv (halve channels) -> concat skip -> conv ----
        def _stage(cin: int, cmid: int, cskip: int, cout: int) -> nn.Module:
            return nn.ModuleDict({
                "up": nn.Sequential(
                    nn.Conv2d(cin, cmid, 3, padding=1, bias=False),
                    nn.BatchNorm2d(cmid), nn.ReLU(inplace=True)),
                "fuse": nn.Sequential(
                    nn.Conv2d(cmid + cskip, cout, 3, padding=1, bias=False),
                    nn.BatchNorm2d(cout), nn.ReLU(inplace=True)),
            })

        self.dec1 = _stage(256, 128, 128, 128)   # + skip2 (128)
        self.dec2 = _stage(128, 64, 64, 64)      # + skip1 (64)
        self.dec3 = _stage(64, 32, 32, 32)       # + skip0 (32); fuse conv is 64 -> 32
        self.trunk_dropout = nn.Dropout(dropout)

        # ---- heads: per-frame Linear over the 32-d trunk, same head_index convention as
        # the other backbones. Optional heads LAST so --init shape migration stays row-aligned. ----
        self.head_index = {"ap": 0}
        n_out = 1
        if m.get("sp_head", False):
            self.head_index["sp"] = n_out
            n_out += 1
        if m.get("boundary_head", False):
            self.head_index["start"] = n_out
            self.head_index["end"] = n_out + 1
            n_out += 2
        for opt_head in (m.get("optional_heads") or []):
            name = str(opt_head).strip().lower()
            if name and name not in self.head_index:
                self.head_index[name] = n_out
                n_out += 1
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(32, n_out))

    def _layers_for(self) -> List:
        """BiMamba layers to call: torch.compile-wrapped in training only; eval always eager (windowed inference would recompile per tail shape)."""
        if self._compiled is None:
            if self.um_compile:
                try:
                    self._compiled = [torch.compile(l, dynamic=False)
                                      for l in self.mamba_layers]
                except Exception as e:                      # pragma: no cover
                    print(f"[unet_mamba] torch.compile unavailable ({e}); eager mode")
                    self._compiled = list(self.mamba_layers)
            else:
                self._compiled = list(self.mamba_layers)
        if self.training:
            return self._compiled
        return [getattr(l, "_orig_mod", l) for l in self.mamba_layers]

    def _run_mamba(self, h: torch.Tensor) -> torch.Tensor:
        """Pre-norm residual BiMamba stack over ``[B, T', d]``."""
        h = self.norm_in(h)
        layers = self._layers_for()
        use_ckpt = self.checkpoint and self.training and h.requires_grad
        for ln, layer in zip(self.mamba_norms, layers):
            if use_ckpt:
                from torch.utils.checkpoint import checkpoint
                h = h + checkpoint(layer, ln(h), use_reentrant=False)
            else:
                h = h + layer(ln(h))
        return self.norm_out(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ``[B, rows, T] -> [B, C, T]`` logits (channel order = head_index)."""
        if self.ch_last and x.device.type == "cuda":
            x4 = x.unsqueeze(1).contiguous(memory_format=torch.channels_last)
        else:
            x4 = x.unsqueeze(1)                                # [B, 1, F, T]
        s0 = self.stem(x4)                                     # [B, 32, F, T]
        s1 = self.eb1(s0)                                      # [B, 64, F/2, T/2]
        s2 = self.eb2(s1)                                      # [B, 128, F/4, T/4]
        n = self.neck(s2)                                      # [B, 256, F/4, T/4] (T/8 if um_neck_stride_t=2)
        h = self.freq_pool(n).squeeze(2).transpose(1, 2)       # [B, T/4, 256]
        h = self._run_mamba(h).transpose(1, 2).unsqueeze(2)    # [B, 256, 1, T/4]

        u1 = self.dec1["up"](F.interpolate(h, size=s2.shape[2:], mode="nearest"))
        d1 = self.dec1["fuse"](torch.cat([u1, s2], dim=1))     # [B, 128, F/4, T/4]
        u2 = self.dec2["up"](F.interpolate(d1, size=s1.shape[2:], mode="nearest"))
        d2 = self.dec2["fuse"](torch.cat([u2, s1], dim=1))     # [B, 64, F/2, T/2]
        u3 = self.dec3["up"](F.interpolate(d2, size=s0.shape[2:], mode="nearest"))
        d3 = self.dec3["fuse"](torch.cat([u3, s0], dim=1))     # [B, 32, F, T]
        trunk = d3.mean(dim=2).transpose(1, 2)                 # freq mean-pool -> [B, T, 32]
        return self.head(self.trunk_dropout(trunk)).transpose(1, 2)   # [B, C, T]
