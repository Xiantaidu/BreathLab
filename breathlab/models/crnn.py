"""Lightweight CRNN baseline: frequency-axis-pooling CNN + BiGRU + per-frame head; time is never downsampled (shortest breath events ~130 ms). Input [B, F, T] -> [B, C, T] per-frame logits (C = 1: AP, or 2: AP + SP)."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions (BN + ReLU each) followed by frequency-axis max-pooling."""

    def __init__(self, cin: int, cout: int, freq_pool: int, dropout: float, fdy_m: int = 0,
                 fdy_mode: str = "mix", fdy_ckpt: bool = True) -> None:
        super().__init__()
        if fdy_m > 0:
            from breathlab.models.fdy import FDYConv2d
            conv1 = FDYConv2d(cin, cout, 3, M=fdy_m, mode=fdy_mode, ckpt=fdy_ckpt)
            conv2 = FDYConv2d(cout, cout, 3, M=fdy_m, mode=fdy_mode, ckpt=fdy_ckpt)
        else:
            conv1 = nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False)
            conv2 = nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False)
        self.conv = nn.Sequential(
            conv1,
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            conv2,
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=(freq_pool, 1))  # pools the frequency axis only
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, F, T] -> [B, cout, F // freq_pool, T]."""
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class CRNN(nn.Module):
    """CNN front end -> bidirectional GRU -> per-frame classification head."""

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        m = cfg["model"]
        # Input rows = n_mels (+ F0 channels, see feature.f0_channels);
        # older checkpoints lack the key -> still n_mels.
        n_rows = int(cfg["feature"].get("input_rows") or cfg["feature"]["n_mels"])
        channels: List[int] = m["cnn_channels"]
        freq_pools: List[int] = m["cnn_freq_pool"]
        dropout = m["dropout"]
        fdy_m = int(m.get("fdy_m", 0))

        fdy_mode = str(m.get("fdy_mode") or "mix")
        fdy_ckpt = bool(m.get("fdy_checkpoint", True))
        blocks = []
        cin = 1
        f = n_rows
        for c, fp in zip(channels, freq_pools):
            blocks.append(ConvBlock(cin, c, fp, dropout, fdy_m=fdy_m,
                                    fdy_mode=fdy_mode, fdy_ckpt=fdy_ckpt))
            cin = c
            f = f // fp
        self.cnn = nn.Sequential(*blocks)
        self.cnn_out_dim = channels[-1] * f

        self.gru = nn.GRU(
            input_size=self.cnn_out_dim,
            hidden_size=m["gru_hidden"],
            num_layers=m["gru_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=dropout if m["gru_layers"] > 1 else 0.0,
        )
        n_out = 2 if m.get("sp_head", False) else 1     # ch0 = AP, ch1 = SP (supervised silence head)
        # Channel order: [ap, sp, start, end, optional...] -- optional heads LAST so
        # --init shape migration from older boundary-head checkpoints stays row-aligned.
        self.head_index = {"ap": 0}
        if m.get("sp_head", False):
            self.head_index["sp"] = 1
        if m.get("boundary_head", False):
            self.head_index["start"] = n_out
            self.head_index["end"] = n_out + 1
            n_out += 2
        for opt_head in (m.get("optional_heads") or []):
            name = str(opt_head).strip().lower()
            if name and name not in self.head_index:
                self.head_index[name] = n_out
                n_out += 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * m["gru_hidden"], n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, F, T] -> [B, C, T] logits, channel order given by ``head_index``."""
        x = x.unsqueeze(1)                 # [B, 1, F, T]
        x = self.cnn(x)                    # [B, C, F', T]
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # [B, T, C*F']
        x, _ = self.gru(x)                 # [B, T, 2H]
        return self.head(x).transpose(1, 2)  # [B, C, T]  C=1: AP | C=2: AP+SP


def count_params(model: nn.Module) -> int:
    """Count the parameters that require gradients."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: Dict) -> nn.Module:
    """Instantiate a backbone by name (``model.name``: crnn | conformer | unet_mamba)."""
    name = cfg["model"]["name"].lower()
    if name == "crnn":
        return CRNN(cfg)
    if name == "conformer":
        from breathlab.models.conformer import Conformer
        return Conformer(cfg)
    if name in ("unet_mamba", "unetmamba"):
        from breathlab.models.unet_mamba import UNetMamba
        return UNetMamba(cfg)
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    from breathlab.common import load_config

    cfg = load_config("breathlab/configs/default.yaml")
    net = build_model(cfg)
    x = torch.randn(2, cfg["feature"]["n_mels"], 600)
    y = net(x)
    print("Output shape:", tuple(y.shape))
    print(f"Parameters: {count_params(net):,}")
