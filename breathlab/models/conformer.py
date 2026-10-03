"""Conformer-family models: convolution-augmented self-attention with per-frame heads; frequency-only CNN pooling (time stays at full resolution) and length-independent rotary embeddings, so inference can run at any length. [B, F, T] -> [B, C, T] logits."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish / SiLU activation: ``x * sigmoid(x)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ConvFront(nn.Module):
    """CNN front end pooling frequency only: [B, F, T] -> [B, T, C*F']; FDY optional per conv layer."""

    def __init__(self, channels: List[int], freq_pools: List[int], n_mels: int, dropout: float,
                 fdy_m: int = 0, fdy_mode: str = "fuse", fdy_ckpt: bool = True,
                 ch_last: bool = False, fdy_layers: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        blocks = []
        self.ch_last = ch_last
        from breathlab.models.fdy import FDYConv2d
        if isinstance(fdy_layers, str):          # also accept the "111100" shorthand
            fdy_layers = [int(v) for v in fdy_layers.strip() if v.strip()]
        n_conv = 2 * len(channels)
        if fdy_layers is not None:
            assert len(fdy_layers) == n_conv, (
                f"fdy_layers should have {n_conv} entries (2x blocks), got {len(fdy_layers)}")
        # fdy_layers: per-conv FDY switch ordered as conv1/conv2 of each block (2 * len(channels)
        # entries); None = FDY everywhere (legacy), e.g. [1,1,1,1,0,0] keeps FDY while F is
        # large (66/33) and uses plain convs after 8x pooling (F = 16).
        cin, f = 1, n_mels
        idx = 0
        for c, fp in zip(channels, freq_pools):
            convs = []
            for k_in in (cin, c):
                if fdy_m > 0 and (fdy_layers is None or int(fdy_layers[idx])):
                    convs.append(FDYConv2d(k_in, c, 3, M=fdy_m, mode=fdy_mode, ckpt=fdy_ckpt))
                else:
                    convs.append(nn.Conv2d(k_in, c, 3, padding=1, bias=False))
                idx += 1
            conv1, conv2 = convs
            blocks += [
                conv1, nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                conv2, nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.MaxPool2d((fp, 1)), nn.Dropout(dropout),
            ]
            cin, f = c, f // fp
        self.net = nn.Sequential(*blocks)
        self.out_dim = channels[-1] * f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, F, T] -> [B, T, C*F']."""
        x = x.unsqueeze(1)                              # [B, 1, F, T]
        if self.ch_last and x.device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)  # faster cuDNN convs
        x = self.net(x)                                 # [B, C, F', T]
        b, c, f, t = x.shape
        return x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # [B, T, C*F']


# Rotary tables depend only on (T, dim, device, dtype), so they are cached across steps.
# Deliberately not a persistent buffer: state_dict keys stay stable for older checkpoints.
_ROT_CACHE: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
_ROT_CACHE_ON = True          # benchmark / fallback switch


def set_rotary_cache_enabled(on: bool) -> None:
    """Enable or disable the rotary table cache (benchmark / fallback switch)."""
    global _ROT_CACHE_ON
    _ROT_CACHE_ON = bool(on)
    if not on:
        _ROT_CACHE.clear()


def _rotary_cache(t: int, dim: int, device: torch.device,
                  dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) rotary tables of shape [T, dim], cached by (T, dim, device, dtype)."""
    key = (int(t), int(dim), str(device), dtype)
    if _ROT_CACHE_ON:
        hit = _ROT_CACHE.get(key)
        if hit is not None:
            return hit
        if len(_ROT_CACHE) > 64:              # inference tail lengths vary; bound growth
            _ROT_CACHE.clear()
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    pos = torch.arange(t, device=device, dtype=torch.float32)
    ang = torch.outer(pos, inv)               # [T, dim/2]
    emb = torch.cat([ang, ang], dim=-1)       # [T, dim]
    val = (emb.cos().to(dtype), emb.sin().to(dtype))
    if _ROT_CACHE_ON:
        _ROT_CACHE[key] = val
    return val


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to a [B, H, T, D] tensor with [T, D] tables."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


_HEAD_PAD_ON = True


def set_head_pad_enabled(on: bool) -> None:
    """Enable or disable the SDPA head-dim padding trick (benchmark / fallback switch)."""
    global _HEAD_PAD_ON
    _HEAD_PAD_ON = bool(on)


def head_pad_enabled() -> bool:
    """Return whether head-dim padding is currently enabled."""
    return _HEAD_PAD_ON


class MHSA(nn.Module):
    """Multi-head self-attention with rotary embeddings and SDPA."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] -> [B, T, d_model]."""
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # [B, H, T, dk]
        cos, sin = _rotary_cache(t, self.dk, x.device, x.dtype)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        # head_dim must be a multiple of 8 or SDPA silently falls back to the slow math
        # backend; q/k/v are padded up (mathematically exact, the padded coordinates add 0)
        # and scale is passed explicitly because SDPA would otherwise derive it from the
        # padded size, which changes the temperature.
        pad = -self.dk % 8 if _HEAD_PAD_ON else 0
        if pad:
            q = F.pad(q, (0, pad))
            k = F.pad(k, (0, pad))
            v = F.pad(v, (0, pad))
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop.p if self.training else 0.0,
            scale=self.dk ** -0.5)
        if pad:
            o = o[..., : self.dk]
        o = o.transpose(1, 2).reshape(b, t, self.h * self.dk)
        return self.out(o)


class FFN(nn.Module):
    """Pre-norm feed-forward block: LayerNorm -> Linear -> Swish -> Dropout -> Linear."""

    def __init__(self, d_model: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * mult), Swish(),
            nn.Dropout(dropout), nn.Linear(d_model * mult, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvModule(nn.Module):
    """Conformer convolution module: pointwise GLU -> depthwise conv -> pointwise."""

    def __init__(self, d_model: int, kernel: int, dropout: float, use_ln: bool = True) -> None:
        super().__init__()
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(d_model) if use_ln else nn.Identity()
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)      # GLU
        self.dw = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = Swish()
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x).transpose(1, 2)        # [B, D, T]
        x = F.glu(self.pw1(x), dim=1)
        x = self.act(self.bn(self.dw(x)))
        x = self.drop(self.pw2(x))
        return x.transpose(1, 2)


class CgMLP(nn.Module):
    """Branchformer conv-gating MLP branch: depthwise temporal conv in a mult*d latent space."""

    def __init__(self, d_model: int, mult: int, kernel: int, dropout: float,
                 use_ln: bool = False) -> None:
        super().__init__()
        h = int(d_model * mult)
        self.ln = nn.LayerNorm(d_model) if use_ln else nn.Identity()
        self.up = nn.Linear(d_model, h)
        self.act = nn.GELU()
        self.dw = nn.Conv1d(h, h, kernel, padding=kernel // 2, groups=h)
        self.down = nn.Linear(h, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.up(self.ln(x)))
        x = self.dw(x.transpose(1, 2)).transpose(1, 2)     # temporal conv in the latent space only
        return self.drop(self.down(x))


class EnhancedMerge(nn.Module):
    """E-Branchformer enhanced merge: concat -> depthwise temporal conv (residual) -> project."""

    def __init__(self, d_model: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.dw = nn.Conv1d(2 * d_model, 2 * d_model, kernel, padding=kernel // 2,
                            groups=2 * d_model)
        self.proj = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        y = torch.cat([a, c], dim=-1)                      # [B, T, 2d]
        z = self.dw(y.transpose(1, 2)).transpose(1, 2)
        return self.proj(self.drop(y + z))


class EBranchformerBlock(nn.Module):
    """Parallel two-branch block (Branchformer / E-Branchformer style): FFN -> Merge(Attn, conv branch) -> FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, conv_kernel: int,
                 dropout: float, branch: str = "conv", merge_kernel: int = 31) -> None:
        super().__init__()
        self.ffn1 = FFN(d_model, ffn_mult, dropout)
        self.ffn2 = FFN(d_model, ffn_mult, dropout)
        self.ln = nn.LayerNorm(d_model)
        self.attn = MHSA(d_model, n_heads, dropout)
        self.att_drop = nn.Dropout(dropout)
        # Both branches run in parallel after this LN, so they must not normalize again.
        if branch.lower() == "cgmlp":
            self.conv = CgMLP(d_model, ffn_mult, merge_kernel, dropout, use_ln=False)
        else:
            self.conv = ConvModule(d_model, conv_kernel, dropout, use_ln=False)
        self.merge = EnhancedMerge(d_model, merge_kernel, dropout)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        h = self.ln(x)
        x = x + self.merge(self.att_drop(self.attn(h)), self.conv(h))
        x = x + 0.5 * self.ffn2(x)
        return self.ln_out(x)


class ConformerBlock(nn.Module):
    """Standard serial Conformer block: FFN -> MHSA -> convolution -> FFN (Macaron half-step FFNs)."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, conv_kernel: int,
                 dropout: float) -> None:
        super().__init__()
        self.ffn1 = FFN(d_model, ffn_mult, dropout)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = MHSA(d_model, n_heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = FFN(d_model, ffn_mult, dropout)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.drop(self.attn(self.ln_attn(x)))
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.ln_out(x)


class Conformer(nn.Module):
    """CNN front end -> projection -> Conformer/E-Branchformer blocks -> per-frame heads (channel order in ``head_index``: AP, then optional SP / onset / offset / extra heads)."""

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        m = cfg["model"]
        # Input rows = n_mels (+ F0 channels, see feature.f0_channels);
        # older checkpoints lack the key -> still n_mels.
        n_rows = int(cfg["feature"].get("input_rows") or cfg["feature"]["n_mels"])
        dropout = m.get("dropout", 0.1)
        fdy_m = int(m.get("fdy_m", 0))
        fdy_mode = str(m.get("fdy_mode") or ("mix" if m.get("fdy_fused", True) else "loop"))
        # default mix (fastest, mathematically equivalent); legacy fdy_fused=false maps to loop
        fdy_ckpt = bool(m.get("fdy_checkpoint", True))
        fdy_layers = m.get("fdy_layers")         # None = all FDY; [1,1,1,1,0,0] = first 4 only
        ch_last = bool(m.get("channels_last", False)) or bool(
            cfg.get("train", {}).get("channels_last", False))
        self.front = ConvFront(m["cnn_channels"], m["cnn_freq_pool"], n_rows, dropout, fdy_m=fdy_m,
                               fdy_mode=fdy_mode, fdy_ckpt=fdy_ckpt, ch_last=ch_last,
                               fdy_layers=fdy_layers)
        d_model = m["d_model"]
        self.proj = nn.Linear(self.front.out_dim, d_model)
        # block: conformer (serial attn -> conv, default) | ebranchformer (parallel branches + merge)
        block_kind = str(m.get("block", "conformer")).lower()
        self.block_kind = block_kind
        if block_kind in ("ebranchformer", "e_branchformer", "branchformer"):
            self.blocks = nn.ModuleList([
                EBranchformerBlock(d_model, m["n_heads"], m.get("ffn_mult", 4),
                                   m.get("conv_kernel", 15), dropout,
                                   branch=str(m.get("branch", "conv")),
                                   merge_kernel=int(m.get("merge_kernel", 31)))
                for _ in range(m["n_layers"])
            ])
        else:
            self.blocks = nn.ModuleList([
                ConformerBlock(d_model, m["n_heads"], m.get("ffn_mult", 4),
                               m.get("conv_kernel", 15), dropout)
                for _ in range(m["n_layers"])
            ])
        self.head_index = {"ap": 0}
        n_out = 1
        if m.get("sp_head", False):
            self.head_index["sp"] = n_out
            n_out += 1
        if m.get("boundary_head", False):
            self.head_index["start"] = n_out
            self.head_index["end"] = n_out + 1
            n_out += 2
        # Optional heads (EP = exhale, GS = glottal stop, ...) come LAST in channel order
        # so --init shape migration from older checkpoints appends rows instead of mixing
        # channel semantics.
        for opt_head in (m.get("optional_heads") or []):
            name = str(opt_head).strip().lower()
            if name and name not in self.head_index:
                self.head_index[name] = n_out
                n_out += 1
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, F, T] -> [B, C, T] logits, channel order given by ``head_index``."""
        x = self.front(x)                     # [B, T, C*F']
        x = self.proj(x)                      # [B, T, d_model]
        for blk in self.blocks:
            x = blk(x)
        return self.head(x).transpose(1, 2)   # [B, C, T]


if __name__ == "__main__":
    import time

    from breathlab.common import load_config
    from breathlab.models.crnn import build_model, count_params

    cfg = load_config("breathlab/configs/default.yaml")
    cfg["model"]["name"] = "conformer"
    n_rows = int(cfg["feature"].get("input_rows") or cfg["feature"]["n_mels"])
    x = torch.randn(2, n_rows, 600)
    for blk in ("conformer", "ebranchformer"):
        cfg["model"]["block"] = blk
        net = build_model(cfg)
        y = net(x)
        t0 = time.time()
        for _ in range(3):
            net(x)
        dt = (time.time() - t0) / 3 * 1000
        print(f"block={blk:14s} output {tuple(y.shape)} params {count_params(net):,} "
              f"forward {dt:.1f}ms (CPU, batch 2 x 600 frames)")
