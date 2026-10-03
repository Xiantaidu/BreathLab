"""Central training/inference speed switches (under ``cfg["train"]``, plus ``fdy_*`` in ``cfg["model"]``); all default to safe values and are implicitly disabled on CPU."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Iterable

import torch
import torch.nn as nn

DEFAULTS = {
    "channels_last": True,
    "cudnn_benchmark": True,
    "fused_adamw": True,
    "compile": False,
    "compile_mode": "default",
    "compile_dynamic": True,
    "sdpa_backend": "auto",
}


def speed_cfg(cfg: Dict) -> Dict:
    """Fill in speed defaults on ``cfg["train"]`` and return that section."""
    tc = cfg.setdefault("train", {})
    for k, v in DEFAULTS.items():
        tc.setdefault(k, v)
    return tc


def on_cuda(device: torch.device | str) -> bool:
    """True when CUDA is available and ``device`` refers to a CUDA device."""
    return torch.cuda.is_available() and getattr(device, "type", str(device)) == "cuda"


def maybe_compile(
    model: nn.Module,
    cfg: Dict,
    device: torch.device | str,
    allow_compile: bool = True,
) -> nn.Module:
    """Optionally wrap the model in ``torch.compile`` (call after loading checkpoint weights)."""
    tc = speed_cfg(cfg)
    if not allow_compile or not on_cuda(device) or not tc.get("compile", False):
        return model
    return torch.compile(
        model,
        mode=str(tc.get("compile_mode", "default")),
        dynamic=bool(tc.get("compile_dynamic", True)),
    )


def setup_speed(model: nn.Module, cfg: Dict, device: torch.device | str) -> nn.Module:
    """Enable channels_last memory format and cuDNN autotuning (after ``.to(device)``); compilation is handled by :func:`maybe_compile`."""
    tc = speed_cfg(cfg)
    if not on_cuda(device):
        return model
    if tc.get("channels_last", True):
        cfg["model"]["channels_last"] = True           # modules created later use NHWC
        for m in model.modules():
            if hasattr(m, "ch_last"):
                m.ch_last = True                       # already-built front-ends switch too
        model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = bool(tc.get("cudnn_benchmark", True))
    return model


def sdpa_context(cfg: Dict):
    """Context manager selecting the SDPA backend; ``auto`` = no-op (note: head_dim=44 has no flash-attention kernel)."""
    name = str(speed_cfg(cfg).get("sdpa_backend", "auto")).lower()
    if name in ("", "auto", "none"):
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    m = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem": SDPBackend.EFFICIENT_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    if name not in m:
        print(f"[speed] unknown sdpa_backend={name}, falling back to auto")
        return nullcontext()
    return sdpa_kernel(m[name])


def make_adamw(
    params: Iterable[nn.Parameter],
    lr: float,
    cfg: Dict,
    device: torch.device | str,
    weight_decay: float | None = None,
) -> torch.optim.AdamW:
    """Build an AdamW (fused kernel on CUDA when enabled); ``weight_decay`` defaults to ``cfg["train"]["weight_decay"]``."""
    tc = speed_cfg(cfg)
    if weight_decay is None:
        weight_decay = float(cfg["train"].get("weight_decay", 1e-4))
    fused = bool(tc.get("fused_adamw", True)) and on_cuda(device)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=fused)


def describe(cfg: Dict, device: torch.device | str) -> str:
    """One-line ``[speed] key=value ...`` summary of the active settings."""
    tc = speed_cfg(cfg)
    m = cfg.get("model", {})
    return (f"[speed] channels_last={tc['channels_last']} cudnn_benchmark={tc['cudnn_benchmark']} "
            f"fused_adamw={tc['fused_adamw']} compile={tc['compile']} "
            f"fdy_mode={m.get('fdy_mode', 'fuse')} fdy_checkpoint={m.get('fdy_checkpoint', True)} "
            f"sdpa={tc['sdpa_backend']}")
