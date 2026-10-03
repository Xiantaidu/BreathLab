"""Shared utilities (config loading, RNG seeding, time/frame conversion); numpy + PyYAML only so data-prep scripts can import without torch."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML configuration file into a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python / NumPy / (if installed) PyTorch RNGs; torch is imported lazily and failures swallowed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (including parents) if missing and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sec_to_frame(sec: float, fps: int) -> int:
    """Convert seconds to a frame index (round-half-to-even)."""
    return int(round(sec * fps))


def frames_from_duration(duration_sec: float, fps: int) -> int:
    """Clip duration -> frame count (``1 + floor(n)``), matching the ``center=True`` frame convention of the feature extractor."""
    return int(np.floor(duration_sec * fps)) + 1
