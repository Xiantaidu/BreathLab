"""Model package: exposes ``build_model`` and ``count_params``; heavier backbones import lazily inside ``build_model``."""
from __future__ import annotations

from breathlab.models.crnn import build_model, count_params

__all__ = ["build_model", "count_params"]
