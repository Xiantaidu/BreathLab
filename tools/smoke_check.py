"""Smoke test: one training step + one inference-window forward + optional legacy-checkpoint shape migration.

Usage: PYTHONPATH=D:/BreathLab python tools/smoke_check.py [--ckpt runs/conformer_v1/best.pt]
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from breathlab.common import load_config
from breathlab.models.crnn import build_model, count_params
from breathlab.speed import describe, make_adamw, sdpa_context, setup_speed
from breathlab.train import bce_loss, focal_loss


def migrate_state_dict(model: nn.Module, state: dict) -> tuple[list[str], list[str], list[str]]:
    """Load a legacy checkpoint into ``model``, growing tensors whose shape changed."""
    sd = dict(state)
    msd = model.state_dict()
    shape_fixed: list[str] = []
    for k in list(sd):
        if k in msd and msd[k].shape != sd[k].shape:
            old, new = sd[k], torch.zeros_like(msd[k])
            sl = tuple(slice(0, min(a, b)) for a, b in zip(new.shape, old.shape))
            new[sl] = old[sl]
            sd[k] = new
            shape_fixed.append(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return shape_fixed, list(missing), list(unexpected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="breathlab/configs/default.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="optional: check whether an old checkpoint still loads")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["model"]["name"] = "conformer"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    model = setup_speed(model, cfg, device)
    print(describe(cfg, device))
    print(f"Parameters: {count_params(model):,}")

    rows = int(cfg["feature"].get("input_rows") or cfg["feature"]["n_mels"])
    t_train = int(cfg["dataset"]["window_sec"] * cfg["fps"])
    x = torch.randn(args.batch, rows, t_train, device=device)
    label = (torch.rand(args.batch, t_train, device=device) > 0.9).float()
    sp = (torch.rand(args.batch, t_train, device=device) > 0.5).float()
    mask = torch.ones(args.batch, t_train, device=device)
    opt = make_adamw(model.parameters(), 1e-4, cfg, device)
    model.train()
    with sdpa_context(cfg):
        for i in range(2):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                out = model(x)
                loss = focal_loss(out[:, 0], label, mask, cfg["train"]["focal_alpha"],
                                  cfg["train"]["focal_gamma"])
                if out.shape[1] >= 2:
                    loss = loss + 0.5 * bce_loss(out[:, 1], sp, mask, 1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
    print(f"[OK] training step: out={tuple(out.shape)} loss={loss.item():.4f} "
          f"peak VRAM={torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

    t_eval = int(cfg["eval"]["infer_window_sec"] * cfg["fps"])
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        y = model(torch.randn(1, rows, t_eval, device=device))
    print(f"[OK] inference window: out={tuple(y.shape)} (T={t_eval})")

    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        shape_fixed, missing, unexpected = migrate_state_dict(model, state["model"])
        print(f"[OK] ckpt {args.ckpt}: missing keys {len(missing)}, "
              f"unexpected keys {len(unexpected)}, shape-migrated {shape_fixed}")
        if missing:
            print(f"     sample missing keys: {missing[:4]}")


if __name__ == "__main__":
    main()
