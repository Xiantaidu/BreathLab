"""Train the CRNN breath detector (masked focal/weighted-BCE loss, bf16 autocast,
AdamW with cosine schedule after warmup; artifacts under ``<runs_dir>/<tag>/``)."""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from breathlab.common import ensure_dir, load_config, set_seed
from breathlab.data.dataset import (
    ClipDataset,
    PreparedClips,
    WindowDataset,
    compute_train_norm,
    read_split,
)
from breathlab.engine import evaluate_probs, group_indices, group_metrics, predict_clip_probs
from breathlab.postprocess import prefilter_probs
from breathlab.models.crnn import build_model, count_params
from breathlab.speed import describe, make_adamw, maybe_compile, sdpa_context, setup_speed


def focal_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    hardneg: torch.Tensor | None = None,
    hardneg_weight: float = 0.0,
) -> torch.Tensor:
    """Compute a masked binary focal loss with optional hard-negative upweighting."""
    p = torch.sigmoid(logit)
    ce = nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = torch.where(target > 0.5, p, 1 - p)
    alpha_t = torch.where(target > 0.5, torch.as_tensor(alpha, device=logit.device),
                          torch.as_tensor(1 - alpha, device=logit.device))
    loss = alpha_t * (1 - pt).pow(gamma) * ce
    loss = loss * mask
    if hardneg is not None and hardneg_weight > 0:
        loss = loss * (1.0 + float(hardneg_weight) * hardneg * (1.0 - target))
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def bce_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float,
    hardneg: torch.Tensor | None = None,
    hardneg_weight: float = 0.0,
) -> torch.Tensor:
    """Compute a masked BCE loss with positive-class weighting and optional
    hard-negative upweighting."""
    pw = torch.as_tensor(pos_weight, device=logit.device)
    ce = nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none",
                                                        pos_weight=pw)
    ce = ce * mask
    if hardneg is not None and hardneg_weight > 0:
        ce = ce * (1.0 + float(hardneg_weight) * hardneg * (1.0 - target))
    return ce.sum() / mask.sum().clamp(min=1.0)


def boundary_targets(label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive onset/offset frame targets from AP labels, bridging frame-wise training
    and the event-level (onset-matching) evaluation the boundary head aligns with."""
    prev = torch.nn.functional.pad(label[:, :-1], (1, 0))
    nxt = torch.nn.functional.pad(label[:, 1:], (0, 1))
    pos = label > 0.5
    return (pos & (prev <= 0.5)).float(), (pos & (nxt <= 0.5)).float()


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a linear-warmup + cosine-decay LR scheduler."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def main() -> None:
    """Run the full training loop: data prep, model build, train, dev sweep, early stop."""
    ap = argparse.ArgumentParser(
        description="Train the breath detector (focal loss + bf16 + cosine(warmup)).")
    ap.add_argument("--config", default="breathlab/configs/default.yaml")
    ap.add_argument("--tag", default="crnn")
    ap.add_argument("--init", default=None,
                    help="Initialize weights from an existing checkpoint (fine-tuning).")
    ap.add_argument("--dev-split", default=None,
                    help="Override the dev split name (default: dev). Use dev_bal, the "
                         "per-singer-capped balanced split built by "
                         "tools/make_bal_splits.py; the original dev.txt draws 2 singers "
                         "at random out of 20, so the aggregated F1 is dominated by a "
                         "single singer.")
    ap.add_argument("--lr", type=float, default=None, help="Override the configured learning rate.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override train.batch_size (the value stored in the checkpoint's "
                         "config follows this override).")
    ap.add_argument("--model", default=None, help="Override model.name (crnn|conformer).")
    ap.add_argument("--compile", action="store_true", default=None,
                    help="Force-enable torch.compile.")
    ap.add_argument("--no-compile", action="store_true", help="Force-disable torch.compile.")
    ap.add_argument("--no-harmony", action="store_true",
                    help="Ablation switch: disable harmony overlay augmentation "
                         "(harmony_overlay_prob=0).")
    ap.add_argument("--harmony-prob", type=float, default=None,
                    help="Ablation switch: override the harmony overlay probability (0~1). "
                         "If given together with --no-harmony, the latter wins.")
    ap.add_argument("--no-f0", action="store_true",
                    help="Ablation switch: drop the F0/periodicity input rows "
                         "(66 -> 64 rows, input_rows is updated accordingly).")
    ap.add_argument("--block", default=None, help="Override model.block: conformer|ebranchformer.")
    ap.add_argument("--branch", default=None, help="ebranchformer convolution branch: conv|cgmlp.")
    ap.add_argument("--norm-mode", default=None, help="Override feature.norm_mode: global|cmvn.")
    ap.add_argument("--hardneg-file", default=None,
                    help="Override data.hardneg_file: mined hard-negative JSON "
                         "(window oversampling).")
    ap.add_argument("--hardneg-weight", type=float, default=None,
                    help="Override train.hardneg_loss_weight: hard-negative frames are "
                         "weighted by (1+w).")
    ap.add_argument("--boundary-head", action="store_true",
                    help="Enable the onset/offset boundary auxiliary head (2 extra output "
                         "channels), aligning training with event-level evaluation.")
    ap.add_argument("--optional-heads", default=None, metavar="EP,GS",
                    help="Comma-separated optional output heads to train (e.g. 'ep,gs': "
                         "EP = exhale events, GS = glottal-stop events, both coming from "
                         "dedicated lists in the manifests). AP and SP are always present; "
                         "optional heads exist "
                         "only when listed here.")
    ap.add_argument("--ep-weight", type=float, default=None,
                    help="Override train.ep_loss_weight (focal loss weight of the EP head).")
    ap.add_argument("--gs-weight", type=float, default=None,
                    help="Override train.gs_loss_weight (focal loss weight of the GS head).")
    ap.add_argument("--boundary-weight", type=float, default=None,
                    help="Override train.boundary_loss_weight (default 0.5); use together "
                         "with --boundary-head.")
    ap.add_argument("--d-model", type=int, default=None,
                    help="Override model.d_model (head_dim=d_model/n_heads; keep it a "
                         "multiple of 8 to hit the efficient SDPA path).")
    ap.add_argument("--n-layers", type=int, default=None, help="Override model.n_layers.")
    ap.add_argument("--no-fdy", action="store_true",
                    help="Disable FDY (swap every FDY block for a plain convolution, "
                         "fdy_m=0). Measured to save 21%% step time; accuracy impact "
                         "still needs an A/B run.")
    ap.add_argument("--corpus-weight", action="append", default=None, metavar="NAME=W",
                    help="Override data.corpus_weights (repeatable), e.g. "
                         "--corpus-weight c10=0.35 --corpus-weight c13=0.8. "
                         "Feeds WeightedRandomSampler and decides the share of training "
                         "windows drawn from each corpus (the log prints the actual "
                         "[dataset] sampling mix).")
    ap.add_argument("--loss", default=None, help="Override train.loss: focal|bce.")
    ap.add_argument("--balance-by", default=None, choices=["none", "corpus", "singer"],
                    help="Equalize the sampling share of every corpus / singer in an epoch "
                         "(train.group_balance). Use 'singer' when the breath density differs "
                         "a lot between singers: without it the model can read the density "
                         "off the corpus identity instead of the audio.")
    ap.add_argument("--vtl", action="store_true",
                    help="Enable VTLN augmentation (train only): each window's mel "
                         "filterbank is warped by a random alpha, so singer-specific "
                         "spectral scale cannot be used as a shortcut. Uses "
                         "augment.vtl_alpha_range from the config (default +/-0.08).")
    ap.add_argument("--vtl-range", type=float, default=None, metavar="R",
                    help="Enable VTLN and set the alpha range to U(1-R, 1+R). "
                         "Evaluation / ONNX always use alpha=1, so checkpoints stay "
                         "compatible.")
    ap.add_argument("--audio-sr", type=int, default=None,
                    help="Override audio.sample_rate and recompute n_fft / win_length "
                         "(25 ms) / hop_length (10 ms) from their durations. Note: raising "
                         "the sample rate without raising --fmax is wasted effort (the mel "
                         "bank barely changes) and only costs extra disk space.")
    ap.add_argument("--fmax", type=int, default=None,
                    help="Override feature.fmax (the mel upper bound = true bandwidth). "
                         "Keeping n_mels fixed leaves the input row count unchanged, so "
                         "existing checkpoints stay 100%% compatible and can be fine-tuned "
                         "directly with --init.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.block:
        cfg["model"]["block"] = args.block
    if args.branch:
        cfg["model"]["branch"] = args.branch
    if args.norm_mode:
        cfg["feature"]["norm_mode"] = args.norm_mode
    if args.hardneg_file:
        cfg["data"]["hardneg_file"] = args.hardneg_file
    if args.hardneg_weight is not None:
        cfg["train"]["hardneg_loss_weight"] = args.hardneg_weight
    if args.boundary_head:
        cfg["model"]["boundary_head"] = True
    if args.optional_heads:
        heads = [h.strip().lower() for h in args.optional_heads.split(",") if h.strip()]
        cfg["model"]["optional_heads"] = heads
        print(f"[cfg] optional heads -> {heads}")
    if args.ep_weight is not None:
        cfg["train"]["ep_loss_weight"] = args.ep_weight
    if args.gs_weight is not None:
        cfg["train"]["gs_loss_weight"] = args.gs_weight
    if args.batch_size:
        cfg["train"]["batch_size"] = int(args.batch_size)
        print(f"[cfg] batch_size -> {cfg['train']['batch_size']}")
    if args.boundary_weight is not None:
        cfg["train"]["boundary_loss_weight"] = args.boundary_weight
    if args.d_model:
        cfg["model"]["d_model"] = args.d_model
    if args.n_layers:
        cfg["model"]["n_layers"] = args.n_layers
    if args.no_fdy:
        cfg["model"]["fdy_m"] = 0
    if args.loss:
        cfg["train"]["loss"] = args.loss
    if args.balance_by:
        cfg["train"]["group_balance"] = args.balance_by
        print(f"[cfg] group_balance -> {args.balance_by}")
    if args.vtl_range is not None:
        cfg["augment"]["vtl_enable"] = True
        cfg["augment"]["vtl_alpha_range"] = float(args.vtl_range)
    elif args.vtl:
        cfg["augment"]["vtl_enable"] = True
    if cfg["augment"].get("vtl_enable"):
        print(f"[cfg] VTLN on: alpha ~ U({1 - float(cfg['augment']['vtl_alpha_range']):.3f}, "
              f"{1 + float(cfg['augment']['vtl_alpha_range']):.3f}), "
              f"prob={cfg['augment'].get('vtl_prob', 1.0)} (training only; eval uses 1.0)")
    if args.corpus_weight:
        cw = cfg["data"].setdefault("corpus_weights", {})
        for kv in args.corpus_weight:
            k, _, v = kv.partition("=")
            if not k or not v:
                raise SystemExit(f"--corpus-weight expects NAME=W, got: {kv}")
            cw[k.strip()] = float(v)
        print(f"[cfg] corpus sampling weights -> {cw}")
    if args.audio_sr:
        sr = int(args.audio_sr)
        f = cfg["feature"]
        cfg["audio"]["sample_rate"] = sr
        f["n_fft"] = int(round(0.025 * sr))
        f["win_length"] = f["n_fft"]
        # hop_length stays time-defined (10 ms): a frame-rate change would silently desync model and labels.
        f["hop_length"] = int(round(0.010 * sr))
        print(f"[cfg] sample rate -> {sr} Hz (n_fft={f['n_fft']}, hop={f['hop_length']} "
              f"samples -> still {cfg['fps']} fps)")
    if args.fmax:
        f = cfg["feature"]
        sr = int(cfg["audio"]["sample_rate"])
        if int(args.fmax) > sr // 2:
            raise SystemExit(f"--fmax {args.fmax} exceeds Nyquist {sr // 2} "
                             f"(current sample rate {sr}); raise the sample rate with "
                             f"--audio-sr as well")
        f["fmax"] = int(args.fmax)
        print(f"[cfg] fmax -> {f['fmax']} Hz (n_mels={f['n_mels']} unchanged -> input rows "
              f"{f['input_rows']} unchanged, old checkpoints can be fine-tuned via --init)")

    # Ablation switches below must apply before any dataset/model is built; the config
    # stored in the checkpoint carries them, so evaluation stays consistent.
    if args.harmony_prob is not None:
        cfg["augment"]["harmony_overlay_prob"] = float(args.harmony_prob)
    if args.no_harmony:
        cfg["augment"]["harmony_overlay_prob"] = 0.0
    if args.no_f0:
        cfg["feature"]["f0_channels"] = 0
        cfg["feature"]["input_rows"] = int(cfg["feature"]["n_mels"])  # or model/data shapes disagree
    toggled: list[str] = []
    if args.no_harmony or args.harmony_prob is not None:
        toggled.append(f"harmony_prob={cfg['augment']['harmony_overlay_prob']}")
    if args.no_f0:
        toggled.append(f"f0_channels=0(input_rows={cfg['feature']['input_rows']})")
    if toggled:
        print("[ablation] " + ", ".join(toggled))

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = ensure_dir(Path(cfg["paths"]["runs_dir"]) / args.tag)
    print(f"[run] {run_dir}  device={device}")

    # ---- data ----
    splits_dir = cfg["paths"]["splits_dir"]
    dev_split = args.dev_split or "dev"
    train_names = read_split(splits_dir, "train")
    dev_names = read_split(splits_dir, dev_split)
    print(f"Loading train ({len(train_names)}) / {dev_split} ({len(dev_names)}) ...")
    # Train is streamed so large sets fit in RAM; dev is small and preloaded.
    train_prep = PreparedClips(cfg, train_names, build_breath_bank=True, preload=False)
    dev_prep = PreparedClips(cfg, dev_names, build_breath_bank=False, preload=True)
    print(f"Breath bank size: {len(train_prep.breath_bank)} segments")

    # The norm-stat cache signature must cover every mel-affecting config entry
    # (sr/fmax/n_mels/hop/norm_mode): stale stats after a change silently collapse
    # training. CMVN additionally needs per-clip whole-segment stats, warmed up here
    # and shared exactly with the evaluation side.
    _f = cfg["feature"]
    sig = (f"{len(train_names)}:{sum(train_prep._n_frames)}:"
           f"{cfg['audio']['sample_rate']}:{_f['fmax']}:{_f['n_mels']}:{_f['hop_length']}:"
           f"{_f.get('norm_mode', 'global')}")
    norm_cache = Path(cfg["paths"]["cache_dir"]) / "norm_stats.npz"
    need_clip_stats = str(cfg["feature"].get("norm_mode", "global")).lower() == "cmvn"
    mean = std = None
    clip_stats = None
    if norm_cache.exists():
        z = np.load(norm_cache, allow_pickle=True)
        if str(z["signature"]) == sig:
            mean, std = z["mean"], z["std"]
            if need_clip_stats and "clip_mu" in z and z["clip_mu"].shape[0] == len(train_prep):
                clip_stats = list(zip(z["clip_mu"], z["clip_sd"]))
            if clip_stats is not None or not need_clip_stats:
                print(f"Reusing cached normalisation: {norm_cache}")
    if mean is None or (need_clip_stats and clip_stats is None):
        print("Computing train-set normalisation statistics (streaming, a few minutes "
              "on the first run)...")
        if need_clip_stats:
            mean, std, clip_stats = compute_train_norm(train_prep, with_clip_stats=True)
        else:
            mean, std = compute_train_norm(train_prep)
        norm_cache.parent.mkdir(parents=True, exist_ok=True)
        extra: dict[str, Any] = {}
        if clip_stats is not None:
            extra = {"clip_mu": np.stack([s[0] for s in clip_stats]),
                     "clip_sd": np.stack([s[1] for s in clip_stats])}
        np.savez(norm_cache, mean=mean, std=std, signature=sig, **extra)
    if clip_stats is not None:
        train_prep.set_clip_cmvn(clip_stats)
        print(f"[cmvn] per-clip whole-segment stats injected into the train set "
              f"({len(clip_stats)} clips, same source as the evaluation side)")
    np.savez(run_dir / "norm.npz", mean=mean, std=std)

    train_ds = WindowDataset(train_prep, cfg, mean, std, augment=True)
    dev_clip_ds = ClipDataset(dev_prep, cfg, mean, std)
    print(f"Training windows: {len(train_ds)}")

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(train_ds.weights, dtype=torch.double),
        num_samples=len(train_ds), replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler,
        num_workers=cfg["train"]["num_workers"], drop_last=True, pin_memory=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
    )

    # ---- model ----
    if args.compile:
        cfg["train"]["compile"] = True
    if args.no_compile:
        cfg["train"]["compile"] = False
    model = build_model(cfg).to(device)
    model = setup_speed(model, cfg, device)
    print(describe(cfg, device))
    if args.init:
        state = torch.load(args.init, map_location="cpu", weights_only=False)
        msd = model.state_dict()
        sd = dict(state["model"])
        shape_fixed = []
        for k in list(sd):
            if k in msd and msd[k].shape != sd[k].shape:
                old, new = sd[k], torch.zeros_like(msd[k])
                sl = tuple(slice(0, min(a, b)) for a, b in zip(new.shape, old.shape))
                new[sl] = old[sl]          # head 1->2 channels: warm-start AP, zero SP
                sd[k] = new
                shape_fixed.append(k)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if shape_fixed:
            print(f"Shape migration: {shape_fixed} (new channels zero-initialised)")
        # strict=False: on structural upgrades (F0 channels / FDY kernels) new keys keep
        # their own initialisation while everything else is warm-started.
        if missing:
            from breathlab.models.fdy import FDYConv2d
            n_re = 0
            for mod_name, m in model.named_modules():
                if isinstance(m, FDYConv2d):
                    src_key = f"{mod_name}.weight"
                    if src_key in state["model"]:
                        w = state["model"][src_key]
                        with torch.no_grad():
                            for c in m.convs:
                                # every candidate kernel gets the old conv weights, so the
                                # FDY start equals a plain conv
                                c.weight.copy_(w)
                        n_re += 1
            if n_re:
                print(f"FDY warm start: convolution kernels copied from old weights at "
                      f"{n_re} positions")
        print(f"Weights initialised from {args.init} (epoch {state.get('epoch')}), entering "
              f"fine-tuning mode (missing keys {len(missing)}, unexpected keys "
              f"{len(unexpected)})")
        # A/B variants are expressed purely by stacking CLI switches; print config diffs
        # vs the checkpoint so a forgotten switch cannot silently drop transferred weights.
        ck_m = (state.get("cfg") or {}).get("model") or {}
        cur_m = cfg["model"]
        diffs = [(k, ck_m.get(k, "<missing>"), cur_m.get(k, "<missing>"))
                 for k in sorted(set(ck_m) | set(cur_m))
                 if k != "channels_last" and not k.startswith("um_")
                 and ck_m.get(k, "<missing>") != cur_m.get(k, "<missing>")]
        if diffs:
            print(f"[init] WARNING: current model config differs from the checkpoint in "
                  f"{len(diffs)} place(s) -> part of the transfer will be lost; add the "
                  f"matching switches to align fully with the checkpoint:")
            for k, a, b in diffs[:8]:
                print(f"       model.{k}: ckpt={a}  current={b}")
    model = maybe_compile(model, cfg, device)   # wrap with compile only after loading weights
    print(f"Model parameters: {count_params(model):,}")
    lr = args.lr if args.lr is not None else cfg["train"]["lr"]
    print(f"Learning rate: {lr}")
    opt = make_adamw(model.parameters(), lr, cfg, device)
    accum = max(1, int(cfg["train"].get("accum_steps", 1)))
    if accum > 1:
        print(f"Gradient accumulation: batch {cfg['train']['batch_size']} x {accum} steps "
              f"= effective batch {cfg['train']['batch_size'] * accum}")
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = steps_per_epoch * cfg["train"]["warmup_epochs"]
    sched = build_scheduler(opt, warmup_steps, total_steps)

    loss_name = cfg["train"]["loss"]
    amp_enabled = device.type == "cuda"

    best_f1, best_epoch, patience = -1.0, -1, 0
    history: list[dict[str, Any]] = []
    n_batches = len(train_loader)
    # SDPA context is entered via an ExitStack so the training loop gains no indentation.
    stack = ExitStack()
    stack.enter_context(sdpa_context(cfg))
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False, unit="batch")
        opt.zero_grad(set_to_none=True)
        hn_w = float(cfg["train"].get("hardneg_loss_weight", 0.0))
        sp_w = float(cfg["train"].get("sp_loss_weight", 0.5))
        ep_w = float(cfg["train"].get("ep_loss_weight", 0.5))
        gs_w = float(cfg["train"].get("gs_loss_weight", 0.5))
        bd_w = float(cfg["train"].get("boundary_loss_weight", 0.0))
        bd_pw = float(cfg["train"].get("boundary_pos_weight", 50.0))
        hi = getattr(model, "head_index", {"ap": 0})
        run_aux: dict[str, float] = {"sp": 0.0, "start": 0.0, "ep": 0.0, "gs": 0.0}
        for it, (lm, label, sp_label, mask, hn, ep_label, gs_label) in enumerate(pbar):
            lm, label, mask = lm.to(device), label.to(device), mask.to(device)
            sp_label = sp_label.to(device)
            ep_label = ep_label.to(device)
            gs_label = gs_label.to(device)
            hn = hn.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                out = model(lm)                      # [B, C, T]
                ap_logit = out[:, int(hi.get("ap", 0))]
                if loss_name == "focal":
                    loss = focal_loss(ap_logit, label, mask, cfg["train"]["focal_alpha"],
                                      cfg["train"]["focal_gamma"], hardneg=hn, hardneg_weight=hn_w)
                else:
                    loss = bce_loss(ap_logit, label, mask, cfg["train"]["bce_pos_weight"],
                                    hardneg=hn, hardneg_weight=hn_w)
                if "sp" in hi and out.shape[1] > int(hi["sp"]):
                    # SP head: fricatives/singing are not silence, so the contrast is learned directly.
                    sp_loss = bce_loss(out[:, int(hi["sp"])], sp_label, mask, pos_weight=1.0)
                    loss = loss + sp_w * sp_loss
                    run_aux['sp'] += sp_loss.item()
                if "ep" in hi and out.shape[1] > int(hi["ep"]) and ep_w > 0:
                    # Optional EP head (exhales): rare positives -> focal, same recipe as AP.
                    ep_loss = focal_loss(out[:, int(hi["ep"])], ep_label, mask,
                                         cfg["train"]["focal_alpha"], cfg["train"]["focal_gamma"])
                    loss = loss + ep_w * ep_loss
                    run_aux['ep'] += ep_loss.item()
                if "gs" in hi and out.shape[1] > int(hi["gs"]) and gs_w > 0:
                    # Optional GS head (glottal stops): extremely rare positives.
                    gs_loss = focal_loss(out[:, int(hi["gs"])], gs_label, mask,
                                         cfg["train"]["focal_alpha"], cfg["train"]["focal_gamma"])
                    loss = loss + gs_w * gs_loss
                    run_aux['gs'] += gs_loss.item()
                if "start" in hi and bd_w > 0:
                    # Boundary head: onset/offset frames are sparse (one per breath), hence the large pos_weight.
                    st_t, en_t = boundary_targets(label)
                    bd_start = bce_loss(out[:, int(hi["start"])], st_t, mask, pos_weight=bd_pw)
                    bd_end = bce_loss(out[:, int(hi["end"])], en_t, mask, pos_weight=bd_pw)
                    bd_loss = 0.5 * (bd_start + bd_end)
                    loss = loss + bd_w * bd_loss
                    run_aux['start'] += bd_loss.item()
            (loss / accum).backward()
            running += loss.item()                   # added once per batch; inside the accum block it would double
            if (it + 1) % accum == 0 or (it + 1) == n_batches:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        train_loss = running / n_batches

        # Sweep thresholds every epoch: a fixed 0.5 threshold is meaningless under focal loss.
        te = time.time()
        n_dev = len(dev_clip_ds)
        print(f"[eval] epoch {epoch}: {dev_split} {n_dev} clips, inference + threshold sweep...")
        # cudnn.benchmark off during dev eval: clip lengths vary wildly and every new shape
        # would trigger an algorithm re-search (training shapes are already cached).
        _cb_prev = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        try:
            probs, sp_probs, extras_probs = predict_clip_probs(model, dev_prep, dev_clip_ds,
                                                               device)
        finally:
            torch.backends.cudnn.benchmark = _cb_prev
        med_probs = prefilter_probs(probs, cfg["postprocess"]["median_filter_frames"])
        t_infer = time.time() - te
        metr, best_t = None, 0.5
        for t in tqdm(np.arange(0.30, 0.96, 0.05), desc="threshold sweep", unit="t", leave=False):
            m = evaluate_probs(probs, dev_prep, dev_clip_ds, cfg, threshold=float(t),
                               sp_probs=sp_probs, med_probs=med_probs,
                               extras_probs=extras_probs)
            if metr is None or (m["event"]["f1"], m["event"]["precision"]) > \
               (metr["event"]["f1"], metr["event"]["precision"]):
                metr, best_t = m, float(t)
        ev, fr = metr["event"], metr["frame"]
        sp_txt = ""
        if "sp_frame" in metr:
            sf = metr["sp_frame"]
            sp_txt = f" | SP frame F1={sf['f1']:.3f}(P={sf['precision']:.2f} R={sf['recall']:.2f})"
        opt_txt = ""
        for h in ("ep", "gs"):
            if f"{h}_frame" in metr:
                hf = metr[f"{h}_frame"]
                opt_txt += (f" | {h.upper()} frame F1={hf['f1']:.3f}"
                            f"(P={hf['precision']:.2f} R={hf['recall']:.2f})")
        # Monitor blends optional heads' frame F1 into AP event F1: AP can plateau while
        # EP/GS still improve (measured: AP peaked at epoch 1, EP at epoch 9).
        mon = ev["f1"]
        mon_txt = ""
        mon_parts = []
        for h in ("ep", "gs"):
            w = float(cfg["train"].get(f"monitor_{h}_weight", 0.0)) if h in hi else 0.0
            if w > 0 and f"{h}_frame" in metr:
                mon_parts.append((h, w, metr[f"{h}_frame"]["f1"]))
        if mon_parts:
            w_ap = float(cfg["train"].get("monitor_ap_weight", 1.0))
            mon = w_ap * ev["f1"] + sum(w * f1 for _, w, f1 in mon_parts)
            mon_txt = (" | monitor={:.3f} ({:.2f}*AP{:.3f} + {})".format(
                mon, w_ap, ev["f1"],
                " + ".join(f"{w:.2f}*{h.upper()}{f1:.3f}" for h, w, f1 in mon_parts)))
        dt = time.time() - t0
        aux_parts = []
        if "sp" in hi:
            aux_parts.append(f"sp={run_aux['sp'] / n_batches:.4f}")
        if "ep" in hi:
            aux_parts.append(f"ep={run_aux['ep'] / n_batches:.4f}")
        if "gs" in hi:
            aux_parts.append(f"gs={run_aux['gs'] / n_batches:.4f}")
        if "start" in hi and bd_w > 0:
            aux_parts.append(f"bd={run_aux['start'] / n_batches:.4f}")
        aux_txt = f" [{', '.join(aux_parts)}]" if aux_parts else ""
        print(f"epoch {epoch:3d} loss={train_loss:.4f}{aux_txt} | dev event F1={ev['f1']:.3f} "
              f"(P={ev['precision']:.3f} R={ev['recall']:.3f}) @t={best_t:.2f} "
              f"frame F1={fr['f1']:.3f}{sp_txt}{opt_txt}{mon_txt} "
              f"| total {dt:.1f}s (dev eval {time.time() - te:.1f}s, "
              f"of which inference {t_infer:.1f}s)")
        hrow = {"epoch": epoch, "train_loss": train_loss, "best_t": best_t,
                "dev_event": ev, "dev_frame": fr, "monitor": mon}
        for h in ("ep", "gs"):
            if f"{h}_frame" in metr:
                hrow[f"dev_{h}_frame"] = metr[f"{h}_frame"]
        history.append(hrow)

        # Per-group breakdown: an aggregate F1 can be dominated by one out-of-domain
        # singer (measured: 54% of dev breaths); only the breakdown shows where the
        # model is actually stuck.
        by_corpus = group_metrics(probs, dev_prep, dev_clip_ds, cfg, best_t,
                                  lambda c: c.corpus, sp_probs=sp_probs, med_probs=med_probs,
                                  extras_probs=extras_probs)
        print("[corpus] " + " | ".join(f"{k} {v['event']['f1']:.3f}" for k, v in by_corpus.items()))
        s_idx = group_indices(dev_prep, lambda c: f"{c.corpus}/{c.singer}")
        s_met = group_metrics(probs, dev_prep, dev_clip_ds, cfg, best_t,
                              lambda c: f"{c.corpus}/{c.singer}",
                              sp_probs=sp_probs, med_probs=med_probs,
                              extras_probs=extras_probs)
        worst = sorted(((k, v["event"]["f1"]) for k, v in s_met.items() if len(s_idx[k]) >= 20),
                       key=lambda x: x[1])[:3]
        if worst:
            print("[worst singers] " + " | ".join(f"{k} {f:.3f}({len(s_idx[k])} clips)"
                                                  for k, f in worst))
        history[-1]["dev_corpus_f1"] = {k: round(v["event"]["f1"], 4) for k, v in by_corpus.items()}

        if mon > best_f1:
            best_f1, best_epoch, patience = mon, epoch, 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                        "mean": mean, "std": std, "threshold": best_t,
                        "dev_split": dev_split}, run_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print(f"Early stop at epoch {epoch} (best epoch {best_epoch}, "
                      f"monitor={best_f1:.3f})")
                break

    stack.close()
    (run_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(f"Training finished. Best monitor={best_f1:.3f} @ epoch {best_epoch}")
    print(f"checkpoint: {run_dir/'best.pt'}")


if __name__ == "__main__":
    main()
