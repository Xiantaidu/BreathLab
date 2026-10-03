"""Run inference: wav files -> AP / SP (/ EP / GS) segmentation, exported as
TextGrid + JSON (default threshold is the best dev threshold stored in the checkpoint)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from breathlab.data.features import LogMel, frame_rms_energy, load_wav_16k
from breathlab.models.crnn import build_model
from breathlab.speed import setup_speed
from breathlab.postprocess import (density_constrained_events,
                                  full_segmentation, prob_to_events)


def collect_wavs(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files/directories into a sorted list of wav paths, skipping macOS
    resource-fork artefacts (``__MACOSX`` dirs, ``._*`` files)."""
    out: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for w in sorted(pp.rglob("*.wav")):
                s = str(w)
                if "__MACOSX" in s or w.name.startswith("._"):
                    continue
                out.append(w)
        elif pp.is_file():
            out.append(pp)
        else:
            print(f"[infer] skipping nonexistent path: {p}")
    return out


def load_runner(
    ckpt_path: str | Path,
    device: torch.device,
) -> tuple[dict[str, Any], Any, Any, float, torch.nn.Module]:
    """Load a checkpoint and build a ready-to-use inference runner
    ``(cfg, mean, std, threshold, model)``."""
    last = None
    for _ in range(3):  # training may be overwriting best.pt, so retry on failure
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            break
        except Exception as e:
            last = e
    else:
        raise last
    cfg = ckpt["cfg"]
    mean, std = ckpt["mean"], ckpt["std"]
    thr = float(ckpt.get("threshold", 0.5))
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    # No torch.compile here: inference sees many clip lengths and would recompile per
    # shape; setup_speed (channels_last / cuDNN autotune) is safe.
    model = setup_speed(model.to(device), cfg, device).eval()
    return cfg, mean, std, thr, model


@torch.no_grad()
def run_one(
    wav_path: str | Path,
    cfg: dict[str, Any],
    mean: Any,
    std: Any,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    norm: str = "global",
    emit: Sequence[str] = ("ap", "sp"),
    density_target: float = 0.0,
) -> tuple[list[tuple[float, float]], list[tuple[str, float, float]], float,
           dict[str, list[tuple[float, float]]]]:
    """Run the detector on a single wav, returning ``(ap_events, segments, duration,
    extra_events)``."""
    sr = cfg["audio"]["sample_rate"]
    y = load_wav_16k(wav_path, sr=sr)
    logmel = LogMel(cfg)
    lm = logmel(y)
    # High-band energy (top third of the log-mel rows, unnormalised): fricatives
    # z/ch/sh/f live up there while true silence sits at the floor.
    n_mels = cfg["feature"]["n_mels"]
    hf_db = lm[2 * n_mels // 3:].mean(axis=0)
    if norm == "perbin":
        # CMVN each mel channel back onto the training distribution (cross-domain robustness).
        bm = lm.mean(axis=1, keepdims=True)
        bs = lm.std(axis=1, keepdims=True)
        lm = (lm - bm) / (bs + 1e-6) * std.reshape(-1, 1) + mean.reshape(-1, 1)
        lm = (lm - mean.reshape(-1, 1)) / std.reshape(-1, 1)
    elif str(cfg["feature"].get("norm_mode", "global")).lower() == "cmvn":
        # Checkpoint trained with per-clip CMVN: stats must cover the whole clip (never
        # a window), matching ClipDataset, or the result silently degrades.
        mu = lm.mean(axis=1, keepdims=True)
        sd = lm.std(axis=1, keepdims=True) + 1e-5
        lm = (lm - mu) / sd
    else:
        lm = (lm - mean.reshape(-1, 1)) / std.reshape(-1, 1)
    if int(cfg["feature"].get("input_rows") or cfg["feature"]["n_mels"]) > cfg["feature"]["n_mels"]:
        # F0 rows have fixed [0, 1] semantics and never take part in normalisation.
        from breathlab.data.features import pitch_channels
        f0 = pitch_channels(y, sr, cfg["feature"]["hop_length"])
        T = min(lm.shape[1], f0.shape[1])
        lm = np.vstack([lm[:, :T], f0[:, :T]])
    from breathlab.engine import predict_probs_windowed
    x = torch.from_numpy(lm.astype(np.float32))
    prob, sp_prob, extras = predict_probs_windowed(model, x, cfg, device)
    energy = frame_rms_energy(y, cfg["feature"]["hop_length"], cfg["feature"]["win_length"])
    n = min(len(prob), len(energy))
    dur = len(y) / sr
    pre_events = None
    if density_target > 0:
        pre_events = density_constrained_events(
            prob[:n], cfg["fps"], dur, density_target, cfg["postprocess"],
            float(cfg["postprocess"].get("density_cand_threshold", 0.15)))
    ap_events, segments = full_segmentation(prob[:n], energy[:n], cfg, threshold,
                                            hf_db=hf_db[:n],
                                            sp_prob=None if sp_prob is None else sp_prob[:n],
                                            ap_events=pre_events)
    # Optional heads: independent per-head thresholding (no energy/SP interplay).
    pp = cfg["postprocess"]
    fps = cfg["fps"]
    extra_events: dict[str, list[tuple[float, float]]] = {}
    for head in ("ep", "gs"):
        if head not in emit:
            continue
        curve = extras.get(head)
        if curve is None:
            continue
        curve = curve[:n]
        thr = float(pp.get(f"{head}_threshold", 0.5))
        extra_events[head.upper()] = prob_to_events(
            curve, fps, thr, pp["median_filter_frames"], pp["min_dur_ms"], pp["merge_gap_ms"])
    return ap_events, segments, dur, extra_events


def _contiguous(
    entries: Iterable[tuple[float, float, str]],
    dur: float,
) -> list[tuple[float, float, str]]:
    """Fill a ``(start, end, label)`` list into contiguous intervals covering ``[0, dur]``."""
    out: list[tuple[float, float, str]] = []
    cur = 0.0
    for s, e, lab in sorted(entries):
        e = min(e, dur)
        if e <= s + 1e-6:
            continue
        s = max(s, cur)
        if e <= s + 1e-6:
            continue
        if s > cur + 1e-6:
            out.append((cur, s, ""))
        out.append((s, e, lab))
        cur = e
    if dur > cur + 1e-6:
        out.append((cur, dur, ""))
    return out


def write_textgrid(
    path: str | Path,
    dur: float,
    segments: Sequence[tuple[str, float, float]],
    ap_events: Sequence[tuple[float, float]],
    extra_events: dict[str, Sequence[tuple[float, float]]] | None = None,
) -> None:
    """Write a short-format TextGrid: a ``labels`` tier, an ``AP`` tier, plus optional
    EP/GS tiers (each made contiguous, gaps left blank)."""
    from praatio import textgrid as tglib

    seg_entries = [(s, min(e, dur), lab) for lab, s, e in segments if min(e, dur) > s + 1e-6]
    ap_entries = _contiguous([(s, e, "AP") for s, e in ap_events], dur)
    tg = tglib.Textgrid()
    tg.addTier(tglib.IntervalTier("labels", seg_entries, 0.0, dur))
    tg.addTier(tglib.IntervalTier("AP", ap_entries, 0.0, dur))
    for name, evs in (extra_events or {}).items():
        tg.addTier(tglib.IntervalTier(name, _contiguous([(s, e, name) for s, e in evs], dur),
                                      0.0, dur))
    try:
        fmt = tglib.TextgridOutputFormat.SHORT_TEXTGRID
    except AttributeError:
        fmt = "short_textgrid"
    tg.save(str(path), format=fmt, includeBlankSpaces=True)


def main() -> None:
    """Run inference over every requested wav and write TextGrid + JSON outputs."""
    ap = argparse.ArgumentParser(
        description="Run breath detection on wav files and export TextGrid + JSON.")
    ap.add_argument("--ckpt", default="runs/crnn_v2/best.pt")
    ap.add_argument("--wav", nargs="+", required=True, help="wav files or directories "
                                                            "(searched recursively)")
    ap.add_argument("--out", default=None, help="output directory, default runs/<tag>/infer")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the best threshold stored in the checkpoint")
    ap.add_argument("--norm", choices=["global", "perbin"], default="global",
                    help="perbin is recommended for out-of-domain audio "
                         "(e.g. pop vocal stems)")
    ap.add_argument("--emit", default="ap,sp", metavar="AP,SP[,EP][,GS]",
                    help="label families to output. AP and SP are the mandatory defaults; "
                         "EP (exhale) and GS (glottal stop) are optional and are only "
                         "emitted when explicitly listed here AND the checkpoint carries "
                         "the matching head.")
    ap.add_argument("--density-target", type=float, default=0.0,
                    help="breaths per second; > 0 replaces thresholding with per-clip "
                         "top-K density-constrained decoding (experimental, measured worse "
                         "than the threshold on v5_24k/test). 0 = off (default)")
    args = ap.parse_args()

    emit = [e.strip().lower() for e in args.emit.split(",") if e.strip()]
    bad = [e for e in emit if e not in ("ap", "sp", "ep", "gs")]
    if bad:
        raise SystemExit(f"--emit: unknown label(s) {bad}; allowed: ap, sp, ep, gs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, mean, std, thr0, model = load_runner(args.ckpt, device)
    # Optional heads exist only when the checkpoint was trained with them.
    have = getattr(model, "head_index", {})
    for head in ("ep", "gs"):
        if head in emit and head not in have:
            print(f"[infer] WARNING: --emit requests {head.upper()} but the checkpoint has "
                  f"no {head.upper()} head; it will not be output")
    emit = [e for e in emit if e in ("ap", "sp") or e in have]
    thr = args.threshold if args.threshold is not None else thr0
    tag = Path(args.ckpt).parent.name
    out_dir = Path(args.out) if args.out else Path(cfg["paths"]["runs_dir"]) / tag / "infer"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[infer] ckpt={args.ckpt}  threshold={thr:.2f}  ->  {out_dir}")

    wavs = collect_wavs(args.wav)
    print(f"{len(wavs)} wav file(s) in total")
    for w in wavs:
        try:
            ap_events, segments, dur, extra_events = run_one(
                w, cfg, mean, std, model, device, thr, norm=args.norm, emit=emit,
                density_target=args.density_target)
        except Exception as e:
            print(f"  [failed] {w.name}: {e}")
            continue
        stem = w.stem
        write_textgrid(out_dir / f"{stem}.TextGrid", dur, segments, ap_events, extra_events)
        meta = {
            "audio_file": w.name,
            "duration_sec": round(dur, 4),
            "threshold": thr,
            "ap_events": [{"start_sec": round(s, 3), "end_sec": round(e, 3)} for s, e in ap_events],
            "segments": [{"label": lab, "start_sec": round(s, 3), "end_sec": round(e, 3)}
                         for lab, s, e in segments],
        }
        for name, evs in extra_events.items():
            meta[f"{name.lower()}_events"] = [
                {"start_sec": round(s, 3), "end_sec": round(e, 3)} for s, e in evs]
        (out_dir / f"{stem}.ap.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        n_ap = sum(1 for lab, _s, _e in segments if lab == "AP")
        ap_sec = sum(e - s for lab, s, e in segments if lab == "AP")
        extra_txt = " | ".join(
            f"{n} {name} segments" for name, evs in sorted(extra_events.items())
            if (n := len(evs)))
        print(f"  {stem}: {n_ap} AP segments, {ap_sec:.2f}s / {dur:.2f}s"
              + (f" | {extra_txt}" if extra_txt else ""))
    print("done")


if __name__ == "__main__":
    main()
