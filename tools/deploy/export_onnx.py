"""Export a BreathLab checkpoint to ONNX (network only; feature extraction stays in numpy in infer_onnx.py).

Exportability rules: `fdy_m`=0 or `fdy_mode` in {loop, fuse} (triton untraceable, mix unsupported
by DirectML), `fdy_checkpoint` off, `channels_last` off, fp32, dynamic time axis
(or pin with `--static-frames N`).

Usage: python tools/deploy/export_onnx.py --ckpt runs/v5_24k/best.pt [--opset 18] [--static-frames 1200]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from breathlab.common import load_config
from breathlab.models.crnn import build_model, count_params


class ExportWrapper(nn.Module):
    """Wraps the model so the ONNX graph ends in probabilities (sigmoid inside the graph)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def load_for_export(
    ckpt_path: str | Path,
    fdy_mode: str = "fuse",
    static_frames: Optional[int] = None,
) -> Tuple[nn.Module, Dict, Dict]:
    """Build an export-friendly model and load the checkpoint weights into it (eval, CPU)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    # --- force the graph into a form that traces and that DirectML supports ---
    mcfg = cfg.setdefault("model", {})
    tcfg = cfg.setdefault("train", {})
    if int(mcfg.get("fdy_m", 0)) > 0:
        # `triton` is a custom autograd Function (untraceable); `mix` emits im2col.
        # `loop` / `fuse` are pure Conv + Mul + Add, which every runtime supports.
        print(f"[export] fdy_m={mcfg['fdy_m']}: switching fdy_mode "
              f"{mcfg.get('fdy_mode')!r} -> {fdy_mode!r} for traceability")
        mcfg["fdy_mode"] = fdy_mode
    mcfg["fdy_checkpoint"] = False
    mcfg["channels_last"] = False
    tcfg["channels_last"] = False
    tcfg["compile"] = False
    tcfg["cudnn_benchmark"] = False

    model = build_model(cfg)
    # strict=True: a silent partial load would export a model that looks fine and scores garbage.
    model.load_state_dict(ckpt["model"], strict=True)
    n_params = count_params(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.n_params = n_params       # count_params() ignores frozen params, so keep it
    return model, cfg, ckpt


def build_metadata(cfg: Dict, ckpt: Dict, out_path: Path, opset: int,
                   head_index: Optional[Dict[str, int]] = None) -> Dict:
    """Collect everything the runtime needs to reproduce the training features."""
    f = cfg["feature"]
    # The model owns the channel layout (AP, then optional SP / onset / offset /
    # EP / GS); the hand-rolled computation is only a fallback for checkpoints
    # whose build_model did not set the attribute.
    heads = dict(head_index) if head_index else {"ap": 0}
    if not head_index:
        if cfg["model"].get("sp_head", False):
            heads["sp"] = 1
        if cfg["model"].get("boundary_head", False):
            heads["start"] = 2
            heads["end"] = 3
    return {
        "onnx_path": str(out_path),
        "opset": opset,
        "source_ckpt_epoch": ckpt.get("epoch"),
        "head_index": heads,
        "audio": {
            "sample_rate": cfg["audio"]["sample_rate"],
            "mono": bool(cfg["audio"].get("mono", True)),
        },
        "feature": {
            "n_fft": f["n_fft"],
            "win_length": f["win_length"],
            "hop_length": f["hop_length"],
            "n_mels": f["n_mels"],
            "fmin": f["fmin"],
            "fmax": f["fmax"],
            "log_offset": f["log_offset"],
            "f0_channels": f.get("f0_channels", 0),
            "input_rows": int(f.get("input_rows") or f["n_mels"]),
            "norm_mode": f.get("norm_mode", "global"),
            # pitch_channels() must use the same `frame` default as the training dataset,
            # otherwise the F0 rows shift by several ms.
            "pitch_frame_samples": 400,
            "pitch_fmin_hz": 70.0,
            "pitch_fmax_hz": 1000.0,
        },
        "fps": cfg["fps"],
        "mean": [float(v) for v in np.asarray(ckpt["mean"]).reshape(-1)],
        "std": [float(v) for v in np.asarray(ckpt["std"]).reshape(-1)],
        "threshold": float(ckpt.get("threshold", 0.5)),
        "postprocess": cfg["postprocess"],
        "eval": cfg["eval"],
    }


def export(
    ckpt_path: str | Path,
    out_path: str | Path | None = None,
    opset: int = 18,
    static_frames: Optional[int] = None,
    batch: int = 1,
    verify: bool = True,
) -> Path:
    """Export a checkpoint to ONNX and (optionally) verify numerics against Torch."""
    ckpt_path = Path(ckpt_path)
    if out_path is None:
        out_path = ckpt_path.parent / "model.onnx"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, cfg, ckpt = load_for_export(ckpt_path)
    rows = int(cfg["feature"].get("input_rows") or cfg["feature"]["n_mels"])
    # The wrapper exports every head channel; the model's head_index is the single
    # source of truth for the layout (AP, SP, onset/offset, EP/GS last).
    head_index = dict(getattr(model, "head_index", {}) or {})
    n_out = (max(head_index.values()) + 1) if head_index else \
        1 + int(bool(cfg["model"].get("sp_head", False))) \
        + 2 * int(bool(cfg["model"].get("boundary_head", False)))

    frames = static_frames if static_frames else 1200
    dummy = torch.zeros(batch, rows, frames, dtype=torch.float32)
    wrapper = ExportWrapper(model)

    print(f"[export] rows={rows} n_out={n_out} params={getattr(model, 'n_params', 0):,} "
          f"T={'dynamic' if static_frames is None else frames}")

    dynamic_axes = None
    if static_frames is None:
        dynamic_axes = {
            "logmel": {0: "batch", 2: "time"},
            "prob": {0: "batch", 2: "time"},
        }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["logmel"],
            output_names=["prob"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
            export_params=True,
        )

    # The TorchScript exporter leaves the module in train() mode (weights untouched);
    # restore eval() before verifying, else Dropout/BN would fire in the Torch reference.
    model.eval()

    import onnx

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    ops = sorted({n.op_type for n in onnx_model.graph.node})
    size_mb = out_path.stat().st_size / 1e6
    print(f"[export] wrote {out_path}  ({size_mb:.1f} MB, opset {opset})")
    print(f"[export] ops ({len(ops)}): {', '.join(ops)}")

    meta = build_metadata(cfg, ckpt, out_path, opset, head_index=head_index or None)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[export] metadata -> {meta_path}")

    if verify:
        _verify(out_path, model, rows, n_out)
    return out_path


def _verify(onnx_path: Path, model: nn.Module, rows: int, n_out: int,
            frames: int = 137, atol: float = 2e-5) -> None:
    """Compare onnxruntime output against the Torch reference on random input."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not installed - skipping numerical check")
        return

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Several lengths (traced / shorter / longer / not a multiple of the window) catch any
    # accidental constant folding of the time axis (e.g. a baked rotary table).
    rng = np.random.default_rng(0)
    worst = 0.0
    for t in (frames, 137, 601, 1200):
        x = rng.standard_normal((2, rows, t), dtype=np.float32)
        with torch.no_grad():
            ref = torch.sigmoid(model(torch.from_numpy(x))).numpy()
        got = sess.run(["prob"], {"logmel": x})[0]
        if got.shape != ref.shape:
            raise SystemExit(f"[verify] shape mismatch at T={t}: "
                             f"onnx {got.shape} vs torch {ref.shape}")
        diff = float(np.max(np.abs(got - ref)))
        worst = max(worst, diff)
        print(f"[verify] T={t:5d}  max|onnx - torch| = {diff:.3e}")

    status = "OK" if worst <= atol else "MISMATCH"
    print(f"[verify] {status}: worst={worst:.3e} (atol {atol:.0e})")
    if worst > atol:
        raise SystemExit("[verify] numerical mismatch - do not ship this model")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export a BreathLab checkpoint to ONNX.",
    )
    ap.add_argument("--ckpt", default="runs/v5_24k/best.pt", help="Source checkpoint.")
    ap.add_argument("--out", default=None, help="Output .onnx (default: <ckpt dir>/model.onnx).")
    ap.add_argument("--opset", type=int, default=18, help="ONNX opset (default 18).")
    ap.add_argument("--static-frames", type=int, default=None,
                    help="Pin the time axis to N frames instead of exporting it dynamic.")
    ap.add_argument("--batch", type=int, default=1, help="Dummy batch size for tracing.")
    ap.add_argument("--no-verify", action="store_true", help="Skip the onnxruntime check.")
    args = ap.parse_args()

    export(
        ckpt_path=args.ckpt,
        out_path=args.out,
        opset=args.opset,
        static_frames=args.static_frames,
        batch=args.batch,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
