"""Run a BreathLab ONNX model end-to-end on DirectML / CPU (no PyTorch needed): wav -> numpy log-mel + F0 -> onnxruntime -> TextGrid + JSON.

`--check-features` verifies the numpy features against torchaudio and `--compare-torch` verifies
the whole pipeline against `breathlab.infer` (both need torch; dev machine only).

Usage: python tools/deploy/infer_onnx.py --onnx runs/v5_24k/model.onnx --wav song.wav --out output/onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

_F_SP = 200.0 / 3.0          # Slaney: linear region slope (Hz per mel)
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    """Slaney mel scale (linear below 1 kHz, logarithmic above)."""
    mel = freq / _F_SP
    log_region = freq >= _MIN_LOG_HZ
    mel[log_region] = _MIN_LOG_MEL + np.log(freq[log_region] / _MIN_LOG_HZ) / _LOGSTEP
    return mel


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_hz_to_mel`."""
    freq = _F_SP * mel
    log_region = mel >= _MIN_LOG_MEL
    freq[log_region] = _MIN_LOG_HZ * np.exp(_LOGSTEP * (mel[log_region] - _MIN_LOG_MEL))
    return freq


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular Slaney filterbank `[n_mels, n_fft // 2 + 1]`, equivalent to torchaudio's melscale_fbanks(norm="slaney", mel_scale="slaney")."""
    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0.0, sr / 2.0, n_freqs)
    f_pts = _mel_to_hz(np.linspace(_hz_to_mel(np.array([fmin]))[0],
                                   _hz_to_mel(np.array([fmax]))[0], n_mels + 2))
    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        low = (all_freqs >= left) & (all_freqs <= center)
        high = (all_freqs >= center) & (all_freqs <= right)
        if center > left:
            fb[i, low] = (all_freqs[low] - left) / (center - left)
        if right > center:
            fb[i, high] = (right - all_freqs[high]) / (right - center)
        fb[i] *= 2.0 / (right - left)
    return fb


def logmel(y: np.ndarray, meta: Dict) -> np.ndarray:
    """log-mel `[n_mels, T]` replicating `breathlab.data.features.LogMel` (reflect padding, periodic Hann, `n_frames = 1 + n_samples // hop`)."""
    f = meta["feature"]
    sr = meta["audio"]["sample_rate"]
    n_fft, win, hop = f["n_fft"], f["win_length"], f["hop_length"]
    y = np.asarray(y, dtype=np.float32)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    pad = n_fft // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(yp) - n_fft) // hop
    # torch.hann_window(M) is periodic; np.hanning(M) is symmetric - they differ.
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win) / win)
    if win < n_fft:                       # torch zero-pads a short window to n_fft
        window = np.pad(window, (0, n_fft - win))
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = yp[idx] * window.astype(np.float32)
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = spec.real ** 2 + spec.imag ** 2          # power=2.0
    fb = mel_filterbank(sr, n_fft, f["n_mels"], f["fmin"], f["fmax"])
    mel = power @ fb.T                                # [T, n_mels]
    return np.log(mel.T + f["log_offset"]).astype(np.float32)


def pitch_channels(y: np.ndarray, sr: int, hop: int, frame: int = 400,
                   fmin: float = 70.0, fmax: float = 1000.0,
                   per_thresh: float = 0.30) -> np.ndarray:
    """Numpy port of `breathlab.data.features.pitch_channels`: `[2, T]` (periodicity, log-F0)."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    pad = frame // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + len(y) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = yp[idx] * np.hanning(frame).astype(np.float32)
    spec = np.fft.rfft(frames, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :frame]
    r = ac / (ac[:, :1] + 1e-9)
    lo, hi = max(1, int(sr / fmax)), min(frame - 1, int(sr / fmin))
    seg = r[:, lo: hi + 1]
    lag = seg.argmax(axis=1) + lo
    per = seg.max(axis=1).clip(0, 1)
    per = np.convolve(per, np.ones(3) / 3.0, mode="same")
    f0 = (sr / lag).astype(np.float32)
    f0[per < per_thresh] = 0.0
    lf = np.zeros_like(per)
    voiced = f0 > 0
    lf[voiced] = np.clip(
        (np.log2(f0[voiced]) - np.log2(fmin)) / (np.log2(1050.0) - np.log2(fmin)), 0.0, 1.0
    )
    return np.vstack([per, lf]).astype(np.float32)


def frame_rms_energy(y: np.ndarray, hop: int, frame_length: int) -> np.ndarray:
    """Per-frame RMS energy in dB, aligned with the mel frames (port of features.py)."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))
    pad = frame_length // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + len(y) // hop
    idx = np.arange(frame_length)[None, :] + hop * np.arange(n_frames)[:, None]
    rms = np.sqrt(np.mean(yp[idx] ** 2, axis=1) + 1e-12)
    return (20.0 * np.log10(rms + 1e-8)).astype(np.float32)


class BreathOnnx:
    """ONNX breath detector: wav -> AP events and AP/SP/V segmentation."""

    def __init__(self, onnx_path: str | Path, meta_path: str | Path | None = None,
                 provider: str = "auto", threshold: float | None = None):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        if meta_path is None:
            meta_path = self.onnx_path.with_suffix(".meta.json")
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.threshold = float(threshold if threshold is not None
                               else self.meta.get("threshold", 0.5))

        available = ort.get_available_providers()
        if provider == "auto":
            order = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            order = [provider, "CPUExecutionProvider"]
        providers = [p for p in order if p in available] or ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(self.onnx_path), opts, providers=providers)
        self.provider = self.sess.get_providers()[0]
        self.head = self.meta.get("head_index", {"ap": 0})
        # Channel count is static in the graph (only batch/time are dynamic), so the
        # optional-head guard can be resolved once here instead of per window.
        out_shape = self.sess.get_outputs()[0].shape
        self.n_channels = int(out_shape[1]) if isinstance(out_shape[1], int) else None
        # Optional event heads (EP = exhale, GS = glottal stop). start/end are boundary
        # channels, not event heads, so they are deliberately excluded here.
        self.opt_heads = {name: int(idx) for name, idx in self.head.items()
                          if name in ("ep", "gs")
                          and (self.n_channels is None or int(idx) < self.n_channels)}

    def load_wav(self, path: str | Path) -> np.ndarray:
        """Decode to mono float32 at the model's sample rate (scipy resampling)."""
        from math import gcd
        from scipy.signal import resample_poly

        y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        sr = self.meta["audio"]["sample_rate"]
        if int(file_sr) != int(sr):
            g = gcd(int(file_sr), int(sr))
            y = resample_poly(y, sr // g, file_sr // g)
        return np.ascontiguousarray(y, dtype=np.float32)

    def features(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(model_input [rows, T], hf_db [T], energy_db [T])`."""
        f = self.meta["feature"]
        lm = logmel(y, self.meta)
        n_mels = f["n_mels"]
        hf_db = lm[2 * n_mels // 3:].mean(axis=0)          # raw, before normalisation
        if f.get("norm_mode", "global") == "cmvn":
            # Per-clip per-bin CMVN over the WHOLE clip, exactly as ClipDataset; never per window.
            mu = lm.mean(axis=1, keepdims=True)
            sd = lm.std(axis=1, keepdims=True) + 1e-5
            mel = (lm - mu) / sd
        else:
            mel = (lm - np.asarray(self.meta["mean"], dtype=np.float32).reshape(-1, 1)) \
                / np.asarray(self.meta["std"], dtype=np.float32).reshape(-1, 1)
        if int(f.get("f0_channels", 0)) > 0:
            # `frame` must stay at the training default (400) or the F0 rows shift.
            f0 = pitch_channels(y, self.meta["audio"]["sample_rate"], f["hop_length"],
                                frame=int(f.get("pitch_frame_samples", 400)),
                                fmin=float(f.get("pitch_fmin_hz", 70.0)),
                                fmax=float(f.get("pitch_fmax_hz", 1000.0)))
            t = min(mel.shape[1], f0.shape[1])
            mel, f0 = mel[:, :t], f0[:, :t]
            mel = np.vstack([mel, f0])                      # F0 rows keep [0,1] semantics
        energy = frame_rms_energy(y, f["hop_length"], f["win_length"])
        n = min(mel.shape[1], len(energy))
        return mel[:, :n], hf_db[:n], energy[:n]

    def probs(self, x: np.ndarray, batch: int = 8) -> Tuple[np.ndarray, Optional[np.ndarray],
                                                            Dict[str, np.ndarray]]:
        """Chunked sliding-window inference; returns `(ap_prob, sp_prob | None, extras)`.

        `extras` holds one averaged probability curve per optional event head (EP/GS)
        present in the graph.
        """
        ev = self.meta["eval"]
        fps = self.meta["fps"]
        win = int(round(ev.get("infer_window_sec", 12.0) * fps))
        hop = int(round(ev.get("infer_hop_sec", 6.0) * fps))
        t = x.shape[1]
        starts = [0] if t <= win else list(range(0, t - win + 1, max(1, hop)))
        if t > win and starts[-1] != t - win:
            starts.append(t - win)
        ap_acc = np.zeros(t, dtype=np.float64)
        sp_acc = np.zeros(t, dtype=np.float64)
        opt_acc = {name: np.zeros(t, dtype=np.float64) for name in self.opt_heads}
        wsum = np.zeros(t, dtype=np.float64)
        for i0 in range(0, len(starts), batch):
            chunk = starts[i0:i0 + batch]
            w = min(win, t - chunk[0])
            xs = np.stack([x[:, s:s + w] for s in chunk]).astype(np.float32)
            out = self.sess.run(["prob"], {"logmel": xs})[0]        # [n, C, w]
            for j, s in enumerate(chunk):
                ap_acc[s:s + w] += out[j, int(self.head.get("ap", 0))]
                if "sp" in self.head and out.shape[1] > int(self.head["sp"]):
                    sp_acc[s:s + w] += out[j, int(self.head["sp"])]
                for name, idx in self.opt_heads.items():
                    if out.shape[1] > idx:
                        opt_acc[name][s:s + w] += out[j, idx]
                wsum[s:s + w] += 1.0
        den = np.maximum(wsum, 1e-8)
        ap = (ap_acc / den).astype(np.float32)
        sp = (sp_acc / den).astype(np.float32) if "sp" in self.head else None
        extras = {name: (acc / den).astype(np.float32) for name, acc in opt_acc.items()}
        return ap, sp, extras

    def __call__(self, wav_path: str | Path,
                 emit: Sequence[str] = ("ap", "sp")
                 ) -> Tuple[List[Tuple[float, float]], List[Tuple[str, float, float]],
                            float, Dict[str, List[Tuple[float, float]]]]:
        from breathlab.postprocess import full_segmentation, prob_to_events

        y = self.load_wav(wav_path)
        x, hf_db, energy = self.features(y)
        ap_prob, sp_prob, extras = self.probs(x)
        ap_events, segments = full_segmentation(
            ap_prob, energy, self.meta, self.threshold, hf_db=hf_db, sp_prob=sp_prob
        )
        # Optional heads: independent per-head thresholding (no energy/SP interplay),
        # identical to breathlab.infer.
        pp = self.meta["postprocess"]
        fps = self.meta["fps"]
        extra_events: Dict[str, List[Tuple[float, float]]] = {}
        for head in ("ep", "gs"):
            if head not in emit:
                continue
            curve = extras.get(head)
            if curve is None:
                continue
            thr = float(pp.get(f"{head}_threshold", 0.5))
            extra_events[head.upper()] = prob_to_events(
                curve, fps, thr, pp["median_filter_frames"], pp["min_dur_ms"],
                pp["merge_gap_ms"])
        return ap_events, segments, len(y) / self.meta["audio"]["sample_rate"], extra_events


def check_features(runner: BreathOnnx, wav_path: str | Path) -> None:
    """Compare the numpy feature stack against the torchaudio one used in training."""
    import torch
    from breathlab.data.features import (LogMel, frame_rms_energy as torch_energy,
                                         load_wav_16k, pitch_channels as torch_pitch)

    cfg = {"audio": {"sample_rate": runner.meta["audio"]["sample_rate"]},
           "feature": runner.meta["feature"]}
    y = runner.load_wav(wav_path)
    y_t = load_wav_16k(wav_path, sr=runner.meta["audio"]["sample_rate"])

    ref_mel = LogMel(cfg)(torch.from_numpy(np.ascontiguousarray(y_t)))
    got_mel = logmel(y, runner.meta)
    n = min(ref_mel.shape[1], got_mel.shape[1])
    d = np.abs(ref_mel[:, :n] - got_mel[:, :n])
    # Gate on p99.9, not max: log(x + 1e-6) amplifies float32 noise in dead bins far below
    # the log offset (measured: max rel err 0.6 at mel 1.7e-12, p99.9 = 1.6e-4); those bins
    # cannot influence the network.
    d_mel = float(np.percentile(d, 99.9))
    d_mel_max = float(d.max())

    f = runner.meta["feature"]
    ref_f0 = torch_pitch(y_t, runner.meta["audio"]["sample_rate"], f["hop_length"])
    got_f0 = pitch_channels(y, runner.meta["audio"]["sample_rate"], f["hop_length"],
                            frame=int(f.get("pitch_frame_samples", 400)))
    n2 = min(ref_f0.shape[1], got_f0.shape[1])
    d_f0 = float(np.max(np.abs(ref_f0[:, :n2] - got_f0[:, :n2])))

    ref_e = torch_energy(y_t, f["hop_length"], f["win_length"])
    got_e = frame_rms_energy(y, f["hop_length"], f["win_length"])
    n3 = min(len(ref_e), len(got_e))
    d_e = float(np.max(np.abs(ref_e[:n3] - got_e[:n3])))

    print(f"[check] wave        : {len(y)} vs {len(y_t)} samples")
    print(f"[check] log-mel     : p99.9|numpy-torchaudio| = {d_mel:.3e} "
          f"(max {d_mel_max:.3e}, in dead bins)  shape {ref_mel.shape}")
    print(f"[check] pitch/F0    : max|numpy-torchaudio| = {d_f0:.3e}")
    print(f"[check] rms energy  : max|numpy-torchaudio| = {d_e:.3e}")
    ok = d_mel < 1e-3 and d_f0 < 1e-5 and d_e < 1e-4
    print("[check] " + ("OK - numpy features match the training front end"
                        if ok else "MISMATCH - do not ship"))
    if not ok:
        raise SystemExit(1)


def compare_torch(runner: BreathOnnx, wav_path: str | Path, ckpt: str | Path) -> None:
    """Compare the full ONNX pipeline against `breathlab.infer` on one wav."""
    import torch
    from breathlab.infer import load_runner, run_one

    cfg, mean, std, thr, model = load_runner(ckpt, torch.device("cpu"))
    ref_events, ref_seg, ref_dur, ref_extra = run_one(wav_path, cfg, mean, std, model,
                                                      torch.device("cpu"), thr)
    got_events, got_seg, got_dur, got_extra = runner(wav_path)
    print(f"[compare] torch: {len(ref_events)} AP events, {ref_dur:.2f}s")
    print(f"[compare] onnx : {len(got_events)} AP events, {got_dur:.2f}s "
          f"(provider {runner.provider})")

    def match(a, b, collar=0.2):
        used = [False] * len(b)
        tp = 0
        for s, _e in a:
            best, bd = -1, None
            for j, (gs, _ge) in enumerate(b):
                if used[j]:
                    continue
                d = abs(s - gs)
                if d <= collar and (bd is None or d < bd):
                    bd, best = d, j
            if best >= 0:
                used[best] = True
                tp += 1
        return tp, len(a) - tp, len(b) - tp

    tp, fp, fn = match(got_events, ref_events)
    print(f"[compare] event match vs torch (200 ms): tp={tp} fp={fp} fn={fn}")
    if got_events and ref_events:
        ds = [abs(a[0] - b[0]) for a, b in zip(got_events, ref_events)]
        print(f"[compare] onset shift vs torch: mean {np.mean(ds)*1000:.1f} ms, "
              f"max {np.max(ds)*1000:.1f} ms")


def write_textgrid(path: str | Path, dur: float, segments, ap_events,
                   extra_events: Dict[str, List[Tuple[float, float]]] | None = None) -> None:
    """Write a short-format TextGrid with a `labels` tier, an `AP` tier, plus optional
    EP/GS tiers (each made contiguous, gaps left blank)."""
    def contiguous(entries):
        out, cur = [], 0.0
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

    tiers = [("labels", [(s, min(e, dur), lab) for lab, s, e in segments
                         if min(e, dur) > s + 1e-6]),
             ("AP", contiguous([(s, e, "AP") for s, e in ap_events]))]
    for name, evs in (extra_events or {}).items():
        tiers.append((name, contiguous([(s, e, name) for s, e in evs])))
    # Full ("normal") TextGrid format: praatio, Praat and Parselmouth all read it.
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
             "xmin = 0.0", f"xmax = {dur:.6f}", "tiers? <exists>",
             f"size = {len(tiers)}", "item []:"]
    for i, (name, items) in enumerate(tiers, start=1):
        lines += [f"    item [{i}]:", '        class = "IntervalTier"',
                  f'        name = "{name}"', "        xmin = 0.0",
                  f"        xmax = {dur:.6f}", f"        intervals: size = {len(items)}"]
        for j, (s, e, lab) in enumerate(items, start=1):
            lines += [f"        intervals [{j}]:", f"            xmin = {s:.6f}",
                      f"            xmax = {e:.6f}", f'            text = "{lab}"']
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a BreathLab ONNX model (DirectML / CPU).")
    ap.add_argument("--onnx", default="runs/v5_24k/model.onnx", help="Exported ONNX model.")
    ap.add_argument("--meta", default=None, help="Metadata JSON (default: <onnx>.meta.json).")
    ap.add_argument("--wav", nargs="+", default=None, help="Wav files or directories.")
    ap.add_argument("--out", default=None, help="Output directory (default: output/onnx).")
    ap.add_argument("--threshold", type=float, default=None, help="Override the threshold.")
    ap.add_argument("--emit", default="ap,sp", metavar="AP,SP[,EP][,GS]",
                    help="Event classes to output. EP (exhale) and GS (glottal stop) are "
                         "optional and are only emitted when explicitly listed here AND "
                         "the exported model carries the head.")
    ap.add_argument("--provider", default="auto",
                    help="'auto' (DML > CUDA > CPU), or an explicit onnxruntime provider.")
    ap.add_argument("--check-features", action="store_true",
                    help="Verify the numpy features against torchaudio (needs torch).")
    ap.add_argument("--compare-torch", action="store_true",
                    help="Compare the whole pipeline against breathlab.infer (needs torch).")
    ap.add_argument("--ckpt", default=None, help="Checkpoint for --compare-torch.")
    args = ap.parse_args()

    runner = BreathOnnx(args.onnx, args.meta, provider=args.provider,
                        threshold=args.threshold)
    print(f"[onnx] {runner.onnx_path}  provider={runner.provider}  "
          f"threshold={runner.threshold:.2f}")
    print(f"[onnx] sr={runner.meta['audio']['sample_rate']} "
          f"rows={runner.meta['feature']['input_rows']} fps={runner.meta['fps']} "
          f"norm={runner.meta['feature'].get('norm_mode')}")

    emit = [e.strip().lower() for e in args.emit.split(",") if e.strip()]
    bad = [e for e in emit if e not in ("ap", "sp", "ep", "gs")]
    if bad:
        raise SystemExit(f"--emit: unknown label(s) {bad}; allowed: ap, sp, ep, gs")
    for head in ("ep", "gs"):
        if head in emit and head not in runner.opt_heads:
            print(f"[onnx] WARNING: --emit requests {head.upper()} but the model has "
                  f"no {head.upper()} head; it will not be output")

    if args.check_features:
        for w in _collect(args.wav or []):
            check_features(runner, w)
    if args.compare_torch:
        if not args.ckpt:
            raise SystemExit("--compare-torch needs --ckpt")
        for w in _collect(args.wav or []):
            compare_torch(runner, w, args.ckpt)

    if not args.wav:
        return
    out_dir = Path(args.out) if args.out else Path("output/onnx")
    out_dir.mkdir(parents=True, exist_ok=True)
    for w in _collect(args.wav):
        ap_events, segments, dur, extra_events = runner(w, emit=emit)
        stem = Path(w).stem
        write_textgrid(out_dir / f"{stem}.TextGrid", dur, segments, ap_events,
                       extra_events)
        meta = {
            "audio_file": Path(w).name,
            "duration_sec": round(dur, 4),
            "threshold": runner.threshold,
            "provider": runner.provider,
            "ap_events": [{"start_sec": round(s, 3), "end_sec": round(e, 3)}
                          for s, e in ap_events],
            "segments": [{"label": lab, "start_sec": round(s, 3), "end_sec": round(e, 3)}
                         for lab, s, e in segments],
        }
        for name, evs in extra_events.items():
            meta[f"{name.lower()}_events"] = [
                {"start_sec": round(s, 3), "end_sec": round(e, 3)} for s, e in evs]
        (out_dir / f"{stem}.ap.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        n_ap = sum(1 for lab, _s, _e in segments if lab == "AP")
        ap_sec = sum(e - s for lab, s, e in segments if lab == "AP")
        extra_txt = " | ".join(
            f"{n} {name} segments" for name, evs in sorted(extra_events.items())
            if (n := len(evs)))
        print(f"  {stem}: {n_ap} AP segments, {ap_sec:.2f}s / {dur:.2f}s"
              + (f" | {extra_txt}" if extra_txt else "") + f" -> {out_dir}")


def _collect(paths) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out += [w for w in sorted(pp.rglob("*.wav"))
                    if "__MACOSX" not in str(w) and not w.name.startswith("._")]
        elif pp.is_file():
            out.append(pp)
    return out


if __name__ == "__main__":
    main()
