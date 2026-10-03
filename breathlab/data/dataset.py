"""Training (windowed) and evaluation (full-clip) datasets sharing one feature/target definition."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from breathlab.common import sec_to_frame
from breathlab.data.augment import (
    add_noise,
    add_noise_floor,
    apply_gain,
    apply_reverb,
    breath_overlay,
    channel_jitter,
    compress,
    harmony_overlay,
    spec_augment,
)
from breathlab.data.features import (
    LogMel,
    load_wav_16k,
    load_wav_16k_cached,
    pitch_channels,
    wav_cache_path,
)
from breathlab.data.parse_labels import Clip, load_all_clips

NormStats = Tuple[np.ndarray, np.ndarray]


# --- Config-derived helpers ---


def f0_enabled(cfg: Dict) -> bool:
    """True when the F0 (pitch) input channels are enabled."""
    return int((cfg["feature"].get("f0_channels") or 0)) > 0


def input_rows(cfg: Dict) -> int:
    """Number of input rows feeding the model front end (n_mels plus F0 rows when enabled)."""
    f = cfg["feature"]
    return int(f.get("input_rows") or f["n_mels"])


# --- Frame targets ---


def build_frame_targets(
    clip: Clip,
    n_frames: int,
    fps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Frame-resolution targets ``(label, mask, hardneg, sp_label, ep_label, gs_label)`` for one clip."""
    label = np.zeros(n_frames, dtype=np.float32)
    mask = np.ones(n_frames, dtype=np.float32)
    hardneg = np.zeros(n_frames, dtype=np.float32)
    sp_label = np.zeros(n_frames, dtype=np.float32)
    ep_label = np.zeros(n_frames, dtype=np.float32)
    gs_label = np.zeros(n_frames, dtype=np.float32)

    def span(ev):
        a = max(0, sec_to_frame(ev.start, fps))
        b = min(n_frames, sec_to_frame(ev.end, fps))
        return a, b

    for ev in clip.sp_events:
        a, b = span(ev)
        sp_label[a:b] = 1.0
    for ev in clip.exhales:
        a, b = span(ev)
        ep_label[a:b] = 1.0
        sp_label[a:b] = 0.0          # EP beats SP wherever they overlap an SP annotation
    for ev in clip.gs_events:
        a, b = span(ev)
        gs_label[a:b] = 1.0
        sp_label[a:b] = 1.0          # GS is acoustically silence -> SP head also fires there
    for ev in clip.breath_events:
        a, b = span(ev)
        label[a:b] = 1.0
        mask[a:b] = 1.0
        sp_label[a:b] = 0.0          # AP beats SP wherever they overlap an SP annotation
        ep_label[a:b] = 0.0
        gs_label[a:b] = 0.0
    for ev in clip.hard_negatives:
        a, b = span(ev)
        hardneg[a:b] = 1.0
    # silent_breaths already went into sp_label; all other frames stay negative
    for ev in clip.uncertain:
        a, b = span(ev)
        # uncertain only masks non-positive frames, so real breaths are never erased
        for f in range(a, b):
            if label[f] == 0.0:
                mask[f] = 0.0
    return label, mask, hardneg, sp_label, ep_label, gs_label


# --- Normalization ---


def clip_cmvn_of_logmel(x: np.ndarray) -> NormStats:
    """Per-mel-bin mean/std of a full-clip log-mel; the single source both training and evaluation use for CMVN."""
    return x.mean(axis=1).astype(np.float32), x.std(axis=1).astype(np.float32)


def normalize_mel(
    lm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mode: str = "global",
    cmvn_stats: Optional[NormStats] = None,
) -> np.ndarray:
    """Normalize the mel rows of a feature map (mode ``"global"`` or ``"cmvn"``); F0 rows are handled by the caller."""
    # CMVN statistics must come from the whole clip, never this window: training consumes
    # windows while evaluation consumes full clips, so window-level statistics would be two
    # different rulers between train and inference.
    if str(mode).lower() == "cmvn":
        if cmvn_stats is None:
            mu = lm.mean(axis=1, keepdims=True)
            sd = lm.std(axis=1, keepdims=True) + 1e-5
        else:
            mu = np.asarray(cmvn_stats[0], dtype=np.float32).reshape(-1, 1)
            sd = np.asarray(cmvn_stats[1], dtype=np.float32).reshape(-1, 1) + 1e-5
        return (lm - mu) / sd
    return (lm - mean) / std


# --- Prepared clip store ---


class PreparedClips:
    """Waveforms (or their cache paths), frame label arrays and an optional breath bank for one split."""

    def __init__(self, cfg: Dict, split_names: Sequence[str], build_breath_bank: bool = False,
                 preload: bool = True):
        self.cfg = cfg
        self.fps = cfg["fps"]
        self.hop = cfg["feature"]["hop_length"]
        self.sr = cfg["audio"]["sample_rate"]
        self.logmel = LogMel(cfg)
        self.preload = preload
        self.cache_dir = cfg["paths"]["cache_dir"]

        all_clips = load_all_clips(cfg)
        by_name = {c.name: c for c in all_clips}
        self.clips: List[Clip] = [by_name[n] for n in split_names if n in by_name]
        missing = [n for n in split_names if n not in by_name]
        if missing:
            listed = ", ".join(missing[:5])
            print(f"[dataset] WARNING: {len(missing)} clip name(s) missing from corpus: {listed}")

        # lazy per-clip CMVN stats; training injects precomputed ones via set_clip_cmvn
        self._clip_cmvn_cache: List[Optional[NormStats]] = [None] * len(self.clips)

        self.waveforms: List[np.ndarray] = []
        self.full_logmels: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.sp_labels: List[np.ndarray] = []
        self.ep_labels: List[np.ndarray] = []
        self.gs_labels: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.hardnegs: List[np.ndarray] = []
        self.breath_bank: List[np.ndarray] = []
        self._n_frames: List[int] = []

        if preload:
            for c in self.clips:
                y = np.asarray(load_wav_16k(c.wav_path, sr=self.sr), dtype=np.float32)
                lm = self.logmel(y)
                T = lm.shape[1]
                label, mask, hardneg, sp_label, ep_label, gs_label = build_frame_targets(
                    c, T, self.fps)
                self.waveforms.append(y)
                self.full_logmels.append(lm)
                self.labels.append(label)
                self.sp_labels.append(sp_label)
                self.ep_labels.append(ep_label)
                self.gs_labels.append(gs_label)
                self.masks.append(mask)
                self.hardnegs.append(hardneg)
        else:
            for i, c in enumerate(self.clips):
                cp = load_wav_16k_cached(c.wav_path, sr=self.sr, cache_dir=self.cache_dir)
                if isinstance(cp, Path):
                    n = int(np.load(cp, mmap_mode="r").shape[0])
                else:
                    n = len(cp)
                T = 1 + n // self.hop
                self._n_frames.append(T)
                label, mask, hardneg, sp_label, ep_label, gs_label = build_frame_targets(
                    c, T, self.fps)
                self.labels.append(label)
                self.sp_labels.append(sp_label)
                self.ep_labels.append(ep_label)
                self.gs_labels.append(gs_label)
                self.masks.append(mask)
                self.hardnegs.append(hardneg)
                if (i + 1) % 200 == 0:
                    print(f"[dataset] cache progress {i + 1}/{len(self.clips)}")

        if build_breath_bank:
            for ci, c in enumerate(self.clips):
                y = self.load_wave(ci)
                for ev in c.breath_events:
                    a = max(0, int(ev.start * self.sr))
                    b = min(len(y), int(ev.end * self.sr))
                    if b - a >= int(0.05 * self.sr):
                        self.breath_bank.append(np.asarray(y[a:b], dtype=np.float32).copy())

    def __len__(self) -> int:
        """Number of prepared clips."""
        return len(self.clips)

    def n_frames(self, ci: int) -> int:
        """Number of frames of clip ``ci``."""
        if self.preload:
            return self.full_logmels[ci].shape[1]
        return self._n_frames[ci]

    def load_wave(self, ci: int) -> np.ndarray:
        """Return the 16 kHz waveform of clip ``ci`` (read-only memmap in streaming mode)."""
        if self.preload:
            return self.waveforms[ci]
        cp = wav_cache_path(self.clips[ci].wav_path, self.sr, self.cache_dir)
        return np.load(cp, mmap_mode="r")

    def full_logmel(self, ci: int) -> np.ndarray:
        """Return the log-mel of the whole clip ``ci``, shape ``[n_mels, T]``."""
        if self.preload:
            return self.full_logmels[ci]
        # np.array() copies: the memmap is read-only while torch needs a writable buffer
        y = np.array(self.load_wave(ci), dtype=np.float32)
        return self.logmel(y)

    def clip_cmvn(self, ci: int) -> NormStats:
        """Whole-clip CMVN statistics ``(mu, sd)`` of clip ``ci``, memoized."""
        st = self._clip_cmvn_cache[ci]
        if st is None:
            st = clip_cmvn_of_logmel(self.full_logmel(ci))
            self._clip_cmvn_cache[ci] = st
        return st

    def set_clip_cmvn(self, stats: List[NormStats]) -> None:
        """Inject externally pre-computed per-clip statistics (from compute_train_norm)."""
        assert len(stats) == len(self.clips), (
            f"cmvn stats count {len(stats)} != clip count {len(self.clips)}"
        )
        self._clip_cmvn_cache = list(stats)

    def _cmvn_for(self, ci: int, mode: str) -> Optional[NormStats]:
        """Return the CMVN statistics for clip ``ci`` when ``mode`` is CMVN, else ``None``."""
        return self.clip_cmvn(ci) if str(mode).lower() == "cmvn" else None


# --- Datasets ---


class WindowDataset(Dataset):
    """Training dataset: fixed-length windows sampled from prepared clips, with augmentation."""

    def __init__(self, prepared: PreparedClips, cfg: Dict, mean: np.ndarray, std: np.ndarray,
                 augment: bool):
        self.p = prepared
        self.cfg = cfg
        self.augment = augment
        self.mean = mean.reshape(-1, 1).astype(np.float32)
        self.std = std.reshape(-1, 1).astype(np.float32)
        self.win_frames = int(round(cfg["dataset"]["window_sec"] * cfg["fps"]))
        self.norm_mode = str(cfg["feature"].get("norm_mode", "global"))
        self.win_samples = self.win_frames * cfg["feature"]["hop_length"]
        hop_frames = int(round(cfg["dataset"]["hop_sec"] * cfg["fps"]))
        # optional VTLN: per-window random mel-filterbank warp; 0 disables (alpha stays 1, front end unchanged)
        acfg = cfg.get("augment", {}) or {}
        self.vtl_range = float(acfg.get("vtl_alpha_range", 0.08)) \
            if (augment and acfg.get("enable", True) and acfg.get("vtl_enable", False)) else 0.0
        self.vtl_prob = float(acfg.get("vtl_prob", 1.0))

        self.index: List[Tuple[int, int]] = []
        self.weights: List[float] = []
        pos_os = cfg["dataset"]["pos_window_oversample"]
        hn_os = cfg["dataset"]["hardneg_window_oversample"]
        cw = (cfg.get("data", {}) or {}).get("corpus_weights", {}) or {}
        corpus_w = [float(cw.get(c.corpus, 1.0)) for c in self.p.clips]
        for ci in range(len(self.p)):
            T = self.p.n_frames(ci)
            starts = list(range(0, max(1, T - self.win_frames + 1), max(1, hop_frames)))
            if not starts:
                starts = [0]
            if starts[-1] != max(0, T - self.win_frames):
                starts.append(max(0, T - self.win_frames))
            for s in starts:
                self.index.append((ci, s))
                lab = self.p.labels[ci][s : s + self.win_frames]
                hn = self.p.hardnegs[ci][s : s + self.win_frames]
                ep = self.p.ep_labels[ci][s : s + self.win_frames]
                gs = self.p.gs_labels[ci][s : s + self.win_frames]
                w = corpus_w[ci]
                # EP/GS windows are as scarce as breath windows (EP ~0.05%, GS ~0.005% of frames) -> same oversampling
                if lab.sum() > 0 or ep.sum() > 0 or gs.sum() > 0:
                    w *= pos_os
                if hn.sum() > 0:
                    w *= hn_os
                self.weights.append(w)
        if cw:
            share: Dict[str, float] = {}
            for (ci, _), w in zip(self.index, self.weights):
                k = self.p.clips[ci].corpus
                share[k] = share.get(k, 0.0) + w
            s = sum(share.values()) or 1.0
            ranked = sorted(share.items(), key=lambda x: -x[1])
            parts = ", ".join(f"{k}={v / s:.1%}" for k, v in ranked)
            print("[dataset] sampling mixture: " + parts)
        gb = str((cfg.get("train", {}) or {}).get("group_balance", "none")).lower()
        if gb in ("corpus", "singer"):
            self._balance_groups(gb)

    def _balance_groups(self, by: str) -> None:
        """Rescale sampling weights so each group (corpus or singer) contributes evenly; "singer" mode preserves the per-corpus totals from ``data.corpus_weights``."""
        key = (lambda ci: self.p.clips[ci].corpus) if by == "corpus" \
            else (lambda ci: f"{self.p.clips[ci].corpus}/{self.p.clips[ci].singer}")
        groups: Dict[str, List[int]] = {}
        for i, (ci, _s) in enumerate(self.index):
            groups.setdefault(key(ci), []).append(i)

        targets: Dict[str, float] = {}
        if by == "corpus":
            for g in groups:
                targets[g] = 1.0 / max(len(groups), 1)
        else:
            # keep the tuned per-corpus totals, then split them evenly within each corpus
            corp_total: Dict[str, float] = {}
            for g, idxs in groups.items():
                corp = g.split("/", 1)[0]
                corp_total[corp] = corp_total.get(corp, 0.0) + sum(self.weights[i] for i in idxs)
            singers_per_corp: Dict[str, int] = {}
            for g in groups:
                corp = g.split("/", 1)[0]
                singers_per_corp[corp] = singers_per_corp.get(corp, 0) + 1
            for g in groups:
                corp = g.split("/", 1)[0]
                targets[g] = corp_total[corp] / max(singers_per_corp[corp], 1)

        for g, idxs in groups.items():
            total = sum(self.weights[i] for i in idxs)
            if total <= 0:
                continue
            f = targets[g] / total
            for i in idxs:
                self.weights[i] *= f

        share: Dict[str, float] = {}
        for (ci, _s), w in zip(self.index, self.weights):
            share[key(ci)] = share.get(key(ci), 0.0) + w
        tot = sum(share.values()) or 1.0
        ranked = sorted(share.items(), key=lambda x: -x[1])
        print(f"[dataset] group-balanced by {by} ({len(groups)} groups): "
              + ", ".join(f"{k}={v / tot:.1%}" for k, v in ranked[:12])
              + (" ..." if len(ranked) > 12 else ""))

    def __len__(self) -> int:
        """Number of available windows."""
        return len(self.index)

    def _norm(self, lm: np.ndarray, ci: int) -> np.ndarray:
        """Normalize a window using whole-clip CMVN statistics where applicable."""
        # always whole-clip statistics (same source as evaluation), never this window's own
        stats = self.p._cmvn_for(ci, self.norm_mode)
        return normalize_mel(lm, self.mean, self.std, self.norm_mode, stats)

    def __getitem__(self, i: int):
        """Return one training window ``(x, label, sp_label, mask, hardneg, ep_label, gs_label)``."""
        ci, s = self.index[i]
        hop = self.cfg["feature"]["hop_length"]
        start_sample = s * hop
        y = self.p.load_wave(ci)
        seg = np.asarray(y[start_sample : start_sample + self.win_samples], dtype=np.float32)
        if len(seg) < self.win_samples:
            seg = np.pad(seg, (0, self.win_samples - len(seg)))
        seg = seg.copy()

        label = self.p.labels[ci][s : s + self.win_frames].copy()
        sp_label = self.p.sp_labels[ci][s : s + self.win_frames].copy()
        ep_label = self.p.ep_labels[ci][s : s + self.win_frames].copy()
        gs_label = self.p.gs_labels[ci][s : s + self.win_frames].copy()
        mask = self.p.masks[ci][s : s + self.win_frames].copy()
        hn = self.p.hardnegs[ci][s : s + self.win_frames].copy()
        if len(label) < self.win_frames:
            pad = self.win_frames - len(label)
            label = np.pad(label, (0, pad))
            sp_label = np.pad(sp_label, (0, pad))
            ep_label = np.pad(ep_label, (0, pad))
            gs_label = np.pad(gs_label, (0, pad))
            mask = np.pad(mask, (0, pad))
            hn = np.pad(hn, (0, pad))

        v_mask = np.zeros(self.win_frames, dtype=bool) if f0_enabled(self.cfg) else None

        if self.augment:
            acfg = self.cfg["augment"]
            if acfg.get("enable", True):
                seg = apply_gain(seg, acfg["gain_db"])
                if np.random.rand() < acfg["noise_prob"]:
                    seg = add_noise(seg, acfg["noise_snr_db"])
                if np.random.rand() < acfg.get("noisefloor_prob", 0.0):
                    seg = add_noise_floor(seg, acfg.get("noisefloor_snr_db", [20.0, 40.0]))
                seg = compress(seg, acfg.get("compress_prob", 0.0))
                seg = harmony_overlay(seg, self.cfg)
                breath_overlay(seg, label, mask, self.p.breath_bank, self.cfg, hop,
                               voicing_mask=v_mask)
                # reverb last: the whole segment, overlaid breaths included, gets one room response
                if np.random.rand() < acfg.get("reverb_prob", 0.0):
                    seg = apply_reverb(seg, self.cfg)

        alpha = 1.0
        if self.vtl_range > 0.0 and np.random.rand() < self.vtl_prob:
            alpha = float(np.random.uniform(1.0 - self.vtl_range, 1.0 + self.vtl_range))
        lm = self.p.logmel(seg, alpha=alpha)[:, : self.win_frames]
        if lm.shape[1] < self.win_frames:
            missing_t = self.win_frames - lm.shape[1]
            lm = np.pad(lm, ((0, 0), (0, missing_t)), constant_values=float(lm.mean()))
        if self.augment and self.cfg["augment"].get("enable", True):
            lm = spec_augment(lm, self.cfg)
            lm = channel_jitter(lm, self.cfg["augment"].get("channel_jitter_db", 0.0))
        lm = self._norm(lm, ci)
        if v_mask is not None:
            # recompute F0 on the final waveform (overlays + reverb included); zero periodicity
            # on overlay frames so the features agree with the labels
            f0 = pitch_channels(seg, self.p.sr, hop)[:, : self.win_frames]
            if f0.shape[1] < self.win_frames:
                f0 = np.pad(f0, ((0, 0), (0, self.win_frames - f0.shape[1])))
            f0[:, v_mask] = 0.0
            lm = np.vstack([lm, f0])

        return (
            torch.from_numpy(lm.astype(np.float32)),
            torch.from_numpy(label.astype(np.float32)),
            torch.from_numpy(sp_label.astype(np.float32)),
            torch.from_numpy(mask.astype(np.float32)),
            torch.from_numpy(hn.astype(np.float32)),
            torch.from_numpy(ep_label.astype(np.float32)),
            torch.from_numpy(gs_label.astype(np.float32)),
        )


class ClipDataset(Dataset):
    """Evaluation dataset: one full clip per item (use ``batch_size=1``)."""

    def __init__(self, prepared: PreparedClips, cfg: Dict, mean: np.ndarray, std: np.ndarray):
        self.p = prepared
        self.cfg = cfg
        self.mean = mean.reshape(-1, 1).astype(np.float32)
        self.std = std.reshape(-1, 1).astype(np.float32)
        self.norm_mode = str(cfg["feature"].get("norm_mode", "global"))
        self.sr = cfg["audio"]["sample_rate"]
        self.hop = cfg["feature"]["hop_length"]

    def __len__(self) -> int:
        """Number of clips."""
        return len(self.p)

    def __getitem__(self, i: int):
        """Return one full clip ``(x, label, sp_label, mask, ep_label, gs_label, i)``."""
        mel = self.p.full_logmel(i)
        f0 = None
        if f0_enabled(self.cfg):
            y = np.asarray(self.p.load_wave(i), dtype=np.float32)
            f0 = pitch_channels(y, self.sr, self.hop)
            T = min(mel.shape[1], f0.shape[1])
            mel, f0 = mel[:, :T], f0[:, :T]
        stats = self.p._cmvn_for(i, self.norm_mode)
        mel = normalize_mel(mel, self.mean, self.std, self.norm_mode, stats)
        # only mel rows are normalized; F0 rows keep their fixed [0, 1] semantics
        if f0 is not None:
            x = np.vstack([mel, f0])
        else:
            x = mel
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(self.p.labels[i].astype(np.float32)),
            torch.from_numpy(self.p.sp_labels[i].astype(np.float32)),
            torch.from_numpy(self.p.masks[i].astype(np.float32)),
            torch.from_numpy(self.p.ep_labels[i].astype(np.float32)),
            torch.from_numpy(self.p.gs_labels[i].astype(np.float32)),
            i,
        )


# --- Statistics / split IO ---


def compute_train_norm(
    prepared: PreparedClips,
    with_clip_stats: bool = False,
) -> Union[NormStats, Tuple[np.ndarray, np.ndarray, List[NormStats]]]:
    """Accumulate per-mel-bin mean/std over a split (streaming); optionally collect per-clip CMVN stats in the same pass."""
    s = None
    ss = None
    n = 0
    clip_stats: List[NormStats] = []
    for ci in range(len(prepared)):
        x = prepared.full_logmel(ci)
        if s is None:
            s = np.zeros(x.shape[0], dtype=np.float64)
            ss = np.zeros(x.shape[0], dtype=np.float64)
        s += x.sum(axis=1)
        ss += (x.astype(np.float64) ** 2).sum(axis=1)
        n += x.shape[1]
        if with_clip_stats:
            clip_stats.append(clip_cmvn_of_logmel(x))
    mean = (s / n).astype(np.float32)
    var = (ss / n - (s / n) ** 2).clip(min=1e-8)
    std = np.sqrt(var).astype(np.float32)
    if with_clip_stats:
        return mean, std, clip_stats
    return mean, std


def read_split(splits_dir: Union[str, Path], name: str) -> List[str]:
    """Read a split file written by ``make_splits`` (one clip name per non-empty line)."""
    p = Path(splits_dir) / f"{name}.txt"
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
