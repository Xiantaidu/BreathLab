"""Feature extraction: wav -> 16 kHz mono -> log-mel at ``sr / hop`` fps (deliberately without librosa, which hangs on import in this environment)."""
from __future__ import annotations

import hashlib
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly

PathLike = Union[str, Path]


def wav_cache_path(path: PathLike, sr: int, cache_dir: PathLike) -> Path:
    """Path of the 16 kHz ``.npy`` cache; the name hashes the source path so same-named wavs from different corpora cannot collide."""
    h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / "wav16k" / f"{h}_{sr}.npy"


def load_wav_16k_cached(
    path: PathLike,
    sr: int = 16000,
    cache_dir: PathLike | None = None,
) -> Union[np.ndarray, Path]:
    """Read and resample to 16 kHz mono float32; with ``cache_dir``, persist the .npy atomically and return its Path (callers stream it via memmap)."""
    if cache_dir is None:
        return load_wav_16k(path, sr)
    cp = wav_cache_path(path, sr, cache_dir)
    if cp.exists():
        return cp
    y = load_wav_16k(path, sr)
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(".tmp.npy")
    np.save(tmp, y)
    tmp.replace(cp)
    return cp


def load_wav_16k(path: PathLike, sr: int = 16000) -> np.ndarray:
    """Read a wav and resample it to mono float32 at ``sr`` (rational ``resample_poly``, no libsoxr)."""
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if int(file_sr) != int(sr):
        g = gcd(int(file_sr), int(sr))
        y = resample_poly(y, sr // g, file_sr // g)
    return np.ascontiguousarray(y, dtype=np.float32)


# VTLN warps the mel filterbank along frequency (vocal-tract length shifts the spectral scale); alpha == 1 is bit-identical to the stock filterbank and is the only value used at inference / ONNX export.

_SLANEY_F_SP = 200.0 / 3
_SLANEY_MIN_LOG_HZ = 1000.0
_SLANEY_LOGSTEP = 0.06875177786290215      # log(6.4) / 27


def hz_to_mel_slaney(f: np.ndarray) -> np.ndarray:
    """Slaney mel scale, matching ``torchaudio``'s ``mel_scale="slaney"`` convention."""
    f = np.asarray(f, dtype=np.float64)
    mel = f / _SLANEY_F_SP
    log_part = (_SLANEY_MIN_LOG_HZ / _SLANEY_F_SP
                + np.log(np.maximum(f, 1.0) / _SLANEY_MIN_LOG_HZ) / _SLANEY_LOGSTEP)
    return np.where(f >= _SLANEY_MIN_LOG_HZ, log_part, mel)


def mel_to_hz_slaney(mel: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hz_to_mel_slaney`."""
    mel = np.asarray(mel, dtype=np.float64)
    f = mel * _SLANEY_F_SP
    min_log_mel = _SLANEY_MIN_LOG_HZ / _SLANEY_F_SP
    log_part = _SLANEY_MIN_LOG_HZ * np.exp(_SLANEY_LOGSTEP * (mel - min_log_mel))
    return np.where(mel >= min_log_mel, log_part, f)


def vtln_warp_hz(f: np.ndarray, alpha: float, f_hi: float, nyquist: float) -> np.ndarray:
    """Kaldi-style smooth VTLN warp: scale tapers from ``alpha`` at DC to 1 at ``f_hi`` (identity at the top of the band)."""
    f = np.asarray(f, dtype=np.float64)
    if abs(alpha - 1.0) < 1e-9:
        return f
    taper = np.clip(1.0 - f / max(f_hi, 1e-9), 0.0, 1.0)
    return np.clip(f * (1.0 + (alpha - 1.0) * taper), 0.0, nyquist)


def triangular_fb(f_pts: np.ndarray, all_freqs: np.ndarray) -> np.ndarray:
    """Slaney-normalised triangular filterbank ``[n_mels, n_bins]`` from band edges (mirrors torchaudio, so warped banks can be built)."""
    n_mels = len(f_pts) - 2
    fb = np.zeros((len(all_freqs), n_mels), dtype=np.float32)
    for i in range(n_mels):
        left, centre, right = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        down = (all_freqs - left) / max(centre - left, 1e-12)
        up = (right - all_freqs) / max(right - centre, 1e-12)
        fb[:, i] = np.clip(np.minimum(down, up), 0.0, None)
    enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
    return (fb * enorm.astype(np.float32)).T


class LogMel:
    """Log-mel front end (center=True with reflect padding; frame count ``1 + n_samples // hop_length``)."""

    def __init__(self, cfg: Dict):
        a = cfg["audio"]
        f = cfg["feature"]
        self.sr = a["sample_rate"]
        self.hop_length = f["hop_length"]
        self.log_offset = f["log_offset"]
        self.n_mels = f["n_mels"]
        self.fmin = f["fmin"]
        self.fmax = f["fmax"]
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=f["n_fft"],
            win_length=f["win_length"],
            hop_length=f["hop_length"],
            f_min=f["fmin"],
            f_max=f["fmax"],
            n_mels=f["n_mels"],
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )
        # uniform mel spacing is what lets a frequency warp be expressed as a row shift
        m_min = float(hz_to_mel_slaney(np.array([self.fmin]))[0])
        m_max = float(hz_to_mel_slaney(np.array([self.fmax]))[0])
        self._m_min, self._m_max = m_min, m_max
        self._m_spacing = (m_max - m_min) / (self.n_mels + 1)
        self._f_pts = mel_to_hz_slaney(np.linspace(m_min, m_max, self.n_mels + 2))
        self._all_freqs = np.linspace(0, self.sr / 2.0, 1 + f["n_fft"] // 2)
        self._fb_cache: Dict[float, torch.Tensor] = {}

    def _stock_fb(self) -> torch.Tensor:
        """The un-warped filterbank of the underlying torchaudio transform (location varies across torchaudio versions)."""
        ms = getattr(self._mel, "mel_scale", None)
        fb = ms.fb if (ms is not None and hasattr(ms, "fb")) else self._mel.fb
        # torchaudio stores it as [n_bins, n_mels]; this module works in [n_mels, n_bins].
        return fb.transpose(0, 1) if fb.shape[0] != self.n_mels else fb

    def fb(self, alpha: float = 1.0) -> torch.Tensor:
        """Mel filterbank ``[n_mels, n_bins]``, optionally VTLN-warped by ``alpha``."""
        key = round(float(alpha), 4)
        hit = self._fb_cache.get(key)
        if hit is not None:
            return hit
        if abs(alpha - 1.0) < 1e-9:
            # identity: hand back torchaudio's own buffer so the default path is bit-exact
            out = self._stock_fb()
        else:
            nyq = self.sr / 2.0
            f_hi = min(self.fmax, 0.95 * nyq)
            warped = vtln_warp_hz(self._f_pts, alpha, f_hi, nyq)
            warped = np.maximum.accumulate(warped + 1e-9) - 1e-9   # keep it strictly monotone
            out = torch.from_numpy(triangular_fb(warped, self._all_freqs))
        self._fb_cache[key] = out
        return out

    def __call__(self, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Compute the log-mel spectrogram ``[n_mels, T]`` of a waveform (``alpha`` = VTLN warp; training-only augmentation, 1.0 at eval / ONNX)."""
        x = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0)
        spec = self._mel.spectrogram(x).squeeze(0)
        mel = torch.matmul(self.fb(alpha), spec)
        return torch.log(mel + self.log_offset).numpy().astype(np.float32)

    def n_frames(self, n_samples: int) -> int:
        """Return the number of frames produced by ``n_samples`` samples."""
        return 1 + n_samples // self.hop_length


def frame_rms_energy(y: np.ndarray, hop_length: int, frame_length: int) -> np.ndarray:
    """Per-frame RMS energy in dB, reflect-padded and frame-aligned with the log-mel front end."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))
    pad = frame_length // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + len(y) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = yp[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return (20.0 * np.log10(rms + 1e-8)).astype(np.float32)


def pitch_channels(
    y: np.ndarray,
    sr: int = 16000,
    hop: int = 160,
    frame: int = 400,
    fmin: float = 70.0,
    fmax: float = 1000.0,
    per_thresh: float = 0.30,
) -> np.ndarray:
    """Dependency-free FFT-autocorrelation pitch tracker; returns ``[2, T]`` (ch0 periodicity in [0, 1], ch1 log-F0 mapped to [0, 1]; unvoiced frames zeroed)."""
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
    per = np.convolve(per, np.ones(3) / 3.0, mode="same")          # light smoothing
    f0 = (sr / lag).astype(np.float32)
    f0[per < per_thresh] = 0.0
    lf = np.zeros_like(per)
    voiced = f0 > 0
    lf[voiced] = np.clip(
        (np.log2(f0[voiced]) - np.log2(fmin)) / (np.log2(1050.0) - np.log2(fmin)), 0.0, 1.0
    )
    return np.vstack([per, lf]).astype(np.float32)


def compute_norm_stats(feats: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate per-mel-bin mean/std over many clips (streaming in float64, all frames pooled)."""
    n = 0
    s = None
    ss = None
    for x in feats:
        if s is None:
            s = np.zeros(x.shape[0], dtype=np.float64)
            ss = np.zeros(x.shape[0], dtype=np.float64)
        s += x.sum(axis=1)
        ss += (x.astype(np.float64) ** 2).sum(axis=1)
        n += x.shape[1]
    mean = (s / n).astype(np.float32)
    var = (ss / n - (s / n) ** 2).clip(min=1e-8)
    std = np.sqrt(var).astype(np.float32)
    return mean, std
