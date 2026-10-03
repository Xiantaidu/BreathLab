"""Duration-preserving training-time augmentation; the labelled source signal is never time-stretched (only derived interference tracks may be)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.signal import fftconvolve, lfilter


def apply_gain(y: np.ndarray, gain_db_range: Sequence[float]) -> np.ndarray:
    """Apply a random gain drawn uniformly (in dB) from a range."""
    g = np.random.uniform(gain_db_range[0], gain_db_range[1])
    return (y * (10.0 ** (g / 20.0))).astype(np.float32)


def add_noise(y: np.ndarray, snr_db_range: Sequence[float]) -> np.ndarray:
    """Add white Gaussian noise at a random SNR (mid-to-high noise levels)."""
    sig_power = float(np.mean(y ** 2)) + 1e-12
    snr = np.random.uniform(snr_db_range[0], snr_db_range[1])
    noise_power = sig_power / (10.0 ** (snr / 10.0))
    noise = np.random.randn(len(y)).astype(np.float32) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)


def _crossfade_add(dst: np.ndarray, src: np.ndarray, pos: int, cf: int) -> None:
    """Add ``src`` into ``dst`` starting at ``pos`` with linear crossfades at both ends (in place)."""
    n = len(src)
    end = min(pos + n, len(dst))
    if end <= pos:
        return
    seg = src[: end - pos].copy()
    m = len(seg)
    if cf > 0 and m > 2 * cf:
        ramp = np.linspace(0.0, 1.0, cf, dtype=np.float32)
        seg[:cf] *= ramp
        seg[-cf:] *= ramp[::-1]
    dst[pos:end] += seg


def breath_overlay(
    y: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
    breath_bank: List[np.ndarray],
    cfg: Dict,
    hop_length: int,
    voicing_mask: Optional[np.ndarray] = None,
) -> None:
    """Overlay breath segments onto the waveform and mark the covered frames as positive (all in place)."""
    if not breath_bank:
        return
    aug = cfg["augment"]
    if np.random.rand() >= aug["breath_overlay_prob"]:
        return
    n_events = np.random.randint(1, aug["breath_overlay_max_events"] + 1)
    cf = int(aug["breath_overlay_crossfade_ms"] * 16000 / 1000)  # ms -> samples @16k
    T = len(label)
    for _ in range(n_events):
        b = breath_bank[np.random.randint(len(breath_bank))]
        if len(b) >= len(y):
            continue
        g = np.random.uniform(aug["breath_overlay_gain_db"][0], aug["breath_overlay_gain_db"][1])
        src = (b * (10.0 ** (g / 20.0))).astype(np.float32)
        pos = np.random.randint(0, len(y) - len(src))
        _crossfade_add(y, src, pos, cf)
        f0 = pos // hop_length
        f1 = min(T, (pos + len(src)) // hop_length + 1)
        label[f0:f1] = 1.0
        mask[f0:f1] = 1.0
        if voicing_mask is not None:
            zp = float(aug.get("breath_overlay_zero_f0_prob", 0.5))
            if np.random.rand() < zp:
                voicing_mask[f0:f1] = True


def channel_jitter(lm: np.ndarray, max_db: float) -> np.ndarray:
    """Random per-mel-bin gain offsets (a crude static recording-chain response model)."""
    if max_db <= 0:
        return lm
    off = np.random.uniform(-max_db, max_db, size=(lm.shape[0], 1)).astype(np.float32)
    return (lm + off).astype(np.float32)


def compress(y: np.ndarray, prob: float) -> np.ndarray:
    """Lightweight tanh dynamic compression (normalized for unit slope at small amplitudes)."""
    if prob <= 0 or np.random.rand() >= prob:
        return y
    k = np.random.uniform(1.0, 3.0)
    return (np.tanh(k * y) / np.tanh(k)).astype(np.float32)


def _pink_noise(n: int) -> np.ndarray:
    """Generate 1/f (pink) noise by spectral shaping of white noise."""
    w = np.random.randn(n).astype(np.float64)
    f = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, d=1.0 / 16000.0)
    shape = 1.0 / np.sqrt(np.maximum(freqs, 1.0))     # -3 dB/oct
    return np.fft.irfft(f * shape, n).astype(np.float32)


def add_noise_floor(
    y: np.ndarray,
    snr_db_range: Sequence[float],
    sr: int = 16000,
    pink_prob: float = 0.5,
) -> np.ndarray:
    """Add a constant low-level noise floor (white or pink hiss)."""
    lo, hi = snr_db_range
    if hi <= 0 or np.random.rand() > 0.999:  # placeholder; the caller controls the probability
        return y
    sig_power = float(np.mean(y ** 2)) + 1e-12
    snr = np.random.uniform(lo, hi)
    noise_power = sig_power / (10.0 ** (snr / 10.0))
    if np.random.rand() < pink_prob:
        n = _pink_noise(len(y))
        n *= np.sqrt(noise_power / (float(np.mean(n ** 2)) + 1e-12))
    else:
        n = np.random.randn(len(y)).astype(np.float32) * np.sqrt(noise_power)
    return (y + n).astype(np.float32)


def _synth_ir(sr: int, rt60: float, predelay_ms: float) -> np.ndarray:
    """Synthesize a room impulse response: exponentially decaying noise + one-pole low-pass."""
    n = max(8, int(rt60 * sr))
    t = np.arange(n) / sr
    decay = 10.0 ** (-3.0 * t / max(rt60, 1e-3))      # -60 dB at t = rt60
    ir = np.random.randn(n) * decay
    alpha = np.random.uniform(0.15, 0.45)             # stronger low-pass = duller room
    ir = lfilter([alpha], [1.0, -(1.0 - alpha)], ir)
    pre = int(predelay_ms * sr / 1000.0)
    if pre > 0:
        ir = np.concatenate([np.zeros(pre, dtype=ir.dtype), ir])
    ir[0] += 1.0                                      # direct sound
    e = float(np.sqrt(np.mean(ir ** 2)) + 1e-12)
    return (ir / e).astype(np.float32)


def apply_reverb(y: np.ndarray, cfg: Dict, sr: int = 16000) -> np.ndarray:
    """Convolve with a synthetic IR and mix a wet path; run last in the waveform chain so overlaid breaths get the same room response."""
    a = cfg["augment"]
    rt60 = np.random.uniform(a["reverb_rt60_s"][0], a["reverb_rt60_s"][1])
    pre = np.random.uniform(a["reverb_predelay_ms"][0], a["reverb_predelay_ms"][1])
    wet_db = np.random.uniform(a["reverb_wet_gain_db"][0], a["reverb_wet_gain_db"][1])
    ir = _synth_ir(sr, rt60, pre)
    wet = fftconvolve(y, ir)[: len(y)].astype(np.float32)
    wet *= 10.0 ** (wet_db / 20.0)
    return (y + wet).astype(np.float32)


def _phase_vocoder_stretch(x: np.ndarray, rate: float, n_fft: int = 512) -> np.ndarray:
    """Time-stretch with an STFT phase vocoder (``rate > 1`` lengthens; pitch is preserved) -- derived tracks only."""
    hop = n_fft // 4
    w = np.hanning(n_fft).astype(np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + max(0, len(x) - n_fft) // hop
    Hs = hop * rate
    bins = n_fft // 2 + 1
    nominal = 2 * np.pi * np.arange(bins) / n_fft
    last_s = int(round((n_frames - 1) * Hs))
    out_len = last_s + n_fft
    out = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    phase_acc = np.zeros(bins, dtype=np.float64)
    prev_ph = None
    for i in range(n_frames):
        seg = x[i * hop : i * hop + n_fft]
        spec = np.fft.rfft(seg * w)
        mag, ph = np.abs(spec), np.angle(spec)
        if prev_ph is None:
            omega = nominal
        else:
            d = ph - prev_ph - nominal * hop
            d -= 2 * np.pi * np.round(d / (2 * np.pi))  # principal argument
            omega = nominal + d / hop
        prev_ph = ph
        phase_acc += omega * Hs                         # advance by synthesis hop -> pitch unchanged
        out_spec = mag * np.exp(1j * phase_acc)
        frame = np.fft.irfft(out_spec, n_fft) * w
        s = int(round(i * Hs))
        out[s : s + n_fft] += frame
        wsum[s : s + n_fft] += w
    return (out / np.maximum(wsum, 1e-6)).astype(np.float32)


def pitch_shift_fixed(y: np.ndarray, semitones: float, sr: int = 16000) -> np.ndarray:
    """Pitch-shift at fixed duration (resample, then phase-vocoder stretch back); for derived interference tracks only."""
    r = 2.0 ** (semitones / 12.0)
    n_new = max(2, int(round(len(y) / r)))
    y2 = np.interp(np.linspace(0.0, len(y) - 1, n_new), np.arange(len(y)), y).astype(np.float32)
    y3 = _phase_vocoder_stretch(y2, r)
    if len(y3) >= len(y):
        return y3[: len(y)]
    return np.pad(y3, (0, len(y) - len(y3)))


def harmony_overlay(y: np.ndarray, cfg: Dict, sr: int = 16000) -> np.ndarray:
    """Overlay a quiet pitch-shifted copy of the segment with a slow amplitude wobble (simulated harmony bed)."""
    a = cfg["augment"]
    if np.random.rand() >= a.get("harmony_overlay_prob", 0.0):
        return y
    dn = a.get("harmony_overlay_semitones", [-5.0, -2.0])
    up = a.get("harmony_overlay_semitones_up", [2.0, 5.0])
    if np.random.rand() < 0.5:
        st = np.random.uniform(dn[0], dn[1])
    else:
        st = np.random.uniform(up[0], up[1])
    gain_db = a.get("harmony_overlay_gain_db", [-18.0, -8.0])
    g = np.random.uniform(gain_db[0], gain_db[1])
    h = pitch_shift_fixed(y, float(st), sr) * (10.0 ** (g / 20.0))
    t = np.arange(len(h), dtype=np.float64) / sr
    wob_hz = np.random.uniform(0.5, 2.0)
    wob_phase = np.random.rand() * 2 * np.pi
    wob = 0.8 + 0.2 * np.sin(2 * np.pi * wob_hz * t + wob_phase)
    return (y + h * wob.astype(np.float32)).astype(np.float32)


def spec_augment(logmel: np.ndarray, cfg: Dict) -> np.ndarray:
    """Time/frequency masking on a log-mel; masked regions are filled with the spectrogram mean (zeros would be an out-of-distribution tell)."""
    sa = cfg["augment"]["specaug"]
    if not sa.get("enable", True):
        return logmel
    x = logmel.copy()
    n_mels, T = x.shape
    fill = float(x.mean())
    for _ in range(sa["n_time_masks"]):
        t = np.random.randint(0, sa["time_mask_param"] + 1)
        if t > 0 and T - t > 0:
            t0 = np.random.randint(0, T - t)
            x[:, t0 : t0 + t] = fill
    for _ in range(sa["n_freq_masks"]):
        fq = np.random.randint(0, sa["freq_mask_param"] + 1)
        if fq > 0 and n_mels - fq > 0:
            f0 = np.random.randint(0, n_mels - fq)
            x[f0 : f0 + fq, :] = fill
    return x
