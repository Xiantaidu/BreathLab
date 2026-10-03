# -*- coding: utf-8 -*-
"""Copy the test split to C:/BreathLab_data/test_cut, cutting clips over 30 s at the quietest point near each boundary; every chunk gets a chunk-local annotation file so evaluators never re-parse the corpora.

Run: python tools/analysis/prepare_test_cut.py
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from breathlab.common import load_config
from breathlab.data.parse_labels import load_all_clips

ROOT = Path("D:/BreathLab")
OUT = Path("C:/BreathLab_data/test_cut")
WAV_DIR = OUT / "wavs"
ANN_DIR = OUT / "annotations"
MAX_SEC = 30.0


def silence_cutpoints(wav: np.ndarray, sr: int, dur: float, max_sec: float) -> list[float]:
    """Cut a long clip at the quietest frame near each ``max_sec`` boundary."""
    hop = max(int(0.05 * sr), 1)
    n = len(wav) // hop
    if n < 2:
        return []
    energy = np.sqrt(np.convolve(wav[:n * hop] ** 2, np.ones(hop) / hop, mode="same"))[::hop]
    cuts, pos = [], 0.0
    while dur - pos > max_sec:
        target = pos + max_sec
        # search window must END at the target, else the cut lands beyond max_sec and the
        # chunk stays over-length (measured: a symmetric +-2 s window -> 32 s chunks)
        lo = max(int((target - 4.0) / 0.05), int((pos + 1.0) / 0.05), 0)
        hi = min(int(target / 0.05), n - 1)
        if hi <= lo:
            cut = target
        else:
            k = lo + int(np.argmin(energy[lo:hi + 1]))
            cut = k * 0.05
            if cut <= pos + 1.0 or cut >= dur - 1.0:
                cut = target
        cuts.append(min(cut, pos + max_sec))       # never exceed the hard limit
        pos = cuts[-1]
    return cuts


def clip_events(events, a: float, b: float) -> list:
    """Clip annotation events to [a, b] and shift into chunk-local time."""
    out = []
    for ev in events:
        s, e = max(ev.start, a), min(ev.end, b)
        if e - s > 1e-4:
            out.append({"start_sec": round(s - a, 4), "end_sec": round(e - a, 4)})
    return out


def main() -> None:
    for d in (WAV_DIR, ANN_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    cfg = load_config("breathlab/configs/default.yaml")
    names = [ln.strip() for ln in (ROOT / "splits" / "test.txt").read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    clips = {c.name: c for c in load_all_clips(cfg)}
    test = [clips[n] for n in names if n in clips]
    print(f"test clips: {len(test)}")

    rows = []
    n_whole = n_cut = 0
    for c in test:
        dur = sf.info(str(c.wav_path)).duration
        stem = c.name.replace("/", "__")
        if dur <= MAX_SEC:
            cid = stem
            shutil.copy2(c.wav_path, WAV_DIR / f"{cid}.wav")
            events = {"breath_events": clip_events(c.breath_events, 0, dur),
                      "sp_events": clip_events(c.sp_events, 0, dur),
                      "source_name": c.name, "corpus": c.corpus,
                      "chunk_start_sec": 0.0, "chunk_end_sec": dur}
            (ANN_DIR / f"{cid}.json").write_text(json.dumps(events, ensure_ascii=False),
                                                 encoding="utf-8")
            rows.append([cid, c.name, c.corpus, 0.0, round(dur, 3), round(dur, 3),
                         f"{cid}.wav"])
            n_whole += 1
            continue
        wav, sr = sf.read(str(c.wav_path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        bounds = [0.0] + silence_cutpoints(wav, sr, dur, MAX_SEC) + [dur]
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1]
            if b - a < 0.5:
                continue
            cid = f"{stem}__part{k:02d}"
            seg = np.clip(wav[int(a * sr):int(b * sr)], -1.0, 1.0)
            sf.write(str(WAV_DIR / f"{cid}.wav"), (seg * 32767).astype("int16"),
                     sr, subtype="PCM_16")
            events = {"breath_events": clip_events(c.breath_events, a, b),
                      "sp_events": clip_events(c.sp_events, a, b),
                      "source_name": c.name, "corpus": c.corpus,
                      "chunk_start_sec": round(a, 4), "chunk_end_sec": round(b, 4)}
            (ANN_DIR / f"{cid}.json").write_text(json.dumps(events, ensure_ascii=False),
                                                 encoding="utf-8")
            rows.append([cid, c.name, c.corpus, round(a, 3), round(b, 3),
                         round(b - a, 3), f"{cid}.wav"])
        n_cut += 1

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "source_name", "corpus", "chunk_start_sec",
                    "chunk_end_sec", "dur_sec", "wav_file"])
        w.writerows(rows)
    durs = [r[5] for r in rows]
    print(f"whole: {n_whole}, cut: {n_cut} -> {len(rows)} chunks | "
          f"max dur {max(durs):.1f}s, over-30s: {sum(1 for d in durs if d > 30.0)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
