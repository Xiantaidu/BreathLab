"""Add AP (breath) labels to already-annotated TextGrids.

``--mode tier`` (default) appends a standalone ``AP`` IntervalTier, leaving every
existing tier untouched (``--only_in_sp`` restricts detections to SP/blank regions).
``--mode phones`` is OpenVPI / SOFA compatible: silence words are split into
``SP ... AP ... SP`` runs written into both the ``words`` and ``phones`` tiers, all
other tiers and phones preserved. Both modes are idempotent: re-running replaces the
existing AP tier / injected AP intervals instead of duplicating them.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from breathlab.infer import load_runner, run_one

# Labels treated as "silence" when looking for regions that may host an AP.
_SILENCE_LABELS = ("SP", "SIL", "PAU", "")
_WORD_TIER_NAMES = ("words", "word")
_PHONE_TIER_NAMES = ("phones", "phone")


def _collect_tgs(paths: Iterable[str | Path]) -> list[Path]:
    """Expand TextGrid files/directories into a list of existing paths."""
    out: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            # Two glob patterns for case-sensitive filesystems; Windows glob is
            # case-insensitive, hence the dedupe.
            found = sorted({*pp.glob("*.TextGrid"), *pp.glob("*.Textgrid")})
            out += found
        elif pp.is_file():
            out.append(pp)
        else:
            print(f"[tg_add_ap] skipping nonexistent path: {p}")
    return out


def _tier_intervals(tier: Any) -> list[tuple[float, float, str]]:
    """Read tier intervals in a way that works across old and new praatio versions."""
    items = getattr(tier, "entries", None)
    if items is None:
        items = tier.intervals
    out: list[tuple[float, float, str]] = []
    for it in items:
        s, e = getattr(it, "start", None), getattr(it, "end", None)
        lab = getattr(it, "label", None)
        if s is None and isinstance(it, (tuple, list)):
            s, e, lab = it[0], it[1], it[2]
        out.append((float(s), float(e), "" if lab is None else str(lab)))
    return out


def _is_silence(label: str) -> bool:
    """Return True when a tier label denotes silence / a blank interval."""
    return label.strip().upper() in _SILENCE_LABELS


def _is_splittable(label: str) -> bool:
    """Return True when a label may host an injected AP (AP included, so re-running
    ``--mode phones`` re-splits instead of duplicating)."""
    return _is_silence(label) or label.strip().upper() == "AP"


def _find_tier(tg: Any, names: Iterable[str]) -> Any | None:
    """Return the first tier whose name matches (case-insensitive), else None."""
    wanted = {n.lower() for n in names}
    for tier in tg.tiers:
        if str(tier.name).lower() in wanted:
            return tier
    return None


def _is_interval_tier(tier: Any) -> bool:
    """Return True for interval tiers (as opposed to point tiers)."""
    return hasattr(tier, "entries") or hasattr(tier, "intervals")


def find_word_phone_tiers(tg: Any) -> tuple[Any | None, Any | None]:
    """Locate the ``words`` and ``phones`` tiers; fall back to the positional convention
    (first two interval tiers) when the names are unusual."""
    words = _find_tier(tg, _WORD_TIER_NAMES)
    phones = _find_tier(tg, _PHONE_TIER_NAMES)
    if words is None or phones is None:
        ivals = [t for t in tg.tiers if _is_interval_tier(t)]
        if len(ivals) == 2:
            words = words or ivals[0]
            phones = phones or ivals[1]
    return words, phones


def _sp_regions(tg: Any) -> list[tuple[float, float]]:
    """Collect the SP / blank intervals of the words and phones tiers (the regions
    the conservative ``--only_in_sp`` mode considers safe)."""
    regions: list[tuple[float, float]] = []
    for tier in tg.tiers:
        if str(tier.name).lower() in _WORD_TIER_NAMES + _PHONE_TIER_NAMES:
            for s, e, lab in _tier_intervals(tier):
                if _is_silence(lab):
                    regions.append((s, e))
    return regions


def _overlap_frac(ev: tuple[float, float], regions: Iterable[tuple[float, float]]) -> float:
    """Compute the largest fraction of an event covered by any of ``regions``."""
    s, e = ev
    best = 0.0
    for rs, re_ in regions:
        ov = min(e, re_) - max(s, rs)
        if ov > 0:
            best = max(best, ov / max(e - s, 1e-9))
    return best


def split_silence_interval(
    sp_start: float,
    sp_end: float,
    ap_events: Sequence[tuple[float, float]],
    sp_dur: float,
) -> list[tuple[float, float, str]]:
    """Split one silence interval into SP / AP runs around the detected breaths,
    absorbing SP fragments shorter than ``sp_dur`` into the neighbouring AP."""
    inside: list[tuple[float, float]] = []
    for s, e in ap_events:
        s2, e2 = max(s, sp_start), min(e, sp_end)
        if e2 - s2 > 1e-6:
            inside.append((s2, e2))
    inside.sort()

    chunks: list[tuple[float, float, str]] = []
    cursor = sp_start
    for s, e in inside:
        if s - cursor >= sp_dur:
            chunks.append((cursor, s, "SP"))
            cursor = s
        elif chunks and chunks[-1][2] == "AP":
            pass                            # too short -> absorbed by the previous AP
        # else: too short and no previous AP -> AP simply starts here
        chunks.append((cursor, e, "AP"))
        cursor = e
    if sp_end - cursor > 1e-6:
        if sp_end - cursor >= sp_dur:
            chunks.append((cursor, sp_end, "SP"))
        elif chunks and chunks[-1][2] == "AP":
            ps, _pe, lab = chunks[-1]
            chunks[-1] = (ps, sp_end, lab)  # too short -> extend the last AP
        else:
            chunks.append((cursor, sp_end, "SP"))
    return chunks or [(sp_start, sp_end, "SP")]


def _split_regions(
    regions: Sequence[tuple[float, float]],
    ap_events: Sequence[tuple[float, float]],
    sp_dur: float,
    ap_dur: float,
) -> tuple[list[tuple[float, float, str]], int]:
    """Split every silence region into SP/AP chunks, returning ``(chunks, n_ap)``."""
    strong = [(s, e) for s, e in ap_events if (e - s) >= ap_dur]
    chunks: list[tuple[float, float, str]] = []
    for rs, re_ in regions:
        chunks += [(s, e, lab) for s, e, lab in split_silence_interval(rs, re_, strong, sp_dur)
                   if e - s > 1e-6]
    n_ap = sum(1 for _s, _e, lab in chunks if lab == "AP")
    return chunks, n_ap


def _rebuild_pair(
    words: Sequence[tuple[float, float, str]],
    phones: Sequence[tuple[float, float, str]],
    ap_events: Sequence[tuple[float, float]],
    sp_dur: float,
    ap_dur: float,
) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]], int]:
    """Rewrite the words + phones pair: silence words are split into ``SP ... AP ... SP``
    runs written to both tiers; other words keep their phones (gaps filled with blanks
    so both tiers stay contiguous). Returns ``(new_words, new_phones, n_ap)``."""
    strong = [(s, e) for s, e in ap_events if (e - s) >= ap_dur]
    new_words: list[tuple[float, float, str]] = []
    new_phones: list[tuple[float, float, str]] = []
    n_ap = 0

    for ws, we, wlab in sorted(words, key=lambda x: x[0]):
        if _is_splittable(wlab):
            chunks = [(s, e, lab) for s, e, lab
                      in split_silence_interval(ws, we, strong, sp_dur) if e - s > 1e-6]
            new_words += chunks
            new_phones += chunks
            n_ap += sum(1 for _s, _e, lab in chunks if lab == "AP")
            continue
        new_words.append((ws, we, wlab))
        # The word's own phones, plus blank fillers over gaps, keep the phones tier
        # contiguous inside the word.
        inner = [p for p in phones if p[0] >= ws - 1e-6 and p[1] <= we + 1e-6]
        inner.sort(key=lambda x: x[0])
        cursor = ws
        for ps, pe, plab in inner:
            if pe <= cursor + 1e-6:
                continue
            ps = max(ps, cursor)
            if ps > cursor + 1e-6:
                new_phones.append((cursor, ps, ""))
            new_phones.append((ps, pe, plab))
            cursor = pe
        if we > cursor + 1e-6:
            new_phones.append((cursor, we, ""))
    return new_words, new_phones, n_ap


def add_ap_to_tg(
    tg_path: str | Path,
    ap_events: Sequence[tuple[float, float]],
    out_path: str | Path,
    only_in_sp: bool = False,
    mode: str = "tier",
    sp_dur: float = 0.1,
    ap_dur: float = 0.08,
) -> dict:
    """Write breath detections into a copy of an existing TextGrid (``tier`` appends a
    standalone AP tier, ``phones`` injects into the words/phones tiers); idempotent,
    returns a report dict. Raises ValueError on unknown mode or missing words/phones."""
    from praatio import textgrid as tglib

    if mode not in ("tier", "phones"):
        raise ValueError(f"unknown mode: {mode!r} (expected 'tier' or 'phones')")

    kw = {"includeEmptyIntervals": True} if "includeEmptyIntervals" in inspect.signature(
        tglib.openTextgrid).parameters else {"includeBlankSpaces": True}
    tg = tglib.openTextgrid(str(tg_path), **kw)
    dur = float(tg.maxTimestamp)
    try:
        fmt = tglib.TextgridOutputFormat.SHORT_TEXTGRID
    except AttributeError:
        fmt = "short_textgrid"

    new_tg = tglib.Textgrid()
    new_tg.minTimestamp = tg.minTimestamp
    new_tg.maxTimestamp = dur

    if mode == "tier":
        regions = _sp_regions(tg) if only_in_sp else None
        kept = [ev for ev in ap_events
                if (regions is None or _overlap_frac(ev, regions) >= 0.5)]
        # An existing AP tier is replaced (idempotent); all other tiers are kept as-is.
        for tier in tg.tiers:
            if str(tier.name).lower() != "ap":
                new_tg.addTier(tier)
        entries = [(s, min(e, dur), "AP") for s, e in kept if min(e, dur) > s + 1e-6]
        ap_entries: list[tuple[float, float, str]] = []
        cur = 0.0
        for s, e, lab in sorted(entries):
            if s > cur + 1e-6:
                ap_entries.append((cur, s, ""))
            ap_entries.append((s, e, lab))
            cur = e
        if dur > cur + 1e-6:
            ap_entries.append((cur, dur, ""))
        new_tg.addTier(tglib.IntervalTier("AP", ap_entries, 0.0, dur))
        report = {"mode": "tier", "only_in_sp": only_in_sp, "written": kept,
                  "n_ap": len(kept), "n_detected": len(ap_events)}
    else:
        words_t, phones_t = find_word_phone_tiers(tg)
        if words_t is None or phones_t is None:
            raise ValueError(
                f"{Path(tg_path).name}: no words/phones tiers found - "
                "'phones' mode needs a two-tier TextGrid (use --mode tier instead)")
        words = _tier_intervals(words_t)
        phones = _tier_intervals(phones_t)
        if any(_is_silence(lab) for _s, _e, lab in words):
            new_words, new_phones, n_ap = _rebuild_pair(words, phones, ap_events,
                                                        sp_dur, ap_dur)
        else:
            # No word-level silence: split the phone tier's own SP/blank intervals,
            # leaving the words tier untouched.
            regions = [(s, e) for s, e, lab in phones if _is_splittable(lab)]
            chunks, n_ap = _split_regions(regions, ap_events, sp_dur, ap_dur)
            new_words = list(words)
            new_phones = _apply_chunks(phones, regions, chunks)
        for tier in tg.tiers:
            name = str(tier.name).lower()
            if name in _WORD_TIER_NAMES:
                new_tg.addTier(tglib.IntervalTier(tier.name, new_words, 0.0, dur))
            elif name in _PHONE_TIER_NAMES:
                new_tg.addTier(tglib.IntervalTier(tier.name, new_phones, 0.0, dur))
            else:
                new_tg.addTier(tier)
        report = {"mode": "phones", "only_in_sp": False,
                  "written": [(s, e) for s, e, lab in new_phones if lab == "AP"],
                  "n_ap": n_ap, "n_detected": len(ap_events)}

    new_tg.save(str(out_path), format=fmt, includeBlankSpaces=True)
    return report


def _apply_chunks(
    phones: Sequence[tuple[float, float, str]],
    regions: Sequence[tuple[float, float]],
    chunks: Sequence[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Replace ``regions`` inside a phone list with the pre-split ``chunks``."""
    region_set = {(round(s, 6), round(e, 6)) for s, e in regions}
    out: list[tuple[float, float, str]] = []
    for s, e, lab in phones:
        if (round(s, 6), round(e, 6)) in region_set:
            continue
        if any(s < re_ - 1e-6 and e > rs + 1e-6 for rs, re_ in regions):
            continue                      # partially covered -> replaced as well
        out.append((s, e, lab))
    out += [(s, e, lab) for s, e, lab in chunks]
    return sorted(out, key=lambda x: x[0])


def main() -> None:
    """Add AP labels to every requested TextGrid."""
    ap = argparse.ArgumentParser(
        description="Add AP (breath) labels to annotated TextGrids.")
    ap.add_argument("--ckpt", default="runs/conformer_v1/best.pt")
    ap.add_argument("--tg", nargs="+", required=True, help="TextGrid files or directories")
    ap.add_argument("--wav_dir", default=None, help="directory holding the wav files, "
                                                    "defaults to each TextGrid's directory")
    ap.add_argument("--out", required=True, help="output directory (mirrors the input layout)")
    ap.add_argument("--mode", choices=["tier", "phones"], default="tier",
                    help="tier = append a standalone AP tier (default); phones = write AP "
                         "into the words+phones tiers (SOFA / OpenVPI style)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold stored in the checkpoint")
    ap.add_argument("--norm", choices=["global", "perbin"], default="global",
                    help="perbin is recommended for out-of-domain audio")
    ap.add_argument("--only_in_sp", action="store_true",
                    help="tier mode only: keep only detections that fall inside "
                         "SP / blank intervals")
    ap.add_argument("--sp-dur", type=float, default=0.1,
                    help="phones mode: SP fragments shorter than this are absorbed into "
                         "the neighbouring AP, in seconds (default 0.1)")
    ap.add_argument("--ap-dur", type=float, default=0.08,
                    help="phones mode: minimum written AP length, in seconds (default 0.08)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, mean, std, thr0, model = load_runner(args.ckpt, device)
    thr = args.threshold if args.threshold is not None else thr0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    desc = {"tier": "standalone AP tier" + (" (SP-only)" if args.only_in_sp else ""),
            "phones": "words+phones injection"}[args.mode]
    print(f"[tg_add_ap] ckpt={args.ckpt}  threshold={thr:.2f}  norm={args.norm}  "
          f"mode={args.mode} ({desc})  sp_dur={args.sp_dur}  ap_dur={args.ap_dur}")

    tgs = _collect_tgs(args.tg)
    print(f"{len(tgs)} TextGrid(s) in total")
    for tg_path in tgs:
        wav_dir = Path(args.wav_dir) if args.wav_dir else tg_path.parent
        wav = wav_dir / (tg_path.stem + ".wav")
        if not wav.exists():
            print(f"  [skip] wav not found: {wav}")
            continue
        out_path = out_dir / tg_path.name
        if out_path.exists() and not args.overwrite:
            print(f"  [skip] already exists: {out_path}")
            continue
        try:
            ap_events, _segments, dur, _extra = run_one(wav, cfg, mean, std, model, device,
                                                        thr, norm=args.norm)
        except Exception as e:
            print(f"  [failed] {tg_path.name}: {e}")
            continue
        try:
            report = add_ap_to_tg(tg_path, ap_events, out_path,
                                  only_in_sp=args.only_in_sp, mode=args.mode,
                                  sp_dur=args.sp_dur, ap_dur=args.ap_dur)
        except Exception as e:
            print(f"  [failed] {tg_path.name}: {e}")
            continue
        kept = report["written"]
        total = sum(e - s for s, e in kept)
        extra = ""
        if report["mode"] == "tier" and args.only_in_sp:
            extra = f", dropped {report['n_detected'] - report['n_ap']} outside SP"
        print(f"  {tg_path.stem}: detected {report['n_detected']} -> wrote "
              f"{report['n_ap']} AP ({total:.2f}s / {dur:.2f}s){extra}")
        meta = {
            "textgrid": tg_path.name,
            "audio_file": wav.name,
            "threshold": thr,
            "mode": report["mode"],
            "only_in_sp": args.only_in_sp,
            "sp_dur": args.sp_dur,
            "ap_dur": args.ap_dur,
            "ap_events": [{"start_sec": round(s, 3), "end_sec": round(e, 3)} for s, e in kept],
            "all_detected": [{"start_sec": round(s, 3), "end_sec": round(e, 3)}
                             for s, e in ap_events],
        }
        (out_dir / f"{tg_path.stem}.ap.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
