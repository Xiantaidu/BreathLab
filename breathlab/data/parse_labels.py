"""Clip loading: turn pre-enumerated corpus manifests into :class:`Clip` objects.

Every corpus is one local, non-committed manifest ``<manifests_dir>/<alias>.jsonl`` with one
row per clip (``name``/``wav``/``singer``/``split_hint``/``duration_sec`` plus the event
lists ``breath``/``sp``/``silent``/``hard_neg``/``uncertain``/``exhale``/``gs`` as
``[start_sec, end_sec]`` pairs). The alias doubles as the clip-name prefix
(``"<alias>/<relative clip id>"``) and the ``data.corpus_weights`` key; real paths and
label semantics never enter the repository.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

PathLike = Union[str, Path]


@dataclass
class Event:
    """A labelled time interval inside a clip, in seconds."""

    start: float
    end: float
    confidence: str = ""
    source: str = ""
    notes: str = ""


@dataclass
class Clip:
    """One recording with all label intervals attached to it.

    ``exhales``/``gs_events`` stay separate from ``breath_events`` on purpose: they
    supervise the optional EP/GS heads, not the AP head. ``sp_events`` are true-silence
    frames for the SP head; ``uncertain`` frames are dropped from the loss.
    """

    name: str
    wav_path: Path
    singer: str
    duration_sec: float
    corpus: str = ""
    split_hint: str = ""
    breath_events: List[Event] = field(default_factory=list)
    sp_events: List[Event] = field(default_factory=list)
    hard_negatives: List[Event] = field(default_factory=list)
    silent_breaths: List[Event] = field(default_factory=list)
    uncertain: List[Event] = field(default_factory=list)
    exhales: List[Event] = field(default_factory=list)
    gs_events: List[Event] = field(default_factory=list)

    @property
    def n_breath(self) -> int:
        return len(self.breath_events)


_EVENT_KEYS = (
    ("breath", "breath_events"),
    ("sp", "sp_events"),
    ("hard_neg", "hard_negatives"),
    ("uncertain", "uncertain"),
    ("exhale", "exhales"),
    ("gs", "gs_events"),
)


def _events(raw) -> List[Event]:
    return [Event(start=float(s), end=float(e), source="manifest")
            for s, e in (raw or []) if float(e) > float(s)]


def _clip_from_row(row: Dict, alias: str, silent_breaths_as_sp: bool) -> Clip:
    silent = _events(row.get("silent"))
    events = {fname: _events(row.get(key)) for key, fname in _EVENT_KEYS}
    return Clip(
        name=str(row["name"]),
        wav_path=Path(row["wav"]),
        singer=str(row.get("singer", alias)),
        duration_sec=float(row.get("duration_sec", 0.0)),
        corpus=alias,
        split_hint=str(row.get("split_hint", "")),
        breath_events=events["breath_events"],
        # silent_breaths_as_sp=True keeps the historical "quiet breath = SP" mapping;
        # False loss-masks those frames via uncertain instead (silent_breaths is always
        # populated for the misfire analysis in evaluate.py)
        sp_events=events["sp_events"] + (silent if silent_breaths_as_sp else []),
        uncertain=events["uncertain"] + ([] if silent_breaths_as_sp else silent),
        hard_negatives=events["hard_negatives"],
        exhales=events["exhales"],
        gs_events=events["gs_events"],
        silent_breaths=silent,
    )


def _manifest_path(cfg: Dict, alias: str) -> Path:
    mdir = Path((cfg.get("data") or {}).get("manifests_dir", "manifests"))
    return mdir / f"{alias}.jsonl"


def load_manifest_clips(cfg: Dict, alias: str,
                        silent_breaths_as_sp: bool = True) -> List[Clip]:
    """Load one corpus from ``<manifests_dir>/<alias>.jsonl``."""
    path = _manifest_path(cfg, alias)
    if not path.exists():
        raise FileNotFoundError(
            f"manifest not found: {path} — generate it with the local manifest tool")
    clips: List[Clip] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                clips.append(_clip_from_row(json.loads(ln), alias, silent_breaths_as_sp))
    print(f"[parse_labels] {alias}: {len(clips)} clip(s) from manifest {path.name}")
    return clips


_ALL_CLIPS_CACHE: Dict[tuple, List[Clip]] = {}


def _corpora_cache_key(cfg: Dict) -> tuple:
    sb_sp = bool((cfg.get("labels") or {}).get("silent_breaths_as_sp", True))
    corpora = (cfg.get("data") or {}).get("corpora") or []
    entries = tuple(json.dumps(c, sort_keys=True, ensure_ascii=False) for c in corpora)
    manifests = []
    for c in corpora:
        p = _manifest_path(cfg, str(c.get("alias", "")))
        if p.exists():
            st = p.stat()
            manifests.append((str(p), st.st_mtime_ns, st.st_size))
    return (entries, sb_sp, tuple(sorted(manifests)))


def _parse_all_clips(cfg: Dict) -> List[Clip]:
    corpora = (cfg.get("data") or {}).get("corpora") or []
    sb_sp = bool((cfg.get("labels") or {}).get("silent_breaths_as_sp", True))
    clips: List[Clip] = []
    for c in corpora:
        if not c.get("enable", True):
            continue
        if "alias" not in c:
            raise ValueError("corpus entry without 'alias' (manifest-based config)")
        clips += load_manifest_clips(cfg, str(c["alias"]), silent_breaths_as_sp=sb_sp)

    hf = (cfg.get("data") or {}).get("hardneg_file")
    if hf:
        hp = Path(hf)
        if hp.exists():
            mined = json.loads(hp.read_text(encoding="utf-8"))
            n_c = n_e = 0
            for c in clips:
                evs = mined.get(c.name)
                if evs:
                    c.hard_negatives = list(c.hard_negatives) + [
                        Event(start=float(s), end=float(e), confidence="high", source="mined")
                        for s, e in evs]
                    n_c += 1
                    n_e += len(evs)
            print(f"[parse_labels] hard negatives injected: "
                  f"{n_e} span(s) / {n_c} clips (from {hf})")
        else:
            print(f"[parse_labels] WARNING: hardneg_file does not exist: {hf}")
    return clips


def load_all_clips(cfg: Dict) -> List[Clip]:
    """Aggregate clips from every corpus manifest, with an in-process cache.

    Returns a shallow list copy so callers can reorder or drop entries without
    corrupting the cache; the :class:`Clip` objects themselves are read-only.
    """
    key = _corpora_cache_key(cfg)
    hit = _ALL_CLIPS_CACHE.get(key)
    if hit is None:
        hit = _parse_all_clips(cfg)
        _ALL_CLIPS_CACHE[key] = hit
        print(f"[parse_labels] corpus parsed and cached: {len(hit)} clips")
    else:
        print(f"[parse_labels] reusing cached corpus: {len(hit)} clips")
    return list(hit)


if __name__ == "__main__":
    import argparse

    from breathlab.common import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="breathlab/configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    clips = load_all_clips(cfg)

    n_pos = sum(1 for c in clips if c.n_breath > 0)
    print(f"total clips: {len(clips)}")
    print(f"clips with breath: {n_pos}  |  all-negative clips: {len(clips) - n_pos}")
    print(f"total breath_events: {sum(c.n_breath for c in clips)}")
    per_alias: Dict[str, List[int]] = {}
    for c in clips:
        d = per_alias.setdefault(c.corpus, [0, 0])
        d[0] += 1
        d[1] += c.n_breath
    print("By corpus alias (clips, breaths):")
    for a in sorted(per_alias):
        print(f"  {a:6s} clips={per_alias[a][0]:6d}  breaths={per_alias[a][1]:6d}")
