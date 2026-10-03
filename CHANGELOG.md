# Changelog

All notable changes to BreathLab are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-29

First organised release. The codebase was prepared for publication: every comment,
docstring, log message and configuration comment was translated into English, the
module layout was tidied, and packaging was added.

### Added

- `breathlab/cli.py` and `breathlab/__main__.py`: a unified entry point
  (`python -m breathlab <command>` / `breathlab <command>`) with the subcommands
  `make-splits`, `train`, `evaluate` (alias `eval`), `infer` and `tg-ap`. Existing
  direct invocations such as `python -m breathlab.train` keep working unchanged.
- `breathlab/__init__.py` with package metadata and `__version__`.
- `pyproject.toml` (editable install, console script, ruff configuration),
  `LICENSE` (Apache-2.0), `NOTICE` (dependency and dataset attribution),
  `.gitignore`, and a reworked `requirements.txt` that no longer pins `torch` so an
  existing CUDA build is never silently replaced.
- `docs/ARCHITECTURE.md` documenting the signal flow and the reasoning behind the
  main design decisions.
- `README_CN.md`, the Chinese counterpart of `README.md`.
- `tools/README.md` indexing every helper script.
- `tools/` was reorganised by purpose: `benchmark/`, `analysis/`, `data/`, `adapt/`,
  with `smoke_check.py` kept at the root.

### Changed

- All code comments, docstrings, argparse help text and runtime log strings are now
  English. Public API names, CLI flags and configuration keys are unchanged.
- `default.yaml` comments were rewritten in English; no key or value changed.
- Google-style docstrings and type hints were added across `breathlab/`.
- `breathlab/data/__init__.py` and `breathlab/models/__init__.py` are now documented
  package inits instead of empty files (the models package exports `build_model` and
  `count_params`).

### Added

- `models_dml/`: a self-contained DirectML inference bundle - `infer.py` (no BreathLab
  package import, no PyTorch), `requirements.txt` (numpy / scipy / soundfile /
  onnxruntime-directml), a README, and two exported models (`v4_all`, `v5_24k`).
- `tools/deploy/export_onnx.py`: exports a checkpoint to ONNX for deployment and
  verifies the graph against Torch on several sequence lengths.
- `tools/deploy/infer_onnx.py`: PyTorch-free end-to-end inference with onnxruntime
  (DirectML by default), plus `--check-features` / `--compare-torch` self-checks.
- `tg-ap --mode phones`: support for SOFA / OpenVPI style TextGrids that carry the
  usual `words` + `phones` pair. Silence intervals (word labelled `SP` or blank) are
  split into `SP ... AP ... SP` runs and the same labels are written into the phones
  tier, while every other tier and all non-silence words keep their original phones.
  New flags `--sp-dur` (default 0.1 s, short leftover SP is absorbed into the
  neighbouring AP) and `--ap-dur` (default 0.08 s). Re-running the mode is idempotent.
  `--mode tier` (the previous behaviour, a standalone AP tier) remains the default.

### Fixed

- `breathlab/infer.py` called `setup_speed(..., allow_compile=False)`, a keyword that
  does not exist, so `python -m breathlab.infer` and `python -m breathlab tg-ap`
  raised `TypeError` on every run.
- `breathlab/infer.py` always applied the global mean/std even when the checkpoint
  was trained with `norm_mode: cmvn`, i.e. inference used a different ruler than
  training. It now honours `feature.norm_mode` (per-clip CMVN over the whole file).

### Removed

- The stray `NUL` file at the repository root (a Windows shell artifact).
- `README_CN.md`: the English `README.md` is now the single documentation file.
- Stale `__pycache__` directories under `tools/`.
- Unused imports in `breathlab/data/dataset.py` and several tools scripts.
