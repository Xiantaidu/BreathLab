# BreathLab

`BreathLab` is a PyTorch toolkit for detecting audible breaths (AP) in singing voice and turning them into ready-to-use annotations.

The detector is a frame-level sequence model that runs at **100 fps on mono 16 kHz audio**. It consumes 64 log-mel bands plus two F0/periodicity channels, and the time axis is never downsampled (in the CRNN and Conformer backbones), because annotated breaths can be as short as a few tens of milliseconds and event-level evaluation depends on onset precision. Per-frame probabilities are post-processed into breath events `(start_sec, end_sec)` and written as TextGrid + JSON files that plug directly into a labelling workflow. The project was developed and validated on tens of hours of singing recordings spanning multiple languages and annotation styles.

The public source tree is intentionally lightweight:

- training code is included
- PyTorch inference and TextGrid annotation utilities are included
- ONNX export and PyTorch-free ONNX Runtime inference are included
- datasets, manifests, splits, and checkpoints are **not** included in the repository

## What This Repository Provides

- A breath-detection training pipeline with three backbones: a lightweight CRNN, a Conformer (serial or E-Branchformer blocks), and a CNN U-Net + BiMamba model, with an optional frequency-dynamic convolution (FDY) front end
- Optional output heads beyond the main breath class: a supervised silence (SP) head, exhale (EP) and glottal-stop (GS) heads, and an onset/offset boundary head
- Duration-preserving augmentation (noise, noise floor, compression, breath/harmony overlay, synthetic-IR reverb, SpecAugment, optional VTLN) so the label timeline stays exact
- Evaluation with a dev-set threshold sweep and frame- and event-level metrics
- PyTorch inference that writes TextGrid + JSON per input file
- A TextGrid utility that inserts breath annotations into existing TextGrids (standalone tier or SOFA/OpenVPI-style `words`+`phones` mode)
- ONNX export and a PyTorch-free ONNX Runtime inference path (DirectML / CPU / CUDA)

## Important Notes

- This repository does not ship audio, annotations, manifests, split files, or model checkpoints.
- Training corpora are referenced through **local, non-committed JSONL manifests** addressed by anonymous aliases (`c1`, `c2`, ...). You supply your own corpora; each is governed by its own license, and you are responsible for complying with it. See `NOTICE`.
- `breathlab/configs/default.yaml` is the baseline configuration and contains local placeholder paths (`paths.cache_dir`, `paths.manifests_dir`, `paths.splits_dir`, `paths.runs_dir`) — edit them for your environment before the first run.
- Do **not** install `librosa`. The whole feature pipeline is built on `soundfile` + `scipy` + `torchaudio`, and librosa's dependency chain is known to hang on some Windows setups.
- `torch` / `torchaudio` are deliberately not hard dependencies so that installing this project never replaces an existing CUDA build.
- The default configuration trains on a single 8 GB consumer GPU (bf16 mixed precision, gradient checkpointing for the Mamba backbone).

## Repository Layout

```text
.
|-- breathlab/                Core package
|   |-- configs/default.yaml  Baseline configuration (single source of truth)
|   |-- data/                 Manifest parsing, splits, features, frame targets, augmentation
|   |-- models/               CRNN, Conformer/E-Branchformer, FDY conv, UNet-BiMamba
|   |-- train.py              Training loop
|   |-- evaluate.py           Threshold sweep on dev + test report
|   |-- infer.py              Breath detection -> TextGrid + JSON
|   |-- postprocess.py        Median filter / threshold / merge / full segmentation
|   `-- tg_add_ap.py          Add an AP tier to existing TextGrids
|-- docs/ARCHITECTURE.md      Design rationale: why the pipeline is shaped the way it is
|-- tools/                    Benchmarking, profiling, data preparation, ONNX deployment
|-- requirements.txt          Base dependencies (torch intentionally not listed)
`-- pyproject.toml            Editable install + `breathlab` console script
```

All commands are dispatched through one entry point:

```bash
python -m breathlab <command> [options]
```

| Command | Aliases | Purpose |
|---|---|---|
| `make-splits` | `splits` | Build train/dev/test splits for all configured corpora |
| `train` | | Train a breath detector |
| `evaluate` | `eval` | Sweep thresholds on dev, then report test metrics |
| `infer` | `predict` | Detect breaths in wav files; write TextGrid + JSON |
| `tg-ap` | | Add an AP tier to existing TextGrid files |

After `pip install -e .` the same commands are available as `breathlab <command>`. Run any command with `--help` for its full flag list.

## Installation

Install PyTorch first, with the CUDA index that matches your driver, so a later `pip install` never swaps in a CPU wheel:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu129
```

Then the base environment:

```bash
pip install -r requirements.txt
pip install -e .    # optional; provides the `breathlab` console script
```

For ONNX export and ONNX Runtime inference, install the extra packages that match your hardware:

```bash
pip install onnx onnxruntime-directml   # Windows GPU via DirectML (default for infer_onnx.py)
```

or `onnxruntime` (CPU) / `onnxruntime-gpu` (CUDA).

## Data Format

The repository knows nothing about specific corpora. Each corpus is described by one local JSONL manifest (not committed), with **one row per clip** and pre-parsed events:

```json
{
  "wav": "path/to/clip.wav",
  "singer": "singer_id",
  "duration_sec": 12.4,
  "breath": [[1.20, 1.55], [7.02, 7.31]],
  "sp": [[0.0, 0.8]],
  "silent": [],
  "uncertain": [],
  "hard_neg": [],
  "exhale": [],
  "gs": []
}
```

All event fields are lists of `[start_sec, end_sec]` pairs and are all optional:

| Field | Meaning |
|---|---|
| `breath` | AP positives — audible breaths |
| `sp` / `silent` | Supervised silence for the SP head (`silent` is treated as `sp` or as loss-masked frames depending on `labels.silent_breaths_as_sp`) |
| `uncertain` | Masked out of the loss (frames keep their position in evaluation) |
| `hard_neg` | Mined false positives, used for window oversampling and optional loss weighting |
| `exhale` / `gs` | Supervise the optional EP / GS heads |

The manifests directory is set once via `data.manifests_dir`; corpora are referred to by aliases (`c1`..`cN`) in the config, with a per-corpus split strategy (`by_singer`, `by_song`, `official_test_by_song`, `by_singer_frac`) so that no singer or song leaks across splits. A 16 kHz waveform cache is built automatically on first use.

Build the splits with:

```bash
python -m breathlab make-splits
```

## Training

The main training entry point is:

```bash
python -m breathlab train --tag my_run --model conformer
```

The run stores its full resolved config inside the checkpoint, so evaluation and inference always reproduce the exact training setup.

Useful flags:

- `--model crnn|conformer|unet_mamba` — backbone override
- `--block conformer|ebranchformer` — Conformer block topology (checkpoints are not interchangeable between the two)
- `--init path/to/other/best.pt` — warm-start from an older checkpoint
- `--norm-mode global|cmvn` — per-bin global normalization vs per-clip CMVN
- `--no-f0` — disable the F0/periodicity input channels
- `--boundary-head` — add the onset/offset auxiliary head
- `--optional-heads EP,GS` — enable the optional output heads
- `--loss focal|bce`, `--lr`, `--batch-size` — optimization overrides
- `--corpus-weight NAME=W` — rebalance a corpus in the sampler

Checkpoints land in `runs/<tag>/` (`best.pt` plus the last state). Early stopping and best-checkpoint selection monitor the dev event-level F1 (blended with the optional-head metric when those heads are enabled).

## Evaluation

`evaluate` sweeps the decision threshold on dev, then reports on the requested splits:

```bash
python -m breathlab evaluate --ckpt runs/my_run/best.pt --splits dev,test
```

It reports two operating points:

- `balanced` — highest dev event F1 (ties broken by precision)
- `precision_first` — among thresholds within 95% of the best F1, the one with the highest precision (fewer false alarms)

Metrics are frame-level precision/recall/F1 and event-level one-to-one matching on onsets with a 200 ms collar. The event-level number is the one to steer by — frame F1 is dominated by the silence-heavy majority class. The chosen threshold is stored in the checkpoint and reused by inference.

## Breath Detection (PyTorch Inference)

Detect breaths in wav files and write TextGrid + JSON outputs:

```bash
python -m breathlab infer --ckpt runs/my_run/best.pt --wav song.wav --out output/
```

`--wav` accepts single files or whole directories. Clips are scored in overlapping 12 s windows with 50% overlap (averaged), so full songs can be processed without blowing up attention memory.

Useful flags:

- `--emit ap,sp[,ep][,gs]` — which event classes to emit (EP/GS only if the head was trained)
- `--threshold` — override; the default is the best dev threshold stored in the checkpoint
- `--norm global|perbin` — normalization mode (checkpoints trained with CMVN are handled automatically)

Per input file the output is `<stem>.TextGrid` (a `labels` tier, an `AP` tier, plus optional SP/EP/GS tiers) and `<stem>.ap.json` with the same events in machine-readable form.

## TextGrid Annotation Utility

`tg-ap` runs the detector over existing TextGrid files and merges the result into them — useful for retrofitting breath labels onto an already phoneme-annotated dataset:

```bash
python -m breathlab tg-ap --tg grid_dir/ --wav_dir wav_dir/ --out out_dir/ --mode tier
```

Two modes:

- `--mode tier` (default) — writes a standalone `AP` tier next to the existing tiers
- `--mode phones` — for SOFA / OpenVPI-style TextGrids carrying the usual `words` + `phones` pair: silence intervals are rewritten as `SP ... AP ... SP` runs, with the same labels mirrored into the phones tier; every non-silence word keeps its original phones

The `phones` mode is idempotent (re-running is safe), and `--sp-dur` / `--ap-dur` control the forced durations of inserted SP/AP intervals. `--only_in_sp` restricts AP insertion to silence intervals.

## ONNX Export and Runtime Inference

Export a checkpoint to ONNX:

```bash
python tools/deploy/export_onnx.py --ckpt runs/my_run/best.pt --opset 18
```

The exporter forces a deployment-friendly graph, writes `model.onnx` + `model.meta.json` next to the checkpoint, and verifies the graph against PyTorch on several sequence lengths.

Run PyTorch-free end-to-end inference with ONNX Runtime (DirectML by default on Windows; CPU and CUDA selectable):

```bash
python tools/deploy/infer_onnx.py --onnx runs/my_run/model.onnx --wav song.wav --out output/onnx
```

The numpy feature front end is verified against torchaudio (`--check-features`) and the whole pipeline against `breathlab.infer` (`--compare-torch`).

## Models

| Backbone | Params | Notes |
|---|---|---|
| `crnn` | ~1.5M | Lightweight baseline: 3-stage CNN (frequency pooling only) + BiGRU |
| `conformer` | ~5.0M | Serial or E-Branchformer blocks, rotary positional encoding, optional FDY front end |
| `unet_mamba` | ~5.4M | CNN U-Net encoder + BiMamba bottleneck + U-Net decoder; downsamples time 4x, so train it from scratch |

The FDY (frequency-dynamic convolution) front end applies per-frequency attention over a bank of candidate kernels, so the front end can specialize per mel band instead of learning an average. Multiple mathematically equivalent compute paths (`loop` / `fuse` / `mix` / `triton`) are provided for speed.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full signal flow, the reasoning behind every major design decision, and the extension points.

## Model Scope

This repository is primarily for:

- automatic breath (AP) annotation of singing-voice datasets
- breath event analysis and segmentation research
- optional silence / exhale / glottal-stop heads built on the same pipeline

It is not intended to be a dataset release repository, and it does not redistribute any audio or annotations.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). BreathLab does not redistribute any third-party corpus; users enable their own corpora through local manifests and are responsible for complying with each corpus's license and terms of use.
