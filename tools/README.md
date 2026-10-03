# BreathLab tools

Standalone command-line scripts for benchmarking, diagnosing, preparing data and adapting the
model to a new singer.

**How to run.** Every script is executed as a plain file from the repository root with the
package on `PYTHONPATH`; there are no `__init__.py` files under `tools/` on purpose:

```bash
cd /d/BreathLab
PYTHONPATH=D:/BreathLab python tools/<group>/<script>.py [flags]
```

Scripts that need a GPU assume CUDA is available; `smoke_check.py` and the CPU-only analysis
scripts fall back to CPU automatically.

## Layout

| Directory | Contents |
|---|---|
| `tools/` | `smoke_check.py` (post-change smoke test entry point) and this README |
| `tools/benchmark/` | Speed, cost and kernel-level profiling; numerical equivalence checks |
| `tools/analysis/` | Metric analysis and reporting (some local-only scripts are not committed) |
| `tools/data/` | Split construction and hard-negative mining |
| `tools/adapt/` | Per-singer few-shot calibration and fine-tuning (local-only, not committed) |
| `tools/deploy/` | ONNX export and PyTorch-free inference (DirectML / CPU) |

## Scripts

| Script | Path | Purpose | Example command |
|---|---|---|---|
| `smoke_check.py` | `tools/smoke_check.py` | Post-change smoke test: one training step, one inference-window forward, and optional legacy-checkpoint compatibility (incl. shape migration). Run this first after any structural edit. | `python tools/smoke_check.py --ckpt runs/conformer_v1/best.pt` |
| `export_onnx.py` | `tools/deploy/export_onnx.py` | Exports a checkpoint to ONNX for deployment. Forces a traceable/DirectML-friendly graph (`fdy_mode` -> `fuse` when FDY is on, `fdy_checkpoint` off, `channels_last` off), writes `<ckpt dir>/model.onnx` + `model.meta.json`, and verifies against Torch on several sequence lengths. | `python tools/deploy/export_onnx.py --ckpt runs/v5_24k/best.pt --opset 18` |
| `infer_onnx.py` | `tools/deploy/infer_onnx.py` | PyTorch-free end-to-end inference: numpy log-mel + F0 + onnxruntime (DirectML by default), writes TextGrid + JSON. `--emit ap,sp[,ep][,gs]` selects the event classes (EP/GS only if the head was exported). `--check-features` proves the numpy front end matches torchaudio; `--compare-torch` proves the whole pipeline matches `breathlab.infer`. | `python tools/deploy/infer_onnx.py --onnx runs/v5_24k/model.onnx --wav song.wav --out output/onnx` |
| `bench_speed.py` | `tools/benchmark/bench_speed.py` | Training-step benchmark over optimization variants (rotary cache, FDY fused/mix/triton, channels_last, fused AdamW, SDPA backend, `torch.compile`); `--parity` / `--attn-parity` verify numerical equivalence; `--data` measures CPU dataloader throughput. | `python tools/benchmark/bench_speed.py --steps 8 --batch 24` |
| `bench_variants.py` | `tools/benchmark/bench_variants.py` | Cost of architecture variants (params / front end / full step / peak VRAM) in about a minute, instead of a multi-hour training run. | `python tools/benchmark/bench_variants.py --batch 24 --frames 600 --only d160` |
| `bench_bandwidth.py` | `tools/benchmark/bench_bandwidth.py` | Shows that sample rate alone buys nothing (VRAM and step time depend on hop x channels); only raising `fmax` **and** `n_mels` adds high-frequency evidence, and that cost scales with `n_mels`. | `python tools/benchmark/bench_bandwidth.py --batch 24 --frames 600` |
| `profile_fdy_layers.py` | `tools/benchmark/profile_fdy_layers.py` | Per-layer timing of the six FDY dynamic convolutions in the CNN front end (FDY triton vs. mix vs. a plain conv of the same shape) to decide which layers are worth cutting. | `python tools/benchmark/profile_fdy_layers.py --batch 24` |
| `profile_front.py` | `tools/benchmark/profile_front.py` | Kernel-level profiler table for the CNN front end (`--front-only`) or the whole model, optionally forcing another FDY path with `--mode`. | `python tools/benchmark/profile_front.py --front-only --steps 3` |
| `check_fdy_kernel.py` | `tools/benchmark/check_fdy_kernel.py` | Fast single-layer regression test: the `mix` and `triton` FDY paths must match the reference `loop` path in fp32 and bf16 (output, dx, dW, dA). | `python tools/benchmark/check_fdy_kernel.py --ci 64 --co 128` |
| `make_bal_splits.py` | `tools/data/make_bal_splits.py` | Rebuilds dev/test with a per-singer cap and round-robin sampling over (technique, song) groups, so no single out-of-domain singer dominates the aggregate metric. `train.txt` is never touched. | `python tools/data/make_bal_splits.py --cap 250` |
| `mine_hard_negatives.py` | `tools/data/mine_hard_negatives.py` | Exports high-confidence false-positive breath spans (with safety valves against learning away missed labels) for window oversampling and per-frame loss weighting. | `python tools/data/mine_hard_negatives.py --ckpt runs/conformer_v2/best.pt --split train --corpora c10` |
| `export_onnx.py` | `tools/deploy/export_onnx.py` | Exports a checkpoint to ONNX for deployment. Forces a traceable/DirectML-friendly graph (`fdy_mode` -> `fuse` when FDY is on, `fdy_checkpoint` off, `channels_last` off), writes `<ckpt dir>/model.onnx` + `model.meta.json`, and verifies against Torch on several sequence lengths. | `python tools/deploy/export_onnx.py --ckpt runs/v5_24k/best.pt --opset 18` |
| `infer_onnx.py` | `tools/deploy/infer_onnx.py` | PyTorch-free end-to-end inference: numpy log-mel + F0 + onnxruntime (DirectML by default), writes TextGrid + JSON. `--emit ap,sp[,ep][,gs]` selects the event classes (EP/GS only if the head was exported). `--check-features` proves the numpy front end matches torchaudio; `--compare-torch` proves the whole pipeline matches `breathlab.infer`. | `python tools/deploy/infer_onnx.py --onnx runs/v5_24k/model.onnx --wav song.wav --out output/onnx` |

## Outputs other code depends on

These paths are contractual -- do not rename them without updating the consumers:

| Output | Produced by | Consumed by |
|---|---|---|
| `splits/dev_bal.txt`, `splits/test_bal.txt` | `tools/data/make_bal_splits.py` | training / evaluation `--split` flags |
| `splits/split_summary_bal.json` | `tools/data/make_bal_splits.py` | split bookkeeping |
