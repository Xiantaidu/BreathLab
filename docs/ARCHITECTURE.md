# BreathLab architecture

This document explains *why* the pipeline is shaped the way it is. For usage, see
[README.md](../README.md).

## 1. Signal flow

```
wav (any SR)
  |  load_wav_16k / load_wav_16k_cached      soundfile + scipy.resample_poly
  v
mono 16 kHz float32
  |  LogMel                                  torchaudio MelSpectrogram, center=True
  v
log-mel [n_mels=64, T]                       T = duration * 100  (hop is 10 ms)
  |
  |-- pitch_channels() ------------------->  [periodicity, log-F0]   (optional, 2 rows)
  v
input [input_rows=66, T]
  |  normalization                           global (train-set per-bin) | cmvn (per clip)
  v
model (CRNN | Conformer)
  v
per-frame logits [C, T]                      C = 1 (AP) | 2 (+SP) | 4 (+onset/offset)
  |
  |-- training : focal / BCE loss with mask, optional SP + boundary losses
  |-- inference: chunked sliding window -> sigmoid -> averaged probabilities
  v
postprocess
  |  median filter -> threshold -> merge -> min duration
  v
AP events [(start_sec, end_sec)]
  |
  +-- energy VAD (or the supervised SP head) -> full AP / SP / V segmentation
  v
TextGrid (labels tier + AP tier) and *.ap.json
```

The frame rate is **100 fps everywhere**: in the label targets, in the model, in the
metrics, and in the exported time axis. Nothing resamples time.

## 2. Why the time axis is never downsampled

Annotated breath durations across the corpora are `min = 33 ms`, `p1 ≈ 130 ms`,
`median ≈ 310 ms`. A single pooling step on the time axis (as in many audio tagging
models) would turn a 130 ms breath into 6 frames or fewer and destroy the onset
precision that event-level evaluation depends on. All capacity is therefore added
along `d_model` / layers / kernel width / frequency axis instead: the CNN front end
pools **only on the frequency axis** (`MaxPool2d((freq_pool, 1))`).

## 3. Models

Both models share the same contract: input `[B, F, T]`, output `[B, C, T]`.

### CRNN

`ConvBlock` (two 3x3 convs + BN + ReLU, frequency pooling, dropout) x3 -> BiGRU ->
dropout -> `Linear`. The CNN output is transposed to `[B, T, C*F']` before the RNN.

### Conformer

`ConvFront` -> `Linear(d_model)` -> N blocks -> `Dropout + Linear(head)`.

Two block topologies are available at comparable parameter count:

- `conformer` — serial `attn -> conv` (Macaron FFN, pre-norm).
- `ebranchformer` — parallel attention and convolution branches merged by
  `EnhancedMerge` (depthwise conv residual + linear projection). Checkpoint keys are
  **not** interchangeable between the two.

Rotary positional encoding makes attention length-independent, so training on 6 s
windows and scoring 264 s clips is consistent.

### Frequency-dynamic convolution (FDY)

The frequency axis is *not* translation invariant: fricatives live at 4-8 kHz, breath
noise has its own band, and vocal harmonics sit low. A static kernel can only learn
an average across bands. `FDYConv2d` applies per-frequency attention over `M`
candidate kernels:

```
out = sum_m A_m(f) * Conv_m(x)
```

Because `A` depends only on frequency, "mixing the outputs" is mathematically
identical to "mixing the kernels per frequency", which gives four equivalent compute
paths: `loop`, `fuse`, `mix` (im2col + batched matmul, 1/M the compute) and `triton`
(a custom kernel that avoids im2col traffic entirely). Initialisation copies one
weight into all candidates and zero-initialises the attention, so an FDY layer starts
exactly equivalent to a plain convolution and can warm-start from an older checkpoint.

## 4. Data pipeline

### Labels

`parse_labels.py` reads **pre-enumerated corpus manifests**: one local, non-committed
`<alias>.jsonl` per corpus (see `data.manifests_dir`), with one row per clip and
pre-parsed events (`breath` / `sp` / `silent` / `hard_neg` / `uncertain` / `exhale` /
`gs`, each a list of `[start_sec, end_sec]` pairs), plus the clip's `wav` path,
`singer` and optional `split_hint`. All corpus-specific knowledge — real paths,
annotation formats, label symbol semantics — lives in the manifests directory and
never enters the repository; the repo only knows aliases (`c1`, `c2`, ...).

`build_frame_targets()` turns events into four `[T]` arrays: `label` (AP),
`mask` (loss participation), `hardneg` (mined hard negatives, used for window
oversampling and optional loss weighting) and `sp_label` (true silence). AP wins over
SP where annotations overlap; `uncertain` only masks frames that are not positive.

### Splitting

Each corpus declares a strategy so no singer or song leaks across splits:
`by_singer`, `by_song`, `official_test_by_song`, `by_singer_frac`.

> `dev.txt` / `test.txt` sample **singers randomly**, so a 20-singer corpus
> contributes only 2 singers to a split and the aggregate F1 is dominated by one
> singer. Use the per-singer-capped `dev_bal` / `test_bal`
> (`python tools/data/make_bal_splits.py`) for any decision-making.

### Loading

- **Training** (`preload=False`): waveforms live on disk as `.npy` and are read
  through `np.load(mmap_mode="r")`; log-mel is computed per window. Only labels,
  masks and the breath bank stay in RAM.
- **Evaluation** (`preload=True`): whole clips are decoded once.

### Normalization

`global` mode subtracts the training-set per-bin mean and divides by the per-bin
std. `cmvn` mode instead removes each clip's own per-bin mean/std, cancelling
recording level and frequency-response differences.

The CMVN statistics must be computed over the **whole clip** and shared between
training (6 s windows) and evaluation (full clips). Computing them inside a training
window produces a different normalisation for long clips (measured: up to 0.4-0.5
difference on a 264 s clip), i.e. two different rulers for training and inference.
`compute_train_norm(with_clip_stats=True)` pre-computes them once and caches them in
`norm_stats.npz` together with a signature that includes sample rate, `fmax`,
`n_mels`, `hop_length` and `norm_mode` — without that signature, changing bandwidth
would silently reuse stale statistics.

### Augmentation

All augmentations preserve duration, because the label timeline must stay exact:
gain, additive noise, noise floor (white/pink), synthetic-IR reverb, breath overlay
from a breath bank (which also flips the corresponding frames to positive), harmony
overlay (a duration-preserving pitch-shifted copy of the window mixed underneath at
low level), dynamics compression, SpecAugment, per-channel mel jitter.

Reverb is applied last so that overlaid breaths are reverberated together with the
signal. When the F0 channels are enabled they are recomputed on the *final*
waveform, and overlaid regions have their periodicity zeroed with probability
0.5 — otherwise the model learns "periodicity present" as evidence *against* breath,
which fails on real harmony-backed recordings.

## 5. Inference

`predict_probs_windowed()` splits a clip into overlapping 12 s windows (50% overlap),
batches them, and averages the overlap region. Batching matters: an earlier
window-by-window version took tens of minutes for a few thousand clips.

Attention models are `O(T^2)` in memory, so a full song cannot be scored in one
pass; 12 s is comfortably larger than any breath, so no long-range context is lost.

## 6. Metrics

- **Frame level**: precision / recall / F1 over masked frames.
- **Event level**: one-to-one greedy matching on **onset** with a 200 ms collar,
  optional offset requirement.

The event-level number is the one to steer by, because it is what a labelling
workflow actually cares about. Frame F1 is dominated by the silence-heavy majority
class and moves very little.

`evaluate` reports two operating points:

- `balanced` — highest dev event F1 (ties broken by precision).
- `precision_first` — among thresholds within 95% of the best F1, the one with the
  highest precision (preferring fewer false alarms).

## 7. Speed switches

`breathlab/speed.py` centralises `channels_last`, `cudnn_benchmark`, `fused_adamw`,
`torch.compile` and the SDPA backend. Two non-obvious constraints:

- **head_dim must be a multiple of 8** for the efficient SDPA backends. With
  `d_model=176, n_heads=4` the head dim is 44, everything silently falls back to the
  math backend, and it is ~6.5x slower. `MHSA` pads head_dim internally (mathematically
  equivalent) and passes `scale` explicitly, because SDPA would otherwise derive the
  temperature from the *padded* dimension.
- **`cudnn.benchmark` is disabled during evaluation.** Dev clips have wildly varying
  lengths, and autotuning would search an algorithm per new shape for thousands of
  clips. Training shapes are fixed and already cached, so they are unaffected.

## 8. Extension points

| Goal | Where to look |
|---|---|
| New corpus | `data/parse_labels.py` + a `data.corpora` entry in the config |
| New block topology | `models/conformer.py` (`ConformerBlock`, `EBranchformerBlock`) |
| New loss | `train.py` (`focal_loss`, `bce_loss`) |
| New output head | `Conformer.head_index` / `CRNN` head width, plus the loss in `train.py` |
| New post-processing rule | `postprocess.py` (`full_segmentation`) |
