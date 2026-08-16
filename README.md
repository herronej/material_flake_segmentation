# flakeseg

Semantic segmentation of exfoliated 2D material flakes in optical micrographs,
targeting the [2DFlakeSemSeg](https://huggingface.co/datasets/openhcsanyu/2DFlakeSemSeg)
dataset (graphene and MoS2, binary `background` / `Thin-Layer`).

This is a PyTorch-native reimplementation rather than a fork of the authors'
MMSegmentation baseline. Every departure from that baseline is deliberate and
documented below.

## Why not just run the released baseline

The dataset ships the authors' config (`upernet_flash_internimage_b_in1k_768.py`)
and checkpoints. Reading it closely surfaces several choices that are wrong for
the task as the authors themselves score it.

| Baseline | Problem | What this repo does |
|---|---|---|
| Global mean/std normalization (channel std ~15-18 of 255) | Discards ~93% of dynamic range; network must model illumination drift | Per-image optical contrast, `(I - I_bg) / I_bg` |
| `PhotoMetricDistortion` with `hue_delta=9` | Per-channel contrast *is* the label; hue jitter destroys it | Physically-grounded augmentation, no hue mixing |
| `cat_max_ratio=1.0` | Crop rejection disabled, so most 768² crops are pure substrate | Forced foreground sampling (`foreground_prob`) |
| SyncBN at batch size 2 | Normalization statistics estimated from two images | LayerNorm encoder, GroupNorm decoder |
| Unweighted cross entropy | A 100 px flake contributes 0.017% of a 600k px flake's gradient | Area-balanced weights + Tversky + boundary |
| UperNet head at stride 4 | 100 px flakes are 2-3 cells wide at stride 4 | Stride-1 output via a detail branch |
| 32 px overlap on a 544 window | Seams break the connected-component analysis RegionIoU depends on | 25% overlap, cosine-taper blending |
| Checkpoint selection on mIoU | mIoU is dominated by background and a few huge flakes | Small-flake F1 at IoU 0.5 |

## The task is low-contrast detection, not segmentation

A monolayer on SiO2/Si differs from bare substrate by a few percent reflectance.
That difference, resolved per colour channel, is what encodes layer number. The
right input representation is therefore optical contrast against the *local*
substrate, not globally standardized RGB:

```
C = (I - I_bg) / I_bg
```

`I_bg` is estimated per image by a robust low-order polynomial fit that
iteratively discards pixels below the current surface (flakes are darker than
substrate under brightfield). `C` is invariant to illumination gain and exposure,
so a model trained on it transfers across microscopes. The same background
convention is used at training and inference time, including the wider context
window used for the fit, so the two never disagree.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # 34 tests, no data required
```

## Data

```bash
huggingface-cli download openhcsanyu/2DFlakeSemSeg --repo-type dataset --local-dir raw
bash scripts/prepare_data.sh raw data
```

Note that the released `.pth` checkpoints are pickled mmengine artifacts and are
flagged as unsafe on the Hub. This repo does not load them. If you want to
compare against the baseline, load with `weights_only=True` in an isolated
environment and port the tensors explicitly.

### hBN, via MaskTerial

[MaskTerial](https://github.com/Jaluus/MaskTerial) ships per-pixel semantic
masks, not just the COCO/RLE instance annotations, so no rasterization is
needed — only a directory remap:

```bash
curl -L -o Real_hBN_Thin.zip "https://zenodo.org/records/15765514/files/Real_hBN_Thin.zip?download=1"
python -m zipfile -e Real_hBN_Thin.zip raw/
flakeseg-prepare-maskterial --src raw/hBN_Thin --dst data/hBN_Thin --val-fraction 0.2
```

Two of those conversions are load-bearing. MaskTerial masks encode *layer
number*, so feeding them to a 2-logit head is an out-of-bounds class index
rather than a silent degradation; the converter collapses them by default and
`--keep-classes` opts out. And the release has only train and test, so
`--val-fraction` holds out images for threshold calibration — sweeping the
threshold on test would invalidate the number it produces.

hBN is not graphene optically. Its contrast is interference-driven rather than
absorptive, so the sign is not fixed with thickness, and the default background
estimator assumes flakes sit *below* the substrate level. `configs/hbn.yaml`
therefore sets `contrast.polarity: both`. Confirm that choice on real images
before a long run:

```bash
flakeseg-audit --root data/hBN_Thin --split train --polarity both --limit 50
```

## Run the audit first

```bash
flakeseg-audit --root data/graphene --split train2024 --limit 50
```

This reports the flake area distribution against the evaluation buckets, the
contrast-to-noise ratio, and a JPEG blocking score. The images ship as `.jpg`;
if they were compressed after acquisition, chroma subsampling has already
degraded the colour contrast that separates monolayer from bilayer. That caps
achievable performance regardless of model and belongs in a limitations section.

## Train

```bash
python -m flakeseg.train --config configs/graphene.yaml                 # single GPU
torchrun --nproc_per_node=8 -m flakeseg.train --config configs/mos2.yaml # single node
sbatch scripts/train_frontier.sbatch configs/graphene.yaml              # Frontier
```

Effective batch size is `batch_size * accum_steps * world_size`. Because no
normalization layer depends on batch statistics, accumulation genuinely recovers
large-batch optimization rather than merely stabilizing BatchNorm.

## Evaluation

`flakeseg.metrics.MetricAccumulator` reimplements the authors' protocol: per-flake
RegionIoU at IoU 0.5 and 0.75, stratified by ground-truth area at 100, 1k, 10k
and 100k pixels, with background excluded. Matching is greedy and one-to-one.
Boundary IoU and standard mIoU/Dice are reported alongside.

Report the area-stratified table, not a single mIoU. A model that misses every
flake below 1000 px can still post a respectable mIoU on this data.

```bash
flakeseg-evaluate --checkpoint runs/graphene/best.pth --calibrate
flakeseg-evaluate --checkpoint runs/graphene/best.pth --data-root data/MoS2 --split val2024
```

The second form is the cross-material transfer evaluation.

## Suggested experiments

The pieces are factored so the ablation table writes itself:

- **Input representation.** `contrast.method` over `polynomial`, `median`,
  `morphological`, `none`. The `none` arm recovers the baseline's global
  normalization and is the comparison that matters.
- **Loss terms.** Set `tversky_weight` or `boundary_weight` to 0; sweep
  `area_weight_alpha` from 0 (off) to 1 (full equalization).
- **Sampling.** `foreground_prob` at 0.0 reproduces the baseline's effective
  behaviour.
- **Resolution.** `use_aux_head` and `detail_channels`, against a stride-4 head.
- **Cross-material transfer.** Train on graphene, evaluate on MoS2 and back. The
  two subsets share a contrast mechanism with material-specific parameters, which
  makes this a clean domain-shift benchmark that no published baseline reports.
- **Calibration.** `inference.calibrate_threshold` sweeps the decision threshold
  on validation. Tune it there, report it, and never sweep it on test.

## Layout

Anything that imports the package lives *in* the package and is exposed as a
console entry point. `scripts/` holds only what is not importable Python:
shell glue and scheduler submission, which are site-specific and not unit-tested.

```
src/flakeseg/
  data/contrast.py     background estimation, optical contrast
  data/dataset.py      foreground-oversampled crops, area-balanced weights
  data/transforms.py   physically-grounded augmentation
  data/maskterial.py   flakeseg-prepare-maskterial  MaskTerial -> this layout
  models/unet.py       ConvNeXt encoder, detail branch, stride-1 head
  losses.py            weighted CE + Tversky + soft boundary
  metrics.py           RegionIoU, boundary IoU, confusion-matrix metrics
  inference.py         overlapped sliding window, threshold calibration
  train.py             flakeseg-train      DDP training loop
  evaluate.py          flakeseg-evaluate   score a checkpoint, calibrate threshold
  audit.py             flakeseg-audit      run this before training anything
scripts/
  prepare_data.sh      reassemble and unpack the split archives
  train_frontier.sbatch
```

## Known gaps

- No pretrained encoder weights are loaded by default. Set `model.timm_encoder`
  (for example `convnext_tiny`) with `model.encoder_weights` to use timm's
  ImageNet checkpoints; the built-in encoder trains from scratch.
- Full-resolution decode per crop is the throughput bottleneck. If dataloading
  starves the GPUs, pre-tile the images into shards; the dataset already accepts
  a directory of tiles without modification.
- `val_max_images` exists because whole-image sliding-window validation over the
  full split is slow. Use it during development, and set it to 0 for reported numbers.
