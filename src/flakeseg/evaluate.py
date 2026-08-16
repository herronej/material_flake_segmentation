"""Evaluate a trained checkpoint against the RegionIoU protocol.

Usage:
    flakeseg-evaluate --checkpoint runs/graphene/best.pth --data-root data/graphene
    python -m flakeseg.evaluate --checkpoint runs/graphene/best.pth --calibrate

Threshold calibration
---------------------
The 0.5 argmax threshold is rarely optimal for a recall-weighted metric on data
this imbalanced, and sweeping it is close to free. `--calibrate` sweeps on the
split given by `--split` and reports the chosen value. Sweep on validation and
report the number; sweeping on the split you report is a leak.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data.contrast import ContrastConfig
from .data.dataset import FlakeEvalDataset
from .inference import (
    InferenceConfig,
    calibrate_threshold,
    probs_to_mask,
    sliding_window_predict,
)
from .metrics import MetricAccumulator, region_iou
from .models.unet import ModelConfig, build_model


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load a checkpoint written by `flakeseg.train`.

    `weights_only=True` is attempted first. Our own checkpoints contain only
    tensors and plain containers, so it should succeed; the fallback exists for
    checkpoints produced by older versions of this code, and should not be used
    on files from an untrusted source.
    """
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        payload = torch.load(path, map_location=device, weights_only=False)

    config = payload.get("config", {})
    model = build_model(ModelConfig(**config.get("model", {})))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, config


def predict_split(
    model: torch.nn.Module,
    dataset: FlakeEvalDataset,
    inference: InferenceConfig,
    contrast: ContrastConfig,
    device: torch.device,
    limit: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Run sliding-window inference over a split, returning probabilities."""
    total = len(dataset) if limit <= 0 else min(limit, len(dataset))
    probabilities, targets, names = [], [], []
    for i in range(total):
        item = dataset[i]
        probs = sliding_window_predict(
            model, item["image"], inference, contrast, device=device
        )
        probabilities.append(probs.astype(np.float32))
        targets.append(item["mask"])
        names.append(item["name"])
        print(f"  [{i + 1}/{total}] {item['name']}", flush=True)
    return probabilities, targets, names


def small_flake_f1(pred: np.ndarray, target: np.ndarray) -> float:
    """Objective used for threshold calibration: the hardest scored bucket."""
    result = region_iou(pred, target, thresholds=(0.5,), area_filters=(100,))[(100, 0.5)]
    value = result.f1
    return 0.0 if not np.isfinite(value) else float(value)


def format_table(accumulator: MetricAccumulator) -> str:
    """Render the area-stratified table that should appear in a paper."""
    lines = [
        f"{'area >=':>10} {'IoU':>6} {'n_gt':>7} {'recall':>8} {'prec':>8} {'F1':>8}",
        "-" * 52,
    ]
    for area in accumulator.area_filters:
        for threshold in accumulator.thresholds:
            r = accumulator.regions[(area, threshold)]
            fmt = lambda v: "  n/a  " if not np.isfinite(v) else f"{v:7.4f}"  # noqa: E731
            lines.append(
                f"{area:>10} {threshold:>6.2f} {r.n_gt:>7} "
                f"{fmt(r.recall)} {fmt(r.precision)} {fmt(r.f1)}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None, help="defaults to the training value")
    parser.add_argument("--split", default="val2024")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--calibrate", action="store_true", help="sweep the threshold")
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--flip-tta", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 for the whole split")
    parser.add_argument("--save-masks", default=None, help="directory for prediction PNGs")
    parser.add_argument("--out", default=None, help="write the metric JSON here")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)

    data_root = args.data_root or config.get("data_root")
    if data_root is None:
        parser.error("--data-root is required when the checkpoint has no stored config")

    inference = InferenceConfig(**config.get("inference", {}))
    inference.flip_tta = args.flip_tta
    inference.min_component_area = args.min_area
    contrast = ContrastConfig(**config.get("contrast", {}))

    dataset = FlakeEvalDataset(data_root, args.split)
    print(f"evaluating {args.checkpoint} on {data_root}:{args.split}")
    probabilities, targets, names = predict_split(
        model, dataset, inference, contrast, device, args.limit
    )

    threshold = args.threshold
    if args.calibrate:
        threshold, score = calibrate_threshold(
            probabilities, targets, small_flake_f1
        )
        print(f"\ncalibrated threshold {threshold:.3f} (small-flake F1 {score:.4f})")

    accumulator = MetricAccumulator()
    predictions = []
    for probs, target in zip(probabilities, targets, strict=True):
        mask = probs_to_mask(probs, threshold, args.min_area)
        predictions.append(mask)
        accumulator.update(mask, target)

    summary = accumulator.summary()
    summary["threshold"] = threshold
    summary["n_images"] = len(predictions)

    print("\n" + format_table(accumulator))
    print(
        f"\nmIoU {summary['mIoU']:.4f}   flake IoU {summary['IoU_flake']:.4f}   "
        f"boundary IoU {summary['boundary_IoU']:.4f}"
    )

    if args.save_masks:
        from PIL import Image

        out_dir = Path(args.save_masks)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, mask in zip(names, predictions, strict=True):
            Image.fromarray(mask.astype(np.uint8)).save(out_dir / f"{name}.png")
        print(f"masks written to {out_dir}")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"metrics written to {args.out}")


if __name__ == "__main__":
    main()
