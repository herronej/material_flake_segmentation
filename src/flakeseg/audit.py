"""Audit the dataset before training anything.

Run this first. It answers four questions that determine whether the modelling
plan is even valid, and two of them can cap achievable performance regardless of
architecture.

1. JPEG damage. The images ship as .jpg. If they were compressed after
   acquisition, chroma subsampling has already degraded the per-channel colour
   contrast that separates monolayer from bilayer. This script measures the
   noise floor on flat substrate regions and tests for 8x8 blocking. If the
   blocking energy is comparable to the flake signal, that is a hard ceiling and
   belongs in the limitations section of any paper.

2. Signal magnitude. Reports the optical contrast distribution of foreground
   against substrate. This is the number that tells you how hard the task is.

3. Flake area distribution. The evaluation buckets are 100 / 1k / 10k / 100k
   pixels. How many flakes actually fall in each bucket determines which
   buckets carry statistical weight.

4. Class balance and image geometry, which set the sampling and window
   parameters.

Usage:
    flakeseg-audit --root data/graphene --split train2024 --limit 50
    python -m flakeseg.audit --root data/graphene --split train2024 --limit 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .data.contrast import ContrastConfig, estimate_background
from .data.dataset import find_pairs

Image.MAX_IMAGE_PIXELS = None


def blocking_score(channel: np.ndarray) -> float:
    """Ratio of gradient energy on 8x8 block boundaries to off-boundary energy.

    A value near 1.0 means no detectable JPEG blocking. Values above roughly
    1.15 indicate visible block structure at the scale of the flake signal.
    """
    diff = np.abs(np.diff(channel.astype(np.float32), axis=1))
    columns = np.arange(diff.shape[1])
    on_block = diff[:, (columns + 1) % 8 == 0]
    off_block = diff[:, (columns + 1) % 8 != 0]
    if off_block.size == 0 or off_block.mean() == 0:
        return float("nan")
    return float(on_block.mean() / off_block.mean())


def flat_region_noise(channel: np.ndarray, mask: np.ndarray, patch: int = 64) -> float:
    """Standard deviation within background-only patches: the effective noise floor."""
    height, width = channel.shape
    values: list[float] = []
    rng = np.random.default_rng(0)
    for _ in range(64):
        if height <= patch or width <= patch:
            break
        top = int(rng.integers(0, height - patch))
        left = int(rng.integers(0, width - patch))
        sub_mask = mask[top : top + patch, left : left + patch]
        if sub_mask.any():
            continue
        values.append(float(channel[top : top + patch, left : left + patch].std()))
    return float(np.median(values)) if values else float("nan")


def analyze_pair(image_path: Path, mask_path: Path, config: ContrastConfig) -> dict:
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]

    foreground = mask > 0
    background = estimate_background(image, config)
    contrast = (image.astype(np.float32) - background) / np.maximum(background, 1e-3)

    labels, count = ndimage.label(
        foreground, structure=ndimage.generate_binary_structure(2, 2)
    )
    areas = np.bincount(labels.ravel())[1:] if count else np.array([], dtype=np.int64)

    record: dict = {
        "name": image_path.name,
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "foreground_fraction": float(foreground.mean()),
        "n_components": int(count),
        "areas": areas.astype(int).tolist(),
        "raw_mean": [float(image[..., c].mean()) for c in range(3)],
        "raw_std": [float(image[..., c].std()) for c in range(3)],
        "blocking": [float(blocking_score(image[..., c])) for c in range(3)],
        "noise_floor": [
            float(flat_region_noise(image[..., c], foreground)) for c in range(3)
        ],
    }

    if foreground.any():
        record["contrast_fg"] = [
            float(np.median(contrast[..., c][foreground])) for c in range(3)
        ]
        record["contrast_bg_std"] = [
            float(contrast[..., c][~foreground].std()) for c in range(3)
        ]
    return record


def summarize(records: list[dict]) -> dict:
    areas = np.concatenate(
        [np.array(r["areas"], dtype=np.int64) for r in records if r["areas"]]
        or [np.array([], dtype=np.int64)]
    )
    buckets = [100, 1_000, 10_000, 100_000]
    area_counts = {
        f">={b}": int((areas >= b).sum()) for b in buckets
    }

    def stack(key: str) -> np.ndarray:
        rows = [r[key] for r in records if key in r and np.all(np.isfinite(r[key]))]
        return np.array(rows) if rows else np.zeros((0, 3))

    contrast_fg = stack("contrast_fg")
    noise = stack("noise_floor")
    blocking = stack("blocking")

    summary: dict = {
        "n_images": len(records),
        "image_shape_median": [
            int(np.median([r["height"] for r in records])),
            int(np.median([r["width"] for r in records])),
        ],
        "foreground_fraction_mean": float(
            np.mean([r["foreground_fraction"] for r in records])
        ),
        "n_components_total": int(sum(r["n_components"] for r in records)),
        "area_bucket_counts": area_counts,
        "area_percentiles": {
            f"p{p}": float(np.percentile(areas, p)) for p in (5, 25, 50, 75, 95)
        }
        if areas.size
        else {},
        "raw_std_mean": np.array([r["raw_std"] for r in records]).mean(axis=0).tolist(),
        "blocking_mean": blocking.mean(axis=0).tolist() if blocking.size else [],
        "noise_floor_mean": noise.mean(axis=0).tolist() if noise.size else [],
        "contrast_fg_median": np.median(contrast_fg, axis=0).tolist()
        if contrast_fg.size
        else [],
    }

    verdicts: list[str] = []
    if blocking.size and float(blocking.mean()) > 1.15:
        verdicts.append(
            f"JPEG blocking detected (score {blocking.mean():.2f}). Compression "
            "artifacts are at the scale of the flake signal; treat this as a "
            "performance ceiling and report it."
        )
    if contrast_fg.size and noise.size:
        signal = float(np.abs(np.median(contrast_fg))) * float(
            np.mean([r["raw_mean"][1] for r in records])
        )
        snr = signal / max(float(noise.mean()), 1e-6)
        summary["approx_snr"] = float(snr)
        if snr < 3.0:
            verdicts.append(
                f"Low contrast-to-noise ratio (~{snr:.1f}). Denoising or "
                "multi-scale aggregation will matter more than backbone choice."
            )
    if area_counts[">=100"] and area_counts[">=10000"] / max(area_counts[">=100"], 1) < 0.05:
        verdicts.append(
            "The flake population is dominated by small components. Area-balanced "
            "loss weighting and foreground oversampling are load-bearing here."
        )
    summary["verdicts"] = verdicts
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="material directory, e.g. data/graphene")
    parser.add_argument("--split", default="train2024")
    parser.add_argument("--limit", type=int, default=50, help="0 for all images")
    parser.add_argument("--out", default=None, help="write full JSON report here")
    parser.add_argument(
        "--polarity",
        default="dark",
        choices=("dark", "bright", "both"),
        help=(
            "which side of the substrate flakes sit on during background estimation; "
            "'dark' suits graphene and the TMDs, 'both' suits hBN"
        ),
    )
    args = parser.parse_args()

    pairs = find_pairs(Path(args.root), args.split)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    config = ContrastConfig(polarity=args.polarity)
    records = []
    for i, (image_path, mask_path) in enumerate(pairs):
        records.append(analyze_pair(image_path, mask_path, config))
        print(f"  [{i + 1}/{len(pairs)}] {image_path.name}", flush=True)

    summary = summarize(records)
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "verdicts"}, indent=2))
    if summary["verdicts"]:
        print("\nFindings:")
        for verdict in summary["verdicts"]:
            print(f"  - {verdict}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"summary": summary, "records": records}, indent=2)
        )
        print(f"\nfull report written to {args.out}")


if __name__ == "__main__":
    main()
