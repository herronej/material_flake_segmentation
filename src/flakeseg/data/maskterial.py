"""Convert a MaskTerial dataset release into the layout `find_pairs` expects.

MaskTerial (https://github.com/Jaluus/MaskTerial, data at zenodo.org/records/15765514)
ships per-pixel semantic masks alongside its COCO/RLE instance annotations, so no
polygon rasterization is needed. Its on-disk layout is

    <material>/
      train_images/          test_images/
      train_semantic_masks/  test_semantic_masks/
      RLE_annotations/       meta_data/

and this repo wants

    <root>/<split>/                       images
    <root>/annotations_semseg/<split>/    masks

Two conversions are not cosmetic.

1. MaskTerial masks encode *layer number* (1, 2, 3, ... for mono-, bi-, tri-layer),
   while `model.num_classes` here is 2. Feeding raw layer ids to a 2-logit head
   is an out-of-bounds class index, not a silent degradation, so `--binary`
   (the default) collapses every positive layer id to 1. `--keep-classes`
   preserves them, in which case `num_classes` must be set to one more than the
   largest id reported below.

2. There is no validation split, only train and test. `flakeseg-evaluate
   --calibrate` sweeps the decision threshold, and doing that on test would
   invalidate the reported number. `--val-fraction` deterministically holds out
   part of train so calibration has somewhere honest to live.

Images are symlinked rather than copied by default; a MaskTerial release is
hundreds of MB to tens of GB and duplicating it on a parallel filesystem is
pure waste.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# MaskTerial's own split names, mapped to the directory pair that holds them.
SOURCE_SPLITS = {
    "train": ("train_images", "train_semantic_masks"),
    "test": ("test_images", "test_semantic_masks"),
}


@dataclass
class SplitReport:
    """What was written for one output split."""

    name: str
    pairs: int = 0
    class_histogram: Counter = field(default_factory=Counter)
    skipped_no_mask: list[str] = field(default_factory=list)
    skipped_shape_mismatch: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "split": self.name,
            "pairs": self.pairs,
            "class_histogram": {str(k): int(v) for k, v in sorted(self.class_histogram.items())},
            "skipped_no_mask": self.skipped_no_mask,
            "skipped_shape_mismatch": self.skipped_shape_mismatch,
        }


def read_mask(path: Path) -> np.ndarray:
    """Read a semantic mask as a 2D array of class ids.

    Palette PNGs decode straight to indices. A mask stored as RGB is only
    unambiguous if every channel carries the same values; a genuine colour-coded
    mask needs an explicit palette mapping, which we refuse to guess at.
    """
    array = np.array(Image.open(path))
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        channels = array[..., : min(3, array.shape[-1])]
        if np.all(channels == channels[..., :1]):
            return channels[..., 0]
        raise ValueError(
            f"{path} is a colour-coded mask with differing channels; "
            "this converter expects palette or greyscale class ids"
        )
    raise ValueError(f"{path}: unexpected mask shape {array.shape}")


def index_masks(mask_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() == ".png"}


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    """Materialize `src` at `dst`, replacing whatever is already there."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if copy:
        dst.write_bytes(src.read_bytes())
    else:
        dst.symlink_to(src.resolve())


def convert_split(
    images: list[Path],
    masks: dict[str, Path],
    dst_root: Path,
    split: str,
    binary: bool,
    copy: bool,
) -> SplitReport:
    """Write one output split, returning what happened."""
    image_out = dst_root / split
    mask_out = dst_root / "annotations_semseg" / split
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    report = SplitReport(name=split)
    for image_path in images:
        mask_path = masks.get(image_path.stem)
        if mask_path is None:
            report.skipped_no_mask.append(image_path.name)
            continue

        mask = read_mask(mask_path)
        with Image.open(image_path) as handle:
            width, height = handle.size
        if mask.shape != (height, width):
            report.skipped_shape_mismatch.append(image_path.name)
            continue

        values, counts = np.unique(mask, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist(), strict=True):
            report.class_histogram[int(value)] += int(count)

        out = (mask > 0).astype(np.uint8) if binary else mask.astype(np.uint8)
        Image.fromarray(out).save(mask_out / f"{image_path.stem}.png")
        link_or_copy(image_path, image_out / image_path.name, copy)
        report.pairs += 1

    return report


def split_train_val(
    images: list[Path], val_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Deterministically partition training images into train and val.

    Held out by image, never by crop: two crops of one micrograph share
    illumination, substrate and often the same flake, so a crop-level split
    would leak straight across.
    """
    if val_fraction <= 0.0:
        return images, []
    count = int(round(len(images) * val_fraction))
    if count == 0:
        logger.warning(
            "val_fraction=%.3f rounds to 0 images out of %d; no val split written",
            val_fraction,
            len(images),
        )
        return images, []
    if count >= len(images):
        raise ValueError(
            f"val_fraction={val_fraction} would consume all {len(images)} training images"
        )

    order = np.random.default_rng(seed).permutation(len(images))
    val_idx = sorted(order[:count].tolist())
    val_set = {images[i] for i in val_idx}
    train = [p for p in images if p not in val_set]
    val = [images[i] for i in val_idx]
    return train, val


def prepare(
    src: Path,
    dst: Path,
    binary: bool = True,
    val_fraction: float = 0.0,
    seed: int = 0,
    copy: bool = False,
) -> dict:
    """Convert a MaskTerial material directory into a flakeseg data root."""
    missing = [
        name
        for pair in SOURCE_SPLITS.values()
        for name in pair
        if not (src / name).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            f"{src} does not look like a MaskTerial release; missing: {', '.join(missing)}"
        )

    dst.mkdir(parents=True, exist_ok=True)
    reports: list[SplitReport] = []

    train_images = list_images(src / SOURCE_SPLITS["train"][0])
    train_only, val_only = split_train_val(train_images, val_fraction, seed)
    train_masks = index_masks(src / SOURCE_SPLITS["train"][1])

    reports.append(convert_split(train_only, train_masks, dst, "train", binary, copy))
    if val_only:
        reports.append(convert_split(val_only, train_masks, dst, "val", binary, copy))

    test_images = list_images(src / SOURCE_SPLITS["test"][0])
    test_masks = index_masks(src / SOURCE_SPLITS["test"][1])
    reports.append(convert_split(test_images, test_masks, dst, "test", binary, copy))

    manifest = {
        "source": str(src.resolve()),
        "binary": binary,
        "val_fraction": val_fraction,
        "seed": seed,
        "splits": [r.as_dict() for r in reports],
    }
    (dst / "maskterial_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def summarize(manifest: dict) -> str:
    """Human-readable summary, including the num_classes the result implies."""
    lines = []
    max_class = 0
    for split in manifest["splits"]:
        histogram = {int(k): v for k, v in split["class_histogram"].items()}
        total = sum(histogram.values())
        foreground = sum(v for k, v in histogram.items() if k > 0)
        fraction = (foreground / total * 100.0) if total else 0.0
        max_class = max([max_class, *histogram.keys()])
        lines.append(
            f"  {split['split']:>5}: {split['pairs']:5d} pairs  "
            f"classes={sorted(histogram)}  foreground={fraction:.3f}%"
        )
        for reason in ("skipped_no_mask", "skipped_shape_mismatch"):
            if split[reason]:
                lines.append(f"         {reason}: {len(split[reason])} -> {split[reason][:5]}")

    lines.append("")
    if manifest["binary"]:
        lines.append("  binary masks written; keep model.num_classes: 2")
    else:
        lines.append(f"  layer ids preserved; set model.num_classes: {max_class + 1}")
    if not any(s["split"] == "val" for s in manifest["splits"]):
        lines.append(
            "  no val split: pass --val-fraction so threshold calibration "
            "does not touch test"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a MaskTerial release into the flakeseg data layout."
    )
    parser.add_argument("--src", required=True, type=Path, help="unpacked material directory")
    parser.add_argument("--dst", required=True, type=Path, help="output data root")
    parser.add_argument(
        "--keep-classes",
        action="store_true",
        help="preserve layer ids instead of collapsing them to a single foreground class",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="fraction of the training images to hold out as a val split",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the train/val partition")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copy images instead of symlinking them",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = prepare(
        src=args.src,
        dst=args.dst,
        binary=not args.keep_classes,
        val_fraction=args.val_fraction,
        seed=args.seed,
        copy=args.copy,
    )
    print(f"wrote {args.dst}")
    print(summarize(manifest))


if __name__ == "__main__":
    main()
