"""Dataset and crop sampling for 2D flake semantic segmentation.

Two defects in the published baseline are addressed here.

1. `cat_max_ratio=1.0` disables crop rejection, so with flakes covering a small
   fraction of each field of view the overwhelming majority of 768x768 crops
   contain no foreground at all. The model spends most of its gradient budget
   confirming that the substrate is substrate. `FlakeCropDataset` instead does
   nnU-Net style forced foreground sampling: a configurable fraction of crops
   are centred on a foreground pixel.

2. The evaluation protocol scores per-flake RegionIoU with area filters down to
   100 pixels, but the training loss is unweighted cross entropy, under which a
   100-pixel flake contributes 0.017% of the gradient of a 600k-pixel one. Each
   sample therefore carries an area-balanced weight map, built from connected
   components of the crop. Components touching the crop border are left at unit
   weight, since their true area is unknown and upweighting a truncated large
   flake would be actively wrong.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

from .contrast import ContrastConfig, optical_contrast
from .transforms import AugmentConfig, FlakeAugment

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # these are whole-chip mosaics, not web images

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass(frozen=True)
class SampleConfig:
    """Crop sampling and weighting parameters."""

    crop_size: int = 768
    foreground_prob: float = 0.4
    context_factor: float = 1.5
    area_weight_alpha: float = 0.5
    area_weight_reference: float = 10_000.0
    area_weight_max: float = 8.0
    ignore_index: int = 255


def find_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    """Match images in `<root>/<split>` to masks in `<root>/annotations_semseg/<split>`."""
    image_dir = root / split
    mask_dir = root / "annotations_semseg" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"mask directory not found: {mask_dir}")

    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() == ".png"}
    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask_path = masks.get(image_path.stem)
        if mask_path is None:
            missing += 1
            continue
        pairs.append((image_path, mask_path))

    if missing:
        logger.warning("%d image(s) in %s had no matching mask", missing, image_dir)
    if not pairs:
        raise RuntimeError(f"no image/mask pairs found under {root} split={split}")
    return pairs


def build_foreground_index(
    pairs: Sequence[tuple[Path, Path]],
    cache_path: Path | None = None,
    max_points_per_image: int = 4096,
    stride: int = 4,
) -> list[np.ndarray]:
    """Collect candidate foreground coordinates for each mask.

    Scanning every mask once up front is far cheaper than rejection-sampling
    crops during training. Coordinates are subsampled on a stride grid and then
    randomly thinned, which is sufficient because the crop is large relative to
    the grid spacing.

    Returns
    -------
    List of (N, 2) int32 arrays of (row, col) coordinates, one per pair.
    Images with no foreground yield an empty array.
    """
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            stored = [data[f"arr_{i}"] for i in range(len(data.files))]
        if len(stored) == len(pairs):
            logger.info("loaded foreground index from %s", cache_path)
            return stored
        logger.warning(
            "foreground cache at %s has %d entries but %d pairs were found; rebuilding",
            cache_path,
            len(stored),
            len(pairs),
        )

    rng = np.random.default_rng(0)
    index: list[np.ndarray] = []
    for i, (_, mask_path) in enumerate(pairs):
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        sub = mask[::stride, ::stride]
        rows, cols = np.nonzero(sub > 0)
        if rows.size > max_points_per_image:
            keep = rng.choice(rows.size, size=max_points_per_image, replace=False)
            rows, cols = rows[keep], cols[keep]
        index.append(np.stack([rows * stride, cols * stride], axis=1).astype(np.int32))
        if (i + 1) % 200 == 0:
            logger.info("indexed %d/%d masks", i + 1, len(pairs))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, *index)
        logger.info("wrote foreground index to %s", cache_path)
    return index


def area_balanced_weights(
    mask: np.ndarray,
    alpha: float,
    reference: float,
    max_weight: float,
) -> np.ndarray:
    """Build a per-pixel weight map that equalizes small and large flakes.

    Weight for a foreground component of area A is
    `clip((reference / A) ** alpha, 1, max_weight)`. Background is unit weight.
    Components touching the crop border keep unit weight because their true
    area extends outside the crop and cannot be measured here.

    Parameters
    ----------
    mask:
        HxW integer array, nonzero values are foreground.
    alpha:
        Exponent. 0 disables balancing, 1 fully equalizes total component weight.
    reference:
        Component area, in pixels, that receives unit weight.
    max_weight:
        Upper clip, guarding against a handful of tiny components dominating.

    Returns
    -------
    HxW float32 weight map.
    """
    weights = np.ones(mask.shape, dtype=np.float32)
    if alpha <= 0.0:
        return weights

    labels, count = ndimage.label(mask > 0)
    if count == 0:
        return weights

    areas = np.bincount(labels.ravel(), minlength=count + 1).astype(np.float64)

    border_ids = np.unique(
        np.concatenate(
            [labels[0, :].ravel(), labels[-1, :].ravel(),
             labels[:, 0].ravel(), labels[:, -1].ravel()]
        )
    )
    truncated = np.zeros(count + 1, dtype=bool)
    truncated[border_ids[border_ids > 0]] = True

    lookup = np.ones(count + 1, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (reference / np.maximum(areas, 1.0)) ** alpha
    value = np.clip(value, 1.0, max_weight).astype(np.float32)
    valid = ~truncated
    valid[0] = False
    lookup[valid] = value[valid]

    return lookup[labels]


class FlakeCropDataset(Dataset):
    """Random-crop training dataset over full-resolution flake micrographs.

    Each item is a dict with keys `image` (CxHxW float32), `mask` (HxW int64),
    and `weight` (HxW float32).

    Notes
    -----
    Background estimation is performed on a context region larger than the
    output crop (`context_factor`), then the centre is taken. This prevents a
    crop that happens to be fully covered by one large flake from having its own
    flake fitted as the substrate level. The same convention is used at
    inference time by `flakeseg.inference.sliding_window_predict`, so train and
    test see identically normalized inputs.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train2024",
        sample: SampleConfig | None = None,
        contrast: ContrastConfig | None = None,
        augment: AugmentConfig | None = None,
        length: int | None = None,
        cache_dir: str | Path | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sample = sample or SampleConfig()
        self.contrast = contrast or ContrastConfig()
        self.pairs = find_pairs(self.root, split)

        cache = Path(cache_dir) if cache_dir is not None else self.root / ".flakeseg"
        self.fg_index = build_foreground_index(
            self.pairs, cache_path=cache / f"fg_{split}.npz"
        )

        self.augment = FlakeAugment(augment) if augment is not None else None
        self._length = length if length is not None else len(self.pairs)
        self._base_seed = seed
        self._rng_instance: np.random.Generator | None = None

    @property
    def _rng(self) -> np.random.Generator:
        """Lazily create the generator inside whichever worker process uses it.

        `numpy.random.default_rng(seed)` built in `__init__` is forked into every
        dataloader worker with identical internal state, so all `num_workers`
        processes draw the same crop coordinates. That silently divides effective
        crop diversity by the worker count, and it is invisible in the loss curve.
        `torch.initial_seed()` is already varied by PyTorch per worker and per
        epoch, so seeding from it gives independent streams for free.
        """
        if self._rng_instance is None:
            import torch

            info = torch.utils.data.get_worker_info()
            if info is None:
                seed = self._base_seed
            else:
                seed = int(torch.initial_seed() % (2**31)) + info.id
            self._rng_instance = np.random.default_rng(seed)
        return self._rng_instance

    def __len__(self) -> int:
        return self._length

    def _read_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.pairs[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"shape mismatch for {image_path.name}: "
                f"image {image.shape[:2]} vs mask {mask.shape}"
            )
        return image, mask

    def _choose_origin(
        self, idx: int, height: int, width: int, size: int
    ) -> tuple[int, int]:
        """Pick the top-left corner of a crop, biased toward foreground."""
        points = self.fg_index[idx]
        use_fg = points.size > 0 and self._rng.random() < self.sample.foreground_prob
        if use_fg:
            row, col = points[self._rng.integers(0, len(points))]
            jitter = size // 4
            top = int(row) - size // 2 + int(self._rng.integers(-jitter, jitter + 1))
            left = int(col) - size // 2 + int(self._rng.integers(-jitter, jitter + 1))
        else:
            top = int(self._rng.integers(0, max(height - size, 0) + 1))
            left = int(self._rng.integers(0, max(width - size, 0) + 1))
        top = int(np.clip(top, 0, max(height - size, 0)))
        left = int(np.clip(left, 0, max(width - size, 0)))
        return top, left

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        idx = int(self._rng.integers(0, len(self.pairs))) if self._length != len(
            self.pairs
        ) else index
        image, mask = self._read_pair(idx)
        height, width = mask.shape
        size = self.sample.crop_size

        if height < size or width < size:
            pad_h, pad_w = max(0, size - height), max(0, size - width)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(
                mask, ((0, pad_h), (0, pad_w)),
                mode="constant", constant_values=self.sample.ignore_index,
            )
            height, width = mask.shape

        top, left = self._choose_origin(idx, height, width, size)

        # Read a larger context window for background estimation, then centre-crop.
        context = int(round(size * self.sample.context_factor))
        pad = (context - size) // 2
        c_top = int(np.clip(top - pad, 0, max(height - context, 0)))
        c_left = int(np.clip(left - pad, 0, max(width - context, 0)))
        c_bot = min(c_top + context, height)
        c_right = min(c_left + context, width)

        context_image = image[c_top:c_bot, c_left:c_right]
        contrast_full = optical_contrast(context_image, self.contrast)

        off_r, off_c = top - c_top, left - c_left
        crop_image = contrast_full[off_r : off_r + size, off_c : off_c + size]
        crop_mask = mask[top : top + size, left : left + size]

        if self.augment is not None:
            crop_image, crop_mask = self.augment(crop_image, crop_mask)

        valid = crop_mask != self.sample.ignore_index
        weight = area_balanced_weights(
            np.where(valid, crop_mask, 0),
            self.sample.area_weight_alpha,
            self.sample.area_weight_reference,
            self.sample.area_weight_max,
        )
        weight = np.where(valid, weight, 0.0).astype(np.float32)

        return {
            "image": np.ascontiguousarray(crop_image.transpose(2, 0, 1), dtype=np.float32),
            "mask": np.ascontiguousarray(crop_mask, dtype=np.int64),
            "weight": np.ascontiguousarray(weight, dtype=np.float32),
        }


class FlakeEvalDataset(Dataset):
    """Whole-image evaluation dataset. Yields raw images; normalization happens
    per sliding window inside `flakeseg.inference` so that it matches training."""

    def __init__(self, root: str | Path, split: str = "val2024") -> None:
        self.root = Path(root)
        self.pairs = find_pairs(self.root, split)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        image_path, mask_path = self.pairs[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        return {"image": image, "mask": mask, "name": image_path.stem}


def write_split_manifest(root: str | Path, out: str | Path) -> None:
    """Record which files were used, so a run can be reproduced exactly."""
    root = Path(root)
    manifest = {
        split: [p.name for p, _ in find_pairs(root, split)]
        for split in ("train2024", "val2024")
    }
    Path(out).write_text(json.dumps(manifest, indent=2))
