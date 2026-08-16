"""Evaluation metrics for flake segmentation.

The dataset authors do not select on mIoU alone. Their config scores a custom
`RegionIoU` at IoU thresholds 0.5 and 0.75, with area filters at 100, 1000,
10000 and 100000 pixels and the background class excluded. That is a per-flake
detection metric stratified by flake size, and it is the right thing to
optimize: mIoU on this data is dominated by background and by a handful of very
large flakes, so a model that misses every small flake can still post a
respectable mIoU.

This module reimplements that protocol, plus boundary IoU, which matters
because flake area feeds directly into downstream thickness and yield
estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class RegionResult:
    """Per-flake detection scores within one area bucket at one IoU threshold."""

    area_filter: int
    threshold: float
    n_gt: int = 0
    n_pred: int = 0
    n_matched: int = 0
    iou_sum: float = 0.0

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_gt if self.n_gt else float("nan")

    @property
    def precision(self) -> float:
        return self.n_matched / self.n_pred if self.n_pred else float("nan")

    @property
    def f1(self) -> float:
        r, p = self.recall, self.precision
        if not np.isfinite(r) or not np.isfinite(p) or (r + p) == 0:
            return float("nan")
        return 2 * r * p / (r + p)

    @property
    def mean_matched_iou(self) -> float:
        return self.iou_sum / self.n_matched if self.n_matched else float("nan")


def sparse_component_iou(
    gt_labels: np.ndarray,
    n_gt: int,
    pred_labels: np.ndarray,
    n_pred: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute IoU only for component pairs that actually overlap.

    Building the dense (n_gt+1) x (n_pred+1) contingency table is wasteful when
    a field of view holds thousands of flakes. Only overlapping pairs can have
    nonzero IoU, so those are enumerated directly.

    Returns
    -------
    gt_idx, pred_idx, iou:
        Parallel arrays over overlapping pairs, 1-based component ids.
    gt_areas, pred_areas:
        1-based area arrays of length n_gt+1 and n_pred+1 (index 0 is background).
    """
    gt_areas = np.bincount(gt_labels.ravel(), minlength=n_gt + 1).astype(np.int64)
    pred_areas = np.bincount(pred_labels.ravel(), minlength=n_pred + 1).astype(np.int64)

    overlap = (gt_labels > 0) & (pred_labels > 0)
    if not overlap.any():
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=np.float64), gt_areas, pred_areas

    key = gt_labels[overlap].astype(np.int64) * (n_pred + 1) + pred_labels[
        overlap
    ].astype(np.int64)
    uniq, inter = np.unique(key, return_counts=True)
    gt_idx = uniq // (n_pred + 1)
    pred_idx = uniq % (n_pred + 1)

    union = gt_areas[gt_idx] + pred_areas[pred_idx] - inter
    iou = inter.astype(np.float64) / np.maximum(union, 1)
    return gt_idx, pred_idx, iou, gt_areas, pred_areas


def region_iou(
    pred: np.ndarray,
    target: np.ndarray,
    thresholds: tuple[float, ...] = (0.5, 0.75),
    area_filters: tuple[int, ...] = (100, 1_000, 10_000, 100_000),
    connectivity: int = 2,
    ignore_index: int = 255,
) -> dict[tuple[int, float], RegionResult]:
    """Per-flake detection scores, stratified by ground-truth flake area.

    Matching is greedy and one-to-one: pairs are considered in descending IoU
    order and a component may be used at most once. A ground-truth flake counts
    as detected if its matched prediction reaches the IoU threshold.

    Parameters
    ----------
    pred, target:
        HxW integer arrays. Nonzero is foreground. Pixels equal to
        `ignore_index` in `target` are excluded from both.
    thresholds:
        IoU thresholds at which to score detection.
    area_filters:
        Minimum ground-truth component area, in pixels, for each bucket.
    connectivity:
        1 for 4-connectivity, 2 for 8-connectivity.

    Returns
    -------
    Mapping from (area_filter, threshold) to `RegionResult`.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")

    valid = target != ignore_index
    gt_fg = (target > 0) & valid
    pred_fg = (pred > 0) & valid

    structure = ndimage.generate_binary_structure(2, connectivity)
    gt_labels, n_gt = ndimage.label(gt_fg, structure=structure)
    pred_labels, n_pred = ndimage.label(pred_fg, structure=structure)

    gt_idx, pred_idx, iou, gt_areas, pred_areas = sparse_component_iou(
        gt_labels, n_gt, pred_labels, n_pred
    )

    order = np.argsort(-iou)
    results: dict[tuple[int, float], RegionResult] = {}

    for threshold in thresholds:
        gt_taken = np.zeros(n_gt + 1, dtype=bool)
        pred_taken = np.zeros(n_pred + 1, dtype=bool)
        matched_iou = np.zeros(n_gt + 1, dtype=np.float64)

        for k in order:
            if iou[k] < threshold:
                break
            g, p = int(gt_idx[k]), int(pred_idx[k])
            if gt_taken[g] or pred_taken[p]:
                continue
            gt_taken[g] = pred_taken[p] = True
            matched_iou[g] = iou[k]

        for area_filter in area_filters:
            gt_keep = np.zeros(n_gt + 1, dtype=bool)
            gt_keep[1:] = gt_areas[1:] >= area_filter
            pred_keep = np.zeros(n_pred + 1, dtype=bool)
            pred_keep[1:] = pred_areas[1:] >= area_filter

            result = RegionResult(area_filter=area_filter, threshold=threshold)
            result.n_gt = int(gt_keep.sum())
            result.n_pred = int(pred_keep.sum())
            result.n_matched = int((gt_taken & gt_keep).sum())
            result.iou_sum = float(matched_iou[gt_taken & gt_keep].sum())
            results[(area_filter, threshold)] = result

    return results


def confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 2,
    ignore_index: int = 255,
) -> np.ndarray:
    """Accumulate a `num_classes` x `num_classes` confusion matrix."""
    valid = target != ignore_index
    p = pred[valid].astype(np.int64)
    t = target[valid].astype(np.int64)
    np.clip(p, 0, num_classes - 1, out=p)
    np.clip(t, 0, num_classes - 1, out=t)
    return np.bincount(
        t * num_classes + p, minlength=num_classes**2
    ).reshape(num_classes, num_classes)


def iou_from_confusion(matrix: np.ndarray) -> np.ndarray:
    """Per-class IoU from a confusion matrix with rows as ground truth."""
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, np.nan)


def dice_from_confusion(matrix: np.ndarray) -> np.ndarray:
    """Per-class Dice from a confusion matrix with rows as ground truth."""
    intersection = np.diag(matrix).astype(np.float64)
    denom = matrix.sum(axis=1) + matrix.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, 2.0 * intersection / denom, np.nan)


def _boundary_region(mask: np.ndarray, dilation: int) -> np.ndarray:
    """Pixels of `mask` within `dilation` of its boundary."""
    if dilation < 1:
        return mask.copy()
    padded = np.pad(mask, dilation, mode="constant", constant_values=0)
    eroded = ndimage.binary_erosion(
        padded, structure=ndimage.generate_binary_structure(2, 1), iterations=dilation
    )
    eroded = eroded[dilation:-dilation, dilation:-dilation]
    return mask & ~eroded


def boundary_iou(
    pred: np.ndarray,
    target: np.ndarray,
    dilation_ratio: float = 0.02,
    ignore_index: int = 255,
) -> float:
    """Boundary IoU (Cheng et al.), sensitive to edge placement rather than bulk area.

    `dilation_ratio` is expressed as a fraction of the image diagonal, following
    the original definition.
    """
    valid = target != ignore_index
    gt = (target > 0) & valid
    pr = (pred > 0) & valid

    diagonal = float(np.sqrt(target.shape[0] ** 2 + target.shape[1] ** 2))
    dilation = max(1, int(round(dilation_ratio * diagonal)))

    gt_band = _boundary_region(gt, dilation)
    pr_band = _boundary_region(pr, dilation)

    intersection = np.count_nonzero(gt_band & pr_band)
    union = np.count_nonzero(gt_band | pr_band)
    return intersection / union if union else float("nan")


@dataclass
class MetricAccumulator:
    """Accumulates all metrics across a validation set."""

    num_classes: int = 2
    thresholds: tuple[float, ...] = (0.5, 0.75)
    area_filters: tuple[int, ...] = (100, 1_000, 10_000, 100_000)
    ignore_index: int = 255

    matrix: np.ndarray = field(init=False)
    regions: dict[tuple[int, float], RegionResult] = field(init=False)
    boundary_ious: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.regions = {
            (a, t): RegionResult(area_filter=a, threshold=t)
            for a in self.area_filters
            for t in self.thresholds
        }
        self.boundary_ious = []

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        self.matrix += confusion_matrix(
            pred, target, self.num_classes, self.ignore_index
        )
        per_image = region_iou(
            pred,
            target,
            thresholds=self.thresholds,
            area_filters=self.area_filters,
            ignore_index=self.ignore_index,
        )
        for key, value in per_image.items():
            acc = self.regions[key]
            acc.n_gt += value.n_gt
            acc.n_pred += value.n_pred
            acc.n_matched += value.n_matched
            acc.iou_sum += value.iou_sum
        self.boundary_ious.append(boundary_iou(pred, target, ignore_index=self.ignore_index))

    def summary(self) -> dict[str, float]:
        """Flatten to a scalar dict suitable for logging."""
        ious = iou_from_confusion(self.matrix)
        dices = dice_from_confusion(self.matrix)
        out: dict[str, float] = {
            "mIoU": float(np.nanmean(ious)),
            "mDice": float(np.nanmean(dices)),
            "IoU_flake": float(ious[1]) if self.num_classes > 1 else float("nan"),
            "boundary_IoU": float(np.nanmean(self.boundary_ious))
            if self.boundary_ious
            else float("nan"),
        }
        for (area, threshold), result in sorted(self.regions.items()):
            tag = f"a{area}_t{threshold:g}"
            out[f"RIoU_recall_{tag}"] = result.recall
            out[f"RIoU_precision_{tag}"] = result.precision
            out[f"RIoU_f1_{tag}"] = result.f1
        return out

    @property
    def selection_metric(self) -> float:
        """Checkpoint selection criterion.

        Small-flake recall at IoU 0.5 is used rather than mIoU. It is the
        hardest bucket, the one mIoU is blindest to, and the one that determines
        whether the model is actually useful for finding candidate flakes.
        """
        smallest = min(self.area_filters)
        result = self.regions[(smallest, min(self.thresholds))]
        value = result.f1
        return 0.0 if not np.isfinite(value) else value
