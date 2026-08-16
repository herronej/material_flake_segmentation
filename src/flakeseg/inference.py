"""Whole-image inference by overlapped sliding window.

The published configs disagree with themselves about inference geometry: the
graphene config slides a 544 window with stride 512, i.e. 32 pixels of overlap,
while the MoS2 config slides 3200 with stride 2944. Thirty-two pixels is not
enough. A flake straddling a tile boundary is seen only in fragments by both
windows, and hard-averaging the seam produces visible discontinuities that
break the connected-component analysis the RegionIoU metric depends on.

This module uses a configurable overlap (25% by default) with cosine-taper
blending, so every pixel's prediction is dominated by the window that saw it
furthest from an edge. Normalization is applied per window with the same
context convention as training, so the model sees inputs from the same
distribution it was fitted on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .data.contrast import ContrastConfig, optical_contrast


@dataclass
class InferenceConfig:
    """Sliding window geometry and test-time behaviour."""

    window: int = 768
    overlap: float = 0.25
    context_factor: float = 1.5
    batch_size: int = 4
    flip_tta: bool = False
    amp: bool = True
    min_component_area: int = 0

    @property
    def stride(self) -> int:
        stride = int(round(self.window * (1.0 - self.overlap)))
        return max(1, min(stride, self.window))


def cosine_window(size: int, taper: int, device: torch.device) -> Tensor:
    """2D separable cosine taper, unity in the interior, zero at the edges."""
    ramp = torch.ones(size, device=device, dtype=torch.float32)
    taper = int(max(1, min(taper, size // 2)))
    edge = 0.5 * (
        1.0 - torch.cos(torch.linspace(0, np.pi, taper + 2, device=device)[1:-1])
    )
    ramp[:taper] = edge
    ramp[size - taper :] = edge.flip(0)
    return ramp[:, None] * ramp[None, :]


def _tile_origins(extent: int, window: int, stride: int) -> list[int]:
    """Origins covering `extent`, with the last tile flush against the far edge."""
    if extent <= window:
        return [0]
    origins = list(range(0, extent - window + 1, stride))
    if origins[-1] != extent - window:
        origins.append(extent - window)
    return origins


@torch.no_grad()
def sliding_window_predict(
    model: nn.Module,
    image: np.ndarray,
    config: InferenceConfig | None = None,
    contrast: ContrastConfig | None = None,
    device: torch.device | str = "cpu",
    num_classes: int = 2,
) -> np.ndarray:
    """Predict class probabilities for a full-resolution micrograph.

    Parameters
    ----------
    model:
        A network returning either a tensor or a dict containing key `out`.
    image:
        HxWx3 raw image array.
    config, contrast:
        Window geometry and normalization parameters.
    device:
        Torch device for the forward passes.

    Returns
    -------
    (num_classes, H, W) float32 array of blended probabilities.
    """
    config = config or InferenceConfig()
    contrast = contrast or ContrastConfig()
    device = torch.device(device)
    model.eval().to(device)

    height, width = image.shape[:2]
    window = min(config.window, height, width)
    stride = max(1, int(round(window * (1.0 - config.overlap))))
    taper = max(1, int(round(window * config.overlap / 2)))

    accumulator = torch.zeros((num_classes, height, width), dtype=torch.float32, device=device)
    normalizer = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    blend = cosine_window(window, taper, device)

    context = int(round(window * config.context_factor))
    pad = (context - window) // 2

    tops = _tile_origins(height, window, stride)
    lefts = _tile_origins(width, window, stride)

    batch: list[Tensor] = []
    positions: list[tuple[int, int]] = []

    def flush() -> None:
        if not batch:
            return
        tensor = torch.stack(batch).to(device, non_blocking=True)
        autocast = torch.autocast(
            device_type=device.type, enabled=config.amp and device.type == "cuda"
        )
        with autocast:
            outputs = model(tensor)
            logits = outputs["out"] if isinstance(outputs, dict) else outputs
            if config.flip_tta:
                for dims in ((-1,), (-2,), (-1, -2)):
                    flipped = model(torch.flip(tensor, dims=dims))
                    flipped = flipped["out"] if isinstance(flipped, dict) else flipped
                    logits = logits + torch.flip(flipped, dims=dims)
                logits = logits / 4.0
        probs = logits.float().softmax(dim=1)
        for k, (top, left) in enumerate(positions):
            accumulator[:, top : top + window, left : left + window] += probs[k] * blend
            normalizer[:, top : top + window, left : left + window] += blend
        batch.clear()
        positions.clear()

    for top in tops:
        for left in lefts:
            c_top = int(np.clip(top - pad, 0, max(height - context, 0)))
            c_left = int(np.clip(left - pad, 0, max(width - context, 0)))
            c_bot = min(c_top + context, height)
            c_right = min(c_left + context, width)

            normalized = optical_contrast(image[c_top:c_bot, c_left:c_right], contrast)
            tile = normalized[
                top - c_top : top - c_top + window, left - c_left : left - c_left + window
            ]
            batch.append(torch.from_numpy(tile.transpose(2, 0, 1).copy()))
            positions.append((top, left))
            if len(batch) >= config.batch_size:
                flush()
    flush()

    probs = accumulator / normalizer.clamp(min=1e-6)
    return probs.cpu().numpy()


def probs_to_mask(
    probs: np.ndarray, threshold: float = 0.5, min_component_area: int = 0
) -> np.ndarray:
    """Convert foreground probability to a labelled mask, dropping tiny components.

    `min_component_area` should be set no higher than the smallest area bucket
    being scored, otherwise the filter itself, rather than the model, decides
    the small-flake recall number.
    """
    from scipy import ndimage

    mask = (probs[1] >= threshold).astype(np.uint8)
    if min_component_area > 0:
        labels, count = ndimage.label(
            mask, structure=ndimage.generate_binary_structure(2, 2)
        )
        if count:
            areas = np.bincount(labels.ravel())
            keep = areas >= min_component_area
            keep[0] = False
            mask = keep[labels].astype(np.uint8)
    return mask


def calibrate_threshold(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    metric_fn,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Sweep the decision threshold on validation data.

    The argmax threshold of 0.5 is rarely optimal for a recall-weighted metric
    on heavily imbalanced data. Tuning it on the validation split is close to
    free and typically worth more than an architecture change. Report the swept
    threshold in any paper, and do not sweep it on the test split.

    Returns
    -------
    (best_threshold, best_score)
    """
    grid = np.linspace(0.1, 0.9, 33) if grid is None else grid
    best_threshold, best_score = 0.5, -np.inf
    for threshold in grid:
        score = float(
            np.mean(
                [
                    metric_fn(probs_to_mask(p, threshold), t)
                    for p, t in zip(probabilities, targets, strict=True)
                ]
            )
        )
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold, best_score
