"""Loss functions for flake segmentation.

Plain cross entropy is a poor match for the evaluation protocol. Flakes occupy
a small fraction of pixels, so an unweighted pixel loss is dominated by
substrate, and within the foreground it is dominated by the largest flakes,
while the metric scores per-flake detection down to 100 pixels.

The composite loss here has three parts:

* weighted cross entropy, taking the per-pixel area-balance weights produced by
  the dataset, so a small flake contributes comparably to a large one;
* Tversky loss, a generalization of Dice in which `beta > alpha` penalizes
  false negatives more than false positives, biasing toward recall (a missed
  flake costs a candidate; a false positive costs one quick manual check);
* a soft boundary term, since flake area drives downstream thickness and yield
  estimates and edge placement is what determines it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def weighted_cross_entropy(
    logits: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = 255,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross entropy with an explicit per-pixel weight map.

    Parameters
    ----------
    logits: (N, C, H, W)
    target: (N, H, W) int64
    weight: (N, H, W) float, or None for uniform weighting.
    """
    per_pixel = F.cross_entropy(
        logits,
        target,
        ignore_index=ignore_index,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    valid = (target != ignore_index).to(per_pixel.dtype)
    if weight is not None:
        valid = valid * weight
    total = valid.sum()
    if total <= 0:
        return per_pixel.sum() * 0.0
    return (per_pixel * valid).sum() / total


def tversky_loss(
    logits: Tensor,
    target: Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    ignore_index: int = 255,
    smooth: float = 1.0,
    foreground_only: bool = True,
) -> Tensor:
    """Tversky loss. `alpha` weights false positives, `beta` false negatives.

    alpha = beta = 0.5 recovers Dice. The default (0.3, 0.7) trades precision
    for recall, which is the correct direction when a missed flake is a lost
    device candidate and a false positive costs a few seconds of review.
    """
    num_classes = logits.shape[1]
    valid = (target != ignore_index).unsqueeze(1)
    safe_target = torch.where(target == ignore_index, torch.zeros_like(target), target)

    probs = logits.softmax(dim=1) * valid
    onehot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).to(probs.dtype)
    onehot = onehot * valid

    dims = (0, 2, 3)
    tp = (probs * onehot).sum(dims)
    fp = (probs * (1 - onehot)).sum(dims)
    fn = ((1 - probs) * onehot).sum(dims)

    index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    if foreground_only and num_classes > 1:
        index = index[1:]
    return 1.0 - index.mean()


def _soft_boundary(prob: Tensor, kernel_size: int = 3) -> Tensor:
    """Extract a soft outer boundary band from a probability map.

    Morphological gradient by max pooling: `maxpool(p) - p` is large where the
    map changes quickly and near zero in flat regions. Fully differentiable and
    requires no distance transform.
    """
    pad = kernel_size // 2
    dilated = F.max_pool2d(prob, kernel_size=kernel_size, stride=1, padding=pad)
    return torch.clamp(dilated - prob, min=0.0)


def boundary_loss(
    logits: Tensor,
    target: Tensor,
    kernel_size: int = 3,
    ignore_index: int = 255,
    smooth: float = 1.0,
) -> Tensor:
    """Soft boundary F1 loss between predicted and ground-truth flake edges."""
    valid = (target != ignore_index).unsqueeze(1).to(logits.dtype)
    prob_fg = logits.softmax(dim=1)[:, 1:2] * valid
    target_fg = ((target > 0) & (target != ignore_index)).unsqueeze(1).to(logits.dtype)

    pred_edge = _soft_boundary(prob_fg, kernel_size)
    true_edge = _soft_boundary(target_fg, kernel_size)

    dims = (1, 2, 3)
    intersection = (pred_edge * true_edge).sum(dims)
    denom = pred_edge.sum(dims) + true_edge.sum(dims)
    f1 = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - f1.mean()


class FlakeLoss(nn.Module):
    """Composite objective. All three terms are on by default.

    Parameters
    ----------
    ce_weight, tversky_weight, boundary_weight:
        Relative term weights. Set any to 0 to disable, which is how the
        ablation table for a paper should be produced.
    aux_weight:
        Weight applied to auxiliary head logits, if the model returns them.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        tversky_weight: float = 1.0,
        boundary_weight: float = 0.5,
        aux_weight: float = 0.4,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        boundary_kernel: int = 3,
        label_smoothing: float = 0.0,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.aux_weight = aux_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.boundary_kernel = boundary_kernel
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def _single(
        self, logits: Tensor, target: Tensor, weight: Tensor | None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        parts: dict[str, Tensor] = {}
        total = logits.sum() * 0.0

        if self.ce_weight > 0:
            ce = weighted_cross_entropy(
                logits, target, weight, self.ignore_index, self.label_smoothing
            )
            parts["ce"] = ce.detach()
            total = total + self.ce_weight * ce

        if self.tversky_weight > 0:
            tv = tversky_loss(
                logits,
                target,
                self.tversky_alpha,
                self.tversky_beta,
                self.ignore_index,
            )
            parts["tversky"] = tv.detach()
            total = total + self.tversky_weight * tv

        if self.boundary_weight > 0:
            bd = boundary_loss(
                logits, target, self.boundary_kernel, self.ignore_index
            )
            parts["boundary"] = bd.detach()
            total = total + self.boundary_weight * bd

        return total, parts

    def forward(
        self,
        outputs: Tensor | dict[str, Tensor],
        target: Tensor,
        weight: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Returns `(loss, parts)` where `parts` holds detached scalars for logging."""
        if isinstance(outputs, Tensor):
            outputs = {"out": outputs}

        main = outputs["out"]
        if main.shape[-2:] != target.shape[-2:]:
            main = F.interpolate(
                main, size=target.shape[-2:], mode="bilinear", align_corners=False
            )

        total, parts = self._single(main, target, weight)

        aux = outputs.get("aux")
        if aux is not None and self.aux_weight > 0:
            if aux.shape[-2:] != target.shape[-2:]:
                aux = F.interpolate(
                    aux, size=target.shape[-2:], mode="bilinear", align_corners=False
                )
            aux_total, aux_parts = self._single(aux, target, weight)
            total = total + self.aux_weight * aux_total
            parts.update({f"aux_{k}": v for k, v in aux_parts.items()})

        parts["total"] = total.detach()
        return total, parts
