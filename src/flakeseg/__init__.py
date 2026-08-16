"""Semantic segmentation of 2D material flakes in optical micrographs."""

__version__ = "0.1.0"

from .data import find_pairs
from .losses import FlakeLoss
from .metrics import MetricAccumulator, region_iou
from .models.unet import FlakeNet, ModelConfig, build_model

__all__ = [
    "FlakeLoss",
    "FlakeNet",
    "MetricAccumulator",
    "ModelConfig",
    "build_model",
    "find_pairs",
    "region_iou",
]
