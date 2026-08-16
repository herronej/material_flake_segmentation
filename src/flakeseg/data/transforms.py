"""Augmentation for optical-contrast flake images.

What is deliberately absent
---------------------------
Hue jitter. The baseline config applies `PhotoMetricDistortion` with
`hue_delta=9` and saturation/contrast jitter over 0.75-1.25. On this data the
per-channel contrast signature *is* the label: monolayer, bilayer, and few-layer
regions are separated by shifts of comparable magnitude to that jitter. Randomly
perturbing hue therefore destroys label-relevant signal and teaches the model to
ignore the one cue that matters.

What replaces it
----------------
Perturbations that correspond to real acquisition variability and that leave the
physical meaning of the contrast intact:

* illumination gain and black level, which cancel in the contrast definition but
  still perturb the background estimate slightly;
* substrate thickness drift, modelled as a smooth per-channel gain on contrast
  (a change in SiO2 thickness rescales the interference contrast per wavelength);
* vignetting, a smooth multiplicative field;
* defocus, modelled as isotropic Gaussian blur;
* sensor noise, additive Gaussian in the raw domain scaled into contrast units;
* the dihedral group, since in-plane orientation carries no information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class AugmentConfig:
    """Probabilities and ranges for training-time augmentation."""

    flip_prob: float = 0.5
    rot90_prob: float = 0.5

    gain_prob: float = 0.5
    gain_range: tuple[float, float] = (0.9, 1.1)

    channel_gain_prob: float = 0.5
    channel_gain_range: tuple[float, float] = (0.85, 1.15)

    vignette_prob: float = 0.3
    vignette_strength: tuple[float, float] = (0.0, 0.15)

    blur_prob: float = 0.3
    blur_sigma: tuple[float, float] = (0.0, 1.5)

    noise_prob: float = 0.5
    noise_sigma: tuple[float, float] = (0.0, 0.15)

    offset_prob: float = 0.3
    offset_range: tuple[float, float] = (-0.2, 0.2)

    seed: int | None = field(default=None, repr=False)


class FlakeAugment:
    """Callable augmentation pipeline operating on contrast-space crops.

    All operations are applied to an HxWxC float32 contrast image and an HxW
    integer mask. Geometric operations are applied to both; photometric
    operations only to the image.
    """

    def __init__(self, config: AugmentConfig | None = None) -> None:
        self.config = config or AugmentConfig()
        self._rng_instance: np.random.Generator | None = None

    @property
    def _rng(self) -> np.random.Generator:
        """Lazily create the generator inside whichever worker process uses it.

        A generator built in `__init__` is forked into every dataloader worker
        with identical state, so all workers would emit the same augmentation
        sequence. Deferring construction and seeding from `torch.initial_seed()`
        gives each worker, and each epoch, an independent stream. See
        `FlakeCropDataset` for the same fix applied to crop sampling.
        """
        if self._rng_instance is None:
            if self.config.seed is not None:
                seed = self.config.seed
            else:
                import torch

                info = torch.utils.data.get_worker_info()
                seed = int(torch.initial_seed() % (2**31)) + (
                    0 if info is None else info.id
                )
            self._rng_instance = np.random.default_rng(seed)
        return self._rng_instance

    def _u(self, low: float, high: float) -> float:
        return float(self._rng.uniform(low, high))

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        image = np.ascontiguousarray(image, dtype=np.float32)
        mask = np.ascontiguousarray(mask)

        # Geometric: the dihedral group of the square.
        if self._rng.random() < cfg.flip_prob:
            axis = int(self._rng.integers(0, 2))
            image = np.flip(image, axis=axis)
            mask = np.flip(mask, axis=axis)
        if self._rng.random() < cfg.rot90_prob:
            k = int(self._rng.integers(1, 4))
            image = np.rot90(image, k=k, axes=(0, 1))
            mask = np.rot90(mask, k=k, axes=(0, 1))
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        # Global illumination gain. Residual effect after contrast normalization.
        if self._rng.random() < cfg.gain_prob:
            image = image * self._u(*cfg.gain_range)

        # Substrate thickness drift: independent smooth gain per channel.
        if self._rng.random() < cfg.channel_gain_prob:
            gains = self._rng.uniform(
                cfg.channel_gain_range[0], cfg.channel_gain_range[1], size=image.shape[-1]
            ).astype(np.float32)
            image = image * gains

        # Vignetting: smooth radial multiplicative field.
        if self._rng.random() < cfg.vignette_prob:
            strength = self._u(*cfg.vignette_strength)
            h, w = image.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            yy = (yy / max(h - 1, 1)) * 2.0 - 1.0
            xx = (xx / max(w - 1, 1)) * 2.0 - 1.0
            radial = np.sqrt(xx**2 + yy**2) / np.sqrt(2.0)
            image = image * (1.0 - strength * radial).astype(np.float32)[..., None]

        # Defocus.
        if self._rng.random() < cfg.blur_prob:
            sigma = self._u(*cfg.blur_sigma)
            if sigma > 1e-3:
                image = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0))

        # Baseline offset, i.e. imperfect background estimation at test time.
        if self._rng.random() < cfg.offset_prob:
            image = image + self._u(*cfg.offset_range)

        # Sensor noise.
        if self._rng.random() < cfg.noise_prob:
            sigma = self._u(*cfg.noise_sigma)
            if sigma > 1e-4:
                image = image + self._rng.normal(0.0, sigma, size=image.shape).astype(
                    np.float32
                )

        return image.astype(np.float32), mask
