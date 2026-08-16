"""Per-image background estimation and optical contrast normalization.

Rationale
---------
The published baseline normalizes with dataset-global statistics
(mean ~= [137.7, 114.6, 127.2], std ~= [18.1, 17.2, 15.0] on a 0-255 scale).
The std is roughly 7% of dynamic range, so after standardization the flake
signal still occupies a narrow band and the network spends capacity modelling
illumination drift, white balance, and SiO2 thickness variation rather than
the flake itself.

For a thin flake on a dielectric-on-silicon substrate the physically meaningful
quantity is the optical contrast against the local substrate:

    C = (I - I_bg) / I_bg

per colour channel, where I_bg is the local substrate reflectance. C is
invariant to illumination gain and to camera exposure, and its per-channel
signature is what encodes layer number. Estimating I_bg per image removes the
dominant nuisance variation before the network ever sees the data.

Three estimators are provided. `polynomial` is the default: it is robust,
smooth, and cheap, and it matches the physics of uneven Koehler illumination
better than a median filter does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

BackgroundMethod = Literal["polynomial", "median", "morphological", "none"]

#: Which side of the substrate level flakes sit on.
#:
#: Graphene and the TMDs on the standard SiO2/Si stack absorb, so under
#: brightfield they are darker than the substrate and `dark` is correct. hBN is
#: far more transparent and its contrast is interference-driven, so the sign
#: varies with thickness, wavelength and oxide thickness; `both` rejects
#: outliers symmetrically and makes no assumption. Getting this wrong does not
#: crash anything, it just fits the substrate surface through the flakes and
#: quietly flattens the signal the network is meant to see.
Polarity = Literal["dark", "bright", "both"]


@dataclass(frozen=True)
class ContrastConfig:
    """Configuration for optical contrast normalization."""

    method: BackgroundMethod = "polynomial"
    polarity: Polarity = "dark"
    poly_degree: int = 2
    poly_iterations: int = 3
    poly_downsample: int = 8
    poly_reject_sigma: float = 1.0
    median_size: int = 129
    median_downsample: int = 4
    morph_size: int = 101
    clip: float = 0.5
    scale: float = 10.0
    eps: float = 1e-3


def _fit_poly_background(
    channel: np.ndarray,
    degree: int,
    iterations: int,
    downsample: int,
    polarity: Polarity = "dark",
    reject_sigma: float = 1.0,
) -> np.ndarray:
    """Robust low-order polynomial fit to the substrate level of one channel.

    Uses iterative reweighting that progressively discards the pixels a flake
    would occupy. With `polarity="dark"` (graphene, TMDs on SiO2/Si under
    brightfield) flakes appear as negative residuals and only those are
    trimmed; `"bright"` mirrors that, and `"both"` trims symmetrically for
    materials such as hBN whose contrast sign is not fixed.

    Parameters
    ----------
    channel:
        2D float array, one colour channel of the raw image.
    degree:
        Total polynomial degree in (x, y).
    iterations:
        Number of reweighting passes. 0 gives a plain least-squares fit.
    downsample:
        Stride used when building the design matrix. The fit is evaluated at
        full resolution regardless.
    polarity:
        Which residual sign is treated as flake rather than substrate.
    reject_sigma:
        Rejection threshold in robust sigma.

    Returns
    -------
    2D float array with the same shape as `channel`.
    """
    height, width = channel.shape
    step = max(1, int(downsample))

    yy_full, xx_full = np.mgrid[0:height, 0:width]
    # Normalize coordinates to [-1, 1] so the Vandermonde matrix stays conditioned.
    yy_full = (yy_full / max(height - 1, 1)) * 2.0 - 1.0
    xx_full = (xx_full / max(width - 1, 1)) * 2.0 - 1.0

    powers = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]

    def design(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return np.stack([(xs**i) * (ys**j) for i, j in powers], axis=-1)

    xs = xx_full[::step, ::step].ravel()
    ys = yy_full[::step, ::step].ravel()
    zs = channel[::step, ::step].ravel().astype(np.float64)

    basis = design(xs, ys)
    mask = np.ones(zs.shape, dtype=bool)

    coeffs = np.zeros(len(powers), dtype=np.float64)
    for _ in range(max(1, iterations)):
        if mask.sum() < basis.shape[1] * 4:
            break
        coeffs, *_ = np.linalg.lstsq(basis[mask], zs[mask], rcond=None)
        residual = zs - basis @ coeffs
        # Keep pixels on the substrate side of the fit, plus anything within one
        # robust sigma of it, so that noise does not bias the surface.
        sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        sigma = float(max(sigma, 1e-6))
        cut = reject_sigma * sigma
        if polarity == "dark":
            mask = residual > -cut
        elif polarity == "bright":
            mask = residual < cut
        elif polarity == "both":
            mask = np.abs(residual) < cut
        else:
            raise ValueError(f"unknown polarity: {polarity!r}")

    basis_full = design(xx_full.ravel(), yy_full.ravel())
    return (basis_full @ coeffs).reshape(height, width)


def _median_background(channel: np.ndarray, size: int, downsample: int) -> np.ndarray:
    """Large-kernel median background, computed on a decimated grid for speed."""
    step = max(1, int(downsample))
    small = channel[::step, ::step]
    kernel = max(3, int(size) // step)
    if kernel % 2 == 0:
        kernel += 1
    smoothed = ndimage.median_filter(small, size=kernel, mode="nearest")
    zoom = (channel.shape[0] / smoothed.shape[0], channel.shape[1] / smoothed.shape[1])
    return ndimage.zoom(smoothed, zoom, order=1)[: channel.shape[0], : channel.shape[1]]


def _morphological_background(
    channel: np.ndarray, size: int, polarity: Polarity = "dark"
) -> np.ndarray:
    """Grey-scale rolling-ball background, on whichever side the flakes sit.

    Closing removes dark features and opening removes bright ones. `both`
    composes the two, which suppresses features of either sign at the cost of
    slightly more smoothing of the substrate itself.
    """
    kernel = max(3, int(size))
    if kernel % 2 == 0:
        kernel += 1
    if polarity == "dark":
        return ndimage.grey_closing(channel, size=kernel, mode="nearest")
    if polarity == "bright":
        return ndimage.grey_opening(channel, size=kernel, mode="nearest")
    if polarity == "both":
        closed = ndimage.grey_closing(channel, size=kernel, mode="nearest")
        return ndimage.grey_opening(closed, size=kernel, mode="nearest")
    raise ValueError(f"unknown polarity: {polarity!r}")


def estimate_background(image: np.ndarray, config: ContrastConfig) -> np.ndarray:
    """Estimate the per-channel substrate level of an image.

    Parameters
    ----------
    image:
        HxWx3 array, raw pixel values in any numeric dtype.
    config:
        Estimator selection and parameters.

    Returns
    -------
    HxWx3 float32 array, the estimated substrate level.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected HxWxC image, got shape {array.shape}")

    if config.method == "none":
        # Fall back to a per-channel scalar so downstream code stays uniform.
        return np.broadcast_to(
            array.reshape(-1, array.shape[-1]).mean(axis=0), array.shape
        ).astype(np.float32)

    out = np.empty_like(array, dtype=np.float32)
    for c in range(array.shape[-1]):
        channel = array[..., c]
        if config.method == "polynomial":
            out[..., c] = _fit_poly_background(
                channel,
                config.poly_degree,
                config.poly_iterations,
                config.poly_downsample,
                config.polarity,
                config.poly_reject_sigma,
            )
        elif config.method == "median":
            out[..., c] = _median_background(
                channel, config.median_size, config.median_downsample
            )
        elif config.method == "morphological":
            out[..., c] = _morphological_background(
                channel, config.morph_size, config.polarity
            )
        else:
            raise ValueError(f"unknown background method: {config.method!r}")
    return out


def optical_contrast(image: np.ndarray, config: ContrastConfig) -> np.ndarray:
    """Convert a raw brightfield image to per-channel optical contrast.

    The output is `scale * clip((I - I_bg) / I_bg, -clip, clip)`, which places
    typical monolayer contrast (a few percent) at order unity. It is invariant
    to illumination gain, so a model trained on it transfers across microscopes
    and exposure settings far better than one trained on globally standardized
    RGB.

    Parameters
    ----------
    image:
        HxWx3 array of raw pixel values.
    config:
        Estimator and output scaling parameters.

    Returns
    -------
    HxWx3 float32 array of scaled contrast, channels-last.
    """
    array = np.asarray(image, dtype=np.float32)
    background = estimate_background(array, config)
    denom = np.maximum(background, config.eps)
    contrast = (array - background) / denom
    np.clip(contrast, -config.clip, config.clip, out=contrast)
    return (contrast * config.scale).astype(np.float32)
