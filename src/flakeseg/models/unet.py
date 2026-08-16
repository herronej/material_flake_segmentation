"""Segmentation network for thin-flake micrographs.

Two departures from the UperNet baseline drive the design.

Full-resolution output. UperNet predicts at stride 4 and upsamples. The
evaluation protocol scores flakes down to 100 pixels total area, roughly 10x10,
which is 2-3 cells at stride 4. Everything that distinguishes a real monolayer
from a contamination speck lives below that. This decoder carries a stride-1
detail branch taken straight from the input and fuses it back at the end, so
the final prediction is made at native resolution.

LayerNorm rather than BatchNorm. Full-resolution micrographs at 768 crops force
small batches; the baseline runs SyncBN at batch size 2, which estimates
normalization statistics from two images. LayerNorm and GroupNorm are
batch-size independent, so gradient accumulation genuinely recovers large-batch
behaviour instead of merely simulating it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CONVNEXT_PRESETS: dict[str, tuple[list[int], list[int]]] = {
    "tiny": ([3, 3, 9, 3], [96, 192, 384, 768]),
    "small": ([3, 3, 27, 3], [96, 192, 384, 768]),
    "base": ([3, 3, 27, 3], [128, 256, 512, 1024]),
}


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dimension of an (N, C, H, W) tensor."""

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """Stochastic depth applied per sample."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class ConvNeXtBlock(nn.Module):
    """Depthwise 7x7, LayerNorm, inverted bottleneck MLP, layer scale."""

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtEncoder(nn.Module):
    """Hierarchical encoder producing features at strides 4, 8, 16, 32."""

    def __init__(
        self,
        in_channels: int = 3,
        variant: str = "tiny",
        drop_path_rate: float = 0.2,
        layer_scale: float = 1e-6,
    ) -> None:
        super().__init__()
        if variant not in CONVNEXT_PRESETS:
            raise ValueError(
                f"unknown variant {variant!r}; choose from {sorted(CONVNEXT_PRESETS)}"
            )
        depths, dims = CONVNEXT_PRESETS[variant]
        self.dims = dims

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0], eps=1e-6),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cursor = 0
        for i, depth in enumerate(depths):
            self.stages.append(
                nn.Sequential(
                    *[
                        ConvNeXtBlock(dims[i], rates[cursor + j], layer_scale)
                        for j in range(depth)
                    ]
                )
            )
            cursor += depth

    def forward(self, x: Tensor) -> list[Tensor]:
        features = []
        for downsample, stage in zip(self.downsample_layers, self.stages, strict=True):
            x = stage(downsample(x))
            features.append(x)
        return features


class ConvBlock(nn.Module):
    """Conv, GroupNorm, GELU. GroupNorm keeps the decoder batch-size independent."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(groups, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DetailBranch(nn.Module):
    """Shallow high-resolution path preserving stride-1 and stride-2 features.

    The encoder's stride-4 stem discards exactly the spatial detail that
    distinguishes a genuine few-hundred-pixel flake from debris. This branch is
    deliberately thin: it supplies localization, not semantics.
    """

    def __init__(self, in_channels: int = 3, channels: int = 32) -> None:
        super().__init__()
        self.level1 = nn.Sequential(
            ConvBlock(in_channels, channels), ConvBlock(channels, channels)
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, channels * 2), channels * 2),
            nn.GELU(),
            ConvBlock(channels * 2, channels * 2),
        )
        self.out_channels = (channels, channels * 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        f1 = self.level1(x)
        f2 = self.level2(f1)
        return f1, f2


@dataclass
class ModelConfig:
    """Network hyperparameters."""

    in_channels: int = 3
    num_classes: int = 2
    variant: str = "tiny"
    decoder_channels: int = 128
    detail_channels: int = 32
    drop_path_rate: float = 0.2
    use_aux_head: bool = True
    encoder_weights: str | None = None
    timm_encoder: str | None = None
    freeze_encoder_stages: list[int] = field(default_factory=list)


class FlakeNet(nn.Module):
    """Encoder, top-down decoder, detail fusion, stride-1 head.

    Forward returns a dict with `out` at input resolution and, when enabled,
    `aux` at stride 16 for deep supervision.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        if cfg.timm_encoder:
            self.encoder, encoder_dims = _build_timm_encoder(cfg)
        else:
            self.encoder = ConvNeXtEncoder(
                in_channels=cfg.in_channels,
                variant=cfg.variant,
                drop_path_rate=cfg.drop_path_rate,
            )
            encoder_dims = self.encoder.dims

        dec = cfg.decoder_channels
        self.lateral = nn.ModuleList(
            [nn.Conv2d(dim, dec, kernel_size=1) for dim in encoder_dims]
        )
        self.smooth = nn.ModuleList([ConvBlock(dec, dec) for _ in encoder_dims])

        self.detail = DetailBranch(cfg.in_channels, cfg.detail_channels)
        d1, d2 = self.detail.out_channels

        self.fuse2 = ConvBlock(dec + d2, dec // 2)
        self.fuse1 = ConvBlock(dec // 2 + d1, dec // 2)
        self.head = nn.Conv2d(dec // 2, cfg.num_classes, kernel_size=1)

        self.aux_head = (
            nn.Sequential(ConvBlock(dec, dec // 2), nn.Conv2d(dec // 2, cfg.num_classes, 1))
            if cfg.use_aux_head
            else None
        )

        self.apply(self._init_weights)
        if cfg.freeze_encoder_stages:
            self.freeze_stages(cfg.freeze_encoder_stages)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def freeze_stages(self, stages: list[int]) -> None:
        """Freeze the listed encoder stage indices, for staged fine-tuning."""
        inner = getattr(self.encoder, "stages", None)
        if inner is None:
            return
        for i in stages:
            for param in inner[i].parameters():
                param.requires_grad_(False)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        size = x.shape[-2:]
        features = self.encoder(x)

        laterals = [conv(f) for conv, f in zip(self.lateral, features, strict=True)]
        top = laterals[-1]
        merged = [self.smooth[-1](top)]
        for i in range(len(laterals) - 2, -1, -1):
            top = laterals[i] + F.interpolate(
                top, size=laterals[i].shape[-2:], mode="bilinear", align_corners=False
            )
            merged.append(self.smooth[i](top))
        merged = merged[::-1]  # strides 4, 8, 16, 32

        aux = self.aux_head(merged[2]) if self.aux_head is not None else None

        detail1, detail2 = self.detail(x)

        up2 = F.interpolate(
            merged[0], size=detail2.shape[-2:], mode="bilinear", align_corners=False
        )
        fused2 = self.fuse2(torch.cat([up2, detail2], dim=1))

        up1 = F.interpolate(
            fused2, size=detail1.shape[-2:], mode="bilinear", align_corners=False
        )
        fused1 = self.fuse1(torch.cat([up1, detail1], dim=1))

        logits = self.head(fused1)
        if logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

        out: dict[str, Tensor] = {"out": logits}
        if aux is not None:
            out["aux"] = aux
        return out


def _build_timm_encoder(cfg: ModelConfig) -> tuple[nn.Module, list[int]]:
    """Optional path to any timm backbone with `features_only=True`.

    Kept optional so the repository has no hard dependency on timm and the
    built-in ConvNeXt remains usable on an air-gapped compute node.
    """
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "timm is required for ModelConfig.timm_encoder; "
            "install it or leave timm_encoder unset to use the built-in encoder"
        ) from exc

    encoder = timm.create_model(
        cfg.timm_encoder,
        pretrained=cfg.encoder_weights is not None,
        features_only=True,
        in_chans=cfg.in_channels,
        drop_path_rate=cfg.drop_path_rate,
    )
    return encoder, list(encoder.feature_info.channels())


def build_model(config: ModelConfig | None = None) -> FlakeNet:
    """Factory used by the training entry point."""
    return FlakeNet(config)
