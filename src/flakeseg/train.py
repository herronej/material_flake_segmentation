"""Training entry point.

Run single-GPU:

    python -m flakeseg.train --config configs/graphene.yaml

Run distributed:

    torchrun --nproc_per_node=8 -m flakeseg.train --config configs/graphene.yaml

Design notes
------------
Effective batch size is `batch_size * accum_steps * world_size`. The baseline
trains at batch size 2 with SyncBN; here all normalization is batch-size
independent (LayerNorm in the encoder, GroupNorm in the decoder), so gradient
accumulation genuinely recovers large-batch optimization rather than merely
stabilizing normalization statistics.

Checkpoints are selected on small-flake F1 at IoU 0.5, not mIoU. See
`flakeseg.metrics.MetricAccumulator.selection_metric` for the reasoning.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .data.contrast import ContrastConfig
from .data.dataset import FlakeCropDataset, FlakeEvalDataset, SampleConfig
from .data.transforms import AugmentConfig
from .inference import InferenceConfig, probs_to_mask, sliding_window_predict
from .losses import FlakeLoss
from .metrics import MetricAccumulator
from .models.unet import ModelConfig, build_model

logger = logging.getLogger("flakeseg")


@dataclass
class TrainConfig:
    """Everything a run needs, serialized alongside the checkpoint."""

    data_root: str = "data/graphene"
    train_split: str = "train2024"
    val_split: str = "val2024"
    output_dir: str = "runs/graphene"

    batch_size: int = 4
    accum_steps: int = 4
    num_workers: int = 8
    max_iters: int = 60_000
    warmup_iters: int = 1_500
    val_interval: int = 4_000
    log_interval: int = 50
    iters_per_epoch: int = 2_000

    lr: float = 3e-4
    encoder_lr_scale: float = 0.1
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    schedule: str = "poly"
    poly_power: float = 0.9

    amp_dtype: str = "bfloat16"
    seed: int = 0
    val_max_images: int = 0

    model: dict = field(default_factory=dict)
    sample: dict = field(default_factory=dict)
    contrast: dict = field(default_factory=dict)
    augment: dict = field(default_factory=dict)
    loss: dict = field(default_factory=dict)
    inference: dict = field(default_factory=dict)


def load_config(path: str | Path) -> TrainConfig:
    """Load YAML into a TrainConfig, rejecting unknown top-level keys."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = set(TrainConfig().__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return TrainConfig(**raw)


def setup_distributed() -> tuple[int, int, int]:
    """Initialize the process group if launched under torchrun."""
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Two learning rates and no weight decay on norms or biases.

    The encoder carries pretrained structure worth preserving; the decoder and
    detail branch are new. Decaying LayerNorm gains or biases is a common
    unforced error that costs a little accuracy for no benefit.
    """
    decay, no_decay, enc_decay, enc_no_decay = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = name.startswith("encoder.") or ".encoder." in name
        skip = param.ndim <= 1 or name.endswith(".bias") or "gamma" in name
        if is_encoder:
            (enc_no_decay if skip else enc_decay).append(param)
        else:
            (no_decay if skip else decay).append(param)

    encoder_lr = config.lr * config.encoder_lr_scale
    groups = [
        {"params": decay, "lr": config.lr, "weight_decay": config.weight_decay},
        {"params": no_decay, "lr": config.lr, "weight_decay": 0.0},
        {"params": enc_decay, "lr": encoder_lr, "weight_decay": config.weight_decay},
        {"params": enc_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def lr_multiplier(step: int, config: TrainConfig) -> float:
    """Warmup then decay, as a multiplier on each group's base learning rate."""
    if step < config.warmup_iters:
        return (step + 1) / max(1, config.warmup_iters)
    progress = (step - config.warmup_iters) / max(
        1, config.max_iters - config.warmup_iters
    )
    progress = min(max(progress, 0.0), 1.0)
    if config.schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return (1.0 - progress) ** config.poly_power


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataset: FlakeEvalDataset,
    config: TrainConfig,
    device: torch.device,
    max_images: int = 0,
) -> dict[str, float]:
    """Full-image sliding-window validation with the full metric suite."""
    inference = InferenceConfig(**config.inference)
    contrast = ContrastConfig(**config.contrast)
    accumulator = MetricAccumulator()

    total = len(dataset) if max_images <= 0 else min(max_images, len(dataset))
    for i in range(total):
        item = dataset[i]
        probs = sliding_window_predict(
            model, item["image"], inference, contrast, device=device
        )
        mask = probs_to_mask(probs, min_component_area=inference.min_component_area)
        accumulator.update(mask, item["mask"])

    summary = accumulator.summary()
    summary["selection"] = accumulator.selection_metric
    return summary


def train(config: TrainConfig) -> None:
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        level=logging.INFO if is_main(rank) else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)

    output_dir = Path(config.output_dir)
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    train_dataset = FlakeCropDataset(
        root=config.data_root,
        split=config.train_split,
        sample=SampleConfig(**config.sample),
        contrast=ContrastConfig(**config.contrast),
        augment=AugmentConfig(**config.augment),
        length=config.iters_per_epoch * config.batch_size,
        seed=config.seed + rank,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    eval_dataset = FlakeEvalDataset(config.data_root, config.val_split)

    model = build_model(ModelConfig(**config.model)).to(device)
    core = model
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None
        )
        core = model.module

    criterion = FlakeLoss(**config.loss)
    optimizer = build_optimizer(model, config)
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        config.amp_dtype
    )
    use_amp = amp_dtype is not None and device.type == "cuda"
    if amp_dtype is None:
        amp_dtype = torch.float32  # autocast requires a concrete dtype even when off
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype is torch.float16)

    model.train()
    step = 0
    best = -np.inf
    start = time.time()
    logger.info("starting training on %d process(es), device=%s", world_size, device)

    while step < config.max_iters:
        for batch in loader:
            if step >= config.max_iters:
                break

            multiplier = lr_multiplier(step, config)
            for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base * multiplier

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            weights = batch["weight"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                outputs = model(images)
                loss, parts = criterion(outputs, masks, weights)

            scaler.scale(loss / config.accum_steps).backward()

            if (step + 1) % config.accum_steps == 0:
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if is_main(rank) and step % config.log_interval == 0:
                elapsed = time.time() - start
                detail = " ".join(f"{k}={v.item():.4f}" for k, v in sorted(parts.items()))
                logger.info(
                    "iter %d/%d lr=%.2e %s (%.1f it/s)",
                    step,
                    config.max_iters,
                    optimizer.param_groups[0]["lr"],
                    detail,
                    (step + 1) / max(elapsed, 1e-6),
                )

            step += 1

            if step % config.val_interval == 0 or step == config.max_iters:
                if is_main(rank):
                    summary = validate(
                        core, eval_dataset, config, device, config.val_max_images
                    )
                    logger.info(
                        "validation @ %d: %s",
                        step,
                        json.dumps({k: round(v, 4) for k, v in summary.items()}),
                    )
                    with (output_dir / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps({"step": step, **summary}) + "\n")

                    payload = {
                        "step": step,
                        "model": core.state_dict(),
                        "config": asdict(config),
                        "metrics": summary,
                    }
                    torch.save(payload, output_dir / "last.pth")
                    if summary["selection"] > best:
                        best = summary["selection"]
                        torch.save(payload, output_dir / "best.pth")
                        logger.info("new best selection metric: %.4f", best)
                if world_size > 1:
                    dist.barrier()
                model.train()

    if world_size > 1:
        dist.destroy_process_group()
    if is_main(rank):
        logger.info(
            "done in %.1f min, best selection metric %.4f",
            (time.time() - start) / 60,
            best,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a flake segmentation model")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--data-root", default=None, help="override data_root")
    parser.add_argument("--output-dir", default=None, help="override output_dir")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config.data_root = args.data_root
    if args.output_dir:
        config.output_dir = args.output_dir
    train(config)


if __name__ == "__main__":
    main()
