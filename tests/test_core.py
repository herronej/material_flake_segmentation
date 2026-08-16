"""Tests for the pieces where a silent bug would corrupt every reported number."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flakeseg.data.contrast import ContrastConfig, optical_contrast
from flakeseg.data.dataset import area_balanced_weights
from flakeseg.data.transforms import AugmentConfig, FlakeAugment
from flakeseg.inference import InferenceConfig, cosine_window, probs_to_mask
from flakeseg.losses import FlakeLoss, boundary_loss, tversky_loss, weighted_cross_entropy
from flakeseg.metrics import (
    MetricAccumulator,
    boundary_iou,
    confusion_matrix,
    iou_from_confusion,
    region_iou,
)
from flakeseg.models.unet import ModelConfig, build_model


def make_mask(shape=(256, 256), blobs=((32, 32, 10), (128, 128, 30))) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for cy, cx, r in blobs:
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r**2] = 1
    return mask


class TestRegionIoU:
    def test_perfect_prediction_recalls_everything(self):
        target = make_mask()
        results = region_iou(target, target, area_filters=(100,), thresholds=(0.5, 0.75))
        for result in results.values():
            assert result.n_gt == 2
            assert result.n_matched == 2
            assert result.recall == 1.0
            assert result.precision == 1.0
            assert result.mean_matched_iou == pytest.approx(1.0)

    def test_empty_prediction_recalls_nothing(self):
        target = make_mask()
        pred = np.zeros_like(target)
        results = region_iou(pred, target, area_filters=(100,), thresholds=(0.5,))
        result = results[(100, 0.5)]
        assert result.n_gt == 2 and result.n_matched == 0
        assert result.recall == 0.0
        assert np.isnan(result.precision)

    def test_area_filter_excludes_small_components(self):
        # radius 10 -> ~314 px, radius 30 -> ~2827 px
        target = make_mask()
        results = region_iou(
            target, target, area_filters=(100, 1_000, 10_000), thresholds=(0.5,)
        )
        assert results[(100, 0.5)].n_gt == 2
        assert results[(1_000, 0.5)].n_gt == 1
        assert results[(10_000, 0.5)].n_gt == 0
        assert np.isnan(results[(10_000, 0.5)].recall)

    def test_matching_is_one_to_one(self):
        # One GT blob, two predicted blobs both overlapping it: only one may match.
        target = make_mask(blobs=((64, 64, 20),))
        pred = np.zeros_like(target)
        yy, xx = np.mgrid[0:256, 0:256]
        pred[(yy - 64) ** 2 + (xx - 64) ** 2 <= 20**2] = 1
        pred[200:210, 200:210] = 1
        results = region_iou(pred, target, area_filters=(1,), thresholds=(0.5,))
        result = results[(1, 0.5)]
        assert result.n_gt == 1 and result.n_pred == 2 and result.n_matched == 1
        assert result.precision == pytest.approx(0.5)

    def test_threshold_is_respected(self):
        target = make_mask(blobs=((128, 128, 40),))
        pred = np.zeros_like(target)
        yy, xx = np.mgrid[0:256, 0:256]
        # Shifted circle: overlaps substantially but not enough for IoU 0.75.
        pred[(yy - 138) ** 2 + (xx - 138) ** 2 <= 40**2] = 1
        results = region_iou(pred, target, area_filters=(1,), thresholds=(0.5, 0.75))
        assert results[(1, 0.5)].n_matched == 1
        assert results[(1, 0.75)].n_matched == 0

    def test_ignore_index_excluded(self):
        target = make_mask()
        target_ignored = target.copy().astype(np.int32)
        target_ignored[:] = 255
        results = region_iou(target, target_ignored, area_filters=(1,), thresholds=(0.5,))
        assert results[(1, 0.5)].n_gt == 0


class TestPixelMetrics:
    def test_confusion_and_iou(self):
        target = make_mask()
        matrix = confusion_matrix(target, target)
        ious = iou_from_confusion(matrix)
        assert np.allclose(ious, 1.0)
        assert matrix.sum() == target.size

    def test_boundary_iou_perfect_and_disjoint(self):
        target = make_mask()
        assert boundary_iou(target, target) == pytest.approx(1.0)
        assert boundary_iou(np.zeros_like(target), target) == pytest.approx(0.0)

    def test_accumulator_selection_metric_is_finite(self):
        accumulator = MetricAccumulator()
        target = make_mask()
        accumulator.update(target, target)
        summary = accumulator.summary()
        assert summary["mIoU"] == pytest.approx(1.0)
        assert accumulator.selection_metric == pytest.approx(1.0)
        assert "RIoU_recall_a100_t0.5" in summary


class TestAreaWeights:
    def test_small_components_upweighted(self):
        mask = make_mask(shape=(256, 256), blobs=((64, 64, 5), (180, 180, 50)))
        weights = area_balanced_weights(mask, alpha=0.5, reference=10_000.0, max_weight=8.0)
        small = weights[64, 64]
        large = weights[180, 180]
        assert small > large
        assert weights[0, 0] == pytest.approx(1.0)
        assert small <= 8.0

    def test_border_components_not_upweighted(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[0:6, 0:6] = 1  # tiny but truncated at the crop edge
        weights = area_balanced_weights(mask, alpha=0.5, reference=10_000.0, max_weight=8.0)
        assert weights[2, 2] == pytest.approx(1.0)

    def test_alpha_zero_disables(self):
        mask = make_mask()
        weights = area_balanced_weights(mask, alpha=0.0, reference=10_000.0, max_weight=8.0)
        assert np.allclose(weights, 1.0)


class TestContrast:
    def test_illumination_gain_invariance(self):
        rng = np.random.default_rng(0)
        base = np.full((128, 128, 3), 140.0, dtype=np.float32)
        base += rng.normal(0, 2, base.shape).astype(np.float32)
        base[40:60, 40:60] *= 0.94  # a flake, 6% darker
        config = ContrastConfig()
        a = optical_contrast(base, config)
        b = optical_contrast(base * 1.35, config)  # exposure change
        assert np.abs(a - b).mean() < 0.05 * np.abs(a).std() + 0.05

    def test_gradient_illumination_removed(self):
        yy = np.linspace(0.8, 1.2, 128)[:, None, None]
        base = (np.full((128, 128, 3), 140.0) * yy).astype(np.float32)
        contrast = optical_contrast(base, ContrastConfig())
        assert np.abs(contrast).max() < 0.5


class TestLosses:
    def _batch(self, n=2, c=2, h=64, w=64):
        torch.manual_seed(0)
        logits = torch.randn(n, c, h, w, requires_grad=True)
        target = torch.zeros(n, h, w, dtype=torch.long)
        target[:, 16:32, 16:32] = 1
        weight = torch.ones(n, h, w)
        return logits, target, weight

    def test_weighted_ce_matches_unweighted_when_uniform(self):
        logits, target, weight = self._batch()
        a = weighted_cross_entropy(logits, target, None)
        b = weighted_cross_entropy(logits, target, weight)
        assert torch.allclose(a, b, atol=1e-6)

    def test_ce_respects_ignore_index(self):
        logits, target, _ = self._batch()
        ignored = target.clone()
        ignored[:, :32] = 255
        loss = weighted_cross_entropy(logits, ignored)
        assert torch.isfinite(loss) and loss.item() > 0

    def test_tversky_zero_for_perfect_prediction(self):
        target = torch.zeros(1, 32, 32, dtype=torch.long)
        target[:, 8:24, 8:24] = 1
        logits = torch.stack([(target == 0).float(), (target == 1).float()], dim=1) * 40.0
        assert tversky_loss(logits, target).item() < 1e-3

    def test_tversky_beta_penalizes_false_negatives_more(self):
        target = torch.zeros(1, 32, 32, dtype=torch.long)
        target[:, 8:24, 8:24] = 1
        miss = torch.stack([torch.ones(1, 32, 32), torch.zeros(1, 32, 32)], dim=1) * 20.0
        over = torch.stack([torch.zeros(1, 32, 32), torch.ones(1, 32, 32)], dim=1) * 20.0
        fn_loss = tversky_loss(miss, target, alpha=0.3, beta=0.7)
        fp_loss = tversky_loss(over, target, alpha=0.3, beta=0.7)
        assert fn_loss > fp_loss

    def test_boundary_loss_zero_for_perfect_prediction(self):
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        target[:, 16:48, 16:48] = 1
        logits = torch.stack([(target == 0).float(), (target == 1).float()], dim=1) * 40.0
        assert boundary_loss(logits, target).item() < 0.05

    def test_composite_backward(self):
        logits, target, weight = self._batch()
        aux = logits.detach().clone().requires_grad_()
        loss, parts = FlakeLoss()({"out": logits, "aux": aux}, target, weight)
        loss.backward()
        assert torch.isfinite(loss)
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        assert {"ce", "tversky", "boundary", "total"} <= set(parts)

    def test_disabling_terms(self):
        logits, target, weight = self._batch()
        loss, parts = FlakeLoss(tversky_weight=0.0, boundary_weight=0.0)(
            logits, target, weight
        )
        assert "tversky" not in parts and "boundary" not in parts


class TestModel:
    def test_output_is_full_resolution(self):
        model = build_model(ModelConfig(variant="tiny", decoder_channels=32, detail_channels=8))
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out["out"].shape == (1, 2, 128, 128)
        assert out["aux"].shape[-2:] == (8, 8)  # stride 16

    def test_non_square_and_odd_sizes(self):
        model = build_model(ModelConfig(variant="tiny", decoder_channels=32, detail_channels=8))
        out = model(torch.randn(1, 3, 96, 160))
        assert out["out"].shape == (1, 2, 96, 160)

    def test_batch_size_one_works(self):
        """LayerNorm/GroupNorm only. BatchNorm would make this brittle."""
        model = build_model(ModelConfig(variant="tiny", decoder_channels=32, detail_channels=8))
        model.train()
        out = model(torch.randn(1, 3, 64, 64))
        out["out"].sum().backward()
        assert torch.isfinite(out["out"]).all()

    def test_eval_deterministic(self):
        model = build_model(ModelConfig(variant="tiny", decoder_channels=32, detail_channels=8))
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            assert torch.allclose(model(x)["out"], model(x)["out"])


class TestInference:
    def test_cosine_window_interior_is_unity(self):
        w = cosine_window(64, 8, torch.device("cpu"))
        assert w[32, 32].item() == pytest.approx(1.0)
        assert w[0, 0].item() < 0.1
        assert (w >= 0).all()

    def test_stride_from_overlap(self):
        assert InferenceConfig(window=768, overlap=0.25).stride == 576
        assert InferenceConfig(window=512, overlap=0.0).stride == 512

    def test_sliding_window_reconstructs_full_image(self):
        from flakeseg.inference import sliding_window_predict

        model = build_model(ModelConfig(variant="tiny", decoder_channels=32, detail_channels=8))
        image = (np.random.default_rng(0).normal(140, 5, (300, 260, 3))).astype(np.uint8)
        probs = sliding_window_predict(
            model, image, InferenceConfig(window=128, overlap=0.25, batch_size=2, amp=False)
        )
        assert probs.shape == (2, 300, 260)
        assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-3)

    def test_min_area_filter(self):
        probs = np.zeros((2, 64, 64), dtype=np.float32)
        probs[0] = 1.0
        probs[1, 10:13, 10:13] = 1.0  # 9 px
        probs[1, 30:45, 30:45] = 1.0  # 225 px
        probs[0] = 1.0 - probs[1]
        assert probs_to_mask(probs, min_component_area=0).sum() == 9 + 225
        assert probs_to_mask(probs, min_component_area=100).sum() == 225


class TestAugment:
    def test_shapes_and_mask_alignment_preserved(self):
        rng = np.random.default_rng(0)
        image = rng.normal(0, 1, (64, 64, 3)).astype(np.float32)
        mask = make_mask((64, 64), blobs=((20, 20, 6),))
        aug = FlakeAugment(AugmentConfig(seed=0))
        for _ in range(20):
            out_image, out_mask = aug(image, mask)
            assert out_image.shape == image.shape
            assert out_mask.shape == mask.shape
            assert out_mask.sum() == mask.sum()  # geometric ops preserve area
            assert np.isfinite(out_image).all()

    def test_no_hue_channel_mixing(self):
        """Channel gains are applied independently; channels are never permuted."""
        image = np.zeros((32, 32, 3), dtype=np.float32)
        image[..., 0] = 1.0
        aug = FlakeAugment(
            AugmentConfig(
                seed=1, blur_prob=0.0, noise_prob=0.0, offset_prob=0.0, vignette_prob=0.0
            )
        )
        for _ in range(20):
            out, _ = aug(image, np.zeros((32, 32), dtype=np.uint8))
            assert out[..., 1].max() == pytest.approx(0.0)
            assert out[..., 2].max() == pytest.approx(0.0)


class TestWorkerRNG:
    """Regression tests for the dataloader fork-duplication bug.

    A numpy Generator constructed in __init__ is copied verbatim into every
    worker process, so all workers draw identical crop coordinates and identical
    augmentations. Nothing about the loss curve reveals this; it just quietly
    reduces effective data diversity by the worker count.
    """

    def test_dataset_rng_differs_across_workers(self, tmp_path):
        from torch.utils.data import DataLoader

        from flakeseg.data.dataset import FlakeCropDataset

        _write_synthetic(tmp_path, n_train=4, size=(200, 220))
        dataset = FlakeCropDataset(
            root=tmp_path,
            split="train2024",
            sample=__import__(
                "flakeseg.data.dataset", fromlist=["SampleConfig"]
            ).SampleConfig(crop_size=64),
            length=16,
        )
        draws = [
            int(dataset._rng.integers(0, 10**9)) for _ in range(4)
        ]  # main process still works
        assert len(set(draws)) == 4

        loader = DataLoader(dataset, batch_size=1, num_workers=2)
        images = [b["image"].numpy().ravel()[:8] for b in loader]
        # With duplicated RNGs, workers alternate identical crops in lockstep.
        assert len({tuple(np.round(v, 4)) for v in images}) > 1

    def test_augment_rng_differs_when_seed_unset(self):
        a = FlakeAugment(AugmentConfig(seed=None))
        b = FlakeAugment(AugmentConfig(seed=None))
        # Independent instances must not be locked to the same stream by default.
        image = np.zeros((16, 16, 3), dtype=np.float32)
        mask = np.zeros((16, 16), dtype=np.uint8)
        for _ in range(3):
            a(image, mask)
        assert a._rng_instance is not None and b._rng_instance is None

    def test_explicit_seed_still_reproducible(self):
        image = np.random.default_rng(0).normal(0, 1, (16, 16, 3)).astype(np.float32)
        mask = np.zeros((16, 16), dtype=np.uint8)
        out_a = FlakeAugment(AugmentConfig(seed=7))(image, mask)[0]
        out_b = FlakeAugment(AugmentConfig(seed=7))(image, mask)[0]
        assert np.allclose(out_a, out_b)


def _write_synthetic(root, n_train=4, size=(200, 220)):
    """Minimal dataset in the layout the real archives unpack to."""
    from PIL import Image

    rng = np.random.default_rng(0)
    h, w = size
    for split, n in (("train2024", n_train), ("val2024", 1)):
        (root / split).mkdir(parents=True, exist_ok=True)
        (root / "annotations_semseg" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.clip(
                np.full((h, w, 3), 137.0) + rng.normal(0, 3, (h, w, 3)), 0, 255
            ).astype(np.uint8)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[40:80, 40:80] = 1
            Image.fromarray(img).save(root / split / f"i{i}.jpg", quality=98)
            Image.fromarray(mask).save(
                root / "annotations_semseg" / split / f"i{i}.png"
            )


def _write_maskterial(root, n_train=6, n_test=2, size=(64, 72), layers=3):
    """Minimal MaskTerial release, in the layout the Zenodo zips unpack to."""
    from PIL import Image

    rng = np.random.default_rng(1)
    h, w = size
    for images_dir, masks_dir, n in (
        ("train_images", "train_semantic_masks", n_train),
        ("test_images", "test_semantic_masks", n_test),
    ):
        (root / images_dir).mkdir(parents=True, exist_ok=True)
        (root / masks_dir).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.clip(
                np.full((h, w, 3), 120.0) + rng.normal(0, 2, (h, w, 3)), 0, 255
            ).astype(np.uint8)
            # Layer ids 1..layers, as MaskTerial encodes them.
            mask = np.zeros((h, w), dtype=np.uint8)
            for layer in range(1, layers + 1):
                mask[10 * layer : 10 * layer + 6, 5:20] = layer
            Image.fromarray(img).save(root / images_dir / f"u{i}.png")
            Image.fromarray(mask).save(root / masks_dir / f"u{i}.png")


class TestMaskTerialConversion:
    def test_layout_matches_find_pairs(self, tmp_path):
        from flakeseg.data.dataset import find_pairs
        from flakeseg.data.maskterial import prepare

        src, dst = tmp_path / "hBN_Thin", tmp_path / "out"
        _write_maskterial(src)
        prepare(src, dst)

        for split, expected in (("train", 6), ("test", 2)):
            pairs = find_pairs(dst, split)
            assert len(pairs) == expected

    def test_binary_collapses_layer_ids(self, tmp_path):
        from PIL import Image

        from flakeseg.data.maskterial import prepare

        src, dst = tmp_path / "hBN_Thin", tmp_path / "out"
        _write_maskterial(src, layers=3)
        prepare(src, dst, binary=True)

        mask = np.array(Image.open(dst / "annotations_semseg" / "train" / "u0.png"))
        assert set(np.unique(mask).tolist()) == {0, 1}

    def test_keep_classes_preserves_layer_ids(self, tmp_path):
        from PIL import Image

        from flakeseg.data.maskterial import prepare

        src, dst = tmp_path / "hBN_Thin", tmp_path / "out"
        _write_maskterial(src, layers=3)
        manifest = prepare(src, dst, binary=False)

        mask = np.array(Image.open(dst / "annotations_semseg" / "train" / "u0.png"))
        assert set(np.unique(mask).tolist()) == {0, 1, 2, 3}
        assert manifest["binary"] is False

    def test_val_split_is_disjoint_and_deterministic(self, tmp_path):
        from flakeseg.data.dataset import find_pairs
        from flakeseg.data.maskterial import prepare

        src = tmp_path / "hBN_Thin"
        _write_maskterial(src, n_train=6)

        names = []
        for run in ("a", "b"):
            dst = tmp_path / run
            prepare(src, dst, val_fraction=0.5, seed=0)
            train = {p.stem for p, _ in find_pairs(dst, "train")}
            val = {p.stem for p, _ in find_pairs(dst, "val")}
            assert not (train & val), "val leaked into train"
            assert len(train) == 3 and len(val) == 3
            names.append((sorted(train), sorted(val)))
        assert names[0] == names[1], "partition is not deterministic across runs"

    def test_val_fraction_consuming_everything_is_rejected(self, tmp_path):
        from flakeseg.data.maskterial import prepare

        src, dst = tmp_path / "hBN_Thin", tmp_path / "out"
        _write_maskterial(src, n_train=4)
        with pytest.raises(ValueError, match="would consume all"):
            prepare(src, dst, val_fraction=1.0)

    def test_rejects_non_maskterial_directory(self, tmp_path):
        from flakeseg.data.maskterial import prepare

        src = tmp_path / "not_maskterial"
        (src / "junk").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="does not look like") as excinfo:
            prepare(src, tmp_path / "out")
        # The error has to name what it saw, or diagnosing it needs a round trip.
        assert "junk" in str(excinfo.value)

    def test_accepts_release_nested_one_level(self, tmp_path):
        from flakeseg.data.dataset import find_pairs
        from flakeseg.data.maskterial import prepare

        # Some zips carry a material directory, some unpack their splits flat.
        src = tmp_path / "raw"
        _write_maskterial(src / "hBN_Thin", n_train=4, n_test=2)
        prepare(src, tmp_path / "out")
        assert len(find_pairs(tmp_path / "out", "train")) == 4

    def test_ambiguous_nesting_is_refused(self, tmp_path):
        from flakeseg.data.maskterial import prepare

        src = tmp_path / "raw"
        _write_maskterial(src / "hBN_Thin", n_train=2, n_test=1)
        _write_maskterial(src / "WSe2", n_train=2, n_test=1)
        with pytest.raises(FileNotFoundError, match="several MaskTerial releases"):
            prepare(src, tmp_path / "out")

    def test_manifest_records_resolved_source(self, tmp_path):
        from flakeseg.data.maskterial import prepare

        src = tmp_path / "raw"
        _write_maskterial(src / "hBN_Thin", n_train=4, n_test=2)
        manifest = prepare(src, tmp_path / "out")
        assert manifest["source"].endswith("hBN_Thin")

    def test_reports_class_histogram_and_pairs(self, tmp_path):
        from flakeseg.data.maskterial import prepare, summarize

        src, dst = tmp_path / "hBN_Thin", tmp_path / "out"
        _write_maskterial(src, n_train=6, n_test=2, layers=2)
        manifest = prepare(src, dst, binary=False)

        train = next(s for s in manifest["splits"] if s["split"] == "train")
        assert train["pairs"] == 6
        assert set(train["class_histogram"]) == {"0", "1", "2"}
        # summarize() tells the user the num_classes their choice implies.
        assert "num_classes: 3" in summarize(manifest)

    def test_colour_coded_mask_is_refused_not_guessed(self, tmp_path):
        from PIL import Image

        from flakeseg.data.maskterial import read_mask

        path = tmp_path / "rgb.png"
        array = np.zeros((8, 8, 3), dtype=np.uint8)
        array[..., 0] = 1  # channels disagree: a real colour map, not class ids
        Image.fromarray(array).save(path)
        with pytest.raises(ValueError, match="colour-coded"):
            read_mask(path)


class TestContrastPolarity:
    def _stack(self, sign):
        """Flat substrate with a flake of the given sign."""
        image = np.full((64, 64, 3), 120.0, dtype=np.float32)
        image[20:40, 20:40] += sign * 20.0
        return image

    def test_dark_polarity_is_unchanged_default(self):
        from flakeseg.data.contrast import ContrastConfig

        assert ContrastConfig().polarity == "dark"

    def test_background_ignores_flake_of_matching_polarity(self):
        from flakeseg.data.contrast import ContrastConfig, estimate_background

        for sign, polarity in ((-1.0, "dark"), (1.0, "bright")):
            image = self._stack(sign)
            background = estimate_background(
                image, ContrastConfig(polarity=polarity, poly_downsample=1)
            )
            # Substrate level recovered despite the flake, so contrast is real.
            assert abs(float(background[0, 0, 0]) - 120.0) < 1.5

    def test_both_polarity_handles_either_sign(self):
        from flakeseg.data.contrast import ContrastConfig, estimate_background

        for sign in (-1.0, 1.0):
            image = self._stack(sign)
            background = estimate_background(
                image, ContrastConfig(polarity="both", poly_downsample=1)
            )
            assert abs(float(background[0, 0, 0]) - 120.0) < 1.5

    def test_morphological_polarity_selects_operator(self):
        from flakeseg.data.contrast import ContrastConfig, estimate_background

        image = self._stack(1.0)  # 20x20 bright flake on a 120 substrate
        # The structuring element has to exceed the flake for opening to erase it.
        dark_cfg = ContrastConfig(method="morphological", polarity="dark", morph_size=31)
        bright_cfg = ContrastConfig(method="morphological", polarity="bright", morph_size=31)
        # Closing leaves a bright flake in the background, so its contrast is
        # lost; opening removes it and the substrate level is recovered.
        assert float(estimate_background(image, dark_cfg)[30, 30, 0]) > 135.0
        assert float(estimate_background(image, bright_cfg)[30, 30, 0]) < 125.0

    def test_unknown_polarity_raises(self):
        from flakeseg.data.contrast import ContrastConfig, estimate_background

        with pytest.raises(ValueError, match="unknown polarity"):
            estimate_background(
                self._stack(-1.0), ContrastConfig(polarity="sideways")
            )
