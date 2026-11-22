"""Unit tests for activation calibration and input quantizers.

Run from the repository root:
    python -m unittest tests.test_calibrate -v
"""

import unittest

import numpy as np
import torch

from models.unet_modern import UNetModern
from quantization.calibrate import InputObserver, amax_from_entropy, amax_from_percentile, kl_divergence, without_zeros
from quantization.int8 import compute_scale, fake_quantize, fake_quantize_
from quantization.quantize_model import attach_input_quantizers, prepare_int8_model, quantizable_layers
from tests.test_bn_fold import randomize_bn


def histogram(values: np.ndarray, bins: int = 2048):
    amax = np.abs(values).max()
    hist, edges = np.histogram(np.abs(values), bins=bins, range=(0, amax))
    return hist.astype(np.float64), edges


class TestThresholds(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_percentile_uniform(self):
        hist, edges = histogram(self.rng.uniform(0, 1, 1_000_000))
        self.assertAlmostEqual(amax_from_percentile(hist, edges, 90), 0.9, delta=2e-3)
        self.assertAlmostEqual(amax_from_percentile(hist, edges, 100), edges[-1], delta=1e-3)

    def test_kl(self):
        p = np.array([1.0, 2.0, 3.0])
        self.assertEqual(kl_divergence(p, p), 0.0)
        self.assertGreater(kl_divergence(p, np.array([3.0, 2.0, 1.0])), 0)
        self.assertEqual(kl_divergence(p, np.array([1.0, 0.0, 3.0])), float("inf"))

    def test_entropy_clips_rare_outliers(self):
        # Mostly Gaussian data, plus a few huge outliers. Max calibration is
        # controlled by the outliers, but entropy should clip them.
        values = np.concatenate([self.rng.normal(0, 1, 2_000_000), [60.0, -55.0, 58.0]])
        hist, edges = histogram(values)
        amax, divergences = amax_from_entropy(hist, edges)
        self.assertLess(amax, 20.0)
        self.assertGreater(amax, 3.0)
        self.assertTrue(np.isfinite(divergences.min()))

    def test_entropy_keeps_range_without_outliers(self):
        # Uniform data has no long tail to clip, so we keep almost all of it.
        hist, edges = histogram(self.rng.uniform(0, 1, 2_000_000))
        amax, _ = amax_from_entropy(hist, edges)
        self.assertGreater(amax, 0.9)


class TestZeroSpike(unittest.TestCase):
    def test_skip_zeros_avoids_overclipping(self):
        # Data like after a ReLU: half of it is exact zeros, and the rest is |N(0, 1)|.
        rng = np.random.default_rng(0)
        bulk = np.abs(rng.normal(0, 1, 1_000_000))
        values = np.concatenate([np.zeros(1_000_000), bulk])
        hist, edges = histogram(values)
        plain, _ = amax_from_entropy(hist, edges)
        skipped, _ = amax_from_entropy(without_zeros(hist, 1_000_000), edges)
        clipped = lambda amax: (bulk > amax).mean()
        # Because of the spike at zero, plain entropy clips much more of the main data.
        self.assertGreater(clipped(plain), 5 * clipped(skipped))
        self.assertLess(clipped(skipped), 0.01)

    def test_without_zeros_only_touches_bin0(self):
        hist = np.array([10.0, 3.0, 2.0])
        np.testing.assert_array_equal(without_zeros(hist, 7), [3.0, 3.0, 2.0])
        np.testing.assert_array_equal(hist, [10.0, 3.0, 2.0])


class TestObserver(unittest.TestCase):
    def test_counts_and_does_not_modify_input(self):
        torch.manual_seed(0)
        x = torch.relu(torch.randn(2, 4, 32, 32))
        before = x.clone()
        observer = InputObserver(bins=64)
        observer(None, (x,))
        observer.mode = "hist"
        observer(None, (x,))
        self.assertTrue(torch.equal(x, before))
        self.assertAlmostEqual(observer.amax, x.max().item())
        self.assertEqual(observer.min, 0.0)
        self.assertEqual(observer.hist.sum().item(), x.numel())
        self.assertEqual(observer.zeros, int((x == 0).sum()))


class TestInputQuantizers(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = UNetModern(n_classes=1, n_channels=3, base_channels=8, batch_norm=True)
        randomize_bn(self.model)
        self.model.eval()
        self.amax = {name: 2.0 for name in quantizable_layers(self.model)}

    def test_inplace_matches_out_of_place(self):
        x = torch.randn(1000)
        amax = torch.tensor(1.5)
        torch.testing.assert_close(fake_quantize_(x.clone(), amax), fake_quantize(x, amax), rtol=0, atol=0)

    def test_inplace_hooks_match_reference(self):
        # In-place quantization must not damage tensors that are used again later (skips).
        inplace, _ = prepare_int8_model(self.model, act_amax=self.amax, inplace=True)
        reference, _ = prepare_int8_model(self.model, act_amax=self.amax, inplace=False)
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(inplace(x.clone()), reference(x.clone()), rtol=0, atol=0)

    def test_layer_inputs_are_on_grid(self):
        quantized, _ = prepare_int8_model(self.model, act_amax=self.amax, fp_layers=["segment"])
        seen = {}
        for name, module in quantizable_layers(quantized).items():
            # We register this after the quantizer hook, so it sees the quantized input.
            module.register_forward_pre_hook(lambda m, a, name=name: seen.__setitem__(name, a[0].clone()))
        with torch.no_grad():
            quantized(torch.rand(1, 3, 64, 64))
        scale = compute_scale(torch.tensor(2.0))
        for name, value in seen.items():
            levels = value * scale
            on_grid = torch.allclose(levels, levels.round(), atol=1e-3) and levels.abs().max() <= 127 + 1e-3
            self.assertEqual(on_grid, name != "segment", name)

    def test_missing_amax(self):
        del self.amax["up1"]
        with self.assertRaises(ValueError):
            prepare_int8_model(self.model, act_amax=self.amax)
        # This is fine if that layer stays in FP.
        prepare_int8_model(self.model, act_amax=self.amax, fp_layers=["up1"])


if __name__ == "__main__":
    unittest.main()
