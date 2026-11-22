"""Unit tests for quantization/int8.py (with unittest from the standard library).

Run from the repository root:
    python -m unittest tests.test_int8 -v
"""

import unittest

import torch

from quantization.int8 import (
    QMAX,
    compute_amax,
    compute_scale,
    dequantize,
    fake_quantize,
    quantization_error,
    quantize,
    weight_axis,
)


class TestScaleQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_paper_example_values(self):
        # amax = 0.5 -> s = 254; 0.123 * 254 = 31.24 -> 31 -> 31/254.
        amax = torch.tensor(0.5)
        self.assertAlmostEqual(compute_scale(amax).item(), 254.0, places=4)
        x_q, scale = quantize(torch.tensor([0.123]), amax)
        self.assertEqual(x_q.item(), 31)
        self.assertAlmostEqual(dequantize(x_q, scale).item(), 31 / 254, places=6)

    def test_integer_range_is_symmetric(self):
        # Values far outside the range saturate at +-127. -128 never appears.
        x = torch.linspace(-10, 10, 10001)
        x_q, _ = quantize(x, torch.tensor(1.0))
        self.assertEqual(x_q.dtype, torch.int8)
        self.assertEqual(x_q.min().item(), -QMAX)
        self.assertEqual(x_q.max().item(), QMAX)

    def test_rounding_error_bound(self):
        # Inside [-amax, amax], the error is at most half a step: amax / 254.
        amax = torch.tensor(3.0)
        x = (torch.rand(100_000) * 2 - 1) * amax
        error = (x - fake_quantize(x, amax)).abs().max().item()
        self.assertLessEqual(error, amax.item() / (2 * QMAX) + 1e-6)

    def test_clipping(self):
        amax = torch.tensor(1.0)
        x = torch.tensor([-5.0, -1.0, 1.0, 5.0])
        torch.testing.assert_close(fake_quantize(x, amax), torch.tensor([-1.0, -1.0, 1.0, 1.0]))

    def test_grid_values_are_exact(self):
        # A value that is already on the grid does not change after quantize and dequantize.
        amax = torch.tensor(2.0)
        grid = torch.arange(-QMAX, QMAX + 1, dtype=torch.float32) / compute_scale(amax)
        torch.testing.assert_close(fake_quantize(grid, amax), grid, rtol=0, atol=1e-7)

    def test_fake_quantize_keeps_dtype(self):
        x = torch.randn(16)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self.assertEqual(fake_quantize(x.to(dtype), compute_amax(x)).dtype, dtype)

    def test_zero_tensor_has_no_nan(self):
        x = torch.zeros(8)
        self.assertTrue(torch.equal(fake_quantize(x, compute_amax(x)), x))


class TestPerChannel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_weight_axis(self):
        self.assertEqual(weight_axis(torch.nn.Conv2d(3, 8, 3)), 0)
        self.assertEqual(weight_axis(torch.nn.ConvTranspose2d(8, 4, 2, stride=2)), 1)
        with self.assertRaises(TypeError):
            weight_axis(torch.nn.ReLU())

    def _check_each_channel_reaches_qmax(self, module):
        axis = weight_axis(module)
        weight = module.weight.detach()
        amax = compute_amax(weight, axis=axis)
        self.assertEqual(list(amax.shape), [weight.shape[d] if d == axis else 1 for d in range(4)])
        x_q, _ = quantize(weight, amax)
        reduce_dims = [d for d in range(4) if d != axis]
        # Max calibration: in every output channel, the largest |w| maps to 127.
        per_channel_max = x_q.abs().amax(dim=reduce_dims).to(torch.int32)
        self.assertTrue(torch.all(per_channel_max == QMAX))

    def test_conv2d_output_channels(self):
        self._check_each_channel_reaches_qmax(torch.nn.Conv2d(16, 32, 3))

    def test_conv_transpose_output_channels(self):
        # The weight is [C_in, C_out, kH, kW], so per output channel means axis 1.
        self._check_each_channel_reaches_qmax(torch.nn.ConvTranspose2d(32, 16, 2, stride=2))

    def test_per_channel_beats_per_tensor_with_uneven_channels(self):
        # Channels with very different sizes (like after BN folding). One scale
        # for the whole tensor wastes resolution on the small channels.
        weight = torch.randn(32, 16, 3, 3) * torch.logspace(-3, 0, 32).view(32, 1, 1, 1)
        per_tensor = quantization_error(weight, compute_amax(weight))
        per_channel = quantization_error(weight, compute_amax(weight, axis=0))
        self.assertGreater(per_channel["sqnr_db"], per_tensor["sqnr_db"] + 10)
        self.assertEqual(per_channel["clipped_fraction"], 0.0)

    def test_zero_channel_has_no_nan(self):
        weight = torch.randn(4, 2, 3, 3)
        weight[1] = 0
        out = fake_quantize(weight, compute_amax(weight, axis=0))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.equal(out[1], weight[1]))


if __name__ == "__main__":
    unittest.main()
