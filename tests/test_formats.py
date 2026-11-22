"""Tests for formats.py (CPU only).

Run from the repository root:
    python -m unittest tests.test_formats -v
"""

import unittest

import ml_dtypes
import numpy as np
import torch

from quantization.formats import FORMATS, QuantSpec, fake_quantize_spec, get_format, round_to_format
from quantization.int8 import compute_amax, fake_quantize

# Separate reference versions of each minifloat grid, written in another way.
REFERENCE_DTYPES = {
    "fp8_e4m3": ml_dtypes.float8_e4m3fn,
    "fp6_e2m3": ml_dtypes.float6_e2m3fn,
    "fp6_e3m2": ml_dtypes.float6_e3m2fn,
    "fp4_e2m1": ml_dtypes.float4_e2m1fn,
}


class TestRounding(unittest.TestCase):
    def test_minifloats_match_ml_dtypes(self):
        torch.manual_seed(0)
        for name, dtype in REFERENCE_DTYPES.items():
            fmt = get_format(name)
            # Values from the whole range, including subnormals and exact ties.
            x = torch.cat([torch.randn(200_000) * fmt.qmax / 3, torch.rand(50_000) * 2 * fmt.qmax - fmt.qmax])
            ours = round_to_format(x, fmt)
            clipped = x.clamp(-fmt.qmax, fmt.qmax).numpy()  # ml_dtypes saturates in a different way above qmax
            reference = torch.from_numpy(clipped.astype(dtype).astype(np.float32))
            self.assertTrue(torch.equal(ours, reference), name)

    def test_e4m3_matches_torch(self):
        x = torch.randn(100_000) * 100
        fmt = get_format("fp8_e4m3")
        expected = x.clamp(-448, 448).to(torch.float8_e4m3fn).float()
        self.assertTrue(torch.equal(round_to_format(x, fmt), expected))

    def test_fp4_values(self):
        values = round_to_format(torch.linspace(-7, 7, 1001), get_format("fp4_e2m1")).unique().tolist()
        self.assertEqual(sorted(set(abs(v) for v in values)), [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    def test_int4_range(self):
        self.assertEqual(round_to_format(torch.tensor([-9.0, -6.6, 0.4, 7.6]), get_format("int4")).tolist(), [-7, -7, 0, 7])


class TestGranularity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.w = torch.randn(16, 32, 3, 3)

    def test_int8_channel_equals_reference(self):
        # The general code must give exactly the same result as int8.py.
        spec = QuantSpec("int8", "channel")
        expected = fake_quantize(self.w, compute_amax(self.w, axis=0))
        self.assertTrue(torch.equal(fake_quantize_spec(self.w, spec, axis=0), expected))

    def test_int8_tensor_with_amax_equals_reference(self):
        x = torch.relu(torch.randn(2, 8, 10, 10))
        amax = torch.tensor(1.7)
        self.assertTrue(torch.equal(fake_quantize_spec(x, QuantSpec("int8", "tensor"), amax=amax), fake_quantize(x, amax)))

    def test_block_scales_follow_the_blocks(self):
        # In each block of 16 input channels, its own maximum maps to qmax.
        spec = QuantSpec("int4", "block", 16)
        out = fake_quantize_spec(self.w, spec, axis=1)
        for start in (0, 16):
            block = self.w[0, start:start + 16, 1, 1]
            step = block.abs().max() / 7
            levels = out[0, start:start + 16, 1, 1] / step
            torch.testing.assert_close(levels, levels.round(), atol=1e-4, rtol=0)

    def test_blocks_reduce_error(self):
        # The sizes of the values are very different along K. With one scale per
        # channel, the small values fall below the grid step. Per-block scales keep them.
        x = torch.randn(64, 256) * torch.logspace(-2, 1, 256)
        small = slice(0, 128)
        err = lambda spec, axis: ((fake_quantize_spec(x, spec, axis=axis) - x)[:, small].norm() / x[:, small].norm()).item()
        for name in ("int4", "fp4_e2m1"):
            # channel: one scale per output row (axis 0). block: blocks along K (axis 1).
            self.assertLess(err(QuantSpec(name, "block", 16), 1), err(QuantSpec(name, "channel"), 0) / 2, name)

    def test_padding_for_uneven_blocks(self):
        w = torch.randn(4, 3, 3, 3)  # first conv: 3 input channels, block size 16
        out = fake_quantize_spec(w, QuantSpec("fp4_e2m1", "block", 16), axis=1)
        self.assertEqual(out.shape, w.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_presets(self):
        for name in ("nvfp4", "mxfp4"):
            spec = QuantSpec.preset(name)
            out = fake_quantize_spec(self.w, spec, axis=1)
            rel = (out - self.w).norm() / self.w.norm()
            self.assertLess(rel.item(), 0.2, name)
        # MX scales are powers of two, so every block maximum lands in [4, 8) * 2^k.
        self.assertEqual(QuantSpec.preset("mxfp4").describe(), "fp4_e2m1 blocks of 32, e8m0 scales")

    def test_zeros(self):
        z = torch.zeros(4, 16, 3, 3)
        for name in FORMATS:
            for spec in (QuantSpec(name, "channel"), QuantSpec(name, "block", 16)):
                self.assertTrue(torch.equal(fake_quantize_spec(z, spec, axis=1), z), (name, spec))

    def test_invalid_specs(self):
        with self.assertRaises(ValueError):
            QuantSpec("int3")
        with self.assertRaises(ValueError):
            QuantSpec("int4", "block")
        with self.assertRaises(ValueError):
            QuantSpec("int4", "channel", scale_format="e8m0")


if __name__ == "__main__":
    unittest.main()
