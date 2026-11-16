"""Unit tests for quantization/quantize_model.py.

Run from the repository root:
    python -m unittest quantization.test_quantize_model -v
"""

import unittest

import torch
import torch.nn as nn

from models.unet_modern import UNetModern
from quantization.int8 import QMAX, compute_amax, compute_scale, weight_axis
from quantization.quantize_model import prepare_int8_model, quantizable_layers
from quantization.test_bn_fold import randomize_bn


def is_on_int8_grid(module) -> bool:
    """Every weight equals k / s_c with integer k in [-127, 127]."""
    weight = module.weight.detach()
    scale = compute_scale(compute_amax(weight, axis=weight_axis(module)))
    levels = weight * scale
    return bool(
        torch.allclose(levels, levels.round(), atol=1e-3)
        and levels.abs().max() <= QMAX + 1e-3
    )


class TestPrepareInt8(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = UNetModern(n_classes=1, n_channels=3, base_channels=8, batch_norm=True)
        randomize_bn(self.model)
        self.model.eval()

    def test_layer_list(self):
        names = list(quantizable_layers(self.model))
        # 9 DoubleConvs x 2 convs + 4 ConvTranspose2d + final 1x1 conv.
        self.assertEqual(len(names), 23)
        self.assertEqual(names[0], "enc1.block.0")
        self.assertEqual(names[-1], "segment")

    def test_weights_on_grid_and_original_untouched(self):
        original = {k: v.clone() for k, v in self.model.state_dict().items()}
        quantized, stats = prepare_int8_model(self.model)

        self.assertEqual(len(stats), 23)
        for name, module in quantizable_layers(quantized).items():
            self.assertTrue(is_on_int8_grid(module), name)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]), key)

    def test_requantizing_is_idempotent(self):
        # Values already on the grid do not move when quantized again.
        quantized, _ = prepare_int8_model(self.model)
        again, _ = prepare_int8_model(quantized)
        for (name, a), (_, b) in zip(quantizable_layers(quantized).items(), quantizable_layers(again).items()):
            torch.testing.assert_close(a.weight, b.weight, rtol=0, atol=1e-6, msg=name)

    def test_fp_layers_are_skipped(self):
        quantized, stats = prepare_int8_model(self.model, fp_layers=["enc1.block.0", "segment"])
        self.assertEqual(len(stats), 21)
        layers = quantizable_layers(quantized)
        # segment has no BN, so kept in FP it equals the original weight.
        self.assertTrue(torch.equal(layers["segment"].weight, self.model.segment.weight))
        # enc1.block.0 is BN-folded but not quantized: off the INT8 grid.
        self.assertFalse(is_on_int8_grid(layers["enc1.block.0"]))
        self.assertTrue(is_on_int8_grid(layers["enc1.block.3"]))

    def test_unknown_layer_name(self):
        with self.assertRaises(ValueError):
            prepare_int8_model(self.model, fp_layers=["enc9"])

    def test_output_close_to_fp32(self):
        quantized, _ = prepare_int8_model(self.model)
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            reference, output = self.model(x), quantized(x)
        relative_error = ((output - reference).norm() / reference.norm()).item()
        self.assertLess(relative_error, 0.05)
        self.assertGreater(relative_error, 0.0)


if __name__ == "__main__":
    unittest.main()
