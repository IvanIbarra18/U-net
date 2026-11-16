"""Unit tests for quantization/bn_fold.py.

Run from the repository root:
    python -m unittest quantization.test_bn_fold -v
"""

import unittest

import torch
import torch.nn as nn

from models.unet_modern import UNetModern
from quantization.bn_fold import fold_batchnorm, fold_conv_bn


def randomize_bn(model: nn.Module) -> None:
    """Non-trivial BN statistics (fresh BN is the identity, which proves nothing)."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-1, 1)
                m.running_var.uniform_(0.1, 4)
                m.weight.uniform_(0.2, 3)
                m.bias.uniform_(-1, 1)


class TestBNFold(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_single_conv_bn(self):
        conv = nn.Conv2d(4, 8, 3, padding=1, bias=False)
        bn = nn.BatchNorm2d(8)
        randomize_bn(bn)
        bn.eval()
        x = torch.randn(2, 4, 16, 16)
        torch.testing.assert_close(fold_conv_bn(conv, bn)(x), bn(conv(x)), rtol=1e-5, atol=1e-5)

    def test_conv_with_bias(self):
        conv = nn.Conv2d(4, 8, 3, padding=1, bias=True)
        bn = nn.BatchNorm2d(8)
        randomize_bn(bn)
        bn.eval()
        x = torch.randn(2, 4, 16, 16)
        torch.testing.assert_close(fold_conv_bn(conv, bn)(x), bn(conv(x)), rtol=1e-5, atol=1e-5)

    def test_unet_same_output(self):
        model = UNetModern(n_classes=1, n_channels=3, base_channels=8, batch_norm=True)
        randomize_bn(model)
        model.eval()
        folded = fold_batchnorm(model)

        # 2 BN per DoubleConv x 9 DoubleConvs (4 enc + bottleneck + 4 dec).
        self.assertEqual(folded.folded_bn_count, 18)
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in folded.modules()))
        # The original model is untouched.
        self.assertEqual(sum(isinstance(m, nn.BatchNorm2d) for m in model.modules()), 18)

        x = torch.rand(2, 3, 64, 80)
        with torch.no_grad():
            torch.testing.assert_close(folded(x), model(x), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
