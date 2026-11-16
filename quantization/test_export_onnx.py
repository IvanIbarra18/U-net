"""Tests for the ONNX export path.

Run from the repository root:
    python -m unittest quantization.test_export_onnx -v
"""

import unittest

import torch
import torch.nn as nn

from models.unet_modern import UNetModern
from quantization.bn_fold import fold_batchnorm
from quantization.export_onnx import ExportableUNet, upsample_to
from quantization.test_bn_fold import randomize_bn

# Odd sizes exercise output_padding = 1 at one or more decoder levels
# (16x17 is the smallest valid input). Kept small so the test runs on the CPU.
SIZES = [(64, 64), (97, 131), (101, 77), (16, 17)]


class TestUpsampleTo(unittest.TestCase):
    def test_matches_output_size(self):
        torch.manual_seed(0)
        up = nn.ConvTranspose2d(6, 4, kernel_size=2, stride=2)
        for h, w in [(5, 7), (6, 6), (4, 9)]:
            x = torch.randn(2, 6, h, w)
            for oh, ow in [(2 * h, 2 * w), (2 * h + 1, 2 * w), (2 * h + 1, 2 * w + 1)]:
                skip = torch.empty(2, 4, oh, ow)
                torch.testing.assert_close(upsample_to(up, x, skip), up(x, output_size=skip.size()), rtol=0, atol=1e-6)

    def test_extra_row_is_bias(self):
        up = nn.ConvTranspose2d(3, 2, kernel_size=2, stride=2)
        y = upsample_to(up, torch.randn(1, 3, 4, 4), torch.empty(1, 2, 9, 9))
        bias = up.bias.view(1, -1, 1).detach()
        torch.testing.assert_close(y[..., -1, :], bias.expand(1, 2, 9), rtol=0, atol=0)


class TestExportableUNet(unittest.TestCase):
    def test_matches_unet(self):
        torch.manual_seed(0)
        model = UNetModern(n_classes=1, n_channels=3, base_channels=8, batch_norm=True)
        randomize_bn(model)
        for net in (model.eval(), fold_batchnorm(model)):
            wrapper = ExportableUNet(net)
            with torch.no_grad():
                for h, w in SIZES:
                    x = torch.rand(1, 3, h, w)
                    torch.testing.assert_close(wrapper(x), net(x), rtol=0, atol=1e-5, msg=f"size {h}x{w}")

    def test_shares_weights(self):
        model = UNetModern(n_classes=1, n_channels=3, base_channels=8)
        self.assertIs(ExportableUNet(model).model, model)


if __name__ == "__main__":
    unittest.main()
