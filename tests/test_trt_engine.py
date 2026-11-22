"""Tests for the TensorRT path (skipped without CUDA or TensorRT).

Run from the repository root:
    python -m unittest tests.test_trt_engine -v
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from models.unet_modern import UNetModern
from quantization.bn_fold import fold_batchnorm
from quantization.export_onnx import export_onnx
from tests.test_bn_fold import randomize_bn
from quantization.trt_engine import TensorRTModel, get_engine, profile_from_sizes

HAS_TRT = torch.cuda.is_available() and importlib.util.find_spec("tensorrt") is not None


class TestProfile(unittest.TestCase):
    def test_profile_covers_all_sizes(self):
        profile = profile_from_sizes([(100, 200), (50, 300), (100, 200)])
        self.assertEqual(profile, {"min": [50, 200], "opt": [100, 200], "max": [100, 300]})


@unittest.skipUnless(HAS_TRT, "needs CUDA and TensorRT")
class TestEngine(unittest.TestCase):
    """Small U-Net -> ONNX -> FP32 engine (TF32 off). The output must match PyTorch."""

    @classmethod
    def setUpClass(cls):
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(0)
        model = UNetModern(n_classes=1, n_channels=3, base_channels=8, batch_norm=True)
        randomize_bn(model)
        cls.model = fold_batchnorm(model.eval()).cuda()
        cls.tmp = tempfile.TemporaryDirectory()
        onnx_path = export_onnx(cls.model, Path(cls.tmp.name) / "unet.onnx", n_channels=3)
        cls.profile = profile_from_sizes([(16, 16), (64, 96), (101, 77)])
        engine_path = get_engine(onnx_path, cls.profile, tf32=False)
        cls.engine = TensorRTModel(engine_path)
        cls.engine_path = engine_path

    @classmethod
    def tearDownClass(cls):
        del cls.engine
        cls.tmp.cleanup()

    def test_matches_pytorch_on_odd_sizes(self):
        with torch.inference_mode():
            for size in [(64, 96), (101, 77), (17, 33)]:
                x = torch.rand(1, 3, *size, device="cuda")
                torch.testing.assert_close(self.engine(x), self.model(x), rtol=1e-4, atol=1e-4, msg=str(size))

    def test_fp32_without_tf32(self):
        summary = self.engine.layer_summary()
        self.assertEqual(summary["tf32_tactics"], 0)
        self.assertEqual(sum(summary["compute_layers"].values()), 23)  # 19 conv + 4 deconv
        self.assertTrue(all("Float->Float" in kind for kind in summary["compute_layers"]), summary)

    def test_cached_engine_is_reused(self):
        self.assertEqual(get_engine(Path(self.tmp.name) / "unet.onnx", self.profile), self.engine_path)

    def test_rejects_shape_outside_profile(self):
        with self.assertRaises(ValueError):
            self.engine(torch.rand(1, 3, 200, 200, device="cuda"))


if __name__ == "__main__":
    unittest.main()
