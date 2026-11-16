"""Tests for the INT8 Q/DQ ONNX (TensorRT parts skipped without CUDA/TensorRT).

Run from the repository root:
    python -m unittest quantization.test_qdq_onnx -v
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper

from models.unet_modern import UNetModern
from quantization.bn_fold import fold_batchnorm
from quantization.export_onnx import export_onnx
from quantization.int8 import QMAX, compute_amax, quantize, weight_axis
from quantization.qdq_onnx import quantize_onnx
from quantization.quantize_model import prepare_int8_model, quantizable_layers
from quantization.test_bn_fold import randomize_bn
from quantization.trt_engine import TensorRTModel, get_engine, kernel_dtypes, profile_from_sizes

HAS_TRT = torch.cuda.is_available() and importlib.util.find_spec("tensorrt") is not None


def small_model():
    torch.manual_seed(0)
    model = UNetModern(n_classes=1, n_channels=3, base_channels=8, batch_norm=True)
    randomize_bn(model)
    return fold_batchnorm(model.eval())


class TestKernelNames(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(kernel_dtypes("sm86_xmma_fprop_implicit_gemm_i8i8_i8i32_f32_nhwckrsc"),
                         {"input": "i8", "output": "i8", "weight": "i8", "accumulator": "i32"})
        self.assertEqual(kernel_dtypes("sm80_xmma_fprop_wngd_f32f32_f32_f32_nchwkcrs")["weight"], "f32")
        self.assertIsNone(kernel_dtypes("some_unknown_kernel"))


class TestQDQGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = small_model()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fp32 = export_onnx(cls.model, Path(cls.tmp.name) / "fp32.onnx", n_channels=3)
        cls.amax = {name: 2.0 for name in quantizable_layers(cls.model)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def quantized_graph(self, **kwargs):
        path = Path(self.tmp.name) / "int8.onnx"
        report = quantize_onnx(self.fp32, path, **kwargs)
        return onnx.load(str(path)).graph, report

    def test_weights_are_int8_and_match_reference(self):
        graph, report = self.quantized_graph(act_amax=self.amax)
        self.assertEqual(len(report["layers"]), 23)
        inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
        for name, module in quantizable_layers(self.model).items():
            stored = inits[f"{name}.weight_int8"]
            self.assertEqual(stored.dtype.name, "int8", name)
            expected, _ = quantize(module.weight.detach(), compute_amax(module.weight, axis=weight_axis(module)))
            self.assertTrue((torch.from_numpy(stored) == expected).all(), name)
            self.assertNotIn(f"model.{name}.weight", inits)  # FP32 copy removed

    def test_weight_dq_axis(self):
        graph, _ = self.quantized_graph()
        axes = {n.name: n.attribute[0].i for n in graph.node if n.name.endswith("/weight_dequantize")}
        self.assertEqual(axes["up1/weight_dequantize"], 1)
        self.assertEqual(axes["enc1.block.0/weight_dequantize"], 0)

    def test_w8a32_has_no_activation_quantizers(self):
        graph, _ = self.quantized_graph()
        self.assertFalse([n for n in graph.node if n.op_type == "QuantizeLinear"])

    def test_activation_scale(self):
        graph, _ = self.quantized_graph(act_amax=self.amax, fp_layers=["segment"])
        inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
        self.assertAlmostEqual(float(inits["enc1.block.0.input_scale"]), 2.0 / QMAX, places=7)
        self.assertNotIn("segment.input_scale", inits)
        self.assertNotIn("segment.weight_int8", inits)


@unittest.skipUnless(HAS_TRT, "needs CUDA and TensorRT")
class TestINT8Engine(unittest.TestCase):
    """W8A8 engine: INT8 kernels everywhere, output close to fake quant."""

    @classmethod
    def setUpClass(cls):
        torch.backends.cudnn.allow_tf32 = False
        model = small_model()
        cls.tmp = tempfile.TemporaryDirectory()
        fp32 = export_onnx(model, Path(cls.tmp.name) / "fp32.onnx", n_channels=3)
        # Realistic amax: max |input| of each layer on a calibration image.
        observed = {}
        hooks = [m.register_forward_pre_hook(lambda m, a, n=n: observed.__setitem__(n, a[0].abs().max().item()))
                 for n, m in quantizable_layers(model).items()]
        cls.x = torch.rand(1, 3, 64, 96)
        with torch.no_grad():
            model(cls.x)
        for hook in hooks:
            hook.remove()
        int8 = Path(cls.tmp.name) / "int8.onnx"
        quantize_onnx(fp32, int8, act_amax=observed)
        cls.engine = TensorRTModel(get_engine(int8, profile_from_sizes([(64, 96)])))
        cls.fake, _ = prepare_int8_model(model, act_amax=observed, inplace=False)

    @classmethod
    def tearDownClass(cls):
        del cls.engine
        cls.tmp.cleanup()

    def test_all_layers_int8(self):
        summary = self.engine.layer_summary()
        self.assertEqual(sum(summary["compute_layers"].values()), 23)
        self.assertTrue(all(":Int8->" in kind for kind in summary["compute_layers"]), summary)
        self.assertEqual(summary["accumulators"], {"i32": 23})

    def test_close_to_fake_quant(self):
        with torch.inference_mode():
            trt_out = self.engine(self.x.cuda()).cpu()
            fake_out = self.fake(self.x.clone())
        # Not bit-exact (INT32 accumulation order, requantization of fused
        # outputs), but the error must be a small fraction of the logit range.
        error = (trt_out - fake_out).abs().max() / fake_out.abs().max()
        self.assertLess(error.item(), 0.05)


if __name__ == "__main__":
    unittest.main()
