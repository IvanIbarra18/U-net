"""Tests for the INT8 Q/DQ ONNX (the TensorRT parts are skipped without CUDA or TensorRT).

Run from the repository root:
    python -m unittest tests.test_qdq_onnx -v
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
from quantization.formats import QuantSpec, fake_quantize_spec, get_format, round_to_format
from quantization.int8 import QMAX, compute_amax, compute_scale, quantize, weight_axis
from quantization.qdq_onnx import layer_name, quantize_onnx
from quantization.quantize_model import prepare_int8_model, quantizable_layers
from tests.test_bn_fold import randomize_bn
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
            self.assertNotIn(f"model.{name}.weight", inits)  # the FP32 copy was removed

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


class TestFP8Graph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = small_model()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fp32 = export_onnx(cls.model, Path(cls.tmp.name) / "fp32_d2s.onnx", n_channels=3, upsample="depth2space")
        cls.amax = {name: 2.0 for name in quantizable_layers(cls.model)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_layer_names(self):
        self.assertEqual(layer_name("up_1x1.up4.weight"), "up4")
        self.assertEqual(layer_name("model.enc1.block.0.weight"), "enc1.block.0")

    def test_fp8_codes_and_scales(self):
        path = Path(self.tmp.name) / "fp8.onnx"
        report = quantize_onnx(self.fp32, path, act_amax=self.amax, fmt="fp8")
        model = onnx.load(str(path))
        self.assertEqual(len(report["layers"]), 23)
        self.assertTrue(all(n.op_type != "ConvTranspose" for n in model.graph.node))
        self.assertGreaterEqual(next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 19)
        inits = {init.name: init for init in model.graph.initializer}
        self.assertEqual(onnx.TensorProto.DataType.Name(inits["enc2.block.0.weight_fp8"].data_type), "FLOAT8E4M3FN")
        self.assertAlmostEqual(float(numpy_helper.to_array(inits["enc1.block.0.input_scale"])), 2.0 / 448, places=7)
        # The codes are w * 448 / amax rounded to E4M3 (448 = max |code| per channel).
        # These are the same values as in the simulation in formats.py.
        w = self.model.enc2.block[0].weight.detach()
        codes = torch.from_numpy(numpy_helper.to_array(inits["enc2.block.0.weight_fp8"]).astype("float32"))
        expected = round_to_format(w * compute_scale(compute_amax(w, axis=0), 448.0), get_format("fp8_e4m3"))
        self.assertTrue(torch.equal(codes, expected))
        self.assertEqual(codes.abs().amax(dim=(1, 2, 3)).max().item(), 448.0)

    def test_4bit_weight_only(self):
        for fmt, name, dtype in (("int4", "int4", "INT4"), ("fp4", "fp4_e2m1", "FLOAT4E2M1")):
            path = Path(self.tmp.name) / f"{fmt}.onnx"
            quantize_onnx(self.fp32, path, fmt=fmt)
            inits = {init.name: init for init in onnx.load(str(path)).graph.initializer}
            init = inits[f"enc2.block.0.weight_{fmt}"]
            self.assertEqual(onnx.TensorProto.DataType.Name(init.data_type), dtype)
            # The dequantized codes are equal to the fake-quant simulation of the same weights.
            codes = torch.from_numpy(numpy_helper.to_array(init).astype("float32"))
            scale = torch.from_numpy(numpy_helper.to_array(inits["enc2.block.0.weight_scale"]).copy()).view(-1, 1, 1, 1)
            w = self.model.enc2.block[0].weight.detach()
            torch.testing.assert_close(codes * scale, fake_quantize_spec(w, QuantSpec(name, "channel"), axis=0), rtol=1e-6, atol=1e-7)
            with self.assertRaises(ValueError):  # TensorRT has no 4-bit activation kernels
                quantize_onnx(self.fp32, path, act_amax=self.amax, fmt=fmt)


@unittest.skipUnless(HAS_TRT, "needs CUDA and TensorRT")
class TestFP8Engine(unittest.TestCase):
    """depth2space + FP8 Q/DQ: every conv (also the upsamplers) runs in FP8."""

    def test_all_layers_fp8(self):
        torch.backends.cudnn.allow_tf32 = False
        model = small_model()
        with tempfile.TemporaryDirectory() as tmp:
            fp32 = export_onnx(model, Path(tmp) / "fp32_d2s.onnx", n_channels=3, upsample="depth2space")
            amax = {name: 4.0 for name in quantizable_layers(model)}
            fp8 = Path(tmp) / "fp8.onnx"
            quantize_onnx(fp32, fp8, act_amax=amax, fmt="fp8")
            engine = TensorRTModel(get_engine(fp8, profile_from_sizes([(64, 96)])))
            summary = engine.layer_summary()
            del engine
        self.assertEqual(sum(summary["compute_layers"].values()), 23, summary)
        self.assertTrue(all(":FP8->" in kind for kind in summary["compute_layers"]), summary)
        self.assertEqual(summary["weights"], {"e4m3": 23})
        self.assertEqual(summary["accumulators"], {"f32": 23})


@unittest.skipUnless(HAS_TRT, "needs CUDA and TensorRT")
class TestINT8Engine(unittest.TestCase):
    """W8A8 engine: INT8 kernels everywhere, and the output is close to fake quant."""

    @classmethod
    def setUpClass(cls):
        torch.backends.cudnn.allow_tf32 = False
        model = small_model()
        cls.tmp = tempfile.TemporaryDirectory()
        fp32 = export_onnx(model, Path(cls.tmp.name) / "fp32.onnx", n_channels=3)
        # A realistic amax: the max |input| of each layer on a calibration image.
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
        # Every kernel whose name we can read must sum in INT32. (Some classic-builder
        # kernels, on the L40S for example, have no readable name: "unparsed".)
        self.assertEqual(set(summary["accumulators"]) - {"unparsed"}, {"i32"}, summary)

    def test_close_to_fake_quant(self):
        with torch.inference_mode():
            trt_out = self.engine(self.x.cuda()).cpu()
            fake_out = self.fake(self.x.clone())
        # The result is not bit-exact (because of the INT32 summing order and the
        # requantization of fused outputs), but the error must be a small part of
        # the logit range.
        error = (trt_out - fake_out).abs().max() / fake_out.abs().max()
        self.assertLess(error.item(), 0.05)


if __name__ == "__main__":
    unittest.main()
