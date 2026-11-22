"""Tests for the command-line quantization modes (CPU only).

Run from the repository root:
    python -m unittest tests.test_modes -v
"""

import argparse
import contextlib
import io
import unittest

from quantization import modes


def parse(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="pytorch")
    modes.add_arguments(parser)
    args = parser.parse_args(list(argv))
    modes.validate(args, parser)
    return args


def fails(*argv):
    with contextlib.redirect_stderr(io.StringIO()), unittest.TestCase().assertRaises(SystemExit):
        parse(*argv)
    return True


class TestModes(unittest.TestCase):
    def test_labels(self):
        calib = ("--act-calib", "max", "--calib-file", "c.json")
        self.assertEqual(modes.mode_name(parse("--quant", "INT8", *calib)), "INT8-W8A8")
        self.assertEqual(modes.mode_name(parse("--quant", "INT4")), "INT4-W4A32")
        self.assertEqual(modes.mode_name(parse("--quant", "INT4", "--act-format", "INT8", *calib)), "INT4-W4A8")
        self.assertEqual(modes.mode_name(parse("--quant", "NVFP4", "--act-dynamic")), "NVFP4-W4A4")

    def test_presets(self):
        args = parse("--quant", "NVFP4", "--act-dynamic")
        self.assertEqual(modes.weight_spec(args).describe(), "fp4_e2m1 blocks of 16, e4m3 scales")
        self.assertEqual(modes.activation_spec(args).granularity, "block")
        self.assertEqual(modes.weight_spec(parse("--quant", "MXFP4")).scale_format, "e8m0")

    def test_defaults(self):
        args = parse("--quant", "FP8")
        self.assertEqual(modes.weight_spec(args).granularity, "channel")
        self.assertIsNone(modes.activation_spec(args))
        self.assertEqual(modes.file_suffix(parse("--quant", "INT4", "--block-size", "32")), "_b32")

    def test_invalid(self):
        self.assertTrue(fails("--block-size", "16"))                                   # no --quant
        self.assertTrue(fails("--quant", "INT8", "--act-calib", "max"))                 # no --calib-file
        self.assertTrue(fails("--quant", "INT4", "--act-dynamic"))                      # dynamic needs blocks too
        self.assertTrue(fails("--quant", "INT8", "--scale-format", "e8m0"))             # block scales need blocks too

    def test_tensorrt_limits(self):
        calib = ("--act-calib", "max", "--calib-file", "c.json")
        parse("--backend", "tensorrt", "--quant", "INT8", *calib)                      # real W8A8
        parse("--backend", "tensorrt", "--quant", "FP4")                               # 4-bit weight storage
        self.assertTrue(fails("--backend", "tensorrt", "--quant", "FP4", *calib))      # no FP4 activations
        self.assertTrue(fails("--backend", "tensorrt", "--quant", "FP6"))              # simulation only
        self.assertTrue(fails("--backend", "tensorrt", "--quant", "NVFP4"))            # no block scales
        self.assertTrue(fails("--backend", "tensorrt", "--quant", "INT4", "--block-size", "32"))


if __name__ == "__main__":
    unittest.main()
