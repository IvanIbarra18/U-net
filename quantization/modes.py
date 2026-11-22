"""The quantization modes that you can choose in test.py.

This module is the one place that turns the command-line options into what
quantize_model.py (simulation) and qdq_onnx.py (TensorRT) need.

    --quant          the weight format (and the default activation format):
                     INT8, INT4, FP8 (E4M3), FP6 (E2M3), FP4 (E2M1), or
                     NVFP4 and MXFP4 (FP4 presets with block scales).
    --block-size N   weights get one scale for every N input channels
                     (0 means one scale per output channel). The default is
                     16 for NVFP4 and 32 for MXFP4.
    --scale-format   how the block scales are stored: fp32, e4m3 (NVFP4) or
                     e8m0 (MX).
    --act-calib M    activations per tensor, with a static amax read from
                     --calib-file (method M: max, percentile-99.99,
                     entropy-skip-zeros, and others).
    --act-dynamic    activations per block of input channels. The amax is
                     computed on the fly, with no calibration (this is how
                     NVFP4 and MX work).
    --act-format F   the activation format, if it is different from the
                     weights (for example, --quant INT4 --act-format INT8 = W4A8).
    --fp-layers      layers that stay in FP32 (both weights and input).

What each backend can do (verified on an RTX 5080 with TensorRT 11.4):
    pytorch   every mode, but SIMULATED. The values are on the format's grid,
              but storage and kernels are FP32.
    tensorrt  INT8 and FP8: real W8A8 (per-channel weights and per-tensor
              activations), or weight-only (then TensorRT folds those weights
              back to FP32). INT4 and FP4: weight-only, per channel. The
              engine stores the weights in 4 bits and dequantizes them for
              FP32 convolutions. There are no 4-bit activations, no block
              scales and no FP6.
"""

from pathlib import Path

from quantization.formats import QuantSpec

QUANT_MODES = ("NONE", "INT8", "INT4", "FP8", "FP6", "FP4", "NVFP4", "MXFP4")
ACT_FORMATS = ("INT8", "INT4", "FP8", "FP6", "FP4")
FORMAT_OF = {"INT8": "int8", "INT4": "int4", "FP8": "fp8_e4m3", "FP6": "fp6_e2m3", "FP4": "fp4_e2m1",
             "NVFP4": "fp4_e2m1", "MXFP4": "fp4_e2m1"}
BITS_OF = {"int8": 8, "int4": 4, "fp8_e4m3": 8, "fp6_e2m3": 6, "fp6_e3m2": 6, "fp4_e2m1": 4}
PRESETS = {"NVFP4": ("nvfp4", 16, "e4m3"), "MXFP4": ("mxfp4", 32, "e8m0")}
# The modes that TensorRT can really build. Each mode maps to
# (qdq_onnx format, whether quantized activations are allowed).
TENSORRT_MODES = {"INT8": ("int8", True), "FP8": ("fp8", True), "INT4": ("int4", False), "FP4": ("fp4", False)}


def add_arguments(parser) -> None:
    parser.add_argument("--quant", type=str.upper, choices=QUANT_MODES, default="NONE",
                        help="Weight format (see quantization/modes.py). --backend pytorch simulates it (fake quant); "
                             "--backend tensorrt builds a real engine for INT8/FP8 (W8A8) and INT4/FP4 (weight-only storage).")
    parser.add_argument("--block-size", type=int, default=None,
                        help="Weights (and --act-dynamic activations): one scale per this many input channels. "
                             "Default: per output channel; 16 for NVFP4, 32 for MXFP4.")
    parser.add_argument("--scale-format", choices=["fp32", "e4m3", "e8m0"], default=None,
                        help="Storage of block scales (default fp32; e4m3 for NVFP4, e8m0 for MXFP4).")
    parser.add_argument("--act-calib", type=str.lower, default=None,
                        help="Static per-tensor activations with this method from --calib-file (max, percentile-99.9, "
                             "percentile-99.99, percentile-99.999, entropy, entropy-skip-zeros). Omit for weight-only.")
    parser.add_argument("--calib-file", type=Path, default=None, help="JSON written by quantization/calibrate.py for this checkpoint.")
    parser.add_argument("--act-dynamic", action="store_true",
                        help="Dynamic per-block activations (amax of each block of input channels computed on the fly).")
    parser.add_argument("--act-format", type=str.upper, choices=ACT_FORMATS, default=None,
                        help="Activation format when different from --quant (e.g. --quant INT4 --act-format INT8 = W4A8).")
    parser.add_argument("--fp-layers", type=lambda value: [name for name in value.split(",") if name], default=[],
                        help="Comma-separated Conv/ConvTranspose layer names kept in floating point (e.g. enc1.block.0,segment).")


def validate(args, parser) -> None:
    """Check that the options work together, and fill in the preset defaults (in place)."""
    quantized = args.quant != "NONE"
    extras = [args.block_size, args.scale_format, args.act_calib, args.calib_file, args.act_format]
    if not quantized:
        if any(value is not None for value in extras) or args.act_dynamic or args.fp_layers:
            parser.error("--block-size, --scale-format, --act-*, --calib-file and --fp-layers need --quant.")
        return
    if args.quant in PRESETS:
        _, block, scale_format = PRESETS[args.quant]
        args.block_size = block if args.block_size is None else args.block_size
        args.scale_format = scale_format if args.scale_format is None else args.scale_format
    args.block_size = args.block_size or 0
    args.scale_format = args.scale_format or "fp32"
    if args.block_size < 0:
        parser.error("--block-size cannot be negative")
    if args.scale_format != "fp32" and not args.block_size:
        parser.error("--scale-format e4m3/e8m0 needs --block-size (block scales).")
    if args.act_calib is not None and args.act_dynamic:
        parser.error("Use --act-calib (static per-tensor) or --act-dynamic (per-block), not both.")
    if (args.act_calib is None) != (args.calib_file is None):
        parser.error("--act-calib and --calib-file go together.")
    if args.act_dynamic and not args.block_size:
        parser.error("--act-dynamic needs --block-size (the activation blocks).")
    if args.act_format is not None and args.act_calib is None and not args.act_dynamic:
        parser.error("--act-format needs --act-calib or --act-dynamic.")

    if args.backend == "tensorrt":
        if args.quant not in TENSORRT_MODES:
            parser.error(f"--quant {args.quant} has no TensorRT implementation (simulation only: --backend pytorch).")
        _, activations_ok = TENSORRT_MODES[args.quant]
        if args.block_size or args.act_dynamic or args.scale_format != "fp32":
            parser.error("--backend tensorrt supports per-channel weights and per-tensor activations only (no block scales).")
        if args.act_calib is not None and not activations_ok:
            parser.error(f"TensorRT has no {args.quant} activation kernels for convolutions: use weight-only (omit --act-calib).")
        if args.act_format is not None and args.act_format != args.quant:
            parser.error("TensorRT convolutions need the same format for inputs and weights (no --act-format).")


def weight_spec(args) -> QuantSpec:
    if args.quant in PRESETS and args.block_size == PRESETS[args.quant][1] and args.scale_format == PRESETS[args.quant][2]:
        return QuantSpec.preset(PRESETS[args.quant][0])
    if args.block_size:
        return QuantSpec(FORMAT_OF[args.quant], "block", args.block_size, args.scale_format)
    return QuantSpec(FORMAT_OF[args.quant], "channel")


def activation_spec(args) -> QuantSpec | None:
    if args.act_calib is None and not args.act_dynamic:
        return None
    fmt = FORMAT_OF[args.act_format or args.quant]
    if args.act_dynamic:
        return QuantSpec(fmt, "block", args.block_size, args.scale_format)
    return QuantSpec(fmt, "tensor")


def mode_name(args) -> str:
    """Return a short label for the CSV, for example INT8-W8A8, INT4-W4A32 or NVFP4-W4A4."""
    if args.quant == "NONE":
        return "none"
    act = activation_spec(args)
    return f"{args.quant}-W{BITS_OF[FORMAT_OF[args.quant]]}A{BITS_OF[act.format] if act else 32}"


def file_suffix(args) -> str:
    """Return extra tags for file names, to tell apart runs with the same mode_name."""
    tags = []
    if args.block_size and args.quant not in PRESETS:
        tags.append(f"b{args.block_size}" + ("" if args.scale_format == "fp32" else f"-{args.scale_format}"))
    if args.act_dynamic:
        tags.append("dyn")
    if args.act_format and args.act_format != args.quant:
        tags.append(f"a-{args.act_format}")
    return "".join(f"_{tag}" for tag in tags)


def describe(args) -> str:
    """Return a description that people can read, for example
    'int4 per-channel weights, int8 per-tensor activations'."""
    if args.quant == "NONE":
        return "fp32"
    text = f"{weight_spec(args).describe()} weights"
    act = activation_spec(args)
    return text + (f", {act.describe()} activations" if act else ", fp32 activations")
