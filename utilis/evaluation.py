"""Helper functions for test.py.

They read some arguments, find the image sizes, and describe which precision
a run really used (representation, storage and execution)."""

import argparse

from PIL import Image

from quantization import modes as quant_modes


def split_arg(value):
    """Accept a predefined split index (0-4), or 'full' for the whole test dataset."""
    if value.lower() == "full":
        return "full"
    try:
        index = int(value)
    except ValueError:
        index = None
    if index not in range(5):
        raise argparse.ArgumentTypeError("must be 0-4 or 'full'")
    return index


def image_sizes(base_dataset, sample_ids):
    """Return the (H, W) of each image. We read it from the file headers, so we do not decode the images."""
    sizes = []
    for sample_id in sample_ids:
        with Image.open(base_dataset.image_files[sample_id][0]) as image:
            width, height = image.size
        sizes.append((height, width))
    return sizes


def describe_precision(args, use_amp, model_format, trt_summary=None):
    """Return the three properties that every result reports separately:
    representation  which values the numbers are limited to.
    storage         how the weights are really kept in memory.
    execution       which precision the kernels really computed in.

    For TensorRT, storage and execution come from the built engine (kernel
    names, tensor dtypes and engine size), not from what the ONNX graph asked for.
    """
    if args.backend == "tensorrt":
        counts = lambda field: ", ".join(f"{count}x {kind}" for kind, count in trt_summary[field].items())
        tf32 = f", {trt_summary['tf32_tactics']} TF32 tactics" if trt_summary["tf32_tactics"] else ""
        representation = quant_modes.describe(args)
        storage = f"TensorRT engine, conv weights [{counts('weights')}]"
        if args.quant != "NONE" and set(trt_summary["weights"]) == {"f32"}:
            # Weight-only: the kernels get FP32 weights. The engine may keep the
            # weights compressed (and dequantize them at run time), or it may fold
            # them to FP32 when it is built. We find out by comparing the engine
            # size with the size of the FP32 weights.
            ratio = trt_summary["engine_bytes"] / (trt_summary["total_params"] * 4)
            if ratio < 0.6:
                storage = (f"TensorRT engine, {args.quant.lower()} weights kept compressed "
                           f"({ratio:.0%} of FP32 size), dequantized to f32 for the convs")
            else:
                storage = f"TensorRT engine, weights folded to f32 at build time ({ratio:.0%} of FP32 size)"
        execution = f"tensorrt [{counts('compute_layers')}; accumulators {counts('accumulators')}{tf32}]"
        return representation, storage, execution
    tf32 = " (TF32 convs)" if args.tf32 == "on" else ""
    if args.quant != "NONE":
        return quant_modes.describe(args), "fp32", f"pytorch fp32{tf32} (simulated {args.quant.lower()})"
    if args.cast is not None:
        return model_format.lower(), model_format.lower(), f"pytorch {model_format.lower()}"
    if use_amp:
        return f"mixed (autocast {args.precision.lower()})", "fp32", f"pytorch autocast {args.precision.lower()}"
    return "fp32", "fp32", f"pytorch fp32{tf32}"
