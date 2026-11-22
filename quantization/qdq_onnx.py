"""INT8 and FP8 ONNX with explicit QuantizeLinear and DequantizeLinear (Q/DQ) nodes.

This module turns the FP32 ONNX export (export_onnx.py) into an INT8 graph
that TensorRT runs with INT8 kernels. It uses the same quantization as the
fake-quant reference (quantize_model.py), but now the quantization is
written into the graph:

    weights      stored as INT8 initializers with one scale per output
                 channel, followed by DequantizeLinear (axis 0 for Conv,
                 axis 1 for ConvTranspose). The ONNX file really holds int8
                 weights.
    activations  a Q node and then a DQ node on the input of every quantized
                 Conv and ConvTranspose. They are per tensor, with
                 scale = amax / 127, and amax comes from calibrate.py.

The ONNX scale rule is x_q = saturate(round(x / scale)) and
x_hat = x_q * scale, so scale = amax / 127. (int8.py uses the inverse,
s = 127 / amax.)

TensorRT does not really run "Q and then DQ". It fuses
DQ -> Conv -> (ReLU) -> Q into one INT8 convolution: the inputs and weights
are int8, the sums are in INT32, and the output is requantized to int8 or
left in float. It also moves Q nodes across MaxPool and Concat when the
result stays exact. We do not assume which layers became INT8. We check it
in the built engine (trt_engine.summarize_layers).

Known differences from the fake-quant reference (all very small and
documented):
  * QuantizeLinear saturates to [-128, 127], but int8.py clips to
    [-127, 127]. Only values below -127.5 * scale are different (they give
    -128 instead of -127). This never happens for weights (max calibration).
    It only happens for clipped negative activations (the ConvTranspose half
    of the dec*.block.0 inputs).
  * x / (amax/127) and x * (127/amax) can differ by one ULP (one step of the
    last bit). So a value that lies exactly on a rounding tie may go to the
    neighboring integer.
  * Biases stay in FP32 (TensorRT handles them inside the INT8 kernel).

FP8 (--format fp8): the same structure with E4M3 codes and
scale = amax / 448. It starts from the depth2space export
(export_onnx.py --upsample depth2space), so all 23 layers are convolutions
with FP8 kernels.

INT4 and FP4 (--format int4 or fp4): weight-only, per output channel
(scale = amax/7 or amax/6). TensorRT stores the 4-bit weights in the engine
and dequantizes them for FP32 convolutions. So the storage is really 4-bit,
but the computation is not. The codes are the same as in the simulation in
formats.py.

Usage (from the repository root):
    python -m quantization.qdq_onnx --onnx exports/best_CarvanaBaseline_fp32_bnfolded.onnx \
        --calib-file calibration/best_CarvanaBaseline_n128.json --act-calib percentile-99.99
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from quantization.export_onnx import sha256
from quantization.formats import get_format, round_to_format
from quantization.int8 import QMAX, compute_amax, compute_scale, quantize

QUANTIZED_OPS = ("Conv", "ConvTranspose")
# ExportableUNet wraps the U-Net under the name `model`. In depth2space mode,
# the 1x1 convs that replace the ConvTranspose layers are stored in `up_1x1`.
WEIGHT_PREFIXES = ("model.", "up_1x1.")

# Formats that use scale quantization: x_q = saturate(round(x / scale)), with
# scale = amax / qmax. For FP8 E4M3, the largest finite value is 448, and we
# round to the nearest even value on the E4M3 grid. FP8 has no INT32 sums;
# TensorRT adds up in FP32.
# int4 and fp4: TensorRT 11.4 accepts them only as WEIGHT-ONLY (per output
# channel). The engine keeps the 4-bit weights and dequantizes them for FP32
# convs. TensorRT rejects 4-bit activations (there are no conv kernels for
# them), so act_amax must be None.
FORMATS = {
    "int8": {"qmax": float(QMAX), "dtype": np.int8, "min_opset": 13, "format": "int8", "activations": True},
    "fp8": {"qmax": 448.0, "dtype": ml_dtypes.float8_e4m3fn, "min_opset": 19, "format": "fp8_e4m3", "activations": True},
    "int4": {"qmax": 7.0, "dtype": ml_dtypes.int4, "min_opset": 21, "format": "int4", "activations": False},
    "fp4": {"qmax": 6.0, "dtype": ml_dtypes.float4_e2m1fn, "min_opset": 23, "format": "fp4_e2m1", "activations": False},
}


def layer_name(weight_name: str) -> str:
    """Turn an ONNX initializer name into a layer name.

    For example, 'model.enc1.block.0.weight' becomes 'enc1.block.0'. These
    are the names that quantize_model.py and the calibration JSON use.
    'up_1x1.up4.weight' becomes 'up4', because it reads the same input as
    the ConvTranspose up4."""
    for prefix in WEIGHT_PREFIXES:
        if weight_name.startswith(prefix):
            weight_name = weight_name[len(prefix):]
            break
    return weight_name.removesuffix(".weight")


def onnx_scale(amax: np.ndarray, qmax: float = QMAX) -> np.ndarray:
    """Return the ONNX scale, amax / qmax.

    A tensor or channel that is all zeros gets scale 1. It becomes zeros in
    any case, just like in int8.compute_scale."""
    amax = np.asarray(amax, dtype=np.float32)
    return np.where(amax > 0, amax / np.float32(qmax), np.float32(1.0)).astype(np.float32)


def quantize_weight(weight: torch.Tensor, amax: torch.Tensor, fmt: str) -> np.ndarray:
    """Return the weight codes in the target format, as a NumPy array that ONNX can store.

    We use the same math as the simulation (formats.py): round_to_format(w * s)
    with s = qmax / amax. So the codes match the fake-quant reference."""
    if fmt == "int8":
        return quantize(weight, amax)[0].numpy()
    spec = FORMATS[fmt]
    codes = round_to_format(weight.float() * compute_scale(amax, spec["qmax"]), get_format(spec["format"]))
    return codes.numpy().astype(spec["dtype"])  # exact, because the codes are already on the grid


def quantize_onnx(fp32_path: Path, output_path: Path, act_amax: dict | None = None, fp_layers=(), fmt: str = "int8") -> dict:
    """Write the Q/DQ version of `fp32_path` in format `fmt` to `output_path`.

    If act_amax is None, only the weights are quantized (only weight DQ
    nodes). Otherwise, both weights and activations are quantized. Layers in
    `fp_layers` keep FP32 weights and inputs. Returns a report for each layer.
    """
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {list(FORMATS)}")
    if act_amax is not None and not FORMATS[fmt]["activations"]:
        raise ValueError(f"{fmt}: TensorRT has no {fmt} activation kernels for convolutions; use weight-only (act_amax=None).")
    qmax, code_dtype = FORMATS[fmt]["qmax"], FORMATS[fmt]["dtype"]
    model = onnx.load(str(fp32_path))
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    conv_layers = [layer_name(n.input[1]) for n in graph.node if n.op_type in QUANTIZED_OPS]
    unknown = set(fp_layers) - set(conv_layers)
    if unknown:
        raise ValueError(f"Unknown layers {sorted(unknown)}. Valid names: {conv_layers}")
    if act_amax is not None:
        missing = set(conv_layers) - set(fp_layers) - set(act_amax)
        if missing:
            raise ValueError(f"No calibrated amax for layers: {sorted(missing)}")

    new_nodes, report, removed = [], [], []
    for node in graph.node:
        if node.op_type not in QUANTIZED_OPS or layer_name(node.input[1]) in fp_layers:
            new_nodes.append(node)
            continue
        layer = layer_name(node.input[1])
        entry = {"layer": layer, "op": node.op_type}

        # Activations: a per-tensor Q -> DQ pair, only in front of this layer.
        # Other users of the same tensor (for example, a skip connection) do not change.
        if act_amax is not None:
            scale = onnx_scale(act_amax[layer], qmax)
            graph.initializer.extend([
                numpy_helper.from_array(scale, f"{layer}.input_scale"),
                numpy_helper.from_array(np.zeros((), code_dtype), f"{layer}.input_zero_point"),
            ])
            source = node.input[0]
            new_nodes += [
                helper.make_node("QuantizeLinear", [source, f"{layer}.input_scale", f"{layer}.input_zero_point"],
                                 [f"{layer}.input_q"], name=f"{layer}/input_quantize"),
                helper.make_node("DequantizeLinear", [f"{layer}.input_q", f"{layer}.input_scale", f"{layer}.input_zero_point"],
                                 [f"{layer}.input_dq"], name=f"{layer}/input_dequantize"),
            ]
            node.input[0] = f"{layer}.input_dq"
            entry["input_amax"] = float(act_amax[layer])

        # Weights: an INT8 initializer with one scale per output channel, then DequantizeLinear.
        weight_name = node.input[1]
        weight = torch.from_numpy(numpy_helper.to_array(initializers[weight_name]).copy())
        axis = 1 if node.op_type == "ConvTranspose" else 0
        amax = compute_amax(weight, axis=axis)
        scale = onnx_scale(amax.flatten().numpy(), qmax)
        weight_q = quantize_weight(weight, amax, fmt)
        graph.initializer.extend([
            numpy_helper.from_array(weight_q, f"{layer}.weight_{fmt}"),
            numpy_helper.from_array(scale, f"{layer}.weight_scale"),
            numpy_helper.from_array(np.zeros(scale.shape, code_dtype), f"{layer}.weight_zero_point"),
        ])
        new_nodes.append(helper.make_node(
            "DequantizeLinear", [f"{layer}.weight_{fmt}", f"{layer}.weight_scale", f"{layer}.weight_zero_point"],
            [f"{layer}.weight_dq"], axis=axis, name=f"{layer}/weight_dequantize"))
        node.input[1] = f"{layer}.weight_dq"
        removed.append(weight_name)
        entry["weight_channels"] = int(scale.size)
        report.append(entry)
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    kept = [init for init in graph.initializer if init.name not in removed]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    # FP8 QuantizeLinear and DequantizeLinear need opset >= 19. The other ops in
    # the graph (Conv, Pad, Concat, DepthToSpace, and so on) work the same there.
    default_domain = next(op for op in model.opset_import if op.domain in ("", "ai.onnx"))
    default_domain.version = max(default_domain.version, FORMATS[fmt]["min_opset"])

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return {"layers": report, "fp_layers": list(fp_layers)}


def weight_bytes(onnx_path: Path) -> dict:
    """Return the bytes of the initializers for each dtype.

    This shows that the weights really are int8."""
    totals = {}
    for init in onnx.load(str(onnx_path)).graph.initializer:
        dtype = TensorProto.DataType.Name(init.data_type)
        totals[dtype] = totals.get(dtype, 0) + numpy_helper.to_array(init).nbytes
    return totals


def quantized_onnx_path(fp32_path: Path, fmt: str, act_calib: str | None, fp_layers=()) -> Path:
    """Return the file name of the quantized ONNX. For example:

    exports/<ckpt>_fp32_bnfolded[_d2s].onnx -> <ckpt>_<fmt>_bnfolded[_d2s]_w8a8_<method>[_fp-...].onnx"""
    stem = fp32_path.stem.replace("_fp32", f"_{fmt}")
    bits = 8 if fmt in ("int8", "fp8") else 4
    mode = f"w{bits}a8_{act_calib}" if act_calib else f"w{bits}a32"
    fp = f"_fp-{'+'.join(fp_layers)}" if fp_layers else ""
    return fp32_path.with_name(f"{stem}_{mode}{fp}.onnx")


def build_quantized_onnx(fp32_path: Path, calib_file: Path | None = None, act_calib: str | None = None, fp_layers=(),
                         fmt: str = "int8", force: bool = False) -> Path:
    """Return the Q/DQ ONNX for these settings, and build it only if needed.

    The settings are: the FP32 ONNX, the format, the calibration file, the
    method and the FP layers. If a matching file already exists, we reuse it."""
    output = quantized_onnx_path(fp32_path, fmt, act_calib, fp_layers)
    act_amax, calib_sha = None, None
    if act_calib is not None:
        calibration = json.loads(calib_file.read_text(encoding="utf-8"))
        if act_calib not in calibration["methods"]:
            raise ValueError(f"act_calib must be one of {calibration['methods']}")
        act_amax = {name: layer["amax"][act_calib] for name, layer in calibration["layers"].items()}
        calib_sha = sha256(calib_file)
    source_sha = sha256(fp32_path)

    metadata_path = output.with_suffix(".json")
    if not force and output.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_sha256") == source_sha and metadata.get("calib_sha256") == calib_sha:
            return output

    report = quantize_onnx(fp32_path, output, act_amax, fp_layers, fmt)
    source_metadata = json.loads(fp32_path.with_suffix(".json").read_text(encoding="utf-8"))
    name = {"int8": "int8 symmetric", "fp8": "fp8 e4m3 (scale = amax/448)",
            "int4": "int4 symmetric [-7, 7]", "fp4": "fp4 e2m1 (scale = amax/6)"}[fmt]
    metadata_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": source_metadata["dataset"],
        "checkpoint": source_metadata["checkpoint"],
        "source_onnx": str(fp32_path),
        "source_sha256": source_sha,
        "upsample": source_metadata.get("upsample", "convtranspose"),
        "format": fmt,
        "mode": f"W{8 if fmt in ('int8', 'fp8') else 4}A{8 if act_calib else 32}",
        "weights": f"{name} per-output-channel, max calibration",
        "activations": f"{name} per-tensor, {act_calib}" if act_calib else "fp32",
        "calib_file": str(calib_file) if calib_file else None,
        "calib_sha256": calib_sha,
        "initializer_bytes": weight_bytes(output),
        "onnx_bytes": output.stat().st_size,
        **report,
    }, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description="Write an INT8 or FP8 Q/DQ ONNX from the FP32 export.")
    parser.add_argument("--onnx", type=Path, required=True, help="FP32 ONNX from quantization/export_onnx.py.")
    parser.add_argument("--format", choices=list(FORMATS), default="int8")
    parser.add_argument("--calib-file", type=Path, default=None)
    parser.add_argument("--act-calib", default=None, help="Calibration method; omit for weight-only (W8A32).")
    parser.add_argument("--fp-layers", type=lambda v: [n for n in v.split(",") if n], default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (args.act_calib is None) != (args.calib_file is None):
        parser.error("--act-calib and --calib-file go together.")
    output = build_quantized_onnx(args.onnx, args.calib_file, args.act_calib, args.fp_layers, args.format, args.force)
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    print(f"{args.format.upper()} ONNX: {output} ({metadata['onnx_bytes'] / 2**20:.1f} MiB)")
    print("Initializer bytes by dtype:", {k: f"{v / 2**20:.2f} MiB" for k, v in metadata["initializer_bytes"].items()})


if __name__ == "__main__":
    main()
