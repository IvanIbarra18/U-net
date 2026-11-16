"""INT8 ONNX: explicit QuantizeLinear / DequantizeLinear (Q/DQ) nodes.

Turns the FP32 ONNX export (export_onnx.py) into an INT8 graph that
TensorRT executes with INT8 kernels. Same quantization as the fake-quant
reference (quantize_model.py), now written into the graph:

    weights      stored as INT8 initializers + per-output-channel scales,
                 followed by DequantizeLinear (axis 0 for Conv, axis 1 for
                 ConvTranspose). The ONNX file really holds int8 weights.
    activations  Q -> DQ on the input of every quantized Conv/ConvTranspose,
                 per-tensor, scale = amax / 127 with amax from calibrate.py.

ONNX scale convention: x_q = saturate(round(x / scale)), x_hat = x_q * scale,
so scale = amax / 127 (int8.py uses the inverse, s = 127 / amax).

TensorRT does not run "Q then DQ": it fuses DQ -> Conv -> (ReLU) -> Q into
one INT8 convolution (int8 inputs and weights, INT32 accumulation, output
requantized to int8 or left in float), and moves Q nodes across MaxPool and
Concat when that is exact. Which layers actually became INT8 is checked from
the built engine (trt_engine.summarize_layers), not assumed.

Known differences from the fake-quant reference (all tiny, documented):
  * QuantizeLinear saturates to [-128, 127]; int8.py clips to [-127, 127].
    Only values below -127.5 * scale differ (-128 instead of -127): never
    for weights (max calibration) and only for clipped negative activations
    (the ConvTranspose half of the dec*.block.0 inputs).
  * x / (amax/127) vs x * (127/amax): can differ by one ULP, so a value on
    an exact rounding tie may land on the neighbouring integer.
  * Biases stay FP32 (TensorRT handles them inside the INT8 kernel).

Usage (from the repository root):
    python -m quantization.qdq_onnx --onnx exports/best_CarvanaBaseline_fp32_bnfolded.onnx \
        --calib-file calibration/best_CarvanaBaseline_n128.json --act-calib percentile-99.99
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from quantization.export_onnx import sha256
from quantization.int8 import QMAX, compute_amax, quantize

QUANTIZED_OPS = ("Conv", "ConvTranspose")
WEIGHT_PREFIX = "model."  # ExportableUNet wraps the U-Net as `model`


def layer_name(weight_name: str) -> str:
    """ONNX initializer 'model.enc1.block.0.weight' -> layer 'enc1.block.0'
    (the names used by quantize_model.py and the calibration JSON)."""
    name = weight_name[len(WEIGHT_PREFIX):] if weight_name.startswith(WEIGHT_PREFIX) else weight_name
    return name.removesuffix(".weight")


def onnx_scale(amax: np.ndarray) -> np.ndarray:
    """ONNX scale = amax / 127; an all-zero tensor/channel gets scale 1
    (it quantizes to zeros either way, as in int8.compute_scale)."""
    amax = np.asarray(amax, dtype=np.float32)
    return np.where(amax > 0, amax / np.float32(QMAX), np.float32(1.0)).astype(np.float32)


def quantize_onnx(fp32_path: Path, output_path: Path, act_amax: dict | None = None, fp_layers=()) -> dict:
    """Write the INT8 Q/DQ version of `fp32_path` to `output_path`.

    act_amax=None -> W8A32 (only weight DQ); otherwise W8A8. Layers in
    `fp_layers` keep FP32 weights and inputs. Returns a per-layer report.
    """
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

        # Activations: per-tensor Q -> DQ in front of this layer only (other
        # consumers of the same tensor, e.g. a skip connection, are untouched).
        if act_amax is not None:
            scale = onnx_scale(act_amax[layer])
            graph.initializer.extend([
                numpy_helper.from_array(scale, f"{layer}.input_scale"),
                numpy_helper.from_array(np.zeros((), np.int8), f"{layer}.input_zero_point"),
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

        # Weights: per-output-channel INT8 initializer + DequantizeLinear.
        weight_name = node.input[1]
        weight = torch.from_numpy(numpy_helper.to_array(initializers[weight_name]).copy())
        axis = 1 if node.op_type == "ConvTranspose" else 0
        amax = compute_amax(weight, axis=axis)
        weight_q, _ = quantize(weight, amax)
        scale = onnx_scale(amax.flatten().numpy())
        graph.initializer.extend([
            numpy_helper.from_array(weight_q.numpy(), f"{layer}.weight_int8"),
            numpy_helper.from_array(scale, f"{layer}.weight_scale"),
            numpy_helper.from_array(np.zeros(scale.shape, np.int8), f"{layer}.weight_zero_point"),
        ])
        new_nodes.append(helper.make_node(
            "DequantizeLinear", [f"{layer}.weight_int8", f"{layer}.weight_scale", f"{layer}.weight_zero_point"],
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

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return {"layers": report, "fp_layers": list(fp_layers)}


def weight_bytes(onnx_path: Path) -> dict:
    """Bytes of the initializers by dtype: shows that the weights really are int8."""
    totals = {}
    for init in onnx.load(str(onnx_path)).graph.initializer:
        dtype = TensorProto.DataType.Name(init.data_type)
        totals[dtype] = totals.get(dtype, 0) + numpy_helper.to_array(init).nbytes
    return totals


def int8_onnx_path(fp32_path: Path, act_calib: str | None, fp_layers=()) -> Path:
    stem = fp32_path.stem.replace("_fp32", "_int8")
    mode = f"w8a8_{act_calib}" if act_calib else "w8a32"
    fp = f"_fp-{'+'.join(fp_layers)}" if fp_layers else ""
    return fp32_path.with_name(f"{stem}_{mode}{fp}.onnx")


def build_int8_onnx(fp32_path: Path, calib_file: Path | None = None, act_calib: str | None = None, fp_layers=(), force: bool = False) -> Path:
    """Cached INT8 ONNX for this (FP32 ONNX, calibration, method, FP layers)."""
    output = int8_onnx_path(fp32_path, act_calib, fp_layers)
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

    report = quantize_onnx(fp32_path, output, act_amax, fp_layers)
    source_metadata = json.loads(fp32_path.with_suffix(".json").read_text(encoding="utf-8"))
    metadata_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": source_metadata["dataset"],
        "checkpoint": source_metadata["checkpoint"],
        "source_onnx": str(fp32_path),
        "source_sha256": source_sha,
        "mode": "W8A8" if act_calib else "W8A32",
        "weights": "int8 symmetric per-output-channel, max calibration",
        "activations": f"int8 symmetric per-tensor, {act_calib}" if act_calib else "fp32",
        "calib_file": str(calib_file) if calib_file else None,
        "calib_sha256": calib_sha,
        "initializer_bytes": weight_bytes(output),
        "onnx_bytes": output.stat().st_size,
        **report,
    }, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description="Write an INT8 Q/DQ ONNX from the FP32 export.")
    parser.add_argument("--onnx", type=Path, required=True, help="FP32 ONNX from quantization/export_onnx.py.")
    parser.add_argument("--calib-file", type=Path, default=None)
    parser.add_argument("--act-calib", default=None, help="Calibration method; omit for weight-only (W8A32).")
    parser.add_argument("--fp-layers", type=lambda v: [n for n in v.split(",") if n], default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (args.act_calib is None) != (args.calib_file is None):
        parser.error("--act-calib and --calib-file go together.")
    output = build_int8_onnx(args.onnx, args.calib_file, args.act_calib, args.fp_layers, args.force)
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    print(f"INT8 ONNX: {output} ({metadata['onnx_bytes'] / 2**20:.1f} MiB)")
    print("Initializer bytes by dtype:", {k: f"{v / 2**20:.2f} MiB" for k, v in metadata["initializer_bytes"].items()})


if __name__ == "__main__":
    main()
