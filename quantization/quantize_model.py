"""Apply simulated ("fake") quantization to a model (Wu et al., 2020).

What each file does:
    int8.py            the INT8 math (amax, scale, quantize and dequantize).
    formats.py         all the other formats (INT4, FP8, FP6, FP4) and the
                       granularities (tensor, channel and block).
    bn_fold.py         merges Conv+BN into one Conv, before we quantize the
                       weights.
    quantize_model.py  (this file) decides WHICH layers are quantized, and
                       applies those formats to them.

We quantize the layers that do most of the computation: every Conv2d and
ConvTranspose2d, including the first conv and the final 1x1 `segment`.
You can keep any of them in floating point by giving its name (useful for a
sensitivity analysis).

Weights: symmetric, with max calibration. This needs no data, because amax
comes from the weights themselves. We use one scale per output channel, or
one per block along the input channels. The weights do not change, so we
fake-quantize them only ONCE, in place. After that, the stored FP32 weight
only has values on the format's grid.

Activations (the input of each quantized layer) use one of two options:
per tensor, with a calibrated amax (static, from calibrate.py), or per block
of input channels, with the amax of each block computed on the fly (dynamic,
for example NVFP4).

Biases stay in FP32. A real INT8 kernel stores the bias as INT32 with scale
s_w * s_x, and the rounding error of that is so small that we can ignore it.
"""

import torch
import torch.nn as nn

from quantization.bn_fold import fold_batchnorm
from quantization.formats import QuantSpec, fake_quantize_spec, get_format
from quantization.int8 import fake_quantize_, weight_axis

QUANTIZABLE_TYPES = (nn.Conv2d, nn.ConvTranspose2d)
INT8_WEIGHTS = QuantSpec("int8", "channel")
INT8_ACTIVATIONS = QuantSpec("int8", "tensor")


def quantizable_layers(model: nn.Module) -> dict:
    """Return {name: module} for the layers that we quantize, in forward order."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, QUANTIZABLE_TYPES)
    }


def check_layer_names(model: nn.Module, names) -> None:
    unknown = set(names) - set(quantizable_layers(model))
    if unknown:
        raise ValueError(
            f"Unknown layers {sorted(unknown)}. "
            f"Valid names: {list(quantizable_layers(model))}"
        )


def reduction_axis(module: nn.Module) -> int:
    """Return the input-channel axis of a convolution weight.

    This is the axis that the convolution sums over: 1 for Conv2d
    [C_out, C_in, kH, kW], and 0 for ConvTranspose2d [C_in, C_out, kH, kW]."""
    return 0 if isinstance(module, nn.ConvTranspose2d) else 1


def _error_stats(original: torch.Tensor, quantized: torch.Tensor) -> dict:
    """Return error statistics. sqnr_db = 10 log10(sum x^2 / sum (x - x_hat)^2).

    Higher is better. On a uniform grid, each bit adds about 6 dB."""
    noise = (original - quantized).float()
    noise_power = noise.pow(2).sum()
    sqnr = float("inf") if noise_power == 0 else (10 * torch.log10(original.float().pow(2).sum() / noise_power)).item()
    return {"max_abs_error": noise.abs().max().item(), "mse": noise.pow(2).mean().item(), "sqnr_db": sqnr}


@torch.no_grad()
def quantize_weights_(model: nn.Module, fp_layers=(), spec: QuantSpec = INT8_WEIGHTS) -> list:
    """Fake-quantize the weights of every quantizable layer in place, using
    `spec` (granularity "channel" or "block"). Layers in `fp_layers` are skipped.

    Returns one dict of error statistics for each layer.
    """
    if spec.granularity == "tensor":
        raise ValueError("weights use per-channel or per-block scales")
    check_layer_names(model, fp_layers)
    stats = []
    for name, module in quantizable_layers(model).items():
        if name in fp_layers:
            continue
        weight = module.weight
        axis = weight_axis(module) if spec.granularity == "channel" else reduction_axis(module)
        quantized = fake_quantize_spec(weight, spec, axis=axis)
        stats.append({"layer": name, **_error_stats(weight, quantized)})
        weight.copy_(quantized)
    return stats


def _input_quantizer(spec: QuantSpec, amax: torch.Tensor | None, inplace: bool):
    """Return the function that we apply to a layer input.

    Per-tensor INT8 and INT4 can work in place, with no full-size temporary
    tensors. We need this at full ISIC resolution. The other cases create
    temporary tensors as big as the activation."""
    fmt = get_format(spec.format)
    if spec.granularity == "tensor" and inplace and fmt.kind == "int":
        if fmt.qmax == 127:
            return lambda x: fake_quantize_(x, amax)
        scale = fmt.qmax / amax if amax > 0 else torch.ones_like(amax)
        return lambda x: x.mul_(scale).round_().clamp_(-fmt.qmax, fmt.qmax).div_(scale)
    if spec.granularity == "tensor":
        return lambda x: fake_quantize_spec(x, spec, amax=amax)
    # Dynamic blocks along the channel axis of the activation, which is [N, C, H, W].
    return lambda x: fake_quantize_spec(x, spec, axis=1)


def attach_input_quantizers(model: nn.Module, act_amax: dict | None, fp_layers=(), inplace: bool = True,
                            spec: QuantSpec = INT8_ACTIVATIONS) -> list:
    """Fake-quantize the INPUT of every quantizable layer on each forward pass.

    Layers in `fp_layers` are skipped. There are two cases:
        spec.granularity == "tensor": symmetric, per tensor, with the
            calibrated amax of each layer (act_amax = {layer: amax}).
        spec.granularity == "block": dynamic scales for each block along the
            input channels (act_amax is not needed).

    We use forward pre-hooks, so the model code does not change.
    With inplace=True we write over the input tensor, so we do not need a
    full-size copy. This is safe in UNetModern, because no conv input is read
    again later. (The skip connections are the DoubleConv outputs, and they
    reach the decoder through copies made by torch.cat.)
    Returns the hook handles (call .remove() to take them off).
    """
    check_layer_names(model, fp_layers)
    if spec.granularity == "channel":
        raise ValueError("activations use per-tensor (static) or per-block (dynamic) scales")
    if spec.granularity == "tensor":
        missing = set(quantizable_layers(model)) - set(fp_layers) - set(act_amax or {})
        if missing:
            raise ValueError(f"No calibrated amax for layers: {sorted(missing)}")

    handles = []
    for name, module in quantizable_layers(model).items():
        if name in fp_layers:
            continue
        amax = None
        if spec.granularity == "tensor":
            amax = torch.tensor(float(act_amax[name]), device=module.weight.device)
        quantizer = _input_quantizer(spec, amax, inplace)

        def hook(_module, args, quantizer=quantizer):
            return (quantizer(args[0]),) + tuple(args[1:])

        handles.append(module.register_forward_pre_hook(hook))
    return handles


def prepare_quantized_model(model: nn.Module, weight_spec: QuantSpec, act_spec: QuantSpec | None = None,
                            act_amax: dict | None = None, fp_layers=(), inplace: bool = True) -> tuple[nn.Module, list]:
    """Turn an FP32 model into a quantized eval-mode copy.

    In the copy, BN is folded and the weights are fake-quantized with
    `weight_spec`. If `act_spec` is given, the layer inputs are quantized too.

    A layer in `fp_layers` keeps both its weight and its input in FP32.
    The input model does not change.
    """
    quantized = fold_batchnorm(model)
    stats = quantize_weights_(quantized, fp_layers=fp_layers, spec=weight_spec)
    if act_spec is not None:
        attach_input_quantizers(quantized, act_amax, fp_layers=fp_layers, inplace=inplace, spec=act_spec)
    return quantized, stats


def prepare_int8_model(model: nn.Module, fp_layers=(), act_amax: dict | None = None, inplace: bool = True) -> tuple[nn.Module, list]:
    """Prepare an INT8 model: W8A32 (when act_amax=None) or W8A8.

    Weights are per channel, and activations are per tensor with calibration.
    This is the same as prepare_quantized_model with the INT8 specs."""
    act_spec = INT8_ACTIVATIONS if act_amax is not None else None
    return prepare_quantized_model(model, INT8_WEIGHTS, act_spec, act_amax, fp_layers, inplace)
