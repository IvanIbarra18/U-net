"""Apply simulated INT8 quantization to a model (Wu et al., 2020).

Responsibilities:
    int8.py            the arithmetic (amax, scale, quantize/dequantize).
    bn_fold.py         Conv+BN -> Conv, before quantizing weights.
    quantize_model.py  (this file) decides WHICH layers are quantized and
                       applies int8.py to them.

Quantized layers are the compute-heavy ones: every Conv2d and
ConvTranspose2d, including the first conv and the final 1x1 `segment`.
Any of them can be kept in floating point by name (sensitivity analysis).

Weights: symmetric, per-output-channel, max calibration (no data needed:
amax comes from the weights themselves). Because the weights are constant,
they are fake-quantized ONCE, in place, instead of on every forward pass:
the stored FP32 weight then only takes values on the INT8 grid.

Biases stay in FP32. In a real INT8 kernel the bias is stored as INT32
with scale s_w * s_x, whose rounding error is negligible.
"""

import torch
import torch.nn as nn

from quantization.bn_fold import fold_batchnorm
from quantization.int8 import compute_amax, fake_quantize, fake_quantize_, quantization_error, weight_axis

QUANTIZABLE_TYPES = (nn.Conv2d, nn.ConvTranspose2d)


def quantizable_layers(model: nn.Module) -> dict:
    """{name: module} of the layers INT8 applies to, in forward order."""
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


@torch.no_grad()
def quantize_weights_(model: nn.Module, fp_layers=()) -> list:
    """Fake-quantize the weights of every quantizable layer in place
    (per-channel, max calibration), except the ones named in `fp_layers`.

    Returns one dict of error statistics per layer, measured before
    replacing the weight.
    """
    check_layer_names(model, fp_layers)
    stats = []
    for name, module in quantizable_layers(model).items():
        if name in fp_layers:
            continue
        weight = module.weight
        amax = compute_amax(weight, axis=weight_axis(module))
        stats.append({"layer": name, **quantization_error(weight, amax)})
        weight.copy_(fake_quantize(weight, amax))
    return stats


def attach_input_quantizers(model: nn.Module, act_amax: dict, fp_layers=(), inplace: bool = True) -> list:
    """Fake-quantize the INPUT of every quantizable layer on each forward
    (per-tensor, symmetric, amax from calibration), except `fp_layers`.

    Implemented as forward pre-hooks, so the model code is unchanged.
    inplace=True overwrites the input tensor to avoid a full-size copy. That
    is safe in UNetModern: no conv input is read again afterwards (skips are
    the DoubleConv outputs, which reach the decoder through torch.cat copies).
    Returns the hook handles (call .remove() to detach them).
    """
    check_layer_names(model, fp_layers)
    missing = set(quantizable_layers(model)) - set(fp_layers) - set(act_amax)
    if missing:
        raise ValueError(f"No calibrated amax for layers: {sorted(missing)}")

    handles = []
    for name, module in quantizable_layers(model).items():
        if name in fp_layers:
            continue
        amax = torch.tensor(float(act_amax[name]), device=module.weight.device)

        def hook(_module, args, amax=amax):
            x = args[0]
            return (fake_quantize_(x, amax) if inplace else fake_quantize(x, amax),) + tuple(args[1:])

        handles.append(module.register_forward_pre_hook(hook))
    return handles


def prepare_int8_model(model: nn.Module, fp_layers=(), act_amax: dict | None = None, inplace: bool = True) -> tuple[nn.Module, list]:
    """FP32 model -> eval-mode copy with BN folded and INT8 weights, plus
    INT8 inputs when `act_amax` ({layer: amax}) is given: W8A32 or W8A8.

    A layer in `fp_layers` keeps both its weight and its input in FP32.
    The input model is not modified.
    """
    quantized = fold_batchnorm(model)
    stats = quantize_weights_(quantized, fp_layers=fp_layers)
    if act_amax is not None:
        attach_input_quantizers(quantized, act_amax, fp_layers=fp_layers, inplace=inplace)
    return quantized, stats
