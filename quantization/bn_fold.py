"""Batch Normalization folding (Wu et al., 2020, Sec. 3; Jacob et al., 2018).

In eval mode, BatchNorm does not learn anymore. For each channel k it only
scales and shifts the input by fixed numbers (an "affine map"):
    y_k = gamma_k * (x_k - mu_k) / sigma_k + beta_k,  sigma_k = sqrt(var_k + eps)
When BatchNorm comes right after a convolution, we can merge ("fold") it
into the convolution by changing the weights and the bias:
    w'_k = w_k * gamma_k / sigma_k
    b'_k = gamma_k * (b_k - mu_k) / sigma_k + beta_k
The folded network gives the same result (apart from small floating-point
rounding), but each block has one operation less. We must quantize the
weights AFTER folding, because the folded weights are the ones that really run.

In UNetModern, every DoubleConv is Sequential(Conv, BN, ReLU, Conv, BN, ReLU).
The ConvTranspose2d upsampling layers and the final 1x1 conv have no BN.
"""

import copy

import torch
import torch.nn as nn


@torch.no_grad()
def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a new Conv2d that gives the same result as bn(conv(x)) in eval mode."""
    if conv.out_channels != bn.num_features:
        raise ValueError("Conv output channels and BatchNorm features differ.")
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BatchNorm has no running statistics to fold.")

    dtype = conv.weight.dtype
    # We compute in FP64, so the folding step itself adds almost no rounding error.
    weight = conv.weight.double()
    bias = conv.bias.double() if conv.bias is not None else torch.zeros_like(bn.running_mean, dtype=torch.float64)
    gamma = bn.weight.double() if bn.affine else torch.ones_like(bias)
    beta = bn.bias.double() if bn.affine else torch.zeros_like(bias)
    factor = gamma / torch.sqrt(bn.running_var.double() + bn.eps)

    folded = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=dtype,
    )
    folded.weight.copy_((weight * factor.view(-1, 1, 1, 1)).to(dtype))
    folded.bias.copy_(((bias - bn.running_mean.double()) * factor + beta).to(dtype))
    return folded


def fold_batchnorm(model: nn.Module) -> nn.Module:
    """Return an eval-mode copy of `model` where BatchNorm is folded.

    We look inside every nn.Sequential for a Conv2d that is followed by a
    BatchNorm2d. We fold each such pair into one Conv2d and replace the
    BatchNorm with nn.Identity (a layer that does nothing).

    The original model does not change.
    """
    folded_model = copy.deepcopy(model).eval()
    folded_count = 0
    for module in folded_model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        children = list(module.named_children())
        for (conv_name, conv), (bn_name, bn) in zip(children, children[1:]):
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(module, conv_name, fold_conv_bn(conv, bn))
                setattr(module, bn_name, nn.Identity())
                folded_count += 1

    remaining = [name for name, m in folded_model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    if remaining:
        raise RuntimeError(f"BatchNorm layers not preceded by a Conv2d: {remaining}")
    folded_model.folded_bn_count = folded_count
    return folded_model
