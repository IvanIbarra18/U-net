"""Batch Normalization folding (Wu et al., 2020, Sec. 3 / Jacob et al., 2018).

In eval mode BatchNorm is a fixed per-channel affine map:
    y_k = gamma_k * (x_k - mu_k) / sigma_k + beta_k,  sigma_k = sqrt(var_k + eps)
When it directly follows a convolution, it can be absorbed into it:
    w'_k = w_k * gamma_k / sigma_k
    b'_k = gamma_k * (b_k - mu_k) / sigma_k + beta_k
The folded network computes the same function (up to FP rounding) with one
fewer op per block. Weights must be quantized AFTER folding, because the
folded weights are the ones that are actually executed.

In UNetModern every DoubleConv is Sequential(Conv, BN, ReLU, Conv, BN, ReLU);
the ConvTranspose2d upsamplers and the final 1x1 conv have no BN.
"""

import copy

import torch
import torch.nn as nn


@torch.no_grad()
def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a new Conv2d equal to bn(conv(x)) in eval mode."""
    if conv.out_channels != bn.num_features:
        raise ValueError("Conv output channels and BatchNorm features differ.")
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BatchNorm has no running statistics to fold.")

    dtype = conv.weight.dtype
    # Compute in FP64 so the folding itself adds no measurable rounding.
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
    """Return an eval-mode copy of `model` with every Conv2d -> BatchNorm2d
    pair inside an nn.Sequential folded (BN replaced by nn.Identity).

    The original model is not modified.
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
