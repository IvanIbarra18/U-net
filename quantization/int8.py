"""Symmetric INT8 scale quantization (Wu et al., 2020, Sec. 3).

Notation (the same as in the paper):
    amax   the clipping threshold alpha. We can store values in [-amax, amax].
    scale  s = qmax / amax, with qmax = 2^(b-1) - 1 = 127 for 8 bits.
    x_q    = clip(round(s * x), -qmax, qmax)   (the integer)
    x_hat  = x_q / s                           (the dequantized value)

The range is [-127, 127]. We never produce -128, so the mapping is exactly
symmetric and the zero point is 0. Because of this, the integer dot product
has no extra cross terms.

Granularity:
    per-tensor   amax is one number (used for activations).
    per-channel  amax has one value for each channel along `axis` (used for
                 weights): axis=0 for Conv2d [C_out, C_in, kH, kW], and
                 axis=1 for ConvTranspose2d [C_in, C_out, kH, kW].

This module only does the math. Everything runs in floating point.
fake_quantize() returns values that are limited to the INT8 grid. This
simulates how INT8 changes the accuracy, but not its speed or memory use.
"""

import torch

NUM_BITS = 8
QMAX = 2 ** (NUM_BITS - 1) - 1  # 127

# Protects the division in compute_scale. (The case amax = 0 is handled on its own.)
_MIN_AMAX = torch.finfo(torch.float32).tiny


def weight_axis(module: torch.nn.Module) -> int:
    """Return the output-channel axis of a convolution weight."""
    if isinstance(module, torch.nn.ConvTranspose2d):
        return 1
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        return 0
    raise TypeError(f"Unsupported module for weight quantization: {type(module).__name__}")


def _broadcast_shape(x: torch.Tensor, axis: int) -> list:
    shape = [1] * x.dim()
    shape[axis] = x.shape[axis]
    return shape


def compute_amax(x: torch.Tensor, axis: int | None = None) -> torch.Tensor:
    """Max calibration: amax = max(|x|), per tensor (axis=None) or per channel.

    Returns a scalar tensor, or a tensor with a shape that broadcasts against
    `x` (for example, [C, 1, 1, 1] for axis=0 on a 4-D weight).
    """
    x = x.detach().float()
    if axis is None:
        return x.abs().amax()
    axis = axis % x.dim()
    reduce_dims = [dim for dim in range(x.dim()) if dim != axis]
    return x.abs().amax(dim=reduce_dims, keepdim=True)


def compute_scale(amax: torch.Tensor, qmax: float = QMAX) -> torch.Tensor:
    """Return s = qmax / amax (in FP32).

    By default qmax is 127, the value for INT8. formats.py passes the qmax of
    other formats (INT4: 7, FP8 E4M3: 448, and so on).

    If a tensor or channel is all zeros (amax = 0), it gets s = 1. Then it
    quantizes to exact zeros, and we avoid 0 * inf = NaN.
    """
    amax = amax.float()
    return torch.where(amax > 0, qmax / amax.clamp_min(_MIN_AMAX), torch.ones_like(amax))


def quantize(x: torch.Tensor, amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert real values to INT8. Returns (x_q as torch.int8, scale).

    torch.round rounds half to even (for example, 2.5 becomes 2), like IEEE
    and most hardware. It only differs from round-half-up on exact ties.
    """
    scale = compute_scale(amax)
    x_q = torch.clamp(torch.round(x.float() * scale), -QMAX, QMAX)
    return x_q.to(torch.int8), scale


def dequantize(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Convert INT8 back to real values: x_hat = x_q / s (in FP32)."""
    return x_q.float() / scale


def fake_quantize(x: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    """Quantize and then dequantize. The result has the dtype of `x`.

    This gives the same result as dequantize(*quantize(x, amax)), but it
    never creates an int8 tensor. A simulated INT8 layer applies this to its
    input and to its weight.
    """
    scale = compute_scale(amax)
    x_hat = torch.clamp(torch.round(x.float() * scale), -QMAX, QMAX) / scale
    return x_hat.to(x.dtype)


def fake_quantize_(x: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    """An in-place fake_quantize for FP32 activations.

    It gives the same result as fake_quantize(), but it writes over `x`.
    The normal version creates four temporary tensors as big as `x`. At full
    ISIC resolution, one activation can take about 12 GB, so the normal
    version does not fit in memory. Only use this on tensors that nothing
    else reads later.
    """
    if x.dtype != torch.float32:
        raise TypeError("fake_quantize_ expects an FP32 tensor.")
    scale = compute_scale(amax).to(x.device)
    return x.mul_(scale).round_().clamp_(-QMAX, QMAX).div_(scale)


def quantization_error(x: torch.Tensor, amax: torch.Tensor) -> dict:
    """Return error statistics for fake-quantizing `x` with threshold `amax`.

    sqnr_db is the signal-to-quantization-noise ratio:
    10 * log10(sum(x^2) / sum((x - x_hat)^2)). Higher is better. Each extra
    bit adds about 6 dB.
    """
    x = x.detach().float()
    x_hat = fake_quantize(x, amax)
    noise = x - x_hat
    signal_power = x.pow(2).sum()
    noise_power = noise.pow(2).sum()
    if noise_power == 0:
        sqnr_db = float("inf")
    else:
        sqnr_db = (10 * torch.log10(signal_power / noise_power)).item()
    amax_full = amax.float().expand_as(x) if amax.dim() else amax.float()
    return {
        "max_abs_error": noise.abs().max().item(),
        "mse": noise.pow(2).mean().item(),
        "sqnr_db": sqnr_db,
        "clipped_fraction": (x.abs() > amax_full).float().mean().item(),
    }
