"""Number formats and granularities for simulated ("fake") quantization.

int8.py covers only one case: symmetric INT8, per tensor or per channel.
This module extends the same idea to every format in the project. All
formats follow the same recipe:

    s     = qmax / amax                 (the scale; one amax per tensor, channel or block)
    x_hat = round_to_format(x * s) / s  (round onto the format's grid and saturate)

Here amax is the clipping threshold (the largest value we keep), and qmax is
the largest value the format can store. The only part that changes from one
format to another is round_to_format():

    format     values after scaling                       qmax   grid
    int8       integers in [-127, 127]                     127    uniform
    int4       integers in [-7, 7]                         7      uniform
    fp8_e4m3   minifloat: 4 exponent, 3 mantissa bits      448    logarithmic
    fp6_e2m3   minifloat: 2 exponent, 3 mantissa bits      7.5    logarithmic
    fp6_e3m2   minifloat: 3 exponent, 2 mantissa bits      28     logarithmic
    fp4_e2m1   minifloat: 2 exponent, 1 mantissa bit       6      {0, 0.5, 1, 1.5, 2, 3, 4, 6}

A minifloat (a very small floating-point format) with E exponent bits and
M mantissa bits has 2^M values in each power of two (each "binade"). So the
step between values grows when the numbers get bigger, and the relative
precision stays the same. The integer grid is different: its step is always
the same. All the formats above are the "fn" variants. They have no
infinities, and values above qmax saturate (they become qmax).

Granularity tells us which values share one amax (and so one scale):
    tensor   one amax for the whole tensor (used for activations; calibrated).
    channel  one amax for each output channel (used for weights; axis 0 for
             Conv2d, axis 1 for ConvTranspose2d).
    block    one amax for each block of `block_size` values in a row along
             the REDUCTION axis (the input channels). This is the axis that a
             real kernel sums over, so a real block-scaled kernel can use the
             same blocks. For weights the blocks are static. For activations
             they are dynamic: we compute the amax of each block on the fly,
             for each pixel, with no calibration.

Real formats can also store the block scales in low precision. We support:
    fp32  exact scales (a pure simulation of the element format).
    e4m3  NVFP4 style: an FP8 E4M3 block scale times one FP32 scale per tensor.
    e8m0  MX style (OCP MXFP4, MXFP6, MXFP8): a power-of-two block scale.

Presets (QuantSpec.preset):
    nvfp4 = fp4_e2m1, blocks of 16, e4m3 scales.
    mxfp4 = fp4_e2m1, blocks of 32, e8m0 scales.

Everything here is a SIMULATION. The values are limited to the format's
grid, but we still store them and compute with them in FP32.
"""

import math
from dataclasses import dataclass

import torch

from quantization.int8 import compute_scale


@dataclass(frozen=True)
class Format:
    name: str
    kind: str  # "int" or "float"
    bits: int
    qmax: float
    exponent_bits: int = 0
    mantissa_bits: int = 0

    @property
    def min_normal_exponent(self) -> int:
        """Return the exponent of the smallest normal value, which is 1 - bias."""
        return 1 - (2 ** (self.exponent_bits - 1) - 1)


FORMATS = {
    "int8": Format("int8", "int", 8, 127.0),
    "int4": Format("int4", "int", 4, 7.0),
    "fp8_e4m3": Format("fp8_e4m3", "float", 8, 448.0, 4, 3),
    "fp6_e2m3": Format("fp6_e2m3", "float", 6, 7.5, 2, 3),
    "fp6_e3m2": Format("fp6_e3m2", "float", 6, 28.0, 3, 2),
    "fp4_e2m1": Format("fp4_e2m1", "float", 4, 6.0, 2, 1),
}
GRANULARITIES = ("tensor", "channel", "block")
SCALE_FORMATS = ("fp32", "e4m3", "e8m0")


def get_format(name: str) -> Format:
    try:
        return FORMATS[name]
    except KeyError:
        raise ValueError(f"Unknown format {name!r}; choose from {list(FORMATS)}") from None


def round_to_format(x: torch.Tensor, fmt: Format) -> torch.Tensor:
    """Round values that are already scaled onto the format's grid.

    Ties are rounded half to even, and values saturate at +-qmax. The work is
    done in FP32."""
    if fmt.kind == "int":
        return torch.clamp(torch.round(x), -fmt.qmax, fmt.qmax)
    magnitude = x.abs().clamp(max=fmt.qmax)
    # The binade exponent of each value. Subnormal values use the same step as
    # the smallest normal binade. (log2(0) = -inf is clamped in the same way.)
    exponent = torch.floor(torch.log2(magnitude)).clamp(min=fmt.min_normal_exponent)
    step = torch.exp2(exponent - fmt.mantissa_bits)
    rounded = (torch.round(magnitude / step) * step).clamp(max=fmt.qmax)
    return torch.copysign(rounded, x)


def _quantize_scale(scale: torch.Tensor, scale_format: str, fmt: Format, global_amax: torch.Tensor | None) -> torch.Tensor:
    """Round the scale s = qmax / amax the way the hardware would store it.

    Returns the FP32 scale that we really multiply by."""
    if scale_format == "fp32":
        return scale
    amax = fmt.qmax / scale  # the amax of each block
    if scale_format == "e8m0":
        # OCP MX: the shared exponent is floor(log2(amax)) - emax_elem. This puts
        # the largest value of the block in the top binade of the element format.
        emax_elem = math.floor(math.log2(fmt.qmax))
        block_scale = torch.exp2(torch.floor(torch.log2(amax)) - emax_elem)  # x_q = x / block_scale
        return torch.where(amax > 0, 1.0 / block_scale, torch.ones_like(scale))
    if scale_format == "e4m3":
        # NVFP4: the decode scale (amax / qmax) is stored in E4M3, relative to
        # one FP32 factor per tensor. We choose that factor so that the largest
        # block scale is 448.
        tensor_factor = torch.where(global_amax > 0, global_amax / (fmt.qmax * 448.0), torch.ones_like(global_amax))
        decode = (amax / fmt.qmax) / tensor_factor
        decode = decode.clamp(max=448.0).to(torch.float8_e4m3fn).float() * tensor_factor
        return torch.where(decode > 0, 1.0 / decode, torch.ones_like(scale))
    raise ValueError(f"scale_format must be one of {SCALE_FORMATS}")


@dataclass(frozen=True)
class QuantSpec:
    """Describe how one tensor is quantized (in simulation).

    format       a key of FORMATS.
    granularity  "tensor", "channel" or "block".
    block_size   the number of values in each block along the reduction axis
                 (only for granularity="block").
    scale_format "fp32", "e4m3" (NVFP4) or "e8m0" (MX). Only used for blocks.
    """

    format: str
    granularity: str = "channel"
    block_size: int = 0
    scale_format: str = "fp32"

    def __post_init__(self):
        get_format(self.format)
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {GRANULARITIES}")
        if self.granularity == "block" and self.block_size < 1:
            raise ValueError("granularity='block' needs block_size >= 1")
        if self.scale_format not in SCALE_FORMATS:
            raise ValueError(f"scale_format must be one of {SCALE_FORMATS}")
        if self.scale_format != "fp32" and self.granularity != "block":
            raise ValueError("low-precision scales (e4m3/e8m0) only apply to block granularity")

    @staticmethod
    def preset(name: str) -> "QuantSpec":
        presets = {
            "nvfp4": QuantSpec("fp4_e2m1", "block", 16, "e4m3"),
            "mxfp4": QuantSpec("fp4_e2m1", "block", 32, "e8m0"),
        }
        return presets[name]

    def describe(self) -> str:
        if self.granularity == "block":
            scales = "" if self.scale_format == "fp32" else f", {self.scale_format} scales"
            return f"{self.format} blocks of {self.block_size}{scales}"
        return f"{self.format} per-{self.granularity}"


def _blocks(x: torch.Tensor, axis: int, block_size: int):
    """Reshape x to [..., n_blocks, block_size], with the blocks along `axis`.

    If needed, we add zeros at the end of that axis. Returns the blocked
    tensor and a function that changes it back to the original shape."""
    axis = axis % x.dim()
    moved = x.movedim(axis, -1)
    length = moved.shape[-1]
    pad = (-length) % block_size
    if pad:
        moved = torch.nn.functional.pad(moved, (0, pad))
    blocked = moved.reshape(*moved.shape[:-1], -1, block_size)

    def restore(y: torch.Tensor) -> torch.Tensor:
        y = y.reshape(*moved.shape)[..., :length]
        return y.movedim(-1, axis)

    return blocked, restore


def fake_quantize_spec(x: torch.Tensor, spec: QuantSpec, axis: int = 0, amax: torch.Tensor | None = None) -> torch.Tensor:
    """Quantize and then dequantize `x` with `spec`. The result has x's dtype.

    axis  the channel axis (granularity="channel") or the reduction axis
          (granularity="block"). It is not used for "tensor".
    amax  the calibrated amax for "tensor" (activations). If it is None, we
          compute it from x. Channels and blocks always use their own max
          (for weights, or for dynamic activations).
    """
    fmt = get_format(spec.format)
    xf = x.float()
    if spec.granularity == "tensor":
        amax = xf.abs().amax() if amax is None else amax.float().to(xf.device)
        scale = compute_scale(amax, fmt.qmax)
        return (round_to_format(xf * scale, fmt) / scale).to(x.dtype)
    if spec.granularity == "channel":
        axis = axis % xf.dim()
        reduce_dims = [d for d in range(xf.dim()) if d != axis]
        scale = compute_scale(xf.abs().amax(dim=reduce_dims, keepdim=True), fmt.qmax)
        return (round_to_format(xf * scale, fmt) / scale).to(x.dtype)
    blocked, restore = _blocks(xf, axis, spec.block_size)
    scale = compute_scale(blocked.abs().amax(dim=-1, keepdim=True), fmt.qmax)
    scale = _quantize_scale(scale, spec.scale_format, fmt, xf.abs().amax())
    return restore(round_to_format(blocked * scale, fmt) / scale).to(x.dtype)
