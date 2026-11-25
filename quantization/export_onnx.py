"""Export the FP32 U-Net (usually BN-folded) to ONNX with dynamic height and width.

Every TensorRT engine starts from this ONNX file: first the FP32 reference,
then INT8, FP8 and the other formats with Q/DQ nodes. So the file must compute
exactly the same thing as UNetModern, for every image size.

The ConvTranspose2d problem
---------------------------
UNetModern.forward calls up(x, output_size=skip.size()). When a feature map
has an odd size (231 of the 900 ISIC images, and all Carvana images), PyTorch
chooses output_padding = 1, so that the upsampled map has the same size as
the skip connection. This value depends on the input size. But ONNX
ConvTranspose stores output_padding as a constant attribute, so an export
with dynamic shapes fails. (The exporter cannot translate aten.convolution
when output_padding is symbolic.)

An exact rewrite (only for export; the model and its weights do not change)
--------------------------------------------------------------------------
When kernel_size == stride == s and padding is 0, each output pixel (i, j)
gets exactly one contribution, from input pixel (i // s, j // s), and only
if i < s*H and j < s*W. The extra row or column that output_padding adds gets
no contribution, so it only holds the bias. This means that

    up(x, output_size=skip.size())
        == pad(conv_transpose(x, weight, no bias), to skip size, value 0) + bias

This version only uses Shape, Pad and Add with dynamic sizes. ExportableUNet
uses it, and tests/test_export_onnx.py checks it against UNetModern.

FP8 needs a second rewrite, which is also exact (--upsample depth2space).
When kernel == stride == 2, a ConvTranspose is the same as a 1x1 Conv that
outputs the 4 taps as channels, followed by DepthToSpace. TensorRT 11.4 has
FP8 Conv kernels, but it runs an FP8 ConvTranspose in FP32 (verified on the
RTX 5080).

Usage (from the repository root):
    python -m quantization.export_onnx --dataset Carvana \
        --checkpoint checkpoints/best_CarvanaBaseline.pth
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet_modern import UNetModern
from quantization.bn_fold import fold_batchnorm
from quantization.env_report import environment_info
from utilis.checkpoint import load_checkpoint

ROOT_DIR = Path(__file__).resolve().parent.parent
MIN_SIZE = 16  # the model has four 2x2 pooling layers
MAX_SIZE = 8192
OPSET = 18


def upsample_to(up: nn.ConvTranspose2d, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Give the same result as up(x, output_size=skip.size()), but in a form
    that ONNX can store with dynamic shapes (see the module docstring)."""
    y = F.conv_transpose2d(x, up.weight, None, stride=up.stride)
    y = F.pad(y, (0, skip.shape[-1] - y.shape[-1], 0, skip.shape[-2] - y.shape[-2]))
    if up.bias is not None:
        y = y + up.bias.view(1, -1, 1, 1)
    return y


def depth2space_weight(up: nn.ConvTranspose2d) -> torch.Tensor:
    """Turn a ConvTranspose2d weight into a 1x1 Conv weight.

    The input is a ConvTranspose2d (k=s=2) weight with shape
    [C_in, C_out, 2, 2]. The output is a 1x1 Conv weight with shape
    [4*C_out, C_in, 1, 1]. Its output channel co*4 + a*2 + b holds tap (a, b).

    When kernel == stride == 2, every input pixel (i, j) writes its own 2x2
    output block: out[co, 2i+a, 2j+b] = sum_ci x[ci, i, j] * W[ci, co, a, b].
    This is the same as a 1x1 convolution that outputs the 4 taps as
    channels, followed by pixel_shuffle(2) (ONNX DepthToSpace, CRD order),
    which moves them to their places in the image.
    """
    c_in, c_out, kh, kw = up.weight.shape
    return up.weight.detach().permute(1, 2, 3, 0).reshape(c_out * kh * kw, c_in, 1, 1).contiguous()


def upsample_to_depth2space(up: nn.ConvTranspose2d, conv1x1: nn.Conv2d, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Give the same result as up(x, output_size=skip.size()), using only
    Conv, DepthToSpace, Pad and Add. (TensorRT has FP8 Conv kernels, but it
    has no FP8 deconvolution kernels.)"""
    y = F.pixel_shuffle(conv1x1(x), up.stride[0])
    y = F.pad(y, (0, skip.shape[-1] - y.shape[-1], 0, skip.shape[-2] - y.shape[-2]))
    if up.bias is not None:
        y = y + up.bias.view(1, -1, 1, 1)
    return y


def check_upsampler(up: nn.ConvTranspose2d) -> None:
    """Check the layer settings. The rewrite is exact only for these settings."""
    ok = (
        up.kernel_size == up.stride
        and up.padding == (0, 0)
        and up.output_padding == (0, 0)
        and up.dilation == (1, 1)
        and up.groups == 1
    )
    if not ok:
        raise ValueError(f"upsample_to() is not exact for {up}")


UPSAMPLERS = ("up4", "up3", "up2", "up1")
UPSAMPLE_MODES = ("convtranspose", "depth2space")


class ExportableUNet(nn.Module):
    """UNetModern.forward, but with an upsampler that is easy to export.

    It does not use output_size. Instead, it uses one of these:
        upsample="convtranspose"  upsample_to(): ConvTranspose + Pad + Add
                                  (the default; used for FP32 and INT8).
        upsample="depth2space"    upsample_to_depth2space(): 1x1 Conv +
                                  DepthToSpace + Pad + Add (used for FP8).

    This class wraps the model and does not copy it, so it uses the same
    weights. In depth2space mode it also holds the 1x1 convs
    (`up_1x1.<name>`). Their weights are the ConvTranspose weights in a new
    order. If you change UNetModern.forward, you must make the same change
    here. The unit tests compare the two.
    """

    def __init__(self, model: UNetModern, upsample: str = "convtranspose"):
        super().__init__()
        if upsample not in UPSAMPLE_MODES:
            raise ValueError(f"upsample must be one of {UPSAMPLE_MODES}")
        for name in UPSAMPLERS:
            check_upsampler(getattr(model, name))
        self.model = model
        self.upsample = upsample
        if upsample == "depth2space":
            self.up_1x1 = nn.ModuleDict()
            for name in UPSAMPLERS:
                weight = depth2space_weight(getattr(model, name))
                conv = nn.Conv2d(weight.shape[1], weight.shape[0], kernel_size=1, bias=False)
                conv.weight.data.copy_(weight)
                self.up_1x1[name] = conv.to(weight.device)

    def up(self, name: str, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampler = getattr(self.model, name)
        if self.upsample == "depth2space":
            return upsample_to_depth2space(upsampler, self.up_1x1[name], x, skip)
        return upsample_to(upsampler, x, skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        e1 = m.enc1(x)
        e2 = m.enc2(m.pool(e1))
        e3 = m.enc3(m.pool(e2))
        e4 = m.enc4(m.pool(e3))
        b = m.bottleneck(m.pool(e4))
        d4 = m.dec4(torch.cat([e4, self.up("up4", b, e4)], dim=1))
        d3 = m.dec3(torch.cat([e3, self.up("up3", d4, e3)], dim=1))
        d2 = m.dec2(torch.cat([e2, self.up("up2", d3, e2)], dim=1))
        d1 = m.dec1(torch.cat([e1, self.up("up1", d2, e1)], dim=1))
        return m.segment(d1)


def export_onnx(model: nn.Module, path: Path, n_channels: int, opset: int = OPSET, upsample: str = "convtranspose") -> Path:
    """Wrap `model` in ExportableUNet and export it with dynamic H and W."""
    wrapper = ExportableUNet(model, upsample).eval()
    device = next(model.parameters()).device
    example = torch.rand(1, n_channels, 97, 131, device=device)  # we use odd sizes on purpose
    height = torch.export.Dim("height", min=MIN_SIZE, max=MAX_SIZE)
    width = torch.export.Dim("width", min=MIN_SIZE, max=MAX_SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (example,), str(path),
            input_names=["image"], output_names=["logits"],
            dynamic_shapes=({2: height, 3: width},),
            opset_version=opset, dynamo=True, external_data=False,
            # No progress messages: they contain an emoji that a Windows console
            # (cp1252) cannot print, and that error stops the export.
            verbose=False,
        )
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Export the FP32 U-Net to ONNX (dynamic H, W).")
    parser.add_argument("--dataset", required=True, choices=["Carvana", "ISBI", "ISIC"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--keep-bn", action="store_true", help="Export BatchNorm instead of folding it.")
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument("--upsample", choices=UPSAMPLE_MODES, default="convtranspose",
                        help="How the ConvTranspose upsamplers are written: convtranspose (FP32/INT8) or "
                             "depth2space = exact 1x1 Conv + DepthToSpace (FP8: TensorRT has no FP8 deconvolution).")
    parser.add_argument("--output", type=Path, default=None, help="Default: exports/<checkpoint>_fp32[_bnfolded][_d2s].onnx")
    args = parser.parse_args()

    # We export on the CPU. The graph and the FP32 weights are the same on any device.
    checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    n_channels = 1 if args.dataset == "ISBI" else 3
    model = UNetModern(n_classes=1, n_channels=n_channels, base_channels=args.base_channels, batch_norm=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if not args.keep_bn:
        model = fold_batchnorm(model)

    suffix = ("fp32" if args.keep_bn else "fp32_bnfolded") + ("_d2s" if args.upsample == "depth2space" else "")
    output = args.output or ROOT_DIR / "exports" / f"{args.checkpoint.stem}_{suffix}.onnx"
    export_onnx(model, output, n_channels, args.opset, args.upsample)

    import onnx

    onnx.checker.check_model(str(output))
    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "base_channels": args.base_channels,
        "bn_folded": not args.keep_bn,
        "precision": "fp32",
        "opset": args.opset,
        "dynamic_shapes": {"height": [MIN_SIZE, MAX_SIZE], "width": [MIN_SIZE, MAX_SIZE]},
        "upsample": args.upsample,
        "convtranspose_rewrite": (
            "pad(conv_transpose(x, w), skip size) + bias" if args.upsample == "convtranspose"
            else "pad(depth_to_space(conv1x1(x, rearranged w)), skip size) + bias"
        ) + "  (exact, see export_onnx.py)",
        "onnx_bytes": output.stat().st_size,
        "onnx_sha256": sha256(output),
        "environment": environment_info(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"ONNX saved to: {output} ({metadata['onnx_bytes'] / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
