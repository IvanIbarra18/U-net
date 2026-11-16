"""Export the (BN-folded) FP32 U-Net to ONNX with dynamic height and width.

This ONNX file is the input of every TensorRT engine (FP32 reference first,
then INT8/FP8/... with Q/DQ nodes), so it must compute exactly what
UNetModern computes, for every image size.

The ConvTranspose2d problem
---------------------------
UNetModern.forward calls up(x, output_size=skip.size()). When a feature map
has odd size (231/900 ISIC images, all Carvana images), PyTorch picks
output_padding = 1 so the upsampled map matches the skip. That value
depends on the input size, but ONNX ConvTranspose stores output_padding as a
constant attribute, so a dynamic-shape export fails
(aten.convolution with symbolic output_padding cannot be translated).

Exact rewrite (export only; the model and its weights are unchanged)
-------------------------------------------------------------------
With kernel_size == stride == s and padding 0, output pixel (i, j) receives
exactly one contribution, from input pixel (i // s, j // s), and only if
i < s*H and j < s*W. The extra row/column created by output_padding receives
no contribution, so it holds just the bias. Therefore

    up(x, output_size=skip.size())
        == pad(conv_transpose(x, weight, no bias), to skip size, value 0) + bias

which only uses Shape/Pad/Add with dynamic sizes. ExportableUNet applies it
and is checked against UNetModern by quantization/test_export_onnx.py.

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
MIN_SIZE = 16  # four 2x2 poolings
MAX_SIZE = 8192
OPSET = 18


def upsample_to(up: nn.ConvTranspose2d, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Same result as up(x, output_size=skip.size()), expressible in ONNX
    with dynamic shapes (see the module docstring)."""
    y = F.conv_transpose2d(x, up.weight, None, stride=up.stride)
    y = F.pad(y, (0, skip.shape[-1] - y.shape[-1], 0, skip.shape[-2] - y.shape[-2]))
    if up.bias is not None:
        y = y + up.bias.view(1, -1, 1, 1)
    return y


def check_upsampler(up: nn.ConvTranspose2d) -> None:
    """The rewrite is only exact for this configuration."""
    ok = (
        up.kernel_size == up.stride
        and up.padding == (0, 0)
        and up.output_padding == (0, 0)
        and up.dilation == (1, 1)
        and up.groups == 1
    )
    if not ok:
        raise ValueError(f"upsample_to() is not exact for {up}")


class ExportableUNet(nn.Module):
    """UNetModern.forward with upsample_to() instead of output_size.

    Wraps (does not copy) the model, so it uses the same weights. Any change
    to UNetModern.forward must be mirrored here; the unit tests compare both.
    """

    def __init__(self, model: UNetModern):
        super().__init__()
        for up in (model.up4, model.up3, model.up2, model.up1):
            check_upsampler(up)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        e1 = m.enc1(x)
        e2 = m.enc2(m.pool(e1))
        e3 = m.enc3(m.pool(e2))
        e4 = m.enc4(m.pool(e3))
        b = m.bottleneck(m.pool(e4))
        d4 = m.dec4(torch.cat([e4, upsample_to(m.up4, b, e4)], dim=1))
        d3 = m.dec3(torch.cat([e3, upsample_to(m.up3, d4, e3)], dim=1))
        d2 = m.dec2(torch.cat([e2, upsample_to(m.up2, d3, e2)], dim=1))
        d1 = m.dec1(torch.cat([e1, upsample_to(m.up1, d2, e1)], dim=1))
        return m.segment(d1)


def export_onnx(model: nn.Module, path: Path, n_channels: int, opset: int = OPSET) -> Path:
    """Export `model` (wrapped in ExportableUNet) with dynamic H and W."""
    wrapper = ExportableUNet(model).eval()
    device = next(model.parameters()).device
    example = torch.rand(1, n_channels, 97, 131, device=device)  # odd sizes on purpose
    height = torch.export.Dim("height", min=MIN_SIZE, max=MAX_SIZE)
    width = torch.export.Dim("width", min=MIN_SIZE, max=MAX_SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (example,), str(path),
            input_names=["image"], output_names=["logits"],
            dynamic_shapes=({2: height, 3: width},),
            opset_version=opset, dynamo=True, external_data=False,
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
    parser.add_argument("--output", type=Path, default=None, help="Default: exports/<checkpoint>_fp32[_bnfolded].onnx")
    args = parser.parse_args()

    # Export on the CPU: the graph and the FP32 weights do not depend on the device.
    checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    n_channels = 1 if args.dataset == "ISBI" else 3
    model = UNetModern(n_classes=1, n_channels=n_channels, base_channels=args.base_channels, batch_norm=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if not args.keep_bn:
        model = fold_batchnorm(model)

    suffix = "fp32" if args.keep_bn else "fp32_bnfolded"
    output = args.output or ROOT_DIR / "exports" / f"{args.checkpoint.stem}_{suffix}.onnx"
    export_onnx(model, output, n_channels, args.opset)

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
        "convtranspose_rewrite": "pad(conv_transpose(x, w), skip size) + bias  (exact, see export_onnx.py)",
        "onnx_bytes": output.stat().st_size,
        "onnx_sha256": sha256(output),
        "environment": environment_info(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"ONNX saved to: {output} ({metadata['onnx_bytes'] / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
