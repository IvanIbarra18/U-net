"""Segmentation metrics. (The code that writes CSV files is in utilis/reporting.py.)"""

import torch


@torch.no_grad()
def dice_score(logits, target, threshold=0.5, as_tensor=False):
    """Calculate the mean binary foreground Dice score for a batch.

    With as_tensor=True, the result stays on the device as a 0-d tensor.
    This avoids the CPU-GPU synchronization that .item() causes.
    """

    if logits.ndim != 4 or target.ndim != 4:
        raise ValueError("Logits and targets must have shape (B, C, H, W).")

    if logits.shape[0] != target.shape[0]:
        raise ValueError("Logits and targets must have the same batch size.")

    if target.shape[1] != 1:
        raise ValueError(f"Expected one target channel, got {target.shape[1]}.")

    if logits.shape[1] == 1:
        foreground_logits = logits
    elif logits.shape[1] == 2:
        foreground_logits = logits[:, 1:2]
    else:
        raise ValueError(f"Expected one or two logit channels, got {logits.shape[1]}.")

    out_h, out_w = foreground_logits.shape[-2:]
    target_h, target_w = target.shape[-2:]

    if out_h > target_h or out_w > target_w:
        raise ValueError(
            f"Output size {out_h}x{out_w} exceeds target "
            f"size {target_h}x{target_w}."
        )

    top = (target_h - out_h) // 2
    left = (target_w - out_w) // 2
    target = target[..., top:top + out_h, left:left + out_w]

    if target.shape[-2:] != foreground_logits.shape[-2:]:
        raise ValueError("Cropped targets and logits must have matching sizes.")

    predictions = torch.sigmoid(foreground_logits) >= threshold
    targets = target >= 0.5

    dims = (1, 2, 3)
    intersection = (predictions & targets).sum(dim=dims).float()
    denominator = predictions.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + 1.0) / (denominator.float() + 1.0)
    dice = dice.mean()
    return dice if as_tensor else dice.item()