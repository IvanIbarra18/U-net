import csv
import tempfile
from pathlib import Path

import torch

EXPERIMENT_CSV_FIELDS = (
    "epoch",
    "model",
    "total_params",
    "train_loss",
    "train_bce_loss",
    "train_dice_loss",
    "train_bce_contribution",
    "train_dice_contribution",
    "train_dice_score",
    "val_loss",
    "val_bce_loss",
    "val_dice_loss",
    "val_bce_contribution",
    "val_dice_contribution",
    "val_dice_score",
    "epoch_time_sec",
    "train_images_per_sec",
    "latency_ms_per_image",
    "throughput_images_per_sec",
    "weights_mib",
    "peak_gpu_allocated_mib",
    "peak_gpu_reserved_mib",
)

EVALUATION_SUMMARY_FIELDS = (
    "timestamp",
    "dataset",
    "split",
    "model",
    "checkpoint",
    "checkpoint_epoch",
    "total_params",
    "model_dtype",
    "autocast",
    "tf32",
    "quantization",
    "act_calibration",
    "calib_samples",
    "fp_layers",
    # What precision the run really had (see test.py describe_precision):
    # values restricted to / stored in / computed in.
    "backend",
    "representation",
    "storage",
    "execution",
    "samples",
    "mean_loss",
    "mean_bce_loss",
    "mean_dice_loss",
    "mean_bce_contribution",
    "mean_dice_contribution",
    "mean_dice",
    "latency_ms_per_image",
    "throughput_images_per_sec",
    "weights_mib",
    "engine_mib",
    "peak_gpu_allocated_mib",
    "peak_gpu_reserved_mib",
    "gpu",
    "driver",
    "torch_version",
    "tensorrt_version",
)


def append_summary_row(summary_path, row):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    if not write_header:
        with summary_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_fields = reader.fieldnames
            existing_rows = list(reader)
        if existing_fields != list(EVALUATION_SUMMARY_FIELDS):
            if (
                existing_fields is None
                or not set(existing_fields).issubset(EVALUATION_SUMMARY_FIELDS)
            ):
                raise ValueError(
                    f"CSV at {summary_path} has columns incompatible with "
                    "the current evaluation summary."
                )
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    newline="",
                    encoding="utf-8",
                    dir=summary_path.parent,
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    writer = csv.DictWriter(
                        temporary_file,
                        fieldnames=EVALUATION_SUMMARY_FIELDS,
                    )
                    writer.writeheader()
                    writer.writerows(existing_rows)
                temporary_path.replace(summary_path)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()

    with summary_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EVALUATION_SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_experiment_metrics(csv_path, metrics):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    if not write_header:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_fields = reader.fieldnames
            existing_metrics = list(reader)
        field_aliases = {
            "train_dice": "train_dice_score",
            "val_dice": "val_dice_score",
        }
        if existing_fields is not None:
            existing_fields = [
                field_aliases.get(field, field) for field in existing_fields
            ]
            for row in existing_metrics:
                for old_name, new_name in field_aliases.items():
                    if old_name in row:
                        row[new_name] = row.pop(old_name)
        if existing_fields != list(EXPERIMENT_CSV_FIELDS):
            if (
                existing_fields is None
                or not set(existing_fields).issubset(EXPERIMENT_CSV_FIELDS)
            ):
                raise ValueError(
                    f"CSV at {csv_path} has columns incompatible with the "
                    "current experiment metrics. Choose a different "
                    "--experiment-name."
                )

            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    newline="",
                    encoding="utf-8",
                    dir=csv_path.parent,
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    writer = csv.DictWriter(
                        temporary_file,
                        fieldnames=EXPERIMENT_CSV_FIELDS,
                    )
                    writer.writeheader()
                    writer.writerows(existing_metrics)
                temporary_path.replace(csv_path)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EXPERIMENT_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


def initialize_experiment_metrics(csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EXPERIMENT_CSV_FIELDS)
        writer.writeheader()


@torch.no_grad()
def dice_score(logits, target, threshold=0.5, as_tensor=False):
    """Calculate mean binary foreground Dice for a batch.

    With as_tensor=True the result stays on the device as a 0-d tensor,
    avoiding the CPU/GPU synchronization caused by .item().
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