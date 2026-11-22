"""Everything that train.py and test.py save, except the checkpoints.

This includes the result CSV files, the peak GPU memory and the plot of the
training curves.

Files written:
    <output>/performance/<experiment>.csv   by train.py, one row for each epoch
    test_results/summary.csv                by test.py, one row for each evaluation run
    test_results/per_sample/<run>.csv       by test.py, one row for each image of a run
"""

import csv
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch

LOSS_COMPONENTS = ("bce_loss", "dice_loss", "bce_contribution", "dice_contribution")

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
    # Which precision the run really used (see utilis/evaluation.py): the values
    # allowed, how the weights are stored, and what the kernels computed in.
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

PER_SAMPLE_FIELDS = ("sample_id", "model", "total_params", "loss", *LOSS_COMPONENTS, "dice")


# ---------------------------------------------------------------------------
# Writing CSV files
# ---------------------------------------------------------------------------

def _append_row(csv_path, fields, row, renamed_columns=None, incompatible_hint=""):
    """Append one row to a CSV file with the columns `fields`.

    If an older version wrote the file with fewer columns, we first write it
    again with the new header. The old rows get empty values in the new
    columns. renamed_columns maps old column names to new ones.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_fields = reader.fieldnames
            existing_rows = list(reader)
        if existing_fields is not None and renamed_columns:
            existing_fields = [renamed_columns.get(field, field) for field in existing_fields]
            for old_row in existing_rows:
                for old_name, new_name in renamed_columns.items():
                    if old_name in old_row:
                        old_row[new_name] = old_row.pop(old_name)
        if existing_fields != list(fields):
            if existing_fields is None or not set(existing_fields).issubset(fields):
                raise ValueError(f"The columns of {csv_path} do not match the current format. {incompatible_hint}")
            # We write to a temporary file first, so a crash cannot leave a half-written CSV.
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=csv_path.parent,
                                                 suffix=".tmp", delete=False) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    writer = csv.DictWriter(temporary_file, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(existing_rows)
                temporary_path.replace(csv_path)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def initialize_experiment_metrics(csv_path):
    """Start a new per-epoch CSV. If the file already exists, it is replaced."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=EXPERIMENT_CSV_FIELDS).writeheader()


def append_experiment_metrics(csv_path, row):
    """Add one epoch to the training CSV."""
    _append_row(csv_path, EXPERIMENT_CSV_FIELDS, row,
                renamed_columns={"train_dice": "train_dice_score", "val_dice": "val_dice_score"},
                incompatible_hint="Choose a different --experiment-name.")


def append_summary_row(summary_path, row):
    """Add one evaluation run to summary.csv."""
    _append_row(summary_path, EVALUATION_SUMMARY_FIELDS, row)


def write_per_sample_csv(path, rows, model_name, total_params):
    """Write one row for each image, so we can compare two runs image by image.

    If `path` already exists, it is replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PER_SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows({**row, "model": model_name, "total_params": total_params} for row in rows)


# ---------------------------------------------------------------------------
# Building rows
# ---------------------------------------------------------------------------

def peak_gpu_memory_mib(device):
    """Return the peak allocated and reserved GPU memory in MiB, or (None, None) on the CPU."""
    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        torch.cuda.max_memory_reserved(device) / (1024 ** 2),
    )


def epoch_row(epoch, model_name, total_params, train, validation, performance):
    """Build one row of the per-epoch training CSV.

    train        {"loss", "dice", "bce_loss", "dice_loss", "bce_contribution", "dice_contribution"}
    validation   the dictionary returned by eval.evaluate()
    performance  {"epoch_time_sec", "train_images_per_sec", "weights_mib",
                  "peak_gpu_allocated_mib", "peak_gpu_reserved_mib"}
    """
    return {
        "epoch": epoch,
        "model": model_name,
        "total_params": total_params,
        "train_loss": train["loss"],
        **{f"train_{name}": train[name] for name in LOSS_COMPONENTS},
        "train_dice_score": train["dice"],
        "val_loss": validation["loss"],
        **{f"val_{name}": validation[name] for name in LOSS_COMPONENTS},
        "val_dice_score": validation["dice"],
        "epoch_time_sec": performance["epoch_time_sec"],
        "train_images_per_sec": performance["train_images_per_sec"],
        "latency_ms_per_image": validation["latency_ms_per_image"],
        "throughput_images_per_sec": validation["throughput_images_per_sec"],
        "weights_mib": performance["weights_mib"],
        "peak_gpu_allocated_mib": performance["peak_gpu_allocated_mib"],
        "peak_gpu_reserved_mib": performance["peak_gpu_reserved_mib"],
    }


def sample_means(per_sample_rows):
    """Return the mean loss, the mean loss parts and the mean Dice over the
    images of a run. Every image counts the same, so these are per-image means."""
    count = len(per_sample_rows)
    mean = lambda field: sum(row[field] for row in per_sample_rows) / count
    return {
        "samples": count,
        "mean_loss": mean("loss"),
        **{f"mean_{name}": mean(name) for name in LOSS_COMPONENTS},
        "mean_dice": mean("dice"),
    }


def summary_row(args, run, means, performance, environment):
    """Build one row of summary.csv.

    args         the test.py arguments
    run          {"split", "checkpoint_epoch", "total_params", "model_dtype", "autocast",
                  "quantization", "calib_samples", "representation", "storage", "execution"}
    means        from sample_means()
    performance  {"latency_ms_per_image", "throughput_images_per_sec", "weights_mib", "engine_mib",
                  "peak_gpu_allocated_mib", "peak_gpu_reserved_mib"}
    environment  from quantization.env_report.environment_info()
    """
    quantized = args.quant != "NONE"
    tensorrt = args.backend == "tensorrt"
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "split": run["split"],
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": run["checkpoint_epoch"],
        "total_params": run["total_params"],
        "model_dtype": run["model_dtype"],
        "autocast": run["autocast"],
        "tf32": args.tf32,
        "quantization": run["quantization"],
        "act_calibration": args.act_calib or ("dynamic" if quantized and args.act_dynamic else ""),
        "calib_samples": run["calib_samples"] or "",
        "fp_layers": ",".join(args.fp_layers),
        # TensorRT graphs with the depth2space upsamplers get their own label.
        "backend": args.backend + (" (depth2space)" if tensorrt and args.upsample == "depth2space" else ""),
        "representation": run["representation"],
        "storage": run["storage"],
        "execution": run["execution"],
        **means,
        **performance,
        "gpu": environment["gpu"],
        "driver": environment["driver"],
        "torch_version": environment["torch"],
        "tensorrt_version": environment["tensorrt"] if tensorrt else "",
    }


def per_sample_filename(args, split_name, quantization_name, use_tf32):
    """Return the file name of the per-image CSV.

    The name includes every setting that changes the results, so two
    different runs never use the same file."""
    from quantization import modes as quant_modes

    tensorrt = args.backend == "tensorrt"
    parts = [
        f"{args.checkpoint.stem}_{args.dataset}_{split_name}_{args.model}_{args.cast or args.precision}",
        "_cast" if args.cast else "",
        "_trt" if tensorrt else "",
        "_d2s" if tensorrt and args.upsample == "depth2space" else "",
        "_tf32" if use_tf32 else "",
        "" if quantization_name == "none" else f"_{quantization_name}",
        f"_{args.act_calib}" if args.act_calib else "",
        quant_modes.file_suffix(args) if args.quant != "NONE" else "",
        f"_fp-{'+'.join(args.fp_layers)}" if args.fp_layers else "",
    ]
    return "".join(parts) + ".csv"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_training_history(history, save_path=None, show=True):
    """Plot the training and validation loss and Dice per epoch.

    history  {"train_loss": [...], "train_dice": [...], "val_loss": [...], "val_dice": [...]}
    """
    if not history["train_loss"]:
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (axes[0, 0], "train_loss", "b-o", "Training loss", "Loss"),
        (axes[0, 1], "train_dice", "b-o", "Training Dice score", "Dice score"),
        (axes[1, 0], "val_loss", "r-o", "Validation loss", "Loss"),
        (axes[1, 1], "val_dice", "r-o", "Validation Dice score", "Dice score"),
    ]
    for axis, key, style, title, ylabel in panels:
        axis.plot(epochs, history[key], style, markersize=3)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
