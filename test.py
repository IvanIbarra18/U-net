import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

from data.data_loading import BasicDataset
from data.splits import get_split_indices
from data.transforms import PairTransformDataset
from models.unet_classic import UNetClassic
from models.unet_modern import UNetModern
from utilis.checkpoint import load_checkpoint
from utilis.loss import DiceBCELoss
from utilis.metrics import dice_score

cudnn.benchmark = True

ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "datasets"

CAST_DTYPES = {"FP16": torch.float16, "BF16": torch.bfloat16}
LOSS_COMPONENTS = ("bce_loss", "dice_loss", "bce_contribution", "dice_contribution")
SUMMARY_FIELDS = (
    "timestamp",
    "dataset",
    "split",
    "model",
    "checkpoint",
    "checkpoint_epoch",
    "model_dtype",
    "autocast",
    "samples",
    "mean_loss",
    *(f"mean_{name}" for name in LOSS_COMPONENTS),
    "mean_dice",
    "latency_ms_per_image",
    "throughput_images_per_sec",
    "weights_mib",
    "peak_gpu_allocated_mib",
    "peak_gpu_reserved_mib",
)
PER_SAMPLE_FIELDS = ("sample_id", "loss", *LOSS_COMPONENTS, "dice")


def split_arg(value):
    """Accept a predefined split index (0-4) or 'full' for the test dataset."""
    if value.lower() == "full":
        return "full"
    try:
        index = int(value)
    except ValueError:
        index = None
    if index not in range(5):
        raise argparse.ArgumentTypeError("must be 0-4 or 'full'")
    return index


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net segmentation model.")
    parser.add_argument("--dataset",required=True,choices=["Carvana", "ISBI", "ISIC"],)
    parser.add_argument("--checkpoint",type=Path,required=True,help="Path to the trained model checkpoint.",)
    parser.add_argument("--split",type=split_arg,default=None,metavar="{0-4,full}",help=("0-4: evaluate valid_N.txt of that split. ""full: evaluate every sample of the held-out datasets/test_<dataset>/ folder. ""If omitted, use the same seeded random split as train.py."),)
    parser.add_argument("--seed",type=int,default=None,help="Seed of the random split (default: the checkpoint's training seed, else 42).",)
    parser.add_argument("--validation","-v",type=float,default=None,help="Validation percentage of the random split (default: the checkpoint's value, else 10).",)
    parser.add_argument("--model",choices=["modern", "classic"],default="modern",help="Architecture used to train the checkpoint.",)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers",type=int,default=min(4, os.cpu_count() or 1),help="DataLoader worker processes (default: min(4, CPU count)).",)
    parser.add_argument("--precision",type=str.upper,choices=["FP32", "FP16", "BF16"],default="FP32",help="Autocast (mixed precision) for inference; weights stay in their original dtype.",)
    parser.add_argument("--cast",type=str.upper,choices=list(CAST_DTYPES),default=None,help=("Convert the whole model (weights and BatchNorm buffers) and its inputs to this dtype. ""If omitted, the model keeps the checkpoint's dtype (usually FP32)."),)
    parser.add_argument("--warmup-batches",type=int,default=3,help="Batches excluded from timing (they are still included in the metrics).",)
    parser.add_argument("--output",type=Path,default=Path("test_results"),help="Directory for evaluation results.",)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.validation is not None and not 0 < args.validation < 100:
        parser.error("--validation must be in (0, 100)")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches cannot be negative")
    if args.cast is not None and args.precision != "FP32":
        parser.error("--cast and --precision (autocast) are separate experiments; use only one.")

    return args


def main():
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = load_checkpoint(args.checkpoint, device)

    # Support the checkpoint dictionary saved by our train.py,
    # as well as a plain model state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        train_args = checkpoint.get("args", {})
    else:
        state_dict = checkpoint
        train_args = {}

    # Reproduce the training split: unless overridden, use the checkpoint's
    # seed and validation percentage, falling back to train.py's defaults.
    seed = args.seed if args.seed is not None else train_args.get("seed", 42)
    validation = (
        args.validation
        if args.validation is not None
        else train_args.get("validation", 10.0)
    )

    if (
        args.split != "full"
        and "split" in train_args
        and train_args["split"] != args.split
    ):
        print(
            f"Warning: the checkpoint was trained with split={train_args['split']}, "
            f"but evaluation uses split={args.split}. The evaluated samples may "
            "include training data."
        )

    # Images are loaded as uint8 and converted on the GPU, which moves 4x
    # less data per batch and keeps the conversion off the CPU workers.
    if args.split == "full":
        # Held-out test set, never used for training or validation.
        dataset_path = DATASETS_DIR / f"test_{args.dataset}"
        if not dataset_path.is_dir():
            raise FileNotFoundError(
                f"--split full needs a test dataset folder: {dataset_path}"
            )
        base_dataset = BasicDataset(
            dataset_path / "imgs",
            dataset_path / "masks",
            image_dtype=np.uint8,
            mask_dtype=np.uint8,
        )
        sample_indices = list(range(len(base_dataset)))
        split_name = "test_full"
    else:
        dataset_path = DATASETS_DIR / f"data_{args.dataset}"
        base_dataset = BasicDataset(
            dataset_path / "imgs",
            dataset_path / "masks",
            image_dtype=np.uint8,
            mask_dtype=np.uint8,
        )

        # Evaluate the validation part of the same split used by train.py.
        _, sample_indices = get_split_indices(
            base_dataset=base_dataset,
            dataset_name=args.dataset,
            split_index=args.split,
            validation=validation,
            seed=seed,
        )
        split_name = (
            f"split_{args.split}"
            if args.split is not None
            else f"random_seed{seed}_val{validation:g}"
        )

    dataset = Subset(
        PairTransformDataset(base_dataset, transform=None),
        sample_indices,
    )
    sample_ids = [base_dataset.ids[index] for index in sample_indices]

    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }

    if args.num_workers > 0:
        loader_options["persistent_workers"] = True

    dataloader = DataLoader(dataset, **loader_options)

    input_channels = 1 if args.dataset == "ISBI" else 3

    if args.model == "classic":
        model = UNetClassic(
            1,
            n_channels=input_channels,
        ).to(device)
    else:
        model = UNetModern(
            n_classes=1,
            n_channels=input_channels,
            base_channels=64,
            batch_norm=True,
        ).to(device)

    # Load the FP32 weights first, then convert the model if requested.
    model.load_state_dict(state_dict)

    if args.cast is not None:
        model = model.to(device=device, dtype=CAST_DTYPES[args.cast])

    model.eval()
    model_dtype = next(model.parameters()).dtype
    weights_mib = sum(
        tensor.numel() * tensor.element_size()
        for tensor in list(model.parameters()) + list(model.buffers())
    ) / (1024 ** 2)

    criterion = DiceBCELoss()

    use_amp = (
        device.type == "cuda"
        and args.precision in ("FP16", "BF16")
    )
    amp_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    per_sample_rows = []
    timed_seconds = 0.0
    timed_samples = 0

    # Warm-up batches are excluded from timing only (cuDNN autotuning).
    warmup_batches = min(args.warmup_batches, len(dataloader) - 1)

    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            # Normalize uint8 images to [0, 1] in FP32 on the device, then
            # round once to the model dtype (as the original FP32 pipeline).
            images = batch["image"].to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            images = (
                images.float()
                .div_(torch.iinfo(batch["image"].dtype).max)
                .to(model_dtype)
            )
            masks = batch["mask"].to(
                device,
                non_blocking=(device.type == "cuda"),
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            start_time = time.perf_counter()

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if use_amp else torch.float32,
                enabled=use_amp,
            ):
                logits = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            elapsed = time.perf_counter() - start_time
            batch_size = images.size(0)

            if batch_index >= warmup_batches:
                timed_seconds += elapsed
                timed_samples += batch_size

            # Loss and Dice are computed in FP32 for every precision, per
            # sample so results can be compared image by image across runs.
            logits = logits.float()
            masks = masks.float()
            for offset in range(batch_size):
                sample_logits = logits[offset:offset + 1]
                sample_mask = masks[offset:offset + 1]
                loss, components = criterion(
                    sample_logits, sample_mask, return_components=True
                )
                per_sample_rows.append({
                    "sample_id": sample_ids[len(per_sample_rows)],
                    "loss": loss.item(),
                    **{name: components[name].item() for name in LOSS_COMPONENTS},
                    "dice": dice_score(sample_logits, sample_mask),
                })

    total_samples = len(per_sample_rows)
    if total_samples == 0:
        raise RuntimeError("The evaluation dataset is empty.")

    # Every sample has the same weight, so batch means are sample means.
    def sample_mean(field):
        return sum(row[field] for row in per_sample_rows) / total_samples

    mean_loss = sample_mean("loss")
    mean_dice = sample_mean("dice")
    latency_ms = (
        timed_seconds / timed_samples * 1000 if timed_samples else float("nan")
    )
    throughput = (
        timed_samples / timed_seconds if timed_seconds > 0 else float("nan")
    )
    model_format = {
        torch.float32: "FP32",
        torch.float16: "FP16",
        torch.bfloat16: "BF16",
    }.get(model_dtype, str(model_dtype))
    precision_name = args.cast or args.precision

    print("\nEvaluation complete")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {split_name} ({total_samples} samples)")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model dtype: {model_format} ({weights_mib:.1f} MiB of weights)")
    print(f"Autocast precision: {args.precision if use_amp else 'off'}")
    print(f"Mean loss: {mean_loss:.6f}")
    print(f"Mean Dice: {mean_dice:.6f}")
    print(
        f"Forward latency: {latency_ms:.2f} ms/image | "
        f"Throughput: {throughput:.2f} images/s "
        f"({timed_samples} images, {warmup_batches} warm-up batches excluded)"
    )

    peak_allocated = peak_reserved = None
    if device.type == "cuda":
        peak_allocated = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )
        print(f"Peak GPU memory allocated: {peak_allocated:.2f} MiB")
        print(f"Peak GPU memory reserved: {peak_reserved:.2f} MiB")

    # 1. One row per run in a shared summary, to compare runs side by side.
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.csv"
    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    with summary_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "split": split_name,
            "model": args.model,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": (
                checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
            ),
            "model_dtype": model_format,
            "autocast": args.precision if use_amp else "off",
            "samples": total_samples,
            "mean_loss": mean_loss,
            **{f"mean_{name}": sample_mean(name) for name in LOSS_COMPONENTS},
            "mean_dice": mean_dice,
            "latency_ms_per_image": latency_ms,
            "throughput_images_per_sec": throughput,
            "weights_mib": weights_mib,
            "peak_gpu_allocated_mib": peak_allocated,
            "peak_gpu_reserved_mib": peak_reserved,
        })

    # 2. Per-sample results of this run, for paired comparisons between
    # precisions on exactly the same images (overwritten on reruns).
    per_sample_path = args.output / "per_sample" / (
        f"{args.checkpoint.stem}_{args.dataset}_{split_name}_{args.model}_"
        f"{precision_name}{'_cast' if args.cast else ''}.csv"
    )
    per_sample_path.parent.mkdir(parents=True, exist_ok=True)
    with per_sample_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PER_SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    print(f"Summary appended to: {summary_path}")
    print(f"Per-sample results saved to: {per_sample_path}")


if __name__ == "__main__":
    main()
