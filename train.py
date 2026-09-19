import argparse
import csv
import logging
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset


# ============================================================
# Configuration
# ============================================================

DATASETS = {
    "Carvana": {
        "images": Path("./data_carvana/imgs/"),
        "masks": Path("./data_carvana/masks/"),
        "mask_suffix": "_mask",
    },
    "ISIC": {
        "images": Path("./data_ISIC/imgs/"),
        "masks": Path("./data_ISIC/masks/"),
        "mask_suffix": "_Segmentation",
    },
}

SUPPORTED_PRECISIONS = ("FP32", "FP16", "BF16")


def set_seed(seed):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Keep CUDA operations reproducible where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(model, loader, device, precision):
    """
    Calculate Dice and IoU on the validation set.

    For the binary segmentation experiments used here, class 1
    is treated as the foreground class.
    """

    model.eval()

    if precision == "FP32":
        amp = False
        autocast_dtype = torch.float32
    elif precision == "FP16":
        amp = True
        autocast_dtype = torch.float16
    elif precision == "BF16":
        amp = True
        autocast_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(
                device=device,
                dtype=torch.float32,
                memory_format=torch.channels_last,
            )

            true_masks = batch["mask"].to(
                device=device,
                dtype=torch.long,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=amp,
            ):

                masks_pred = model(images)

            # Binary segmentation with two output classes.
            pred_masks = masks_pred.argmax(dim=1)

            # Foreground class = 1
            pred_foreground = pred_masks == 1
            true_foreground = true_masks == 1

            intersection = (
                pred_foreground & true_foreground
            ).sum().float()

            pred_area = pred_foreground.sum().float()
            true_area = true_foreground.sum().float()

            union = pred_area + true_area - intersection

            dice = (
                (2.0 * intersection + 1e-8)
                / (pred_area + true_area + 1e-8)
            )

            iou = (
                (intersection + 1e-8)
                / (union + 1e-8)
            )

            total_dice += dice.item()
            total_iou += iou.item()
            num_batches += 1

    model.train()

    if num_batches == 0:
        return 0.0, 0.0

    return (
        total_dice / num_batches,
        total_iou / num_batches,
    )


# ============================================================
# GPU memory
# ============================================================

def get_gpu_memory():
    """
    Return current and peak PyTorch GPU memory usage in GB.
    """

    if not torch.cuda.is_available():
        return 0.0, 0.0

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return allocated, peak


# ============================================================
# Plotting
# ============================================================

def save_plots(history, results_dir):

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = history["epoch"]

    plots = [
        ("train_loss", "Training Loss", "loss.png"),
        ("val_dice", "Validation Dice", "dice.png"),
        ("val_iou", "Validation IoU", "iou.png"),
        ("epoch_time", "Epoch Time (seconds)", "epoch_time.png"),
        ("throughput", "Throughput (images/second)", "throughput.png"),
        ("peak_vram", "Peak GPU Memory (GB)", "peak_vram.png"),
    ]

    for key, ylabel, filename in plots:

        plt.figure()

        plt.plot(
            epochs,
            history[key],
            marker="o",
        )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            results_dir / filename,
            dpi=150,
        )

        plt.close()


# ============================================================
# CSV
# ============================================================

def save_metrics_csv(history, results_dir):

    path = results_dir / "metrics.csv"

    fieldnames = [
        "epoch",
        "train_loss",
        "val_dice",
        "val_iou",
        "epoch_time",
        "throughput",
        "peak_vram",
        "gpu_memory_allocated",
        "gpu_memory_reserved",
        "learning_rate",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i in range(len(history["epoch"])):

            writer.writerow({
                key: history[key][i]
                for key in fieldnames
            })


def save_summary(
    history,
    results_dir,
    dataset_name,
    precision,
    total_time,
):

    best_dice_index = max(
        range(len(history["val_dice"])),
        key=lambda i: history["val_dice"][i],
    )

    summary = {
        "dataset": dataset_name,
        "precision": precision,
        "epochs": len(history["epoch"]),

        "best_dice_epoch":
            history["epoch"][best_dice_index],

        "best_dice":
            history["val_dice"][best_dice_index],

        "best_iou_at_best_dice":
            history["val_iou"][best_dice_index],

        "final_dice":
            history["val_dice"][-1],

        "final_iou":
            history["val_iou"][-1],

        "final_loss":
            history["train_loss"][-1],

        "total_training_time":
            total_time,

        "average_epoch_time":
            sum(history["epoch_time"])
            / len(history["epoch_time"]),

        "average_throughput":
            sum(history["throughput"])
            / len(history["throughput"]),

        "max_peak_vram":
            max(history["peak_vram"]),
    }

    path = results_dir / "summary.csv"

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(summary.keys()),
        )

        writer.writeheader()
        writer.writerow(summary)


# ============================================================
# Training
# ============================================================

def train_model(
        model,
        device,
        dataset_name,
        precision,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        workers: int = 0,
        seed: int = 42,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):

    # --------------------------------------------------------
    # Precision configuration
    # --------------------------------------------------------

    if precision == "FP32":

        amp = False
        autocast_dtype = torch.float32

    elif precision == "FP16":

        amp = True
        autocast_dtype = torch.float16

    elif precision == "BF16":

        amp = True
        autocast_dtype = torch.bfloat16

    else:

        raise ValueError(
            f"Unsupported precision: {precision}"
        )

    # --------------------------------------------------------
    # Results directory
    # --------------------------------------------------------

    results_dir = (
        Path("./results")
        / dataset_name
        / precision
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Start each experiment with fresh CSV files.
    for filename in [
        "metrics.csv",
        "summary.csv",
    ]:

        path = results_dir / filename

        if path.exists():
            path.unlink()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_info = DATASETS[dataset_name]

    logging.info(
        f"Creating {dataset_name} dataset"
    )

    if dataset_name == "Carvana":

        dataset = CarvanaDataset(
            dataset_info["images"],
            dataset_info["masks"],
            scale=img_scale,
        )

    else:

        dataset = BasicDataset(
            dataset_info["images"],
            dataset_info["masks"],
            scale=img_scale,
            mask_suffix=dataset_info["mask_suffix"],
        )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    n_val = int(
        len(dataset) * val_percent
    )

    n_train = len(dataset) - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    logging.info(
        f"Total images:       {len(dataset)}"
    )

    logging.info(
        f"Training images:    {n_train}"
    )

    logging.info(
        f"Validation images:  {n_val}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    loader_generator = torch.Generator().manual_seed(seed)

    loader_args = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=loader_generator,
        **loader_args,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **loader_args,
    )


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging.info(
        f"""Starting training:
        Dataset:          {dataset_name}
        Precision:        {precision}
        Epochs:           {epochs}
        Batch size:       {batch_size}
        Learning rate:    {learning_rate}
        Validation:       {val_percent * 100:.1f}%
        Checkpoints:      {save_checkpoint}
        Device:           {device}
        Image scaling:    {img_scale}
        Mixed Precision:  {amp}
        Autocast dtype:   {autocast_dtype}
        DataLoader workers: {workers}
        """
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "max",
        patience=5,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = (
        nn.CrossEntropyLoss()
        if model.n_classes > 1
        else nn.BCEWithLogitsLoss()
    )

    # GradScaler is needed for FP16.
    # It is disabled for FP32 and BF16.
    grad_scaler = torch.cuda.amp.GradScaler(
        enabled=(precision == "FP16")
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = {
        "epoch": [],
        "train_loss": [],
        "val_dice": [],
        "val_iou": [],
        "epoch_time": [],
        "throughput": [],
        "peak_vram": [],
        "gpu_memory_allocated": [],
        "gpu_memory_reserved": [],
        "learning_rate": [],
    }

    total_start_time = time.perf_counter()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, epochs + 1):

        epoch_start_time = time.perf_counter()

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()

        model.train()

        epoch_loss = 0.0

        with tqdm(
            total=n_train,
            desc=(
                f"{dataset_name} {precision} | "
                f"Epoch {epoch}/{epochs}"
            ),
            unit="img",
        ) as pbar:

            for batch in train_loader:

                images = batch["image"]

                true_masks = batch["mask"]

                assert images.shape[1] == model.n_channels, \
                    f"Network has been defined with {model.n_channels} input channels, " \
                    f"but loaded images have {images.shape[1]} channels."

                images = images.to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )

                true_masks = true_masks.to(
                    device=device,
                    dtype=torch.long,
                )

                # ------------------------------------------------
                # Forward
                # ------------------------------------------------

                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=amp,
                ):

                    masks_pred = model(images)

                    if model.n_classes == 1:

                        probabilities = torch.sigmoid(
                            masks_pred.squeeze(1)
                        )

                        loss = criterion(
                            masks_pred.squeeze(1),
                            true_masks.float(),
                        )

                        loss += dice_loss_binary(
                            probabilities,
                            true_masks.float(),
                        )

                    else:

                        loss = criterion(
                            masks_pred,
                            true_masks,
                        )

                        loss += dice_loss_multiclass(
                            masks_pred,
                            true_masks,
                            model.n_classes,
                        )

                # ------------------------------------------------
                # Backward
                # ------------------------------------------------

                optimizer.zero_grad(
                    set_to_none=True
                )

                grad_scaler.scale(loss).backward()

                grad_scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clipping,
                )

                grad_scaler.step(optimizer)

                grad_scaler.update()

                # ------------------------------------------------
                # Logging
                # ------------------------------------------------

                pbar.update(
                    images.shape[0]
                )

                epoch_loss += loss.item()

                pbar.set_postfix(
                    loss=f"{loss.item():.6f}"
                )

        # --------------------------------------------------------
        # Epoch metrics
        # --------------------------------------------------------

        avg_train_loss = (
            epoch_loss / len(train_loader)
        )

        val_dice, val_iou = calculate_metrics(
            model,
            val_loader,
            device,
            precision,
        )

        scheduler.step(val_dice)

        epoch_time = (
            time.perf_counter()
            - epoch_start_time
        )

        throughput = (
            n_train / epoch_time
            if epoch_time > 0
            else 0.0
        )

        gpu_allocated, peak_vram = (
            get_gpu_memory()
        )

        if torch.cuda.is_available():

            gpu_reserved = (
                torch.cuda.max_memory_reserved()
                / (1024 ** 3)
            )

        else:

            gpu_reserved = 0.0

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # --------------------------------------------------------
        # Save history
        # --------------------------------------------------------

        history["epoch"].append(epoch)

        history["train_loss"].append(
            avg_train_loss
        )

        history["val_dice"].append(
            val_dice
        )

        history["val_iou"].append(
            val_iou
        )

        history["epoch_time"].append(
            epoch_time
        )

        history["throughput"].append(
            throughput
        )

        history["peak_vram"].append(
            peak_vram
        )

        history["gpu_memory_allocated"].append(
            gpu_allocated
        )

        history["gpu_memory_reserved"].append(
            gpu_reserved
        )

        history["learning_rate"].append(
            current_lr
        )

        # --------------------------------------------------------
        # Save CSV after every epoch
        # --------------------------------------------------------

        save_metrics_csv(
            history,
            results_dir,
        )


        # --------------------------------------------------------
        # Checkpoint
        # --------------------------------------------------------

        if save_checkpoint:

            checkpoint_dir = (
                Path("./checkpoints")
                / dataset_name
                / precision
            )

            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            state_dict = model.state_dict()

            state_dict["mask_values"] = (
                dataset.mask_values
            )

            checkpoint_path = (
                checkpoint_dir
                / f"checkpoint_epoch{epoch}.pth"
            )

            torch.save(
                state_dict,
                str(checkpoint_path),
            )

            logging.info(
                f"Checkpoint saved: "
                f"{checkpoint_path}"
            )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start_time
    )

    save_metrics_csv(
        history,
        results_dir,
    )

    save_summary(
        history,
        results_dir,
        dataset_name,
        precision,
        total_time,
    )

    save_plots(
        history,
        results_dir,
    )

    logging.info(
        f"Training completed in "
        f"{total_time / 60:.2f} minutes"
    )

    logging.info(
        f"Results saved to: "
        f"{results_dir}"
    )


# ============================================================
# Dice loss for binary segmentation
# ============================================================

def dice_loss_binary(
    probabilities,
    true_masks,
):

    smooth = 1e-6

    intersection = (
        probabilities * true_masks
    ).sum()

    denominator = (
        probabilities + true_masks
    ).sum()

    dice = (
        (2 * intersection + smooth)
        / (denominator + smooth)
    )

    return 1 - dice


# ============================================================
# Dice loss for multiclass segmentation
# ============================================================

def dice_loss_multiclass(
    logits,
    true_masks,
    num_classes,
):

    probabilities = F.softmax(
        logits,
        dim=1,
    )

    true_one_hot = F.one_hot(
        true_masks,
        num_classes,
    ).permute(
        0,
        3,
        1,
        2,
    ).float()

    smooth = 1e-6

    intersection = (
        probabilities * true_one_hot
    ).sum(
        dim=(0, 2, 3)
    )

    denominator = (
        probabilities + true_one_hot
    ).sum(
        dim=(0, 2, 3)
    )

    dice = (
        (2 * intersection + smooth)
        / (denominator + smooth)
    )

    return 1 - dice.mean()


# ============================================================
# Arguments
# ============================================================

def get_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train U-Net on Carvana or ISIC "
            "using FP32, FP16, or BF16"
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["Carvana", "ISIC"],
        default="Carvana",
        help="Dataset to use",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        metavar="E",
        type=int,
        default=5,
        help="Number of epochs",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=1,
        help="Batch size",
    )

    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=1e-5,
        dest="lr",
        help="Learning rate",
    )

    parser.add_argument(
        "--load",
        "-f",
        type=str,
        default=False,
        help="Load model from a .pth file",
    )

    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=0.5,
        help="Downscaling factor of the images",
    )

    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=10.0,
        help=(
            "Percent of the data used as "
            "validation (0-100)"
        ),
    )

    parser.add_argument(
        "--precision",
        type=str,
        choices=SUPPORTED_PRECISIONS,
        default="FP32",
        help=(
            "Floating-point precision: "
            "FP32, FP16, or BF16"
        ),
    )

    parser.add_argument(
        "--bilinear",
        action="store_true",
        default=False,
        help="Use bilinear upsampling",
    )

    parser.add_argument(
        "--classes",
        "-c",
        type=int,
        default=2,
        help="Number of classes",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=0,
        help="Number of DataLoader workers",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible experiments",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save model checkpoints",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    args = get_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    set_seed(args.seed)

    logging.info(
        f"Using device: {device}"
    )

    logging.info(
        f"Random seed: {args.seed}"
    )

    model = UNet(
        n_channels=3,
        n_classes=args.classes,
        bilinear=args.bilinear,
    )

    model = model.to(
        memory_format=torch.channels_last
    )

    logging.info(
        f"Total parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels\n"
        f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    if args.load:

        state_dict = torch.load(
            args.load,
            map_location=device,
        )

        if "mask_values" in state_dict:
            del state_dict["mask_values"]

        model.load_state_dict(
            state_dict
        )

        logging.info(
            f"Model loaded from {args.load}"
        )

    model.to(device=device)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_model(
        model=model,
        device=device,
        dataset_name=args.dataset,
        precision=args.precision,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_percent=args.val / 100,
        save_checkpoint=not args.no_save,
        img_scale=args.scale,
        workers=args.workers,
        seed=args.seed,
    )
