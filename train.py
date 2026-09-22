import argparse
import csv
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader, Sampler, random_split
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
    "ISIC_processed": {
        "images": Path("./data_ISIC_processed/imgs/"),
        "masks": Path("./data_ISIC_processed/masks/"),
        "mask_suffix": "_Segmentation",
    },
}

SUPPORTED_PRECISIONS = ("FP32", "FP16", "BF16")


def ensure_processed_dataset():
    """
    Ensure that data_ISIC_processed exists.

    If the processed dataset is missing or empty, automatically run
    preprocess_isic.py from the project root.
    """

    processed_dir = Path("./data_ISIC_processed")
    images_dir = processed_dir / "imgs"
    masks_dir = processed_dir / "masks"
    preprocess_script = Path("./preprocess_isic.py")

    images_exist = (
        images_dir.exists()
        and any(images_dir.iterdir())
    )

    masks_exist = (
        masks_dir.exists()
        and any(masks_dir.iterdir())
    )

    if images_exist and masks_exist:
        logging.info(
            "Found existing data_ISIC_processed. "
            "Skipping preprocessing."
        )
        return

    logging.info(
        "data_ISIC_processed was not found or appears incomplete."
    )

    if not preprocess_script.exists():
        raise FileNotFoundError(
            "data_ISIC_processed is missing and "
            "preprocess_isic.py was not found."
        )

    logging.info(
        "Running preprocess_isic.py..."
    )

    subprocess.run(
        [sys.executable, str(preprocess_script)],
        check=True,
    )

    if not (
        images_dir.exists()
        and masks_dir.exists()
        and any(images_dir.iterdir())
        and any(masks_dir.iterdir())
    ):
        raise RuntimeError(
            "preprocess_isic.py finished, but "
            "data_ISIC_processed still appears to be incomplete."
        )

    logging.info(
        "ISIC preprocessing completed successfully."
    )



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
# Distributed training helpers
# ============================================================

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    """
    Initialize Distributed Data Parallel when launched with
    torch.distributed.launch.

    Returns:
        local_rank, device
    """

    if "LOCAL_RANK" not in os.environ:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        return 0, device

    local_rank = int(os.environ["LOCAL_RANK"])

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DDP was requested, but CUDA is not available."
        )

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl"
    )

    device = torch.device(
        "cuda",
        local_rank,
    )

    return local_rank, device


def cleanup_distributed():
    """Destroy the distributed process group."""

    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()

class DistributedEvalSampler(Sampler):
    """Distributed sampler for evaluation without duplicating samples."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = num_replicas if num_replicas is not None else get_world_size()
        self.rank = rank if rank is not None else get_rank()
        self.indices = list(range(self.rank, len(dataset), self.num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


# ============================================================
# Metrics
# ============================================================
def calculate_metrics(
    model,
    loader,
    device,
    precision,
):
    """
    Calculate Dice and IoU on the validation set.

    In distributed mode, metrics from all processes are
    synchronized using all_reduce.
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
        raise ValueError(
            f"Unsupported precision: {precision}"
        )

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

            pred_masks = masks_pred.argmax(dim=1)

            valid = true_masks != 255

            pred_foreground = (
                (pred_masks == 1)
                & valid
            )

            true_foreground = (
                (true_masks == 1)
                & valid
            )

            intersection = (
                pred_foreground
                & true_foreground
            ).sum().float()

            pred_area = (
                pred_foreground.sum().float()
            )

            true_area = (
                true_foreground.sum().float()
            )

            union = (
                pred_area
                + true_area
                - intersection
            )

            dice = (
                (2.0 * intersection + 1e-8)
                /
                (pred_area + true_area + 1e-8)
            )

            iou = (
                (intersection + 1e-8)
                /
                (union + 1e-8)
            )

            total_dice += dice.item()
            total_iou += iou.item()
            num_batches += 1

    # --------------------------------------------------------
    # Synchronize validation metrics across GPUs
    # --------------------------------------------------------

    if is_distributed():

        metrics = torch.tensor(
            [
                total_dice,
                total_iou,
                float(num_batches),
            ],
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(
            metrics,
            op=dist.ReduceOp.SUM,
        )

        total_dice = metrics[0].item()
        total_iou = metrics[1].item()
        num_batches = int(
            metrics[2].item()
        )

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
    experiment_name="default",
):

    best_dice_index = max(
        range(len(history["val_dice"])),
        key=lambda i: history["val_dice"][i],
    )

    summary = {
        "experiment": experiment_name,
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
        experiment_name: str = "default",
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1.0,
        workers: int = 0,
        seed: int = 42,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):

    # DDP wraps the real U-Net in model.module. Keep a reference to the
    # underlying model for architecture attributes and checkpoint saving.
    base_model = model.module if is_distributed() else model

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
        / experiment_name
        / dataset_name
        / precision
    )

    if is_main_process():
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

    if is_distributed():
        dist.barrier()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_info = DATASETS[dataset_name]

    # ISIC_processed has already been resized and padded to
    # 1024x768 by preprocess_isic.py. Do not resize it again.
    effective_img_scale = (
        1.0
        if dataset_name == "ISIC_processed"
        else img_scale
    )

    logging.info(
        f"Creating {dataset_name} dataset"
    )

    if dataset_name == "Carvana":

        dataset = CarvanaDataset(
            dataset_info["images"],
            dataset_info["masks"],
            scale=effective_img_scale,
        )

    else:

        dataset = BasicDataset(
            dataset_info["images"],
            dataset_info["masks"],
            scale=effective_img_scale,
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

    if is_distributed():

        train_sampler = DistributedSampler(
            train_set,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
        )

        val_sampler = DistributedEvalSampler(
            val_set,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )

        train_loader = DataLoader(
            train_set,
            shuffle=False,
            sampler=train_sampler,
            **loader_args,
        )

        val_loader = DataLoader(
            val_set,
            shuffle=False,
            sampler=val_sampler,
            **loader_args,
        )

    else:

        train_sampler = None

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

    if is_main_process():
        logging.info(
            f"""Starting training:
        Experiment:        {experiment_name}
        Dataset:           {dataset_name}
        Precision:         {precision}
        Epochs:            {epochs}
        Batch size/GPU:    {batch_size}
        Global batch size: {batch_size * get_world_size()}
        Learning rate:     {learning_rate}
        Validation:        {val_percent * 100:.1f}%
        Checkpoints:       {save_checkpoint}
        Device:            {device}
        Image scaling:     {effective_img_scale}
        Mixed Precision:   {amp}
        Autocast dtype:    {autocast_dtype}
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
        nn.CrossEntropyLoss(ignore_index=255)
        if base_model.n_classes > 1
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

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.perf_counter()

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()

        model.train()

        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        progress_total = (
            len(train_sampler) * get_world_size()
            if train_sampler is not None
            else n_train
        )

        progress = tqdm(
            total=progress_total,
            desc=(
                f"{dataset_name} {precision} | "
                f"Epoch {epoch}/{epochs}"
            ),
            unit="img",
            disable=not is_main_process(),
        )

        with progress as pbar:

            for batch in train_loader:

                images = batch["image"]

                true_masks = batch["mask"]

                assert images.shape[1] == base_model.n_channels, \
                    f"Network has been defined with {base_model.n_channels} input channels, " \
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

                    if base_model.n_classes == 1:

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
                            base_model.n_classes,
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

                local_batch_size = images.shape[0]

                pbar.update(
                    local_batch_size * get_world_size()
                    if is_distributed()
                    else local_batch_size
                )

                epoch_loss_sum += loss.item() * local_batch_size
                epoch_sample_count += local_batch_size

                if is_main_process():
                    pbar.set_postfix(
                        loss=f"{loss.item():.6f}"
                    )

        # --------------------------------------------------------
        # Epoch metrics
        # --------------------------------------------------------

        loss_stats = torch.tensor(
            [epoch_loss_sum, float(epoch_sample_count)],
            dtype=torch.float64,
            device=device,
        )

        if is_distributed():
            dist.all_reduce(
                loss_stats,
                op=dist.ReduceOp.SUM,
            )

        global_loss_sum = loss_stats[0].item()
        global_sample_count = loss_stats[1].item()

        avg_train_loss = (
            global_loss_sum / global_sample_count
            if global_sample_count > 0
            else 0.0
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
            global_sample_count / epoch_time
            if epoch_time > 0
            else 0.0
        )

        gpu_allocated, peak_vram = get_gpu_memory()

        if torch.cuda.is_available():
            gpu_reserved = (
                torch.cuda.max_memory_reserved()
                / (1024 ** 3)
            )
        else:
            gpu_reserved = 0.0

        if is_distributed():
            memory_stats = torch.tensor(
                [gpu_allocated, peak_vram, gpu_reserved],
                dtype=torch.float64,
                device=device,
            )

            dist.all_reduce(
                memory_stats,
                op=dist.ReduceOp.MAX,
            )

            gpu_allocated = memory_stats[0].item()
            peak_vram = memory_stats[1].item()
            gpu_reserved = memory_stats[2].item()

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

        if is_main_process():
            save_metrics_csv(
                history,
                results_dir,
            )

        # --------------------------------------------------------
        # Checkpoint
        # --------------------------------------------------------

        if save_checkpoint and is_main_process():

            checkpoint_dir = (
                Path("./checkpoints")
                / experiment_name
                / dataset_name
                / precision
            )

            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            state_dict = base_model.state_dict()

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

    if is_main_process():
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
            experiment_name,
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

    # Processed ISIC masks use raw value 254 for artificial padding.
    # BasicDataset converts raw 254 to tensor value 255 (ignore_index).
    valid = true_masks != 255

    safe_masks = true_masks.clamp(
        min=0,
        max=num_classes - 1,
    )

    true_one_hot = F.one_hot(
        safe_masks,
        num_classes,
    ).permute(
        0,
        3,
        1,
        2,
    ).float()

    valid = valid.unsqueeze(1)

    probabilities = probabilities * valid
    true_one_hot = true_one_hot * valid

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
            "Train U-Net on Carvana, ISIC, or ISIC_processed "
            "using FP32, FP16, or BF16"
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["Carvana", "ISIC", "ISIC_processed"],
        default="Carvana",
        help="Dataset to use",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help=(
            "Experiment name used to separate results and checkpoints. "
            "Example: carvana_vs_isic_processed"
        ),
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
        default=1.0,
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

    # --------------------------------------------------------
    # Distributed setup
    # --------------------------------------------------------

    local_rank, device = setup_distributed()

    rank = get_rank()
    world_size = get_world_size()

    # --------------------------------------------------------
    # Dataset preprocessing
    # --------------------------------------------------------

    if args.dataset == "ISIC_processed":

        if is_main_process():
            ensure_processed_dataset()

        if is_distributed():
            dist.barrier()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        args.seed + rank
    )

    if is_main_process():

        logging.info(
            f"Using device: {device}"
        )

        logging.info(
            f"Experiment: {args.experiment}"
        )

        logging.info(
            f"Random seed: {args.seed}"
        )

        logging.info(
            f"Distributed training: "
            f"{is_distributed()}"
        )

        logging.info(
            f"World size: {world_size}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet(
        n_channels=3,
        n_classes=args.classes,
        bilinear=args.bilinear,
    )

    model = model.to(
        memory_format=torch.channels_last
    )

    if is_main_process():

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

        # Allow loading checkpoints produced by an older DDP version
        # that saved keys with the "module." prefix.
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }

        model.load_state_dict(
            state_dict
        )

        if is_main_process():

            logging.info(
                f"Model loaded from {args.load}"
            )

    model.to(device=device)

    # --------------------------------------------------------
    # DDP
    # --------------------------------------------------------

    if is_distributed():

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    try:

        train_model(
            model=model,
            device=device,
            dataset_name=args.dataset,
            precision=args.precision,
            experiment_name=args.experiment,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.val / 100,
            save_checkpoint=not args.no_save,
            img_scale=args.scale,
            workers=args.workers,
            seed=args.seed,
        )

    finally:

        cleanup_distributed()