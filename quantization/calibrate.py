"""Activation calibration for simulated INT8 (Wu et al., 2020, Sec. 4.2).

For weights, we find the range directly from the weights themselves
(see quantize_model.py). Activations are different: they change with every
input image. So we estimate their clipping threshold amax (the largest value
we keep; bigger values are cut to amax) only once, using typical TRAINING
images, and we save the result to a JSON file. Later, test.py reads that file
(--calib-file) and chooses one method from it (--act-calib).

What we watch: the INPUT of every Conv2d and ConvTranspose2d in the FP32
model after BatchNorm folding. The weights are not quantized here, just like
in TensorRT and pytorch-quantization. Each layer gets one amax, because
activations are quantized per tensor.

We go over the same images two times:
    1. max pass        amax_max = max |x| for each layer.
    2. histogram pass  a histogram of |x| with `bins` bins on [0, amax_max].
From the histogram we compute:
    percentile-P  the smallest amax that keeps P% of the values below it
                  (not clipped).
    entropy       the amax that gives the smallest KL(P || Q). P is the
                  clipped reference distribution and Q is its quantized
                  version with 128 levels (TensorRT's entropy calibrator,
                  Migacz 2017).
    entropy-skip-zeros  the same as entropy, but we first remove the values
                  that are exactly zero from the histogram (our own
                  variant; see without_zeros()).

Usage (from the repository root):
    python -m quantization.calibrate --dataset Carvana \
        --checkpoint checkpoints/best_CarvanaBaseline.pth --samples 128
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.data_loading import BasicDataset
from data.splits import get_split_indices
from data.transforms import PairTransformDataset
from models.unet_modern import UNetModern
from quantization.bn_fold import fold_batchnorm
from quantization.int8 import QMAX
from quantization.quantize_model import quantizable_layers
from utilis.checkpoint import load_checkpoint

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
PERCENTILES = (99.9, 99.99, 99.999)
# torch.histc counts in FP32, which is exact only up to 2^24 per bin. So we
# give it chunks of at most this many elements and add up the counts in FP64.
HIST_CHUNK = 2 ** 24


# ---------------------------------------------------------------------------
# Choosing the threshold from a histogram of |x| (only NumPy, with unit tests).
# ---------------------------------------------------------------------------

def amax_from_percentile(hist: np.ndarray, edges: np.ndarray, percentile: float) -> float:
    """Return the upper edge of the bin where the CDF of |x| reaches `percentile`."""
    cdf = np.cumsum(hist) / hist.sum()
    index = int(np.searchsorted(cdf, percentile / 100.0))
    return float(edges[min(index + 1, len(edges) - 1)])


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    mask = p > 0
    if np.any(q[mask] == 0):
        return float("inf")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def amax_from_entropy(hist: np.ndarray, edges: np.ndarray, levels: int = QMAX + 1, start_bin: int = 128) -> tuple:
    """TensorRT entropy calibration.

    We try every possible threshold bin i and build two distributions:
        P = hist[:i], and we add all the counts after bin i (the clipped
            values) to P[i-1]. This is the data the layer would really get.
        Q = hist[:i] merged into `levels` groups (the INT8 levels; 128 for
            non-negative data on [-127, 127]). Then each group's count is
            spread back evenly over the bins that are not zero in P.
    If we clip too much, P[i-1] gets a lot of extra counts that Q does not
    have. If we clip too little, the 128 groups become too wide. We pick the
    best i, and amax is the upper edge of that bin.
    Returns (amax, divergences).
    """
    hist = hist.astype(np.float64)
    divergences = np.full(len(hist) + 1, np.inf)
    for i in range(max(start_bin, levels), len(hist) + 1):
        p = hist[:i].copy()
        p[-1] += hist[i:].sum()
        nonzero = p != 0
        group = np.minimum(np.arange(i) // (i // levels), levels - 1)
        group_sum = np.bincount(group, weights=hist[:i], minlength=levels)
        group_nonzero = np.bincount(group, weights=nonzero, minlength=levels)
        q = np.where(nonzero, (group_sum / np.maximum(group_nonzero, 1))[group], 0.0)
        divergences[i] = kl_divergence(p, q)
    best = int(np.argmin(divergences))
    return float(edges[best]), divergences


def without_zeros(hist: np.ndarray, zeros: int) -> np.ndarray:
    """Return the histogram with the exact zeros removed from bin 0.

    This is a variant of entropy calibration ("entropy-skip-zeros"). It is not
    part of TensorRT's original algorithm or of Wu et al. The reason: after a
    ReLU, 17-57% of the values are exactly 0. INT8 can store 0 with no error,
    but this big spike controls the KL divergence. Because of it, entropy
    clips 1-8% of the nonzero values. When we remove the zeros, the divergence
    only measures the values that really get a quantization error.
    """
    hist = hist.astype(np.float64).copy()
    hist[0] = max(hist[0] - zeros, 0.0)
    return hist


# ---------------------------------------------------------------------------
# Collecting statistics with forward pre-hooks.
# ---------------------------------------------------------------------------

class InputObserver:
    """Watch the input of one layer. It never changes the input."""

    def __init__(self, bins: int):
        self.bins = bins
        self.amax = 0.0
        self.min = float("inf")
        self.hist = None  # FP64 counts, kept on the CPU
        self.zeros = 0  # number of exact zeros (x == 0); they all fall in hist[0]
        self.mode = "max"

    def __call__(self, module, args):
        x = args[0].detach()
        if self.mode == "max":
            # max(|x|) = max(x.max(), -x.min()). This way we do not need memory for |x|.
            self.amax = max(self.amax, x.max().item(), -x.min().item())
            self.min = min(self.min, x.min().item())
        elif self.amax > 0:
            counts = torch.zeros(self.bins, dtype=torch.float64, device=x.device)
            non_negative = x.min().item() >= 0
            for chunk in x.reshape(-1).split(HIST_CHUNK):
                values = chunk if non_negative else chunk.abs()
                counts += torch.histc(values, bins=self.bins, min=0.0, max=self.amax).double()
                self.zeros += int((chunk == 0).sum().item())
            counts = counts.cpu()
            self.hist = counts if self.hist is None else self.hist + counts


def calibration_loader(args, train_args):
    dataset_path = DATASETS_DIR / f"data_{args.dataset}"
    base_dataset = BasicDataset(
        dataset_path / "imgs", dataset_path / "masks",
        image_dtype=np.uint8, mask_dtype=np.uint8,
    )
    seed = train_args.get("seed", 42)
    validation = train_args.get("validation", 10.0)
    train_indices, _ = get_split_indices(
        base_dataset=base_dataset, dataset_name=args.dataset,
        split_index=train_args.get("split"), validation=validation, seed=seed,
    )
    # A fixed random subset of the TRAINING images. We never use validation or test images.
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train_indices), generator=generator)[: args.samples]
    indices = [train_indices[i] for i in order.tolist()]
    loader = DataLoader(
        Subset(PairTransformDataset(base_dataset, transform=None), indices),
        batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    split = {"seed": seed, "validation": validation, "split": train_args.get("split")}
    return loader, [base_dataset.ids[i] for i in indices], split


@torch.inference_mode()
def run_pass(model, loader, device):
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        images = images.float().div_(torch.iinfo(batch["image"].dtype).max)
        model(images)


def main():
    parser = argparse.ArgumentParser(description="Calibrate INT8 activation ranges on training samples.")
    parser.add_argument("--dataset", required=True, choices=["Carvana", "ISBI", "ISIC"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--samples", type=int, default=128, help="Training images used for calibration.")
    parser.add_argument("--bins", type=int, default=2048, help="Histogram bins (TensorRT uses 2048).")
    parser.add_argument("--seed", type=int, default=42, help="Seed that picks the calibration images.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None, help="JSON path (default: calibration/<checkpoint>_n<samples>.json).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False  # we want real FP32 statistics
    torch.backends.cuda.matmul.allow_tf32 = False

    checkpoint = load_checkpoint(args.checkpoint, device)
    train_args = checkpoint.get("args", {})
    model = UNetModern(
        n_classes=1, n_channels=1 if args.dataset == "ISBI" else 3,
        base_channels=args.base_channels, batch_norm=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = fold_batchnorm(model.eval())

    loader, sample_ids, split = calibration_loader(args, train_args)
    observers = {name: InputObserver(args.bins) for name in quantizable_layers(model)}
    handles = [
        module.register_forward_pre_hook(observers[name])
        for name, module in quantizable_layers(model).items()
    ]

    start = time.perf_counter()
    for mode in ("max", "hist"):
        for observer in observers.values():
            observer.mode = mode
        run_pass(model, loader, device)
        print(f"{mode} pass done ({time.perf_counter() - start:.0f} s)")
    for handle in handles:
        handle.remove()

    layers = {}
    print(f"\n{'layer':20s} {'min':>8s} {'max':>9s} " + " ".join(f"{'p' + str(p):>9s}" for p in PERCENTILES) + f" {'entropy':>9s} {'ent-nz':>9s} {'zeros':>6s}")
    for name, observer in observers.items():
        edges = np.linspace(0.0, observer.amax, args.bins + 1)
        hist = observer.hist.numpy()
        amax = {"max": observer.amax}
        for p in PERCENTILES:
            amax[f"percentile-{p:g}"] = amax_from_percentile(hist, edges, p)
        amax["entropy"], _ = amax_from_entropy(hist, edges)
        amax["entropy-skip-zeros"], _ = amax_from_entropy(without_zeros(hist, observer.zeros), edges)
        layers[name] = {
            "input_min": observer.min,
            "zero_fraction": observer.zeros / float(hist.sum()),
            "zeros": observer.zeros,
            "amax": amax,
            "histogram": hist.tolist(),
        }
        print(f"{name:20s} {observer.min:8.3f} {observer.amax:9.3f} "
              + " ".join(f"{amax[f'percentile-{p:g}']:9.3f}" for p in PERCENTILES)
              + f" {amax['entropy']:9.3f} {amax['entropy-skip-zeros']:9.3f}"
              + f" {layers[name]['zero_fraction']:6.2f}")

    output = args.output or ROOT_DIR / "calibration" / f"{args.checkpoint.stem}_n{args.samples}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "checkpoint": str(args.checkpoint),
            "base_channels": args.base_channels,
            "source": "train split",
            "split": split,
            "samples": len(sample_ids),
            "sample_seed": args.seed,
            "sample_ids": sample_ids,
            "bins": args.bins,
            "model": "BN-folded FP32, weights not quantized",
            "methods": ["max", *(f"percentile-{p:g}" for p in PERCENTILES), "entropy", "entropy-skip-zeros"],
            "layers": layers,
        }, file)
    print(f"\nCalibration saved to: {output}")


if __name__ == "__main__":
    main()
