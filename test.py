import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.data_loading import BasicDataset
from data.splits import get_split_indices
from data.transforms import PairTransformDataset
from models.unet_classic import UNetClassic
from models.unet_modern import UNetModern
from PIL import Image

from quantization.bn_fold import fold_batchnorm
from quantization.env_report import environment_info
from quantization.quantize_model import prepare_int8_model
from utilis.checkpoint import load_checkpoint
from utilis.loss import DiceBCELoss
from utilis.metrics import append_summary_row, dice_score

cudnn.benchmark = True

ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "datasets"

CAST_DTYPES = {"FP16": torch.float16, "BF16": torch.bfloat16}
LOSS_COMPONENTS = ("bce_loss", "dice_loss", "bce_contribution", "dice_contribution")
PER_SAMPLE_FIELDS = (
    "sample_id",
    "model",
    "total_params",
    "loss",
    *LOSS_COMPONENTS,
    "dice",
)


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
    parser.add_argument("--base-channels",type=int,default=64,help="Base channel count used by the selected U-Net (default: 64).",)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers",type=int,default=min(4, os.cpu_count() or 1),help="DataLoader worker processes (default: min(4, CPU count)).",)
    parser.add_argument("--precision",type=str.upper,choices=["FP32", "FP16", "BF16"],default="FP32",help="Autocast (mixed precision) for inference; weights stay in their original dtype.",)
    parser.add_argument("--cast",type=str.upper,choices=list(CAST_DTYPES),default=None,help=("Convert the whole model (weights and BatchNorm buffers) and its inputs to this dtype. ""If omitted, the model keeps the checkpoint's dtype (usually FP32)."),)
    parser.add_argument("--tf32",type=str.lower,choices=["on", "off"],default="off",help=("Allow TF32 (10-bit mantissa inputs) in FP32 convolutions and matmuls on Ampere+ GPUs. ""Default off: FP32 means true FP32. PyTorch's own default for cuDNN convolutions is on."),)
    parser.add_argument("--fold-bn",action="store_true",help="Fold BatchNorm into the preceding convolutions (same function; sanity check for quantization).",)
    parser.add_argument("--quant",type=str.upper,choices=["NONE", "INT8"],default="NONE",help=("Simulated (fake) quantization, run in FP32. INT8: BN folding + symmetric per-channel ""max-calibrated INT8 weights (W8A32). Measures accuracy, not INT8 speed."),)
    parser.add_argument("--act-calib",type=str.lower,default=None,help=("Also quantize the input of every quantized layer (W8A8), per-tensor, with this method from --calib-file: ""max, percentile-99.9, percentile-99.99, percentile-99.999 or entropy. Omit for weight-only (W8A32)."),)
    parser.add_argument("--calib-file",type=Path,default=None,help="JSON written by quantization/calibrate.py for this checkpoint.",)
    parser.add_argument("--fp-layers",type=lambda value: [name for name in value.split(",") if name],default=[],help="Comma-separated Conv/ConvTranspose layer names kept in floating point (e.g. enc1.block.0,segment).",)
    parser.add_argument("--backend",type=str.lower,choices=["pytorch", "tensorrt"],default="pytorch",help=("pytorch: run the model in PyTorch (FP32, --cast, --precision, fake --quant). ""tensorrt: run a TensorRT engine built from the ONNX export (strongly typed, batch 1)."),)
    parser.add_argument("--onnx",type=Path,default=None,help="ONNX file for --backend tensorrt (default: exports/<checkpoint>_fp32_bnfolded.onnx from quantization/export_onnx.py).",)
    parser.add_argument("--rebuild-engine",action="store_true",help="Rebuild the TensorRT engine even if a cached one matches.",)
    parser.add_argument("--warmup-batches",type=int,default=3,help="Batches excluded from timing (they are still included in the metrics).",)
    parser.add_argument("--output",type=Path,default=Path("test_results"),help="Directory for evaluation results.",)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.base_channels < 1:
        parser.error("--base-channels must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.validation is not None and not 0 < args.validation < 100:
        parser.error("--validation must be in (0, 100)")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches cannot be negative")
    if args.cast is not None and args.precision != "FP32":
        parser.error("--cast and --precision (autocast) are separate experiments; use only one.")
    if args.quant != "NONE" and (args.cast is not None or args.precision != "FP32"):
        parser.error("--quant is simulated in FP32; do not combine it with --cast or --precision.")
    if args.fp_layers and args.quant == "NONE":
        parser.error("--fp-layers only applies with --quant.")
    if args.act_calib is not None and (args.quant == "NONE" or args.calib_file is None):
        parser.error("--act-calib needs --quant INT8 and --calib-file.")
    if args.calib_file is not None and args.act_calib is None:
        parser.error("--calib-file needs --act-calib.")
    if args.backend == "tensorrt":
        if args.model != "modern":
            parser.error("--backend tensorrt is implemented for --model modern only.")
        if args.cast is not None or args.precision != "FP32" or args.fold_bn:
            parser.error("--backend tensorrt: the precision comes from the ONNX graph (FP32, or INT8 Q/DQ with --quant); "
                         "do not use --cast, --precision or --fold-bn (BN is always folded in the export).")
        if args.batch_size != 1:
            parser.error("--backend tensorrt engines are built for --batch-size 1 (images have different sizes).")
        if not torch.cuda.is_available():
            parser.error("--backend tensorrt needs a CUDA GPU.")
    elif args.onnx is not None or args.rebuild_engine:
        parser.error("--onnx and --rebuild-engine only apply with --backend tensorrt.")

    return args


def describe_precision(args, quantization_name, use_amp, model_format, trt_summary=None):
    """The three properties every result must state separately:
    representation  which values the numbers are restricted to;
    storage         how weights are actually held in memory;
    execution       what the kernels actually computed in.
    """
    if args.backend == "tensorrt":
        # Storage and execution come from the built engine (kernel names and
        # tensor dtypes), not from what the ONNX graph asked for.
        counts = lambda field: ", ".join(f"{count}x {kind}" for kind, count in trt_summary[field].items())
        tf32 = f", {trt_summary['tf32_tactics']} TF32 tactics" if trt_summary["tf32_tactics"] else ""
        if args.quant == "INT8":
            what = "weights+activations" if quantization_name == "INT8-W8A8" else "weights"
            representation = f"int8 grid ({what})"
        else:
            representation = "fp32"
        storage = f"TensorRT engine, conv weights [{counts('weights')}]"
        execution = f"tensorrt [{counts('compute_layers')}; accumulators {counts('accumulators')}{tf32}]"
        return representation, storage, execution
    tf32 = " (TF32 convs)" if args.tf32 == "on" else ""
    if args.quant == "INT8":
        what = "weights+activations" if quantization_name == "INT8-W8A8" else "weights"
        return f"int8 grid ({what})", "fp32", f"pytorch fp32{tf32} (simulated int8)"
    if args.cast is not None:
        return model_format.lower(), model_format.lower(), f"pytorch {model_format.lower()}"
    if use_amp:
        return f"mixed (autocast {args.precision.lower()})", "fp32", f"pytorch autocast {args.precision.lower()}"
    return "fp32", "fp32", f"pytorch fp32{tf32}"


def image_sizes(base_dataset, sample_ids):
    """(H, W) of each image, read from the file headers (no decoding)."""
    sizes = []
    for sample_id in sample_ids:
        with Image.open(base_dataset.image_files[sample_id][0]) as image:
            width, height = image.size
        sizes.append((height, width))
    return sizes


def main():
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TF32 only changes FP32 convolutions/matmuls; FP16/BF16 kernels are
    # unaffected. Set both flags explicitly so the run does not depend on
    # PyTorch's defaults (cuDNN: on, matmul: off).
    use_tf32 = args.tf32 == "on"
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.backends.cuda.matmul.allow_tf32 = use_tf32

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # With TensorRT the PyTorch model is only needed for its parameter count:
    # keep the checkpoint and the model on the CPU so they take no GPU memory.
    model_device = torch.device("cpu") if args.backend == "tensorrt" else device

    checkpoint = load_checkpoint(args.checkpoint, model_device)

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

    dataset = Subset(PairTransformDataset(base_dataset, transform=None),sample_indices,)

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
            base_channels=args.base_channels,
        ).to(model_device)
    else:
        model = UNetModern(
            n_classes=1,
            n_channels=input_channels,
            base_channels=args.base_channels,
            batch_norm=True,
        ).to(model_device)

    # Load the FP32 weights first, then convert the model if requested.
    model.load_state_dict(state_dict)
    total_params = sum(parameter.numel() for parameter in model.parameters())

    if args.cast is not None:
        model = model.to(device=device, dtype=CAST_DTYPES[args.cast])

    # Quantization works on an eval-mode copy; the checkpoint is never changed.
    act_amax = None
    calib_samples = None
    if args.act_calib is not None:
        with args.calib_file.open("r", encoding="utf-8") as file:
            calibration = json.load(file)
        if args.act_calib not in calibration["methods"]:
            raise ValueError(f"--act-calib must be one of {calibration['methods']}")
        if calibration["dataset"] != args.dataset or Path(calibration["checkpoint"]).name != args.checkpoint.name:
            raise ValueError(
                f"{args.calib_file} was calibrated for {calibration['dataset']} / "
                f"{calibration['checkpoint']}, not {args.dataset} / {args.checkpoint}."
            )
        act_amax = {name: layer["amax"][args.act_calib] for name, layer in calibration["layers"].items()}
        calib_samples = calibration["samples"]

    if args.quant == "INT8":
        model.eval()
        model, weight_stats = prepare_int8_model(model, fp_layers=args.fp_layers, act_amax=act_amax)
        quantization_name = "INT8-W8A8" if act_amax is not None else "INT8-W8A32"
        worst = min(weight_stats, key=lambda row: row["sqnr_db"])
        print(
            f"INT8 weights: {len(weight_stats)} layers quantized "
            f"(per-channel, max), {len(args.fp_layers)} kept in FP. "
            f"Lowest weight SQNR: {worst['sqnr_db']:.1f} dB ({worst['layer']})"
        )
        if act_amax is not None:
            print(
                f"INT8 activations: per-tensor, {args.act_calib} calibration "
                f"({calib_samples} training samples, {args.calib_file})"
            )
        if args.tf32 == "on":
            print("Warning: TF32 adds its own rounding on top of the simulated INT8 error.")
    elif args.fold_bn:
        model.eval()
        model = fold_batchnorm(model)
        quantization_name = "BN-folded"
    else:
        quantization_name = "none"

    model.eval()
    model_dtype = next(model.parameters()).dtype
    weights_mib = sum(
        tensor.numel() * tensor.element_size()
        for tensor in list(model.parameters()) + list(model.buffers())
    ) / (1024 ** 2)

    # The callable that maps images to logits: the PyTorch model, or a
    # TensorRT engine built from the ONNX export of the same checkpoint.
    forward = model
    engine_mib = None
    trt_summary = None
    if args.backend == "tensorrt":
        from quantization.trt_engine import TensorRTModel, get_engine, profile_from_sizes

        onnx_path = args.onnx or ROOT_DIR / "exports" / f"{args.checkpoint.stem}_fp32_bnfolded.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"{onnx_path} not found. Export it first:\n  python -m quantization.export_onnx "
                f"--dataset {args.dataset} --checkpoint {args.checkpoint}"
            )
        export_info = json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
        if export_info["dataset"] != args.dataset or Path(export_info["checkpoint"]).name != args.checkpoint.name:
            raise ValueError(f"{onnx_path} was exported from {export_info['checkpoint']} ({export_info['dataset']}).")
        if args.quant == "INT8":
            # Same quantization as the fake-quant reference, written into the
            # graph as INT8 weights + Q/DQ nodes (cached next to the FP32 ONNX).
            from quantization.qdq_onnx import build_int8_onnx

            onnx_path = build_int8_onnx(onnx_path, args.calib_file, args.act_calib, args.fp_layers)
            print(f"INT8 ONNX: {onnx_path.name}")
        # The engine accepts exactly the image sizes of the evaluated samples.
        profile = profile_from_sizes(image_sizes(base_dataset, sample_ids))
        engine_path = get_engine(onnx_path, profile, tf32=use_tf32, n_channels=input_channels, rebuild=args.rebuild_engine)
        forward = TensorRTModel(engine_path)
        trt_summary = forward.layer_summary()
        engine_mib = forward.engine_bytes / (1024 ** 2)
        weights_mib = None  # the weights live inside the engine (engine_mib)
        print(f"TensorRT engine: {engine_path.name} ({engine_mib:.1f} MiB)")
        print(f"Engine layers: {trt_summary['compute_layers']} | TF32 tactics: {trt_summary['tf32_tactics']}")
        print(f"Per-layer details: {engine_path.with_suffix('.layers.json')}")

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
        for batch_index, batch in enumerate(
            tqdm(
                dataloader,
                desc=f"Evaluation {args.dataset}",
                unit="batch",
                dynamic_ncols=True,
            )
        ):
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
                logits = forward(images)

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
    latency_ms = (timed_seconds / timed_samples * 1000 if timed_samples else float("nan"))
    throughput = (timed_samples / timed_seconds if timed_seconds > 0 else float("nan"))

    model_format = {
        torch.float32: "FP32",
        torch.float16: "FP16",
        torch.bfloat16: "BF16",
    }.get(model_dtype, str(model_dtype))
    precision_name = args.cast or args.precision
    representation, storage, execution = describe_precision(args, quantization_name, use_amp, model_format, trt_summary)
    environment = environment_info()

    print("\nEvaluation complete")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {split_name} ({total_samples} samples)")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backend: {args.backend}")
    if weights_mib is not None:
        print(f"Model dtype: {model_format} ({weights_mib:.1f} MiB of weights)")
    print(f"Representation: {representation} | Storage: {storage}")
    print(f"Execution: {execution}")
    print(f"Autocast precision: {args.precision if use_amp else 'off'}")
    print(f"TF32: {args.tf32}")
    print(f"Quantization: {quantization_name}"
          + (f" (FP layers: {','.join(args.fp_layers)})" if args.fp_layers else ""))
    print(f"Mean loss: {mean_loss:.6f}")
    print(f"Mean Dice: {mean_dice:.6f}")
    print(
        f"Forward latency: {latency_ms:.2f} ms/image | "
        f"Throughput: {throughput:.2f} images/s "
        f"({timed_samples} images, {warmup_batches} warm-up batches excluded)"
    )

    peak_allocated = peak_reserved = None
    if device.type == "cuda":
        peak_allocated = (torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        peak_reserved = (torch.cuda.max_memory_reserved(device) / (1024 ** 2))
        print(f"Peak GPU memory allocated: {peak_allocated:.2f} MiB")
        print(f"Peak GPU memory reserved: {peak_reserved:.2f} MiB")

    # 1. One row per run in a shared summary, to compare runs side by side.
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.csv"
    append_summary_row(summary_path, {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "split": split_name,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": (checkpoint.get("epoch") if isinstance(checkpoint, dict) else None),
        "total_params": total_params,
        "model_dtype": model_format,
        "autocast": args.precision if use_amp else "off",
        "tf32": args.tf32,
        "quantization": quantization_name,
        "act_calibration": args.act_calib or "",
        "calib_samples": calib_samples or "",
        "fp_layers": ",".join(args.fp_layers),
        "backend": args.backend,
        "representation": representation,
        "storage": storage,
        "execution": execution,
        "samples": total_samples,
        "mean_loss": mean_loss,
        **{f"mean_{name}": sample_mean(name) for name in LOSS_COMPONENTS},
        "mean_dice": mean_dice,
        "latency_ms_per_image": latency_ms,
        "throughput_images_per_sec": throughput,
        "weights_mib": weights_mib,
        "engine_mib": engine_mib,
        "peak_gpu_allocated_mib": peak_allocated,
        "peak_gpu_reserved_mib": peak_reserved,
        "gpu": environment["gpu"],
        "driver": environment["driver"],
        "torch_version": environment["torch"],
        "tensorrt_version": environment["tensorrt"] if args.backend == "tensorrt" else "",
    })

    # 2. Per-sample results of this run, for paired comparisons between
    # precisions on exactly the same images (overwritten on reruns).
    per_sample_path = args.output / "per_sample" / (
        f"{args.checkpoint.stem}_{args.dataset}_{split_name}_{args.model}_"
        f"{precision_name}{'_cast' if args.cast else ''}"
        f"{'_trt' if args.backend == 'tensorrt' else ''}"
        f"{'_tf32' if use_tf32 else ''}"
        f"{'' if quantization_name == 'none' else '_' + quantization_name}"
        f"{'_' + args.act_calib if args.act_calib else ''}"
        f"{'_fp-' + '+'.join(args.fp_layers) if args.fp_layers else ''}.csv"
    )
    per_sample_path.parent.mkdir(parents=True, exist_ok=True)
    with per_sample_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PER_SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "model": args.model,
                "total_params": total_params,
            }
            for row in per_sample_rows
        )

    print(f"Summary appended to: {summary_path}")
    print(f"Per-sample results saved to: {per_sample_path}")

if __name__ == "__main__":
    main()
