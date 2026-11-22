import argparse
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
from quantization import modes as quant_modes
from quantization.bn_fold import fold_batchnorm
from quantization.env_report import environment_info
from quantization.quantize_model import prepare_quantized_model
from utilis.checkpoint import load_checkpoint
from utilis.evaluation import describe_precision, image_sizes, split_arg
from utilis.loss import DiceBCELoss
from utilis.metrics import dice_score
from utilis.reporting import (
    LOSS_COMPONENTS,
    append_summary_row,
    peak_gpu_memory_mib,
    per_sample_filename,
    sample_means,
    summary_row,
    write_per_sample_csv,
)

cudnn.benchmark = True

ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "datasets"

CAST_DTYPES = {"FP16": torch.float16, "BF16": torch.bfloat16}


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
    quant_modes.add_arguments(parser)  # --quant, --block-size, --act-*, --calib-file, --fp-layers
    parser.add_argument("--backend",type=str.lower,choices=["pytorch", "tensorrt"],default="pytorch",help=("pytorch: run the model in PyTorch (FP32, --cast, --precision, fake --quant). ""tensorrt: run a TensorRT engine built from the ONNX export (strongly typed, batch 1)."),)
    parser.add_argument("--onnx",type=Path,default=None,help="ONNX file for --backend tensorrt (default: exports/<checkpoint>_fp32_bnfolded.onnx from quantization/export_onnx.py).",)
    parser.add_argument("--upsample",choices=["convtranspose", "depth2space"],default=None,help=("TensorRT export of the ConvTranspose upsamplers: convtranspose, or depth2space = exact 1x1 Conv + DepthToSpace ""(default for --quant FP8: TensorRT has no FP8 deconvolution kernels; default convtranspose otherwise)."),)
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
    quant_modes.validate(args, parser)
    if args.backend == "tensorrt":
        if args.model != "modern":
            parser.error("--backend tensorrt is implemented for --model modern only.")
        if args.cast is not None or args.precision != "FP32" or args.fold_bn:
            parser.error("--backend tensorrt: the precision comes from the ONNX graph (FP32, or Q/DQ with --quant); "
                         "do not use --cast, --precision or --fold-bn (BN is always folded in the export).")
        if args.batch_size != 1:
            parser.error("--backend tensorrt engines are built for --batch-size 1 (images have different sizes).")
        if not torch.cuda.is_available():
            parser.error("--backend tensorrt needs a CUDA GPU.")
        if args.upsample is None:
            args.upsample = "depth2space" if args.quant == "FP8" else "convtranspose"
    elif args.onnx is not None or args.rebuild_engine or args.upsample is not None:
        parser.error("--onnx, --upsample and --rebuild-engine only apply with --backend tensorrt.")

    return args


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

    # With TensorRT, we only need the PyTorch model to count its parameters.
    # So we keep the checkpoint and the model on the CPU, where they use no GPU memory.
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

    # Quantization works on an eval-mode copy. The checkpoint never changes.
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

    if args.quant != "NONE":
        quantization_name = quant_modes.mode_name(args)
        print(f"Quantization {quantization_name}: {quant_modes.describe(args)}"
              + (f" ({args.act_calib} calibration, {calib_samples} training samples)" if args.act_calib else ""))
        if args.backend == "pytorch":
            # Simulation (fake quantization) on an eval-mode, BN-folded copy.
            model.eval()
            model, weight_stats = prepare_quantized_model(
                model, quant_modes.weight_spec(args), quant_modes.activation_spec(args), act_amax, args.fp_layers)
            worst = min(weight_stats, key=lambda row: row["sqnr_db"])
            print(f"Simulated weights: {len(weight_stats)} layers quantized, {len(args.fp_layers)} kept in FP. "
                  f"Lowest weight SQNR: {worst['sqnr_db']:.1f} dB ({worst['layer']})")
            if args.tf32 == "on":
                print("Warning: TF32 adds its own rounding on top of the simulated quantization error.")
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

        d2s = args.upsample == "depth2space"
        onnx_path = args.onnx or ROOT_DIR / "exports" / f"{args.checkpoint.stem}_fp32_bnfolded{'_d2s' if d2s else ''}.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"{onnx_path} not found. Export it first:\n  python -m quantization.export_onnx "
                f"--dataset {args.dataset} --checkpoint {args.checkpoint}{' --upsample depth2space' if d2s else ''}"
            )
        export_info = json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
        if export_info["dataset"] != args.dataset or Path(export_info["checkpoint"]).name != args.checkpoint.name:
            raise ValueError(f"{onnx_path} was exported from {export_info['checkpoint']} ({export_info['dataset']}).")
        if export_info.get("upsample", "convtranspose") != args.upsample:
            raise ValueError(f"{onnx_path} was exported with --upsample {export_info.get('upsample', 'convtranspose')}, not {args.upsample}.")
        if args.quant != "NONE":
            # INT8 uses the same quantization as the fake-quant reference. FP8 uses
            # E4M3 with scale amax/448. Both are written into the graph as
            # low-precision weights and Q/DQ nodes (saved next to the FP32 ONNX).
            from quantization.qdq_onnx import build_quantized_onnx

            fmt, _ = quant_modes.TENSORRT_MODES[args.quant]
            onnx_path = build_quantized_onnx(onnx_path, args.calib_file, args.act_calib, args.fp_layers, fmt=fmt)
            print(f"{args.quant} ONNX: {onnx_path.name}")
        # The engine accepts exactly the image sizes of the evaluated samples.
        profile = profile_from_sizes(image_sizes(base_dataset, sample_ids))
        engine_path = get_engine(onnx_path, profile, tf32=use_tf32, n_channels=input_channels, rebuild=args.rebuild_engine)
        forward = TensorRTModel(engine_path)
        trt_summary = {**forward.layer_summary(), "engine_bytes": forward.engine_bytes, "total_params": total_params}
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
            tqdm(dataloader,desc=f"Evaluation {args.dataset}",unit="batch",dynamic_ncols=True,)):
            # Normalize uint8 images to [0, 1] in FP32 on the device, then
            # round once to the model dtype (as the original FP32 pipeline).
            images = batch["image"].to(device,non_blocking=(device.type == "cuda"),)
            images = (images.float().div_(torch.iinfo(batch["image"].dtype).max).to(model_dtype))
            masks = batch["mask"].to(device,non_blocking=(device.type == "cuda"),)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            start_time = time.perf_counter()

            with torch.autocast(device_type=device.type,dtype=amp_dtype if use_amp else torch.float32,enabled=use_amp,):
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

    means = sample_means(per_sample_rows)
    latency_ms = (timed_seconds / timed_samples * 1000 if timed_samples else float("nan"))
    throughput = (timed_samples / timed_seconds if timed_seconds > 0 else float("nan"))

    model_format = {
        torch.float32: "FP32",
        torch.float16: "FP16",
        torch.bfloat16: "BF16",
    }.get(model_dtype, str(model_dtype))
    representation, storage, execution = describe_precision(args, use_amp, model_format, trt_summary)
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
    print(f"Mean loss: {means['mean_loss']:.6f}")
    print(f"Mean Dice: {means['mean_dice']:.6f}")
    print(
        f"Forward latency: {latency_ms:.2f} ms/image | "
        f"Throughput: {throughput:.2f} images/s "
        f"({timed_samples} images, {warmup_batches} warm-up batches excluded)"
    )

    peak_allocated, peak_reserved = peak_gpu_memory_mib(device)
    if peak_allocated is not None:
        print(f"Peak GPU memory allocated: {peak_allocated:.2f} MiB")
        print(f"Peak GPU memory reserved: {peak_reserved:.2f} MiB")

    # 1. One row per run in a shared summary, to compare runs side by side.
    summary_path = args.output / "summary.csv"
    run = {
        "split": split_name,
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "total_params": total_params,
        "model_dtype": model_format,
        "autocast": args.precision if use_amp else "off",
        "quantization": quantization_name,
        "calib_samples": calib_samples,
        "representation": representation,
        "storage": storage,
        "execution": execution,
    }
    performance = {
        "latency_ms_per_image": latency_ms,
        "throughput_images_per_sec": throughput,
        "weights_mib": weights_mib,
        "engine_mib": engine_mib,
        "peak_gpu_allocated_mib": peak_allocated,
        "peak_gpu_reserved_mib": peak_reserved,
    }
    append_summary_row(summary_path, summary_row(args, run, means, performance, environment))

    # 2. Per-image results of this run, to compare two runs on exactly the
    # same images (a rerun with the same settings overwrites the file).
    per_sample_path = args.output / "per_sample" / per_sample_filename(args, split_name, quantization_name, use_tf32)
    write_per_sample_csv(per_sample_path, per_sample_rows, args.model, total_params)

    print(f"Summary appended to: {summary_path}")
    print(f"Per-sample results saved to: {per_sample_path}")

if __name__ == "__main__":
    main()
