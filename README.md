# Binary Image Segmentation with U-Net (PyTorch)

A PyTorch project for binary image segmentation. It includes two U-Net
implementations, image/mask pair loading and preprocessing, training and
evaluation scripts, post-training quantization experiments, ONNX/TensorRT
inference tools, and utilities for analyzing the ISIC dataset.

![U-Net architecture diagram](graphs/unet_architecture.png)

## Architectures

The `models/` directory contains two models:

- **Classic U-Net** (`models/unet_classic.py`, `UNetClassic`): the original
  five-level architecture, with valid convolutions (no padding) and
  center-cropped skip connections. As a result, its spatial output is smaller
  than its input.
- **Modern U-Net** (`models/unet_modern.py`, `UNetModern`): uses padded
  convolutions, four encoder levels, and a bottleneck. It preserves the input
  height and width and allows the base channel count and BatchNorm to be
  configured.

Both models return logits for a binary mask. `train.py` uses the modern U-Net
with 64 base channels and BatchNorm by default. Choose either architecture with
`--model modern` or `--model classic`; `--base-channels` configures the channel
count of either model.

## Project structure

```text
.
├── data/
│   ├── data_loading.py       # Image and mask loading and normalization
│   ├── splits.py             # Reproducible split generation and loading
│   └── transforms.py         # Synchronized image/mask transforms
├── datasets/                 # Locally extracted data (not tracked by Git)
│   ├── data_Carvana/
│   │   ├── imgs/
│   │   └── masks/
│   ├── data_ISBI/
│   │   ├── imgs/
│   │   └── masks/
│   ├── data_ISIC/
│   │   ├── imgs/
│   │   └── masks/
│   └── subset_ISIC/          # Optional local dataset subset
├── models/
│   ├── unet_classic.py
│   └── unet_modern.py
├── quantization/
│   ├── bn_fold.py
│   ├── calibrate.py
│   ├── env_report.py
│   ├── export_onnx.py
│   ├── int8.py
│   ├── qdq_onnx.py
│   ├── quantize_model.py
│   ├── trt_engine.py
│   └── test_*.py
├── graphs/
│   └── unet_architecture.png # Retained architecture diagram
├── profiling/
│   └── performance_logger.py
├── splits/                   # Saved splits for each dataset
│   ├── Carvana/
│   ├── ISBI/
│   └── ISIC/
├── utilis/
│   ├── checkpoint.py
│   ├── loss.py
│   └── metrics.py
├── checkpoints/              # Generated checkpoints, TensorBoard logs, and CSVs
├── calibration/              # Generated activation-calibration JSON files
├── exports/                  # Generated ONNX files and TensorRT engine cache
├── test_results/             # Evaluation summary and per-sample CSVs
├── datasets.zip              # Dataset archive (requires Git LFS for GitHub)
├── .gitignore
├── README.md
├── eval.py
├── inference.py
├── test.py
└── train.py
```

`data/`, `datasets/`, `splits/`, `models/`, `quantization/`, `profiling/`, and
`utilis/` are separate directories. `calibration/`, `exports/`, and
`test_results/` contain generated artifacts. Git ignores the extracted
`datasets/` directory and `checkpoints/`, but does not ignore `datasets.zip`.
The current archive is about 1.2 GB, so uploading it to GitHub requires Git LFS
or an external file hosting service: GitHub rejects files larger than 100 MB
in regular Git repositories.

## Preparing the data

Extract `datasets.zip` into the project root and check that the directories
have the following structure:

```text
datasets/
├── data_Carvana/
│   ├── imgs/
│   └── masks/
├── data_ISBI/
│   ├── imgs/
│   └── masks/
└── data_ISIC/
    ├── imgs/
    └── masks/
```

Each image must have a corresponding mask with the same base filename (for
example, `sample_01.jpg` and `sample_01.png`), and both files must have the
same dimensions. ISBI is loaded as a single-channel image; Carvana and ISIC
are loaded as three-channel RGB images. Images are converted to values between
0 and 1, and masks are converted to binary values.

## Installation

Install PyTorch and the dependencies used by the scripts:

```bash
pip install torch torchvision numpy Pillow tqdm tensorboard matplotlib scikit-learn prefetch-generator
```

Training uses CUDA when available and otherwise runs on the CPU.

ONNX export and Q/DQ graph generation also require the ONNX packages supported
by the installed PyTorch ONNX exporter:

```bash
pip install onnx onnxscript
```

TensorRT inference is optional. Install a TensorRT Python package compatible
with the system's NVIDIA GPU, CUDA, and PyTorch environment using NVIDIA's
installation instructions; there is no portable TensorRT install command for
all platforms. The PyTorch evaluation backend does not require TensorRT.
`nvidia-ml-py` is optional and enables NVIDIA driver-version reporting:

```bash
pip install nvidia-ml-py
```

The project does not pin package versions in a dependency file. Record the
environment used for an experiment with:

```bash
python -m quantization.env_report
```

## Training

Run commands from the repository root. For example:

```bash
python train.py --dataset ISBI --precision FP16 --epochs 50
python train.py --dataset Carvana --precision FP32 --epochs 10
python train.py --dataset ISIC --precision BF16 --epochs 10
python train.py --dataset ISIC --model modern --base-channels 32
python train.py --dataset ISIC --model classic --base-channels 32
```

Training applies paired transforms to images and masks: random crops, elastic
deformation, and horizontal and vertical flips. Validation does not apply
these transforms and uses a batch size of one.

By default, training uses a reproducible random split with seed 42 and 10%
validation data. You can select one of the saved splits, from 0 to 4:

```bash
python train.py --dataset ISIC --split 0 --precision FP32 --epochs 10
```

Saved splits are stored in `splits/<dataset>/train_N.txt` and
`valid_N.txt`. To regenerate them (five independent splits, reproducible by
seed and with 10% validation data):

```bash
python data/splits.py --dataset ISIC --folds 5 --validation 10 --seed 42
```

### Main arguments

| Argument | Default | Description |
|---|---:|---|
| `--dataset` | Required | `Carvana`, `ISBI`, or `ISIC` |
| `--model` | `modern` | `modern` or `classic` |
| `--base-channels` | `64` | Base channel count for the selected U-Net |
| `--precision` | `FP32` | `FP32`, `FP16`, or `BF16` |
| `--epochs` | `10` | Number of epochs |
| `--batch-size` | `5` | Training batch size |
| `--num-workers` | `min(4, CPU count)` | DataLoader worker processes |
| `--crop` | `512` | Training crop size; accepts `N` or `H W` |
| `--validation` | `10` | Validation percentage for the random split |
| `--split` | Random | Saved split, from `0` to `4` |
| `--seed` | `42` | Random seed |
| `--lr` | `0.0001` | Learning rate |
| `--output` | `checkpoints` | Output directory |
| `--experiment-name` | No CSV | Optional name for the per-epoch CSV (`--experiment_name` is also accepted) |

Example using a rectangular crop:

```bash
python train.py --dataset ISBI --precision FP16 --epochs 50 --crop 256 384
```

FP16 and BF16 use automatic mixed precision (AMP) when training on CUDA.
FP16 also uses gradient scaling. On CPU, training runs without AMP. The loss
combines binary cross-entropy and soft Dice. The Dice score is calculated on
the binary segmentation. When `--experiment-name` is set, the per-epoch CSV
includes the training BCE and Dice losses, their weighted contributions to
the total loss, the corresponding validation loss components, and the training
and validation Dice scores. It also records the model name and parameter count,
validation forward latency and throughput, model weight size, and peak GPU
memory use per epoch.

## Results and monitoring

- The best checkpoint is saved to
  `<output>/best_unet_<dataset>_new.pth` by default, based on the lowest
  validation loss (`checkpoints/` is the default output directory). With
  `--experiment-name NAME`, it is saved as `<output>/best_NAME.pth`. It
  includes the model weights, optimizer state, epoch, configuration, and
  losses.
- TensorBoard logs and example images, ground-truth masks, and predictions
  are saved under `checkpoints/tensorboard/<dataset>/`.
- Enable a per-epoch summary CSV with `--experiment_name`. It is saved to
  `checkpoints/performance/<name>.csv` and includes training and validation
  loss and Dice, model name and parameter count, epoch duration, training
  throughput, validation forward latency and throughput, model weight size,
  and peak GPU memory. Validation latency excludes up to three warm-up batches.
  This CSV is not generated if the argument is omitted.
- `PerformanceLogger` also measures timing and memory usage during training.
- When training finishes, `train.py` displays the loss and Dice curves for
  that run.

To launch TensorBoard:

```bash
tensorboard --logdir checkpoints/tensorboard
```

## Evaluating a checkpoint

`test.py` evaluates a checkpoint using either PyTorch or TensorRT. For
example, the default PyTorch backend evaluates a saved validation split:

```bash
python test.py --dataset ISBI --checkpoint checkpoints/best_unet_ISBI_new.pth --model modern --split 0
```

Use `--split full` to evaluate `datasets/test_<dataset>/`; without `--split`,
the script recreates the checkpoint's training split and evaluates its
validation portion. The split seed and validation percentage default to the
values saved in the checkpoint (or 42 and 10%, respectively, if unavailable).
For a classic U-Net checkpoint, use `--model classic`; pass the matching
`--base-channels` if it differs from 64.

The default `--backend pytorch` supports FP32, FP16/BF16 autocast on CUDA,
whole-model `--cast FP16` or `--cast BF16`, and optional `--tf32 on`. The
`--fold-bn` flag folds BatchNorm for a floating-point evaluation. `--precision`
and `--cast` are separate experiments and cannot be combined. Results are
written under `test_results/` by default; use `--output` to choose another
directory.

`summary.csv` appends one row per run. Per-image loss components and Dice are
written to `per_sample/`; the file name includes the checkpoint, dataset,
split, model, precision, backend, and quantization settings. A repeated run
with the same name overwrites that per-sample file but still appends another
summary row. The summary distinguishes numerical representation, storage,
and execution backend, and includes available GPU, driver, PyTorch, and
TensorRT information. On TensorRT runs, engine size and inspected layer
precision are also recorded.

`inference.py` contains prediction and evaluation utilities, including an
option to average predictions from flipped inputs (test-time augmentation).

## Post-training quantization

The project has two different INT8 evaluation paths. They should not be
interpreted as equivalent in terms of execution or speed.

### PyTorch fake quantization

With `--backend pytorch --quant INT8`, `test.py` folds BatchNorm into
convolutions and fake-quantizes the weights. Weights are symmetric and
per-output-channel, with thresholds derived directly from the weights. By
default this is W8A32: activations are not quantized.

```bash
python test.py --dataset Carvana --checkpoint checkpoints/best_CarvanaBaseline.pth --quant INT8
```

For W8A8, first calibrate activation input ranges using training samples:

```bash
python -m quantization.calibrate --dataset Carvana --checkpoint checkpoints/best_CarvanaBaseline.pth --samples 128
```

The default output is
`calibration/best_CarvanaBaseline_n128.json`. Then select one of the
calibration methods when evaluating:

```bash
python test.py --dataset Carvana --checkpoint checkpoints/best_CarvanaBaseline.pth --quant INT8 --act-calib percentile-99.99 --calib-file calibration/best_CarvanaBaseline_n128.json
```

Calibration methods are `max`, `percentile-99.9`, `percentile-99.99`,
`percentile-99.999`, `entropy`, and `entropy-skip-zeros`. Calibration samples
come from the checkpoint's training split; the JSON stores the sampled IDs,
per-layer histograms, thresholds, and split metadata. The `--seed` option in
this command (default 42) selects the calibration subset; it is separate from
the seed used to define the training split. Set `--base-channels` to match a
non-default checkpoint when calibrating and evaluating. `--samples` defaults
to 128, `--bins` to 2048, and `--num-workers` to 4. Use `--output` to override
the default JSON path.

Calibration uses a folded, FP32 Modern U-Net and observes the inputs to each
Conv2d and ConvTranspose2d without modifying them. Its first pass records each
layer's maximum absolute input value; its second pass builds the histograms
used by the percentile and entropy methods. `entropy-skip-zeros` removes exact
zero counts before applying the entropy threshold selection.

This PyTorch path is a **numerical simulation**, not actual INT8 execution:
values are rounded/clipped to INT8 codes and dequantized to FP32 before
convolution. Parameters and activations remain FP32, and CUDA uses FP32
convolution computation (or TF32 convolution tactics if `--tf32 on`). It
measures the effect of quantization error; it does not provide INT8 model
compression or INT8 acceleration.

`--fp-layers enc1.block.0,segment` leaves the named convolution weights and
their inputs in floating point. The names must match layers in the selected
model. `--act-calib` and `--calib-file` must be supplied together and require
`--quant INT8`. INT8 simulation requires FP32 evaluation: do not combine it
with `--cast` or non-FP32 `--precision`. BatchNorm folding is automatic for
`--quant INT8`; `--fold-bn` is only needed when testing folding without INT8.

### ONNX export and TensorRT

Export the Modern U-Net checkpoint to a dynamic-shape FP32 ONNX graph before
using TensorRT:

```bash
python -m quantization.export_onnx --dataset Carvana --checkpoint checkpoints/best_CarvanaBaseline.pth
```

By default this writes
`exports/best_CarvanaBaseline_fp32_bnfolded.onnx` and a JSON metadata file.
BatchNorm is folded unless `--keep-bn` is specified. The exporter supports
dynamic image height and width from 16 to 8192 pixels and rewrites the U-Net
transpose-convolution upsampling to preserve its dynamic-size behavior.
Use `--output` to select a different ONNX path.

The TensorRT backend requires a CUDA GPU, `--model modern`, and
`--batch-size 1`; do not pass `--cast`, `--precision`, or `--fold-bn`. The
engine profile is built from the evaluated image sizes. For FP32 TensorRT
inference:

```bash
python test.py --dataset Carvana --checkpoint checkpoints/best_CarvanaBaseline.pth --backend tensorrt
```

TensorRT builds and caches an engine beside the ONNX file under `engines/`.
The cache is specific to the ONNX graph, GPU, TensorRT version, image-size
profile, and TF32 setting. Use `--rebuild-engine` to rebuild it, or `--onnx`
to select another exported graph. The engine inspector reports the actual
convolution layers, data types, and tactics used; the requested graph precision
alone is not treated as proof of the precision that ran. Engine metadata and
per-layer inspection are saved next to each engine as `.json` and
`.layers.json` files.

To run TensorRT with the INT8 Q/DQ graph, add `--quant INT8`. W8A32 needs no
calibration JSON; W8A8 also needs `--act-calib` and `--calib-file`:

```bash
python test.py --dataset Carvana --checkpoint checkpoints/best_CarvanaBaseline.pth --backend tensorrt --quant INT8 --act-calib percentile-99.99 --calib-file calibration/best_CarvanaBaseline_n128.json
```

`test.py` automatically turns the FP32 ONNX export into an INT8 Q/DQ ONNX
graph and builds or reuses the TensorRT engine. In this path, weights are
stored as INT8 initializers in the ONNX graph, and TensorRT may fuse Q/DQ
operations into INT8 convolution kernels. Actual layer precision is reported
from the built engine. TensorRT's Q/DQ rounding and saturation can differ
slightly from the PyTorch fake-quant reference.

The ONNX conversion can also be run explicitly. For example, to create a
W8A8 graph without running evaluation:

```bash
python -m quantization.qdq_onnx --onnx exports/best_CarvanaBaseline_fp32_bnfolded.onnx --calib-file calibration/best_CarvanaBaseline_n128.json --act-calib percentile-99.99
```

Omit both `--calib-file` and `--act-calib` for W8A32; use `--fp-layers` to
leave selected layers in FP32. The script writes a Q/DQ ONNX file and metadata
including source hashes and initializer sizes. Pass `--force` to regenerate a
cached Q/DQ graph.

## Quantization tests

The unit tests are not needed to run training or evaluation. Run the core
PyTorch quantization tests from the repository root with:

```bash
python -m unittest quantization.test_int8 quantization.test_bn_fold quantization.test_quantize_model quantization.test_calibrate -v
```

ONNX export and Q/DQ tests are separate:

```bash
python -m unittest quantization.test_export_onnx quantization.test_qdq_onnx -v
```

TensorRT engine tests require a compatible CUDA/TensorRT installation; the
engine-specific tests skip when those components are unavailable:

```bash
python -m unittest quantization.test_trt_engine -v
```
