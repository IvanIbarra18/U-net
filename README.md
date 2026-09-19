# U-Net Floating-Point Precision Experiments

This project evaluates the effect of floating-point precision on U-Net image segmentation.

The experiments compare:

- FP32
- FP16
- BF16

Two segmentation datasets are used:

- Carvana
- ISIC 2016

## Experimental Configuration

The baseline experiments use:

- U-Net architecture
- Image scale: 0.5
- Batch size: 1
- 100 training epochs
- 10% validation split
- RMSprop optimizer
- Learning rate: 1e-5

Each precision configuration is evaluated using:

- Dice coefficient
- Intersection over Union (IoU)
- Training loss
- Epoch time
- Images per second
- Peak GPU VRAM usage

## Datasets

Datasets are not included in this repository.

Expected directory structure:

Pytorch-UNet/
├── train_experiment.py
├── requirements.txt
├── README.md
├── .gitignore
├── data_carvana/
│   ├── imgs/
│   └── masks/
└── data_ISIC/
    ├── imgs/
    └── masks/

## Baseline Experiments

The baseline experiments consist of running the same configuration using three floating-point precisions.

### ISIC

| Precision | Command |
|---|---|
| FP32 | `python train_experiment.py --dataset ISIC --precision FP32 -e 100 -b 1` |
| FP16 | `python train_experiment.py --dataset ISIC --precision FP16 -e 100 -b 1` |
| BF16 | `python train_experiment.py --dataset ISIC --precision BF16 -e 100 -b 1` |

### Carvana

| Precision | Command |
|---|---|
| FP32 | `python train_experiment.py --dataset Carvana --precision FP32 -e 100 -b 1` |
| FP16 | `python train_experiment.py --dataset Carvana --precision FP16 -e 100 -b 1` |
| BF16 | `python train_experiment.py --dataset Carvana --precision BF16 -e 100 -b 1` |