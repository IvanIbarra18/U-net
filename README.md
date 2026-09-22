# U-Net Floating-Point Precision Experiments

This project evaluates the effect of floating-point precision on U-Net image segmentation.

The experiments compare:

- FP32
- FP16
- BF16

Two segmentation datasets are used:

- Carvana
- ISIC 2016

The project also evaluates the effect of image preprocessing and batch size on GPU memory usage and training performance.

## Experimental Configuration

The experiments use:

- U-Net architecture
- RMSprop optimizer
- Learning rate: 1e-5
- 100 training epochs
- 10% validation split

Each precision configuration is evaluated using:

- Dice coefficient
- Intersection over Union (IoU)
- Training loss
- Epoch time
- Images per second
- Peak GPU VRAM usage

## Datasets

Datasets are not included in this repository.

### Carvana

Expected directory structure:

```text
Pytorch-UNet/

├── train.py
├── preprocess_isic.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data_carvana/
│   ├── imgs/
│   └── masks/
│
├── data_ISIC/
│   ├── imgs/
│   └── masks/
│
└── data_ISIC_processed/
    ├── imgs/
    └── masks/

### ISIC

| Precision | Command |
|---|---|
| FP32 | `python train.py --dataset ISIC --precision FP32 -e 100 -b 1` |
| FP16 | `python train.py --dataset ISIC --precision FP16 -e 100 -b 1` |
| BF16 | `python train.py --dataset ISIC --precision BF16 -e 100 -b 1` |

### Carvana

| Precision | Command |
|---|---|
| FP32 | `python train.py --dataset Carvana --precision FP32 -e 100 -b 1` |
| FP16 | `python train.py --dataset Carvana --precision FP16 -e 100 -b 1` |
| BF16 | `python train.py --dataset Carvana --precision BF16 -e 100 -b 1` |