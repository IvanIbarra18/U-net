from pathlib import Path
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path("data_ISIC")
IMAGES_DIR = SOURCE_DIR / "imgs"
MASKS_DIR = SOURCE_DIR / "masks"

OUTPUT_DIR = Path("data_ISIC_processed")
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "imgs"
OUTPUT_MASKS_DIR = OUTPUT_DIR / "masks"

# Target canvas: width x height
TARGET_W = 1024
TARGET_H = 768

# Raw mask value used exclusively for artificial padding.
#
# ISIC masks use:
#     0   = background
#     255 = lesion
#
# Therefore, 254 is used for padding so it cannot be
# confused with either real class.
MASK_PADDING_VALUE = 254


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_IMAGES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_MASKS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def resize_and_pad(
    image,
    target_w,
    target_h,
    resampling,
    pad_value
):
    """
    Resize an image while preserving its aspect ratio,
    then center it inside a fixed-size canvas.

    Args:
        image: PIL Image.
        target_w: Target canvas width.
        target_h: Target canvas height.
        resampling: PIL resampling method.
        pad_value: Value used for the padding.

    Returns:
        processed_image:
            Image after resizing and padding.

        resized_width:
            Width of the resized original image.

        resized_height:
            Height of the resized original image.
    """

    original_w, original_h = image.size

    # --------------------------------------------------------
    # Calculate scale while preserving aspect ratio
    # --------------------------------------------------------

    scale = min(
        target_w / original_w,
        target_h / original_h
    )

    resized_w = round(
        original_w * scale
    )

    resized_h = round(
        original_h * scale
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = image.resize(
        (resized_w, resized_h),
        resampling
    )

    # --------------------------------------------------------
    # Create target canvas
    # --------------------------------------------------------

    processed = Image.new(
        image.mode,
        (target_w, target_h),
        pad_value
    )

    # --------------------------------------------------------
    # Center the resized image
    # --------------------------------------------------------

    left = (
        target_w - resized_w
    ) // 2

    top = (
        target_h - resized_h
    ) // 2

    processed.paste(
        resized,
        (left, top)
    )

    return (
        processed,
        resized_w,
        resized_h
    )


# ============================================================
# FIND INPUT IMAGES
# ============================================================

image_paths = sorted(
    IMAGES_DIR.glob("*.jpg")
)

if not image_paths:

    raise RuntimeError(
        f"No .jpg images found in {IMAGES_DIR}"
    )


# ============================================================
# STATISTICS
# ============================================================

processed_count = 0

missing_masks = []

errors = []

total_original_pixels = 0

total_resized_pixels = 0

total_padding_pixels = 0


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ISIC DATASET PREPROCESSING")
print("=" * 70)

print(
    f"Source images : {IMAGES_DIR}"
)

print(
    f"Source masks  : {MASKS_DIR}"
)

print(
    f"Output        : {OUTPUT_DIR}"
)

print(
    f"Target canvas : {TARGET_W} x {TARGET_H}"
)

print(
    "Image resize  : Bilinear"
)

print(
    "Mask resize   : Nearest"
)

print(
    "Aspect ratio  : Preserved"
)

print(
    "Padding       : Centered"
)

print(
    "Image padding : Black (0, 0, 0)"
)

print(
    f"Mask padding  : {MASK_PADDING_VALUE}"
)

print(
    "Image format  : PNG"
)

print(
    "\nISIC mask values:"
)

print(
    "  0   = background"
)

print(
    "  255 = lesion"
)

print(
    f"  {MASK_PADDING_VALUE} = artificial padding"
)

print(
    "\nProcessing..."
)


# ============================================================
# PROCESS DATASET
# ============================================================

for image_path in image_paths:

    image_id = image_path.stem

    # Expected ISIC 2016 mask name:
    #
    # ISIC_xxxxxxx_Segmentation.png

    mask_path = (
        MASKS_DIR
        / f"{image_id}_Segmentation.png"
    )

    # --------------------------------------------------------
    # Check mask exists
    # --------------------------------------------------------

    if not mask_path.exists():

        missing_masks.append(
            image_id
        )

        print(
            f"[WARNING] Missing mask: "
            f"{mask_path.name}"
        )

        continue

    try:

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        with Image.open(image_path) as image:

            # Force RGB.
            image = image.convert(
                "RGB"
            )

            original_w, original_h = (
                image.size
            )

            total_original_pixels += (
                original_w * original_h
            )

            # ------------------------------------------------
            # Resize + pad image
            # ------------------------------------------------

            (
                processed_image,
                resized_w,
                resized_h
            ) = resize_and_pad(
                image,
                TARGET_W,
                TARGET_H,
                Image.Resampling.BILINEAR,
                (0, 0, 0)
            )

        # ====================================================
        # LOAD MASK
        # ====================================================

        with Image.open(mask_path) as mask:

            # Force grayscale.
            mask = mask.convert(
                "L"
            )

            mask_w, mask_h = (
                mask.size
            )

            # ------------------------------------------------
            # Verify image/mask dimensions
            # ------------------------------------------------

            if (
                mask_w != original_w
                or mask_h != original_h
            ):

                raise ValueError(
                    f"Image/mask size mismatch: "
                    f"image={original_w}x{original_h}, "
                    f"mask={mask_w}x{mask_h}"
                )

            # ------------------------------------------------
            # Resize + pad mask
            #
            # IMPORTANT:
            #     Nearest Neighbor must be used for masks.
            #
            # Padding uses 254 instead of 0 because:
            #
            #     0   = real background
            #     255 = real lesion
            #     254 = artificial padding
            # ------------------------------------------------

            (
                processed_mask,
                mask_resized_w,
                mask_resized_h
            ) = resize_and_pad(
                mask,
                TARGET_W,
                TARGET_H,
                Image.Resampling.NEAREST,
                MASK_PADDING_VALUE
            )

        # ====================================================
        # VERIFY IMAGE/MASK RESIZE MATCH
        # ====================================================

        if (
            resized_w != mask_resized_w
            or resized_h != mask_resized_h
        ):

            raise RuntimeError(
                f"Image/mask resize mismatch "
                f"for {image_id}"
            )

        # ====================================================
        # CALCULATE PIXEL STATISTICS
        # ====================================================

        resized_pixels = (
            resized_w * resized_h
        )

        padding_pixels = (
            TARGET_W * TARGET_H
            - resized_pixels
        )

        total_resized_pixels += (
            resized_pixels
        )

        total_padding_pixels += (
            padding_pixels
        )

        # ====================================================
        # SAVE PROCESSED IMAGE
        # ====================================================

        output_image_path = (
            OUTPUT_IMAGES_DIR
            / f"{image_id}.png"
        )

        output_mask_path = (
            OUTPUT_MASKS_DIR
            / f"{image_id}_Segmentation.png"
        )

        processed_image.save(
            output_image_path,
            format="PNG"
        )

        processed_mask.save(
            output_mask_path,
            format="PNG"
        )

        processed_count += 1

        # ====================================================
        # LOG
        # ====================================================

        print(
            f"[{processed_count:3}/{len(image_paths)}] "
            f"{image_id}: "
            f"{original_w}x{original_h} "
            f"-> "
            f"{resized_w}x{resized_h} "
            f"+ padding "
            f"-> "
            f"{TARGET_W}x{TARGET_H}"
        )

    except Exception as exc:

        errors.append(
            (image_id, str(exc))
        )

        print(
            f"[ERROR] {image_id}: {exc}"
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PREPROCESSING COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Images found       : "
    f"{len(image_paths)}"
)

print(
    f"Images processed   : "
    f"{processed_count}"
)

print(
    f"Missing masks      : "
    f"{len(missing_masks)}"
)

print(
    f"Processing errors  : "
    f"{len(errors)}"
)

print(
    f"Output image dir   : "
    f"{OUTPUT_IMAGES_DIR}"
)

print(
    f"Output mask dir    : "
    f"{OUTPUT_MASKS_DIR}"
)

print(
    f"Mask padding value : "
    f"{MASK_PADDING_VALUE}"
)


# ============================================================
# PIXEL STATISTICS
# ============================================================

if processed_count > 0:

    print(
        f"\nTotal original pixels: "
        f"{total_original_pixels:,}"
    )

    print(
        f"Total resized pixels: "
        f"{total_resized_pixels:,}"
    )

    print(
        f"Total padding pixels: "
        f"{total_padding_pixels:,}"
    )

    print(
        f"Canvas pixels/image: "
        f"{TARGET_W * TARGET_H:,}"
    )


# ============================================================
# REPORT MISSING MASKS
# ============================================================

if missing_masks:

    print(
        "\nMissing masks:"
    )

    for image_id in missing_masks:

        print(
            f"  - {image_id}"
        )


# ============================================================
# REPORT PROCESSING ERRORS
# ============================================================

if errors:

    print(
        "\nProcessing errors:"
    )

    for image_id, error in errors:

        print(
            f"  - {image_id}: {error}"
        )


# ============================================================
# FINAL OUTPUT CHECK
# ============================================================

if processed_count == len(image_paths):

    print(
        "\nAll images were successfully processed."
    )

    print(
        "Processed masks use 254 for artificial padding."
    )

else:

    print(
        "\nWARNING: Not every image was processed."
    )

    print(
        "Check the messages above."
    )