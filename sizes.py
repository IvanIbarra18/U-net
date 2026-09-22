from pathlib import Path
from PIL import Image
from collections import Counter
import statistics
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

images_dir = Path("data_ISIC/imgs")

# Candidate 4:3 canvases.
# All dimensions are divisible by 32, which is convenient for U-Net.
candidate_4_3 = [
    (512, 384),
    (640, 480),
    (768, 576),
    (896, 672),
    (1024, 768),
    (1152, 864),
    (1280, 960),
    (1408, 1056),
    (1536, 1152),
    (1664, 1248),
    (1792, 1344),
    (1920, 1440),
    (2048, 1536),
]


# ============================================================
# LOAD IMAGE DIMENSIONS
# ============================================================

images = []

for path in images_dir.glob("*.jpg"):

    with Image.open(path) as img:

        w, h = img.size

        images.append({
            "name": path.name,
            "width": w,
            "height": h,
            "pixels": w * h,
            "aspect_ratio": w / h,
        })


if not images:
    raise RuntimeError(
        f"No JPG images found in {images_dir}"
    )


total_images = len(images)

total_original_pixels = sum(
    img["pixels"]
    for img in images
)


print("=" * 80)
print("ISIC CANVAS OPTIMIZATION ANALYSIS")
print("=" * 80)

print(f"\nImages: {total_images}")

print(
    f"Total original pixels: "
    f"{total_original_pixels:,} "
    f"({total_original_pixels / 1_000_000:.2f} MP)"
)


# ============================================================
# EXACT RESOLUTION DISTRIBUTION
# ============================================================

resolution_counter = Counter(
    (img["width"], img["height"])
    for img in images
)


print("\n" + "=" * 80)
print("MOST COMMON EXACT RESOLUTIONS")
print("=" * 80)


for (w, h), count in resolution_counter.most_common(30):

    percentage = (
        count / total_images * 100
    )

    print(
        f"{w:5} x {h:<5} "
        f"{count:3} images "
        f"({percentage:5.1f}%)"
    )


# ============================================================
# FUNCTION: ANALYZE ONE CANVAS
# ============================================================

def analyze_canvas(target_w, target_h):

    pixels_destroyed = 0
    pixels_created = 0

    downscaled_images = 0
    upscaled_images = 0
    unchanged_images = 0

    relative_changes = []
    scale_factors = []

    padding_pixels = 0

    for img in images:

        original_w = img["width"]
        original_h = img["height"]

        original_pixels = img["pixels"]

        # ----------------------------------------------------
        # Preserve aspect ratio.
        # Find the largest image that fits inside the canvas.
        # ----------------------------------------------------

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

        resized_pixels = (
            resized_w * resized_h
        )

        scale_factors.append(scale)

        # ----------------------------------------------------
        # Pixel change
        # ----------------------------------------------------

        difference = (
            resized_pixels
            - original_pixels
        )

        if difference < 0:

            pixels_destroyed += abs(difference)

            downscaled_images += 1

        elif difference > 0:

            pixels_created += difference

            upscaled_images += 1

        else:

            unchanged_images += 1

        # ----------------------------------------------------
        # Relative change
        # ----------------------------------------------------

        relative_change = (
            abs(difference)
            / original_pixels
        )

        relative_changes.append(
            relative_change
        )

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        canvas_pixels = (
            target_w * target_h
        )

        padding = (
            canvas_pixels
            - resized_pixels
        )

        padding_pixels += padding


    # ========================================================
    # Aggregate metrics
    # ========================================================

    total_absolute_change = (
        pixels_destroyed
        + pixels_created
    )

    mean_relative_change = (
        statistics.mean(relative_changes)
    )

    median_relative_change = (
        statistics.median(relative_changes)
    )

    return {

        "width": target_w,
        "height": target_h,

        "canvas_pixels": (
            target_w * target_h
        ),

        "destroyed": pixels_destroyed,
        "created": pixels_created,
        "absolute_change": total_absolute_change,

        "downscaled": downscaled_images,
        "upscaled": upscaled_images,
        "unchanged": unchanged_images,

        "mean_relative_change":
            mean_relative_change,

        "median_relative_change":
            median_relative_change,

        "median_scale":
            statistics.median(scale_factors),

        "padding_pixels":
            padding_pixels,
    }


# ============================================================
# ANALYZE ALL CANVASES
# ============================================================

results = []

for w, h in candidate_4_3:

    results.append(
        analyze_canvas(w, h)
    )


# ============================================================
# FIND BEST CANVAS
# ============================================================

results_by_change = sorted(
    results,
    key=lambda x: x["absolute_change"]
)

best = results_by_change[0]


# ============================================================
# PRINT FULL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("4:3 CANVAS COMPARISON")
print("=" * 80)


for result in results:

    print(
        f"\nCanvas: "
        f'{result["width"]} x {result["height"]}'
    )

    print(
        f'  Downscaled images : '
        f'{result["downscaled"]:3}'
    )

    print(
        f'  Upscaled images   : '
        f'{result["upscaled"]:3}'
    )

    print(
        f'  Unchanged images  : '
        f'{result["unchanged"]:3}'
    )

    print(
        f'  Pixels destroyed  : '
        f'{result["destroyed"] / 1_000_000:10.2f} MP'
    )

    print(
        f'  Pixels created    : '
        f'{result["created"] / 1_000_000:10.2f} MP'
    )

    print(
        f'  Absolute change   : '
        f'{result["absolute_change"] / 1_000_000:10.2f} MP'
    )

    print(
        f'  Mean rel. change  : '
        f'{result["mean_relative_change"] * 100:9.2f}%'
    )

    print(
        f'  Median rel. change: '
        f'{result["median_relative_change"] * 100:9.2f}%'
    )

    print(
        f'  Median scale      : '
        f'{result["median_scale"]:.3f}x'
    )

    print(
        f'  Padding generated : '
        f'{result["padding_pixels"] / 1_000_000:10.2f} MP'
    )


# ============================================================
# BEST CANVAS
# ============================================================

print("\n" + "=" * 80)
print("MINIMUM ABSOLUTE PIXEL CHANGE")
print("=" * 80)

print(
    f'\nBest 4:3 canvas: '
    f'{best["width"]} x {best["height"]}'
)

print(
    f'Total absolute pixel change: '
    f'{best["absolute_change"] / 1_000_000:.2f} MP'
)

print(
    f'Pixels destroyed: '
    f'{best["destroyed"] / 1_000_000:.2f} MP'
)

print(
    f'Pixels created: '
    f'{best["created"] / 1_000_000:.2f} MP'
)

print(
    f'Downscaled images: '
    f'{best["downscaled"]}'
)

print(
    f'Upscaled images: '
    f'{best["upscaled"]}'
)

print(
    f'Unchanged images: '
    f'{best["unchanged"]}'
)


# ============================================================
# RANKING
# ============================================================

print("\n" + "=" * 80)
print("RANKING BY TOTAL ABSOLUTE PIXEL CHANGE")
print("=" * 80)


for rank, result in enumerate(
    results_by_change,
    start=1
):

    print(
        f'{rank:2}. '
        f'{result["width"]:4} x '
        f'{result["height"]:<4} | '
        f'change: '
        f'{result["absolute_change"] / 1_000_000:9.2f} MP | '
        f'lost: '
        f'{result["destroyed"] / 1_000_000:9.2f} MP | '
        f'created: '
        f'{result["created"] / 1_000_000:9.2f} MP'
    )


# ============================================================
# GRAPH 1
# PIXELS DESTROYED VS CREATED
# ============================================================

labels = [
    f'{r["width"]}×{r["height"]}'
    for r in results
]

destroyed_mp = [
    r["destroyed"] / 1_000_000
    for r in results
]

created_mp = [
    r["created"] / 1_000_000
    for r in results
]

x = range(len(results))

bar_width = 0.38


plt.figure(figsize=(14, 7))

plt.bar(
    [i - bar_width / 2 for i in x],
    destroyed_mp,
    bar_width,
    label="Pixels destroyed"
)

plt.bar(
    [i + bar_width / 2 for i in x],
    created_mp,
    bar_width,
    label="Pixels created"
)

plt.xticks(
    list(x),
    labels,
    rotation=45,
    ha="right"
)

plt.xlabel("Target canvas")

plt.ylabel(
    "Total pixel change (MP)"
)

plt.title(
    "ISIC: Pixels Destroyed vs Created by Canvas Resolution"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "isic_pixels_destroyed_created.png",
    dpi=200
)

plt.show()


# ============================================================
# GRAPH 2
# TOTAL ABSOLUTE PIXEL CHANGE
# ============================================================

absolute_mp = [
    r["absolute_change"] / 1_000_000
    for r in results
]


plt.figure(figsize=(14, 7))

plt.plot(
    labels,
    absolute_mp,
    marker="o"
)

plt.xlabel("Target canvas")

plt.ylabel(
    "Total absolute pixel change (MP)"
)

plt.title(
    "ISIC: Total Pixel-Area Change by Canvas Resolution"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "isic_total_pixel_change.png",
    dpi=200
)

plt.show()


# ============================================================
# GRAPH 3
# RELATIVE CHANGE
# ============================================================

mean_relative = [
    r["mean_relative_change"] * 100
    for r in results
]

median_relative = [
    r["median_relative_change"] * 100
    for r in results
]


plt.figure(figsize=(14, 7))

plt.plot(
    labels,
    mean_relative,
    marker="o",
    label="Mean"
)

plt.plot(
    labels,
    median_relative,
    marker="o",
    label="Median"
)

plt.xlabel("Target canvas")

plt.ylabel(
    "Absolute pixel-area change (%)"
)

plt.title(
    "ISIC: Relative Resolution Change"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "isic_relative_pixel_change.png",
    dpi=200
)

plt.show()


# ============================================================
# GRAPH 4
# PADDING OVERHEAD
# ============================================================

padding_mp = [
    r["padding_pixels"] / 1_000_000
    for r in results
]


plt.figure(figsize=(14, 7))

plt.bar(
    labels,
    padding_mp
)

plt.xlabel("Target canvas")

plt.ylabel(
    "Total padding (MP)"
)

plt.title(
    "ISIC: Padding Introduced by Canvas Resolution"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "isic_padding_overhead.png",
    dpi=200
)

plt.show()


# ============================================================
# GRAPH 5
# PERCENTAGE OF IMAGES UPSCALED / DOWNSCALED / UNCHANGED
# ============================================================

downscaled_percent = [
    r["downscaled"] / total_images * 100
    for r in results
]

upscaled_percent = [
    r["upscaled"] / total_images * 100
    for r in results
]

unchanged_percent = [
    r["unchanged"] / total_images * 100
    for r in results
]


plt.figure(figsize=(14, 7))

plt.bar(
    labels,
    downscaled_percent,
    label="Downscaled"
)

plt.bar(
    labels,
    upscaled_percent,
    bottom=downscaled_percent,
    label="Upscaled"
)

plt.bar(
    labels,
    unchanged_percent,
    bottom=[
        d + u
        for d, u in zip(
            downscaled_percent,
            upscaled_percent
        )
    ],
    label="Unchanged"
)

plt.xlabel("Target canvas")

plt.ylabel(
    "Percentage of images (%)"
)

plt.title(
    "ISIC: Percentage of Images Affected by Resizing"
)

plt.ylim(0, 100)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "isic_upscale_downscale_percentage.png",
    dpi=200
)

plt.show()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("GENERATED GRAPHS")
print("=" * 80)

print(
    "1. isic_pixels_destroyed_created.png"
)

print(
    "2. isic_total_pixel_change.png"
)

print(
    "3. isic_relative_pixel_change.png"
)

print(
    "4. isic_padding_overhead.png"
)

print(
    "5. isic_upscale_downscale_percentage.png"
)

print("\nAnalysis complete.")