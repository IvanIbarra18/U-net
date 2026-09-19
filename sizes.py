from pathlib import Path
from PIL import Image

images_dir = Path("data_ISIC/imgs")

sizes = []

for path in images_dir.glob("*.jpg"):
    with Image.open(path) as img:
        w, h = img.size
        pixels = w * h
        sizes.append((w, h, pixels, path.name))

print(f"Images: {len(sizes)}")

# Sort by number of pixels
sizes.sort(key=lambda x: x[2], reverse=True)

print("\nTop 20 largest images:")
for w, h, pixels, name in sizes[:20]:
    print(f"{name:30} {w:4} x {h:4} = {pixels:,} pixels")

carvana_pixels = 1920 * 1200

print("\nPixel distribution:")

ranges = [
    (0, 1_000_000, "< 1 MP"),
    (1_000_000, 2_000_000, "1–2 MP"),
    (2_000_000, 3_000_000, "2–3 MP"),
    (3_000_000, 5_000_000, "3–5 MP"),
    (5_000_000, 10_000_000, "5–10 MP"),
    (10_000_000, float("inf"), "> 10 MP"),
]

for low, high, label in ranges:
    count = sum(low <= pixels < high for _, _, pixels, _ in sizes)
    print(f"{label:10}: {count}")

print("\nCompared with Carvana 1920x1200:")
count = sum(pixels > carvana_pixels for _, _, pixels, _ in sizes)
print(f"ISIC images larger than 1920x1200: {count}/{len(sizes)}")

print("\nAfter scale=0.5:")
largest = sizes[0]
w, h = largest[0] // 2, largest[1] // 2
print(f"Largest scaled image: {w} x {h}")
print(f"Scaled pixels: {w*h:,}")