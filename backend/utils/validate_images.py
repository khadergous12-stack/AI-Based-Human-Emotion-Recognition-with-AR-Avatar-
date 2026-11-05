
from pathlib import Path
from PIL import Image

DATASET_PATH = Path("datasets/raw/fer2013")

EMOTIONS = [
    "angry", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
]

def validate_images():
    total_checked = 0
    corrupted = 0
    dimensions = {}

    print("\n========== IMAGE VALIDATION ==========\n")

    for split in ["train", "test"]:
        print(f"\nChecking {split.upper()} dataset...")

        for emotion in EMOTIONS:
            folder = DATASET_PATH / split / emotion

            for image_path in folder.iterdir():
                if image_path.suffix.lower() not in {
                    ".jpg", ".jpeg", ".png", ".bmp"
                }:
                    continue

                total_checked += 1

                try:
                    with Image.open(image_path) as img:
                        img.verify()

                    with Image.open(image_path) as img:
                        size = img.size
                        dimensions[size] = dimensions.get(size, 0) + 1

                except Exception:
                    corrupted += 1
                    print(f"Corrupted: {image_path}")

    print("\n========== VALIDATION RESULTS ==========")
    print(f"Total images checked: {total_checked}")
    print(f"Corrupted images: {corrupted}")
    print(f"Valid images: {total_checked - corrupted}")

    print("\nImage dimensions:")
    for size, count in sorted(dimensions.items()):
        print(f"{size}: {count} images")

    print("\n========================================")

if __name__ == "__main__":
    validate_images()
