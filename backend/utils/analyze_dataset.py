
from pathlib import Path
from collections import Counter

# Dataset location
DATASET_PATH = Path("datasets/raw/fer2013")

# Emotion categories
EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def analyze_dataset():
    print("\n========== FER2013 DATASET ANALYSIS ==========\n")

    if not DATASET_PATH.exists():
        print("ERROR: Dataset folder not found!")
        print(f"Expected location: {DATASET_PATH.resolve()}")
        return

    total_images = 0

    for split in ["train", "test"]:
        split_path = DATASET_PATH / split

        print(f"\n--- {split.upper()} DATASET ---")

        if not split_path.exists():
            print(f"ERROR: {split} folder not found!")
            continue

        split_total = 0

        for emotion in EMOTIONS:
            emotion_path = split_path / emotion

            if not emotion_path.exists():
                print(f"{emotion:10}: Folder missing")
                continue

            image_count = sum(
                1 for file in emotion_path.iterdir()
                if file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            )

            print(f"{emotion:10}: {image_count:5} images")

            split_total += image_count

        print(f"\nTotal {split} images: {split_total}")

        total_images += split_total

    print("\n==============================================")
    print(f"TOTAL DATASET IMAGES: {total_images}")
    print("==============================================\n")


if __name__ == "__main__":
    analyze_dataset()
