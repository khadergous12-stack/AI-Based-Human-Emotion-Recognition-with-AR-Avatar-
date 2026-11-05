
from pathlib import Path
from collections import defaultdict
import random
import shutil

# Original dataset
RAW_PATH = Path("datasets/raw/fer2013")

# Processed dataset
OUTPUT_PATH = Path("datasets/processed/fer2013")

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

VALIDATION_RATIO = 0.20
RANDOM_SEED = 42


def copy_images(images, destination):
    destination.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        shutil.copy2(
            image_path,
            destination / image_path.name
        )


def prepare_dataset():

    print("\n========== DATASET PREPROCESSING ==========\n")

    if not (RAW_PATH / "train").exists():
        print("ERROR: Original train folder not found!")
        return

    if not (RAW_PATH / "test").exists():
        print("ERROR: Original test folder not found!")
        return

    if OUTPUT_PATH.exists() and any(OUTPUT_PATH.iterdir()):
        print("ERROR: Processed dataset already contains files.")
        print("Check the existing folder before running again.")
        return

    rng = random.Random(RANDOM_SEED)

    totals = defaultdict(int)

    # Split original training data
    for emotion in EMOTIONS:

        source_folder = RAW_PATH / "train" / emotion

        if not source_folder.exists():
            print(f"ERROR: Missing folder: {source_folder}")
            return

        images = [
            file for file in source_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            print(f"ERROR: No images found for {emotion}")
            return

        rng.shuffle(images)

        validation_count = round(
            len(images) * VALIDATION_RATIO
        )

        validation_images = images[:validation_count]
        training_images = images[validation_count:]

        copy_images(
            training_images,
            OUTPUT_PATH / "train" / emotion
        )

        copy_images(
            validation_images,
            OUTPUT_PATH / "validation" / emotion
        )

        totals["train"] += len(training_images)
        totals["validation"] += len(validation_images)

        print(
            f"{emotion:10} | "
            f"Train: {len(training_images):5} | "
            f"Validation: {len(validation_images):5}"
        )

    # Copy original test data without changing it
    print("\nCopying original test dataset...")

    for emotion in EMOTIONS:

        source_folder = RAW_PATH / "test" / emotion

        if not source_folder.exists():
            print(f"ERROR: Missing test folder: {source_folder}")
            return

        images = [
            file for file in source_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        copy_images(
            images,
            OUTPUT_PATH / "test" / emotion
        )

        totals["test"] += len(images)

    print("\n========== PREPROCESSING SUMMARY ==========")
    print(f"Training images:   {totals['train']}")
    print(f"Validation images: {totals['validation']}")
    print(f"Testing images:    {totals['test']}")

    grand_total = sum(totals.values())

    print(f"Total images:      {grand_total}")
    print(f"Output location:   {OUTPUT_PATH.resolve()}")
    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    prepare_dataset()
