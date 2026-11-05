
from pathlib import Path

DATASET = Path("datasets/processed/fer2013")

EMOTIONS = [
    "angry", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
]

for split in ["train", "validation", "test"]:
    print(f"\n{split.upper()} DATASET")

    total = 0

    for emotion in EMOTIONS:
        folder = DATASET / split / emotion

        count = len([
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in
            {".jpg", ".jpeg", ".png", ".bmp"}
        ])

        print(f"{emotion:10}: {count}")
        total += count

    print(f"Total: {total}")

print("\nProcessed dataset verification completed.")
