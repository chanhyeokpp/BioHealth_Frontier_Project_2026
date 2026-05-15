from collections import Counter
from pathlib import Path

from PIL import Image

from config import DATA_DIR, TEST_DIR, TRAIN_DIR


def describe_split(split_dir):
    print(f"\n[{split_dir.name}]")
    total = 0
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        files = [
            p
            for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        total += len(files)
        print(f"{class_dir.name:12s}: {len(files)}")
    print(f"{'total':12s}: {total}")


def inspect_images(root, max_images=200):
    image_paths = [
        p
        for p in root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ][:max_images]

    sizes = Counter()
    modes = Counter()
    for path in image_paths:
        with Image.open(path) as img:
            sizes[img.size] += 1
            modes[img.mode] += 1

    print("\n[Sample image metadata]")
    print(f"inspected images: {len(image_paths)}")
    print(f"modes: {dict(modes)}")
    print("top sizes:")
    for size, count in sizes.most_common(10):
        print(f"  {size}: {count}")


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    describe_split(TRAIN_DIR)
    describe_split(TEST_DIR)
    inspect_images(DATA_DIR)


if __name__ == "__main__":
    main()
