from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


def stems(folder: Path, exts: set[str] | None = None) -> set[str]:
    files = folder.glob("*")
    if exts is None:
        return {path.stem for path in files if path.is_file()}
    return {path.stem for path in files if path.is_file() and path.suffix.lower() in exts}


def main() -> None:
    failed = False

    for split in ("train", "val", "test"):
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split
        image_stems = stems(image_dir, IMAGE_EXTS)
        label_stems = stems(label_dir, {".txt"})

        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)

        print(f"{split}: {len(image_stems)} images, {len(label_stems)} labels")
        if missing_labels:
            failed = True
            print(f"  missing labels: {missing_labels[:10]}")
        if missing_images:
            failed = True
            print(f"  missing images: {missing_images[:10]}")

    if failed:
        raise SystemExit(1)

    print("dataset check passed")


if __name__ == "__main__":
    main()
