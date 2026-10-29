
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "RSD-YOLO-Dataset-v1.0"
DATA_YAML = DATASET_DIR / "data.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset path: {DATASET_DIR}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {DATA_YAML}. "
            "Check the extracted dataset structure."
        )

    config = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))

    print("\nDataset YAML:")
    print(yaml.safe_dump(config, sort_keys=False))

    print("\nImage and label counts:")

    for split in ("train", "val", "test"):
        image_dir = DATASET_DIR / "images" / split
        label_dir = DATASET_DIR / "labels" / split

        if not image_dir.exists():
            print(f"{split}: MISSING image directory: {image_dir}")
            continue

        images = [
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"\n{split}: {len(images)} images")

        if not label_dir.exists():
            print(f"  MISSING label directory: {label_dir}")
            continue

        missing_labels = []
        empty_labels = []
        malformed_labels = []
        class_counts = {}

        for image_path in images:
            label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")

            if not label_path.exists():
                missing_labels.append(image_path.name)
                continue

            lines = label_path.read_text(encoding="utf-8").splitlines()

            if not lines:
                empty_labels.append(image_path.name)
                continue

            for line in lines:
                parts = line.split()

                if len(parts) != 5:
                    malformed_labels.append((image_path.name, line))
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    malformed_labels.append((image_path.name, line))
                    continue

                if (
                    class_id < 0
                    or not all(0 <= v <= 1 for v in (x, y, w, h))
                    or w <= 0
                    or h <= 0
                ):
                    malformed_labels.append((image_path.name, line))
                    continue

                class_counts[class_id] = class_counts.get(class_id, 0) + 1

        print(f"  Missing label files: {len(missing_labels)}")
        print(f"  Empty label files: {len(empty_labels)}")
        print(f"  Malformed/invalid annotations: {len(malformed_labels)}")
        print(f"  Valid boxes by class ID: {class_counts}")

        if missing_labels:
            print("  Missing-label examples:", missing_labels[:5])

        if malformed_labels:
            print("  Invalid-label examples:", malformed_labels[:5])

    print("\nDataset inspection complete.")


if __name__ == "__main__":
    main()