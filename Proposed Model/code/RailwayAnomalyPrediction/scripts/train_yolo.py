
from pathlib import Path
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "RSD-YOLO-Dataset-v1.0"
SOURCE_YAML = DATASET_DIR / "data.yaml"
CONFIG_PATH = PROJECT_ROOT / "configs" / "rsd_yolo.yaml"
GENERATED_YAML = PROJECT_ROOT / "configs" / "rsd_yolo_local.yaml"


def main():
    if not SOURCE_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {SOURCE_YAML}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = yaml.safe_load(SOURCE_YAML.read_text(encoding="utf-8"))

    names = dataset.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]

    if len(names) != 3:
        raise ValueError(f"Expected 3 classes, found: {names}")

    # Build a local YAML file. Ultralytics receives a path, not a dict.
    local_dataset = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }

    GENERATED_YAML.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_YAML.write_text(
        yaml.safe_dump(local_dataset, sort_keys=False),
        encoding="utf-8",
    )

    print("Classes:", names)
    print("Dataset YAML:", GENERATED_YAML)
    print("Training device:", config["device"])
    print("Starting YOLO training...")

    model = YOLO(config["model"])

    model.train(
        data=str(GENERATED_YAML.resolve()),
        imgsz=config["imgsz"],
        epochs=config["epochs"],
        batch=config["batch"],
        device=config["device"],
        workers=config["workers"],
        patience=config["patience"],
        seed=config["seed"],
        deterministic=config["deterministic"],
        project=str(PROJECT_ROOT / config["project"]),
        name=config["name"],
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    best_weights = (
        PROJECT_ROOT / config["project"] / config["name"]
        / "weights" / "best.pt"
    )
    print(f"\nTraining finished. Best weights: {best_weights}")


if __name__ == "__main__":
    main()