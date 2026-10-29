
from pathlib import Path
import yaml
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "RSD-YOLO-Dataset-v1.0"
DATA_YAML = DATASET_DIR / "data.yaml"
CONFIG_PATH = PROJECT_ROOT / "configs" / "rsd_yolo.yaml"


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    LOCAL_DATA_YAML = PROJECT_ROOT / "configs" / "rsd_yolo_local.yaml"

    if not LOCAL_DATA_YAML.exists():
        raise FileNotFoundError(
            "Local dataset YAML not found. Run scripts/train_yolo.py first."
        )

    best_weights = (
        PROJECT_ROOT
        / config["project"]
        / config["name"]
        / "weights"
        / "best.pt"
    )

    if not best_weights.exists():
        raise FileNotFoundError(
            f"Trained weights not found: {best_weights}\n"
            "Run scripts/train_yolo.py first."
        )

    model = YOLO(str(best_weights))

    results = model.val(
        data=str(LOCAL_DATA_YAML.resolve()),
        split="test",
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        plots=True,
        project=str(PROJECT_ROOT / config["project"]),
        name="rsd_yolov8n_test",
        exist_ok=True,
    )

    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    }

    output_dir = PROJECT_ROOT / config["project"] / "rsd_yolov8n_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(
        output_dir / "test_metrics.csv", index=False
    )

    print("\nHeld-out test metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"\nMetrics saved to: {output_dir / 'test_metrics.csv'}")
    print(f"Plots and evaluation artifacts: {output_dir}")


if __name__ == "__main__":
    main()