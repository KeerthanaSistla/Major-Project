
from pathlib import Path
import json

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "pjay-in/indian-railways-delays"

print("Checking available dataset configurations...")

configs = get_dataset_config_names(DATASET_ID)
print("\nAvailable configurations:")
for config in configs:
    print(" -", config)

# Prefer the documented historical configuration.
preferred = "2025-02-08_2026-02-07"

if preferred in configs:
    config = preferred
else:
    raise RuntimeError(
        f"Expected configuration {preferred!r} was not found. "
        "Check the printed configurations before choosing another."
    )

print(f"\nLoading dataset: {DATASET_ID}")
print(f"Configuration: {config}")

dataset = load_dataset(
    DATASET_ID,
    name=config,
    split="train",
)

print(f"\nNumber of records: {len(dataset):,}")
print("\nColumn names:")
print(dataset.column_names)

print("\nFirst three records:")
for record in dataset.select(range(min(3, len(dataset)))):
    print(record)

print("\nConverting to pandas...")
df = dataset.to_pandas()

print("\nDataset shape:", df.shape)
print("\nData types:")
print(df.dtypes.to_string())

print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False).to_string())

print("\nSaving raw dataset...")

raw_file = RAW_DIR / "railway_delays_raw.parquet"
df.to_parquet(raw_file, index=False)

# Save schema information for later processing.
schema = {
    "dataset_id": DATASET_ID,
    "configuration": config,
    "rows": int(len(df)),
    "columns": list(df.columns),
    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    "missing_values": {
        col: int(count)
        for col, count in df.isna().sum().items()
    },
}

schema_file = OUTPUT_DIR / "dataset_inspection.json"
with open(schema_file, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)

print("\nDownload and inspection complete.")
print(f"Raw data: {raw_file}")
print(f"Inspection report: {schema_file}")