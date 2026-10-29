
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]

PAIR_FILE = ROOT / "data" / "processed" / "candidate_train_pairs.parquet"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading candidate train pairs...")

columns = [
    "arrival_day",
    "station_code",
    "earlier_delay_min",
    "later_delay_min",
    "scheduled_gap_min",
]

df = pd.read_parquet(PAIR_FILE, columns=columns)

df["arrival_day"] = pd.to_datetime(df["arrival_day"])

df = df.dropna(
    subset=[
        "arrival_day",
        "station_code",
        "earlier_delay_min",
        "later_delay_min",
        "scheduled_gap_min",
    ]
).copy()

df = df[
    np.isfinite(df["earlier_delay_min"])
    & np.isfinite(df["later_delay_min"])
    & np.isfinite(df["scheduled_gap_min"])
].copy()

df["station_code"] = df["station_code"].astype(str)

# Split by calendar date so pairs from the same date
# cannot appear in both training and test sets.
dates = np.sort(df["arrival_day"].unique())
cutoff_index = int(len(dates) * 0.70)
cutoff_date = pd.Timestamp(dates[cutoff_index])

train = df[df["arrival_day"] < cutoff_date].copy()
test = df[df["arrival_day"] >= cutoff_date].copy()

print("\n========== CHRONOLOGICAL SPLIT ==========")
print(f"Total usable pairs : {len(df):,}")
print(f"Distinct dates     : {len(dates):,}")
print(f"Training rows      : {len(train):,}")
print(f"Test rows          : {len(test):,}")
print(f"Training ends      : {train['arrival_day'].max().date()}")
print(f"Test begins        : {test['arrival_day'].min().date()}")

if train.empty or test.empty:
    raise ValueError("Training or test set is empty. Check arrival_day values.")

y_train = train["later_delay_min"]
y_test = test["later_delay_min"]

results = []


def evaluate(name, predictions):
    predictions = np.asarray(predictions)

    results.append(
        {
            "model": name,
            "MAE_min": mean_absolute_error(y_test, predictions),
            "RMSE_min": np.sqrt(
                mean_squared_error(y_test, predictions)
            ),
            "R2": r2_score(y_test, predictions),
        }
    )


# Baseline 1: global median delay from training data.
global_median = y_train.median()

evaluate(
    "Global median baseline",
    np.full(len(test), global_median),
)


# Baseline 2: median delay for each station, learned from training data.
station_medians = train.groupby("station_code")[
    "later_delay_min"
].median()

station_predictions = (
    test["station_code"]
    .map(station_medians)
    .fillna(global_median)
)

evaluate(
    "Station median baseline",
    station_predictions,
)


# Model 1: earlier train's delay only.
model_earlier_delay = LinearRegression()

model_earlier_delay.fit(
    train[["earlier_delay_min"]],
    y_train,
)

predictions_earlier = model_earlier_delay.predict(
    test[["earlier_delay_min"]]
)

evaluate(
    "Earlier-train delay only",
    predictions_earlier,
)


# Model 2: earlier delay + scheduled gap + station identity.
numeric_features = [
    "earlier_delay_min",
    "scheduled_gap_min",
]

categorical_features = ["station_code"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "station",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

model_station_aware = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0, solver="lsqr")),
    ]
)

model_station_aware.fit(
    train[numeric_features + categorical_features],
    y_train,
)

predictions_station_aware = model_station_aware.predict(
    test[numeric_features + categorical_features]
)

evaluate(
    "Station-aware Ridge regression",
    predictions_station_aware,
)


# Save and display the evaluation.
results_df = pd.DataFrame(results).sort_values("MAE_min")

results_path = OUTPUT_DIR / "propagation_model_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\n========== FUTURE-PERIOD TEST RESULTS ==========")
print(results_df.round(3).to_string(index=False))

print(f"\nResults saved to: {results_path}")

# A separate diagnostic: correlation in the held-out period.
test_correlation = test[
    ["earlier_delay_min", "later_delay_min"]
].corr().iloc[0, 1]

print(
    "\nEarlier/later delay correlation on test period: "
    f"{test_correlation:.4f}"
)

print("\nEvaluation complete.")