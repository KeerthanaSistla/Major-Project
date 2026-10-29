from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "lstm_sequences.parquet"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_FILE = MODEL_DIR / "railway_lstm_best.pt"
FEATURE_SCALER_FILE = MODEL_DIR / "lstm_feature_scaler.joblib"
TARGET_SCALER_FILE = MODEL_DIR / "lstm_target_scaler.joblib"

SAMPLE_PERCENT = 5
SEQUENCE_LENGTH = 5
FEATURES_PER_STEP = 5
BATCH_SIZE = 2048
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

# Only include trains with at least this many test observations
# in the train-level report.
MIN_TRAIN_SAMPLES = 50

DEVICE = torch.device("cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "delay_t5", "dwell_t5", "travel_t5", "hour_sin_t5", "hour_cos_t5",
    "delay_t4", "dwell_t4", "travel_t4", "hour_sin_t4", "hour_cos_t4",
    "delay_t3", "dwell_t3", "travel_t3", "hour_sin_t3", "hour_cos_t3",
    "delay_t2", "dwell_t2", "travel_t2", "hour_sin_t2", "hour_cos_t2",
    "delay_t1", "dwell_t1", "travel_t1", "hour_sin_t1", "hour_cos_t1",
]

TARGET = "target_delay_min"

print("=" * 75)
print("RAILWAY DELAY MODEL SUBGROUP EVALUATION")
print("=" * 75)


# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

for path in [
    DATA_FILE,
    MODEL_FILE,
    FEATURE_SCALER_FILE,
    TARGET_SCALER_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

feature_scaler = joblib.load(FEATURE_SCALER_FILE)
target_scaler = joblib.load(TARGET_SCALER_FILE)


# ============================================================
# 3. RECREATE THE ORIGINAL CHRONOLOGICAL SPLIT
# ============================================================

con = duckdb.connect()

try:
    con.execute("SET threads TO 4")

    date_df = con.execute(
        """
        SELECT DISTINCT service_date
        FROM read_parquet(?)
        WHERE service_date IS NOT NULL
        ORDER BY service_date
        """,
        [str(DATA_FILE)],
    ).fetchdf()

    dates = pd.to_datetime(date_df["service_date"]).tolist()

    if len(dates) < 20:
        raise ValueError("Not enough distinct dates.")

    n_dates = len(dates)

    train_end_index = int(n_dates * 0.70) - 1
    val_end_index = int(n_dates * 0.85) - 1

    train_end = dates[train_end_index].date()
    val_end = dates[val_end_index].date()

    print(f"\nDistinct dates: {n_dates}")
    print(f"Training through: {train_end}")
    print(f"Validation through: {val_end}")
    print(f"Test begins after: {val_end}")

    selected_columns = [
        "service_date",
        "train_number",
        "target_station_code",
        "target_station_seq",
    ] + FEATURES + [TARGET]

    select_sql = ", ".join(selected_columns)

    def load_split(split_name):
        if split_name == "train":
            date_filter = "service_date <= ?"
            params = [str(DATA_FILE), train_end]

        elif split_name == "validation":
            date_filter = "service_date > ? AND service_date <= ?"
            params = [str(DATA_FILE), train_end, val_end]

        elif split_name == "test":
            date_filter = "service_date > ?"
            params = [str(DATA_FILE), val_end]

        else:
            raise ValueError(f"Unknown split: {split_name}")

        query = f"""
            SELECT {select_sql}
            FROM read_parquet(?)
            WHERE {date_filter}
              AND hash(
                    train_number,
                    service_date,
                    target_station_seq
                  ) % 100 < {SAMPLE_PERCENT}
            ORDER BY service_date
        """

        df = con.execute(query, params).fetchdf()

        numeric_columns = FEATURES + [TARGET]

        df[numeric_columns] = df[numeric_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        df = df.dropna(subset=numeric_columns).reset_index(drop=True)

        if df.empty:
            raise ValueError(f"The {split_name} split is empty.")

        print(f"{split_name.title()} samples: {len(df):,}")

        return df

    train_df = load_split("train")
    val_df = load_split("validation")
    test_df = load_split("test")

finally:
    con.close()


# ============================================================
# 4. PREPARE INPUTS USING THE EXISTING SCALERS
# ============================================================

def extract_xy(df):
    X = df[FEATURES].to_numpy(dtype=np.float32)

    X = X.reshape(
        len(df),
        SEQUENCE_LENGTH,
        FEATURES_PER_STEP,
    )

    y = df[TARGET].to_numpy(dtype=np.float32)

    return X, y


X_train, y_train = extract_xy(train_df)
X_test, y_test = extract_xy(test_df)

def scale_features(X):
    original_shape = X.shape

    scaled = feature_scaler.transform(
        X.reshape(-1, FEATURES_PER_STEP)
    )

    return scaled.reshape(original_shape).astype(np.float32)


X_train = scale_features(X_train)
X_test = scale_features(X_test)

print(f"\nTraining rows used for Ridge: {len(X_train):,}")
print(f"Test rows evaluated:          {len(X_test):,}")


# ============================================================
# 5. RECREATE THE ORIGINAL LSTM ARCHITECTURE
# ============================================================

class RailwayLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=FEATURES_PER_STEP,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        last_output = sequence_output[:, -1, :]
        return self.regressor(last_output).squeeze(-1)


model = RailwayLSTM().to(DEVICE)

state_dict = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()

print("\nExisting LSTM checkpoint loaded.")


# ============================================================
# 6. GENERATE PREDICTIONS FROM ALL MODELS
# ============================================================

def predict_lstm(X):
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.from_numpy(
                X[start:start + BATCH_SIZE]
            ).to(DEVICE)

            output = model(batch).cpu().numpy()
            predictions.append(output)

    scaled_predictions = np.concatenate(predictions)

    return target_scaler.inverse_transform(
        scaled_predictions.reshape(-1, 1)
    ).ravel()


print("\nGenerating LSTM predictions...")
lstm_predictions = predict_lstm(X_test)

print("Generating persistence predictions...")
persistence_predictions = test_df[
    "delay_t1"
].to_numpy(dtype=np.float32)

print("Fitting Ridge baseline on the original training sample...")
ridge = Ridge(alpha=1.0)

ridge.fit(
    X_train.reshape(len(X_train), -1),
    y_train,
)

ridge_predictions = ridge.predict(
    X_test.reshape(len(X_test), -1)
)

print("Generating training-median baseline...")
median_predictions = np.full(
    len(y_test),
    np.median(y_train),
    dtype=np.float32,
)


# ============================================================
# 7. BUILD A COMMON TEST-PREDICTION TABLE
# ============================================================

predictions_df = test_df[
    [
        "service_date",
        "train_number",
        "target_station_code",
        "target_station_seq",
    ]
].copy()

predictions_df["actual_delay"] = y_test
predictions_df["persistence_prediction"] = persistence_predictions
predictions_df["ridge_prediction"] = ridge_predictions
predictions_df["lstm_prediction"] = lstm_predictions
predictions_df["median_prediction"] = median_predictions

predictions_df["delay_t1"] = test_df["delay_t1"].to_numpy()

predictions_df["actual_delay_range"] = pd.cut(
    predictions_df["actual_delay"],
    bins=[-np.inf, 0, 15, 60, 120, np.inf],
    labels=[
        "Early (<0)",
        "0 to <15 min",
        "15 to <60 min",
        "60 to <120 min",
        "120+ min",
    ],
    right=False,
)

# Handle the boundary at zero correctly:
# negative delays are early; zero and above are assigned
# to the corresponding non-negative range.
predictions_df["actual_delay_range"] = pd.cut(
    predictions_df["actual_delay"],
    bins=[-np.inf, 0, 15, 60, 120, np.inf],
    labels=[
        "Early (<0)",
        "0 to <15 min",
        "15 to <60 min",
        "60 to <120 min",
        "120+ min",
    ],
    right=False,
    include_lowest=True,
)

predictions_df["abs_error_persistence"] = np.abs(
    predictions_df["actual_delay"]
    - predictions_df["persistence_prediction"]
)

predictions_df["abs_error_ridge"] = np.abs(
    predictions_df["actual_delay"]
    - predictions_df["ridge_prediction"]
)

predictions_df["abs_error_lstm"] = np.abs(
    predictions_df["actual_delay"]
    - predictions_df["lstm_prediction"]
)


# ============================================================
# 8. METRIC HELPER
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
        }

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted) if len(actual) >= 2 else np.nan,
    }


# ============================================================
# 9. METRICS BY ACTUAL DELAY RANGE
# ============================================================

print("\n" + "=" * 75)
print("PERFORMANCE BY ACTUAL TARGET DELAY RANGE")
print("=" * 75)

delay_results = []

for delay_range, group in predictions_df.groupby(
    "actual_delay_range",
    observed=False,
):
    if group.empty:
        continue

    actual = group["actual_delay"].to_numpy()

    for model_name, column in [
        ("Persistence", "persistence_prediction"),
        ("Ridge", "ridge_prediction"),
        ("LSTM", "lstm_prediction"),
        ("Training median", "median_prediction"),
    ]:
        metrics = calculate_metrics(
            actual,
            group[column].to_numpy(),
        )

        delay_results.append({
            "delay_range": str(delay_range),
            "model": model_name,
            "samples": len(group),
            **metrics,
        })

delay_results_df = pd.DataFrame(delay_results)

delay_results_path = (
    OUTPUT_DIR / "subgroup_metrics_by_delay_range.csv"
)

delay_results_df.to_csv(delay_results_path, index=False)

print(
    delay_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ============================================================
# 10. PERFORMANCE BY TRAIN
# ============================================================

print("\n" + "=" * 75)
print("PERFORMANCE BY TRAIN")
print("=" * 75)

train_results = []

for train_number, group in predictions_df.groupby("train_number"):
    if len(group) < MIN_TRAIN_SAMPLES:
        continue

    actual = group["actual_delay"].to_numpy()

    persistence_metrics = calculate_metrics(
        actual,
        group["persistence_prediction"].to_numpy(),
    )

    ridge_metrics = calculate_metrics(
        actual,
        group["ridge_prediction"].to_numpy(),
    )

    lstm_metrics = calculate_metrics(
        actual,
        group["lstm_prediction"].to_numpy(),
    )

    train_results.append({
        "train_number": str(train_number),
        "samples": len(group),

        "persistence_MAE": persistence_metrics["MAE"],
        "persistence_RMSE": persistence_metrics["RMSE"],

        "ridge_MAE": ridge_metrics["MAE"],
        "ridge_RMSE": ridge_metrics["RMSE"],

        "lstm_MAE": lstm_metrics["MAE"],
        "lstm_RMSE": lstm_metrics["RMSE"],

        # Positive means LSTM has lower MAE than persistence.
        "LSTM_MAE_improvement_vs_persistence":
            persistence_metrics["MAE"] - lstm_metrics["MAE"],

        # Positive means LSTM has lower MAE than Ridge.
        "LSTM_MAE_improvement_vs_ridge":
            ridge_metrics["MAE"] - lstm_metrics["MAE"],
    })

train_results_df = pd.DataFrame(train_results)

train_results_path = OUTPUT_DIR / "per_train_metrics.csv"

if not train_results_df.empty:
    train_results_df = train_results_df.sort_values(
        "LSTM_MAE_improvement_vs_persistence",
        ascending=False,
    )

    train_results_df.to_csv(train_results_path, index=False)

    print(f"Trains with at least {MIN_TRAIN_SAMPLES} test rows: "
          f"{len(train_results_df):,}")

    print("\nTop 10 trains where LSTM improves on persistence:")
    print(
        train_results_df.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nTop 10 trains where LSTM performs worse than persistence:")
    print(
        train_results_df.tail(10).sort_values(
            "LSTM_MAE_improvement_vs_persistence"
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print(
        "\nFraction of evaluated trains where LSTM beats persistence "
        f"on MAE: "
        f"{(train_results_df['LSTM_MAE_improvement_vs_persistence'] > 0).mean():.1%}"
    )

else:
    print(
        f"No trains had at least {MIN_TRAIN_SAMPLES} test samples. "
        "Lower MIN_TRAIN_SAMPLES if needed."
    )


# ============================================================
# 11. SAVE ALL TEST PREDICTIONS
# ============================================================

prediction_path = OUTPUT_DIR / "test_predictions_subgroups.csv"

predictions_df.to_csv(prediction_path, index=False)


# ============================================================
# 12. OVERALL METRICS FOR REFERENCE
# ============================================================

print("\n" + "=" * 75)
print("OVERALL TEST PERFORMANCE")
print("=" * 75)

overall_results = []

for model_name, column in [
    ("Persistence", "persistence_prediction"),
    ("Ridge", "ridge_prediction"),
    ("LSTM", "lstm_prediction"),
    ("Training median", "median_prediction"),
]:
    metrics = calculate_metrics(
        predictions_df["actual_delay"],
        predictions_df[column],
    )

    overall_results.append({
        "model": model_name,
        "samples": len(predictions_df),
        **metrics,
    })

overall_df = pd.DataFrame(overall_results)

print(
    overall_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

overall_path = OUTPUT_DIR / "overall_test_metrics.csv"
overall_df.to_csv(overall_path, index=False)


# ============================================================
# 13. COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("EVALUATION COMPLETE")
print("=" * 75)

print(f"Delay-range metrics: {delay_results_path}")
print(f"Per-train metrics:   {train_results_path}")
print(f"Test predictions:    {prediction_path}")
print(f"Overall metrics:     {overall_path}")