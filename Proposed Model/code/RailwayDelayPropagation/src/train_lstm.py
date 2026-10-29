from pathlib import Path
import copy
import random

import duckdb
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "lstm_sequences.parquet"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

SEQUENCE_LENGTH = 5
FEATURES_PER_STEP = 5

# Deterministic sample of each split to keep CPU training manageable.
SAMPLE_PERCENT = 5

BATCH_SIZE = 1024
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
MAX_EPOCHS = 20
PATIENCE = 4

DEVICE = torch.device("cpu")

FEATURES = [
    "delay_t5", "dwell_t5", "travel_t5", "hour_sin_t5", "hour_cos_t5",
    "delay_t4", "dwell_t4", "travel_t4", "hour_sin_t4", "hour_cos_t4",
    "delay_t3", "dwell_t3", "travel_t3", "hour_sin_t3", "hour_cos_t3",
    "delay_t2", "dwell_t2", "travel_t2", "hour_sin_t2", "hour_cos_t2",
    "delay_t1", "dwell_t1", "travel_t1", "hour_sin_t1", "hour_cos_t1",
]

TARGET = "target_delay_min"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("RAILWAY ARRIVAL-DELAY PREDICTION")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Features per timestep: {FEATURES_PER_STEP}")
print(f"Sampling rate: {SAMPLE_PERCENT}%")


# ============================================================
# 2. VALIDATE DATASET AND DEFINE CHRONOLOGICAL SPLITS
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Run src/build_lstm_sequences.py first."
    )

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
        raise ValueError("Not enough distinct dates for chronological splits.")

    n_dates = len(dates)

    train_end_index = int(n_dates * 0.70) - 1
    val_end_index = int(n_dates * 0.85) - 1

    train_end = dates[train_end_index].date()
    val_end = dates[val_end_index].date()

    print("\n========== CHRONOLOGICAL SPLIT ==========")
    print(f"Distinct service dates: {n_dates}")
    print(f"Train:      through {train_end}")
    print(f"Validation: after {train_end} through {val_end}")
    print(f"Test:       after {val_end}")

    feature_sql = ", ".join(FEATURES)

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
            SELECT
                service_date,
                train_number,
                target_station_code,
                target_station_seq,
                {feature_sql},
                {TARGET}
            FROM read_parquet(?)

            WHERE {date_filter}

              AND hash(
                  train_number,
                  service_date,
                  target_station_seq
              ) % 100 < {SAMPLE_PERCENT}

            ORDER BY service_date
        """

        print(f"\nLoading {split_name} split...")

        df = con.execute(query, params).fetchdf()

        numeric_cols = FEATURES + [TARGET]

        df[numeric_cols] = df[numeric_cols].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        df = df.dropna(subset=numeric_cols).reset_index(drop=True)

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
# 3. CHECK CHRONOLOGICAL SEPARATION
# ============================================================

train_dates = set(train_df["service_date"].astype(str))
val_dates = set(val_df["service_date"].astype(str))
test_dates = set(test_df["service_date"].astype(str))

assert train_dates.isdisjoint(val_dates), (
    "Training and validation dates overlap."
)

assert train_dates.isdisjoint(test_dates), (
    "Training and test dates overlap."
)

assert val_dates.isdisjoint(test_dates), (
    "Validation and test dates overlap."
)

print("\nChronological date separation verified.")

print("\n========== DATA SHAPES ==========")
print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(val_df):,}")
print(f"Test:       {len(test_df):,}")


# ============================================================
# 4. CONVERT DATA TO LSTM INPUTS
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
X_val, y_val = extract_xy(val_df)
X_test, y_test = extract_xy(test_df)

print("\nLSTM input shape:", X_train.shape)


# ============================================================
# 5. SCALE FEATURES USING TRAINING DATA ONLY
# ============================================================

feature_scaler = StandardScaler()

feature_scaler.fit(
    X_train.reshape(-1, FEATURES_PER_STEP)
)


def scale_features(X):
    original_shape = X.shape

    scaled = feature_scaler.transform(
        X.reshape(-1, FEATURES_PER_STEP)
    )

    return scaled.reshape(original_shape).astype(np.float32)


X_train = scale_features(X_train)
X_val = scale_features(X_val)
X_test = scale_features(X_test)


# Scale the target using the training target distribution only.
target_scaler = StandardScaler()

y_train_scaled = target_scaler.fit_transform(
    y_train.reshape(-1, 1)
).astype(np.float32).ravel()

y_val_scaled = target_scaler.transform(
    y_val.reshape(-1, 1)
).astype(np.float32).ravel()


joblib.dump(
    feature_scaler,
    MODEL_DIR / "lstm_feature_scaler.joblib",
)

joblib.dump(
    target_scaler,
    MODEL_DIR / "lstm_target_scaler.joblib",
)


# ============================================================
# 6. CREATE DATA LOADERS
# ============================================================

train_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train_scaled),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val_scaled),
    ),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    ),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# 7. DEFINE THE LSTM
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

criterion = nn.HuberLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5,
)

print("\n========== MODEL ==========")
print(model)


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def inverse_target_scale(values):
    return target_scaler.inverse_transform(
        np.asarray(values).reshape(-1, 1)
    ).ravel()


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2": float(r2_score(y_true, y_pred)),
    }


def predict_model(loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)

            predictions.append(
                model(X_batch).cpu().numpy()
            )

    return np.concatenate(predictions)


# ============================================================
# 9. TRAIN WITH EARLY STOPPING
# ============================================================

best_val_mae = float("inf")
best_state = None
epochs_without_improvement = 0

best_model_path = MODEL_DIR / "railway_lstm_best.pt"

print("\n========== TRAINING ==========")

for epoch in range(1, MAX_EPOCHS + 1):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(predictions, y_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = len(X_batch)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = total_loss / total_samples

    val_predictions_scaled = predict_model(val_loader)

    val_predictions = inverse_target_scale(
        val_predictions_scaled
    )

    val_mae = mean_absolute_error(
        y_val,
        val_predictions,
    )

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train Huber Loss: {average_loss:.5f} | "
        f"Validation MAE: {val_mae:.3f} min"
    )

    if val_mae < best_val_mae:

        best_val_mae = val_mae

        best_state = copy.deepcopy(model.state_dict())

        torch.save(
            best_state,
            best_model_path,
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping after epoch {epoch}.")
        break


if best_state is None:
    raise RuntimeError("No valid model checkpoint was produced.")

model.load_state_dict(best_state)
model.eval()

print(f"\nBest validation MAE: {best_val_mae:.3f} minutes")


# ============================================================
# 10. PREDICT THE HELD-OUT TEST SET
# ============================================================

test_predictions_scaled = predict_model(test_loader)

lstm_predictions = inverse_target_scale(
    test_predictions_scaled
)

lstm_metrics = calculate_metrics(
    y_test,
    lstm_predictions,
)


# ============================================================
# 11. BASELINE 1: TRAINING MEDIAN
# ============================================================

median_predictions = np.full(
    len(y_test),
    np.median(y_train),
    dtype=np.float32,
)

median_metrics = calculate_metrics(
    y_test,
    median_predictions,
)


# ============================================================
# 12. BASELINE 2: DELAY PERSISTENCE
# ============================================================

# Predict that the most recently observed delay persists
# unchanged to the target station.

persistence_predictions = test_df[
    "delay_t1"
].to_numpy(dtype=np.float32)

persistence_metrics = calculate_metrics(
    y_test,
    persistence_predictions,
)


# ============================================================
# 13. BASELINE 3: RIDGE REGRESSION
# ============================================================

ridge = Ridge(alpha=1.0)

ridge.fit(
    X_train.reshape(len(X_train), -1),
    y_train,
)

ridge_predictions = ridge.predict(
    X_test.reshape(len(X_test), -1)
)

ridge_metrics = calculate_metrics(
    y_test,
    ridge_predictions,
)


# ============================================================
# 14. SAVE COMPARISON
# ============================================================

results = pd.DataFrame(
    [
        {
            "Model": "Training median baseline",
            **median_metrics,
        },
        {
            "Model": "Delay persistence",
            **persistence_metrics,
        },
        {
            "Model": "Ridge regression",
            **ridge_metrics,
        },
        {
            "Model": "LSTM",
            **lstm_metrics,
        },
    ]
)

results_path = OUTPUT_DIR / "lstm_model_comparison.csv"

results.to_csv(
    results_path,
    index=False,
)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Test samples: {len(y_test):,}")
print(
    f"Test period: {test_df['service_date'].min()} "
    f"to {test_df['service_date'].max()}"
)

print("\nLower MAE/RMSE is better; higher R² is better.\n")

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print(f"\nResults saved to: {results_path}")
print(f"Model saved to:   {best_model_path}")
print("\nExperiment completed.")