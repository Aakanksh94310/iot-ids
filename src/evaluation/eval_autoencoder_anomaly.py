import numpy as np
import pandas as pd
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --------------------------------------------------
# Config
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "merged_placeholder.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "autoencoder_normal_bot_iot.pth"
SCALER_PATH = PROJECT_ROOT / "models" / "autoencoder_normal_scaler.joblib"
ANOMALY_METRICS_CSV = PROJECT_ROOT / "data" / "metrics" / "anomaly_metrics.csv"

BATCH_SIZE = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_SOURCE = "Bot-IoT+N-BaIoT_reduced"
LABEL_COLUMN = "label"


# --------------------------------------------------
# Load YOUR Autoencoder class from the training file
# --------------------------------------------------
# (This is the correct path because the file you showed is here:)
#   src/training/train_autoencoder_normal.py
from src.training.train_autoencoder_normal import SimpleAutoencoder


# --------------------------------------------------
# Metric helper
# --------------------------------------------------
def evaluate_and_log(model_name: str, dataset_source: str, y_true, y_pred, rows_list: list):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    rows_list.append({
        "model": model_name,
        "dataset": dataset_source,
        "accuracy": acc,
        "precision_0": prec[0],
        "recall_0": rec[0],
        "f1_0": f1[0],
        "precision_1": prec[1],
        "recall_1": rec[1],
        "f1_1": f1[1],
    })


# --------------------------------------------------
# Reconstruction error helper
# --------------------------------------------------
def compute_reconstruction_errors(model, X, batch_size=BATCH_SIZE, device=DEVICE):
    model.eval()
    model.to(device)

    X_tensor = torch.from_numpy(X.astype(np.float32))
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            recon = model(batch_X)
            mse = torch.mean((recon - batch_X) ** 2, dim=1)
            errors.append(mse.cpu().numpy())

    return np.concatenate(errors, axis=0)


def main():
    # --------------------------------------------------
    # Load processed dataset
    # --------------------------------------------------
    df = pd.read_csv(PROCESSED_CSV)
    if LABEL_COLUMN not in df.columns:
        raise KeyError(f"Missing label column: {LABEL_COLUMN}")

    feature_df = df.drop(columns=[LABEL_COLUMN])
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns
    X = feature_df[numeric_cols].values.astype(np.float32)
    y = df[LABEL_COLUMN].astype(int).values

    print(f"[eval_autoencoder] Using numeric features: {list(numeric_cols)}")
    print(f"[eval_autoencoder] X shape = {X.shape}, y shape = {y.shape}")
    print("[eval_autoencoder] Label distribution:")
    print(pd.Series(y).value_counts())

    # --------------------------------------------------
    # Load scaler
    # --------------------------------------------------
    print(f"[eval_autoencoder] Loading scaler: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)

    # --------------------------------------------------
    # Recreate model architecture + load weights
    # --------------------------------------------------
    input_dim = X_scaled.shape[1]
    model = SimpleAutoencoder(input_dim=input_dim).to(DEVICE)

    print(f"[eval_autoencoder] Loading state_dict: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # --------------------------------------------------
    # Compute reconstruction errors
    # --------------------------------------------------
    print("[eval_autoencoder] Computing reconstruction errors...")
    recon_errors = compute_reconstruction_errors(model, X_scaled)
    print(f"  min={recon_errors.min():.6f}, max={recon_errors.max():.6f}, mean={recon_errors.mean():.6f}")

    # --------------------------------------------------
    # Threshold from NORMAL samples
    # --------------------------------------------------
    normal_errors = recon_errors[y == 0]
    threshold = float(np.quantile(normal_errors, 0.99))
    print(f"[eval_autoencoder] Threshold (99th percentile normal): {threshold:.6f}")

    # --------------------------------------------------
    # Predict anomalies
    # --------------------------------------------------
    y_pred = (recon_errors > threshold).astype(int)

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------
    metrics_rows = []
    model_name = f"Autoencoder_ReconError_thr_{threshold:.4f}"

    evaluate_and_log(model_name, DATASET_SOURCE, y, y_pred, metrics_rows)

    metrics_df = pd.DataFrame(metrics_rows)
    ANOMALY_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(ANOMALY_METRICS_CSV, index=False)

    print(f"[eval_autoencoder] Saved anomaly metrics → {ANOMALY_METRICS_CSV}")
    print(metrics_df)


if __name__ == "__main__":
    main()
