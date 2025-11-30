import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "merged_placeholder.csv"
ANOMALY_METRICS_CSV = PROJECT_ROOT / "data" / "metrics" / "anomaly_metrics.csv"

DATASET_SOURCE = "Bot-IoT+N-BaIoT_reduced"
LABEL_COLUMN = "label"

FEATURE_COLS = [
    "packet_count",
    "byte_count",
    "flow_duration",
    "avg_packet_size",
    "pkt_rate",
    "byte_rate",
    "tcp_flag_syn",
    "tcp_flag_ack",
]


def evaluate_and_log(model_name, dataset_source, y_true, y_pred, rows_list):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    rows_list.append(
        {
            "model": model_name,
            "dataset": dataset_source,
            "accuracy": acc,
            "precision_0": prec[0],
            "recall_0": rec[0],
            "f1_0": f1[0],
            "precision_1": prec[1],
            "recall_1": rec[1],
            "f1_1": f1[1],
        }
    )


def main():
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(f"Processed CSV not found at: {PROCESSED_CSV}")

    df = pd.read_csv(PROCESSED_CSV)

    if LABEL_COLUMN not in df.columns:
        raise KeyError(
            f"Expected label column '{LABEL_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    for col in FEATURE_COLS:
        if col not in df.columns:
            raise KeyError(f"Expected feature column '{col}' not in {PROCESSED_CSV}")

    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_all = df[LABEL_COLUMN].astype(int).values

    print(f"[eval_isoforest] X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")
    print("[eval_isoforest] Label distribution:")
    print(pd.Series(y_all).value_counts())

    # Normal-only subset for training (label == 0)
    normal_mask = (y_all == 0)
    X_normal = X_all[normal_mask]
    print(f"[eval_isoforest] Normal-only shape: {X_normal.shape}")

    # Scale features (fit on normal, apply to all)
    scaler = StandardScaler()
    scaler.fit(X_normal)
    X_normal_scaled = scaler.transform(X_normal)
    X_all_scaled = scaler.transform(X_all)

    # Isolation Forest trained on normal-only data
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # assume ~5% anomalies in "normal world"
        random_state=42,
        n_jobs=-1,
    )
    print("[eval_isoforest] Fitting IsolationForest on normal-only data...")
    iso.fit(X_normal_scaled)

    # Higher "score" = more normal; we invert to get "anomaly score"
    scores_all = -iso.decision_function(X_all_scaled)

    print("[eval_isoforest] Example scores stats:")
    print(
        f"  min={scores_all.min():.6f}, max={scores_all.max():.6f}, mean={scores_all.mean():.6f}"
    )

    # Choose threshold from normal scores (same logic style as AE)
    scores_normal = scores_all[normal_mask]
    thr = float(np.quantile(scores_normal, 0.99))
    print(f"[eval_isoforest] Threshold (99th percentile normal scores): {thr:.6f}")

    # Anomaly prediction: score > thr => anomaly/attack (1)
    y_pred = (scores_all > thr).astype(int)

    metrics_rows = []
    model_name = f"IsolationForest_scores_thr_{thr:.4f}"

    evaluate_and_log(model_name, DATASET_SOURCE, y_all, y_pred, metrics_rows)

    metrics_df = pd.DataFrame(metrics_rows)

    # Append to anomaly_metrics.csv if it exists, else create
    ANOMALY_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if ANOMALY_METRICS_CSV.exists():
        existing = pd.read_csv(ANOMALY_METRICS_CSV)
        combined = pd.concat([existing, metrics_df], ignore_index=True)
    else:
        combined = metrics_df

    combined.to_csv(ANOMALY_METRICS_CSV, index=False)

    print(f"[eval_isoforest] Saved updated anomaly metrics → {ANOMALY_METRICS_CSV}")
    print(combined)


if __name__ == "__main__":
    main()
