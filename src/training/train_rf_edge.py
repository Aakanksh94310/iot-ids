# src/training/train_rf_edge.py

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "merged_placeholder.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# 8 numeric features for edge
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


def main() -> None:
    print(f"[train_rf_edge] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # drop rows with missing values in features/label
    df = df.dropna(subset=FEATURE_COLS + ["label"])

    X = df[FEATURE_COLS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    print("[train_rf_edge] Training RF edge model on 8 features...")
    pipe.fit(X_train, y_train)

    print("[train_rf_edge] Evaluation on held-out test set:")
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    scaler = pipe.named_steps["scaler"]
    rf = pipe.named_steps["rf"]

    model_path = MODELS_DIR / "rf_edge_8f.joblib"
    scaler_path = MODELS_DIR / "rf_edge_8f_scaler.joblib"

    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"[train_rf_edge] Saved RF edge model to: {model_path}")
    print(f"[train_rf_edge] Saved scaler to:       {scaler_path}")


if __name__ == "__main__":
    main()
