# src/evaluation/eval_rf_by_source.py

import pandas as pd
from pathlib import Path
import joblib

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.features.build_basic_features import load_processed_df, build_X_y


MODEL_FILES = {
    "rf_baseline": "rf_bot_iot_baseline.joblib",
    "rf_smote": "rf_bot_iot_smote.joblib",
    "rf_undersampled": "rf_bot_iot_undersampled.joblib",
}


def eval_model_on_splits(model_name: str, model_path: Path, df: pd.DataFrame):
    print("=" * 80)
    print(f"[eval_rf_by_source] Evaluating model: {model_name}")
    print(f"[eval_rf_by_source] Loading model from: {model_path}")
    clf = joblib.load(model_path)

    # Build features from the SAME processed CSV
    X, y = build_X_y(df)
    print(f"[eval_rf_by_source] Feature matrix shape: {X.shape}, labels shape: {y.shape}")

    # 1) Overall performance on the full merged dataset
    print("\n[eval_rf_by_source] === Overall (Bot-IoT + N-BaIoT) ===")
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"[eval_rf_by_source] Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nClassification Report:")
    print(classification_report(y, y_pred, digits=4))

    # 2) Per-dataset_source performance
    if "dataset_source" not in df.columns:
        print("\n[eval_rf_by_source] WARNING: 'dataset_source' column not found. Skipping per-source evaluation.")
        return

    for src, df_src in df.groupby("dataset_source"):
        print("\n" + "-" * 80)
        print(f"[eval_rf_by_source] Dataset source: {src}")
        idx = df_src.index

        X_src = X.loc[idx]
        y_src = y.loc[idx]

        print(f"[eval_rf_by_source]   X_src shape: {X_src.shape}, y_src shape: {y_src.shape}")
        y_pred_src = clf.predict(X_src)

        acc_src = accuracy_score(y_src, y_pred_src)
        print(f"[eval_rf_by_source]   Accuracy: {acc_src:.4f}")
        print("  Confusion Matrix:")
        print(confusion_matrix(y_src, y_pred_src))
        print("\n  Classification Report:")
        print(classification_report(y_src, y_pred_src, digits=4))


def main():
    processed_name = "merged_placeholder.csv"
    path = PROCESSED_DATA_DIR / processed_name
    print(f"[eval_rf_by_source] Loading processed dataset from: {path}")
    df = load_processed_df(processed_name)

    print(f"[eval_rf_by_source] Data shape: {df.shape}")
    print(f"[eval_rf_by_source] Columns: {list(df.columns)}\n")

    for key, fname in MODEL_FILES.items():
        model_path = MODELS_DIR / fname
        if not model_path.exists():
            print(f"[eval_rf_by_source] SKIP {key}: {model_path} not found.")
            continue

        eval_model_on_splits(key, model_path, df)


if __name__ == "__main__":
    main()
