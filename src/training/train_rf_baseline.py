# src/training/train_rf_baseline.py

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from src.config import MODELS_DIR
from src.features.build_basic_features import load_processed_df, build_X_y


def train_rf_baseline():
    # 1. Load processed dataset
    df = load_processed_df("merged_placeholder.csv")

    # 2. Build X, y
    X, y = build_X_y(df)

    # 3. Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[train_rf_baseline] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Define model
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",  # Bot-IoT is usually imbalanced
    )

    # 5. Train
    print("[train_rf_baseline] Training RandomForest...")
    clf.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[train_rf_baseline] Accuracy: {acc:.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # 7. Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "rf_bot_iot_baseline.joblib"
    joblib.dump(clf, model_path)
    print(f"\n[train_rf_baseline] Saved model to: {model_path}")


if __name__ == "__main__":
    train_rf_baseline()
