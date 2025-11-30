# src/training/train_rf_undersampled.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, LABEL_COLUMN
from src.features.build_basic_features import build_X_y, load_processed_df


def undersample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Undersamples the attack class to match the normal class count.
    """
    normal_df = df[df[LABEL_COLUMN] == 0]
    attack_df = df[df[LABEL_COLUMN] == 1]

    normal_count = len(normal_df)
    print(f"[undersample] Normal samples: {normal_count}")
    print(f"[undersample] Attack samples: {len(attack_df)}")

    # Random sample attacks to match normal class count
    attack_df_sampled = attack_df.sample(n=normal_count, random_state=42)

    balanced = pd.concat([normal_df, attack_df_sampled], ignore_index=True)
    print(f"[undersample] Balanced dataset shape: {balanced.shape}")

    return balanced


def train_rf_undersampled():
    # Step 1: Load processed data
    df = load_processed_df("merged_placeholder.csv")

    # Step 2: Build X, y for full dataset (for evaluation later)
    X_full, y_full = build_X_y(df)

    # Step 3: Create balanced training dataset
    balanced_df = undersample(df)

    # Step 4: Build X, y for balanced training
    X_bal, y_bal = build_X_y(balanced_df)

    # Step 5: Train-test split (train only on balanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    print(f"[train_rf_undersampled] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Step 6: Train model
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )

    print("[train_rf_undersampled] Training RandomForest on BALANCED data...")
    clf.fit(X_train, y_train)

    # Step 7: Evaluate on full dataset
    y_pred_full = clf.predict(X_full)

    acc = accuracy_score(y_full, y_pred_full)
    print(f"\n[train_rf_undersampled] Accuracy (on FULL set): {acc:.4f}\n")

    print("Confusion Matrix (FULL set):")
    print(confusion_matrix(y_full, y_pred_full))

    print("\nClassification Report (FULL set):")
    print(classification_report(y_full, y_pred_full, digits=4))

    # Step 8: Save the model
    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / "rf_bot_iot_undersampled.joblib"
    joblib.dump(clf, out_path)
    print(f"\n[train_rf_undersampled] Saved model to: {out_path}")


if __name__ == "__main__":
    train_rf_undersampled()
