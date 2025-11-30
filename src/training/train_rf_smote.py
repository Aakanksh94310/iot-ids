# src/training/train_rf_smote.py

import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from imblearn.over_sampling import SMOTE  # <- new

from src.config import MODELS_DIR
from src.features.build_basic_features import load_processed_df, build_X_y


def train_rf_smote():
    # 1. Load processed dataset
    df = load_processed_df("merged_placeholder.csv")

    # 2. Build initial X, y
    X, y = build_X_y(df)

    # 3. Train/test split on ORIGINAL imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[train_rf_smote] Original train shape: {X_train.shape}, test shape: {X_test.shape}")
    print("[train_rf_smote] Original train label distribution:")
    print(y_train.value_counts())

    # 4. Apply SMOTE ONLY on the training data
    #    sampling_strategy=0.1 -> minority (normal) ~ 10% of majority (attack)
    smote = SMOTE(
        sampling_strategy=0.1,
        random_state=42,
    )

    print("\n[train_rf_smote] Applying SMOTE to training data...")
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"[train_rf_smote] Resampled train shape: {X_train_res.shape}")
    print("[train_rf_smote] Resampled train label distribution:")
    print(y_train_res.value_counts())

    # 5. Define model (no class_weight; SMOTE already balances)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )

    print("\n[train_rf_smote] Training RandomForest on SMOTE-resampled data...")
    clf.fit(X_train_res, y_train_res)

    # 6. Evaluate on untouched test set (original distribution)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[train_rf_smote] Accuracy (on TEST set): {acc:.4f}\n")

    print("Confusion Matrix (TEST set):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report (TEST set):")
    print(classification_report(y_test, y_pred, digits=4))

    # 7. Save model
    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / "rf_bot_iot_smote.joblib"
    joblib.dump(clf, out_path)
    print(f"\n[train_rf_smote] Saved model to: {out_path}")


if __name__ == "__main__":
    train_rf_smote()
