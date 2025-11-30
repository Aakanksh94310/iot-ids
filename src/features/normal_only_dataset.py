# src/features/normal_only_dataset.py

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import LABEL_COLUMN
from src.features.build_basic_features import load_processed_df, build_X_y


def build_normal_only_dataset(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Builds a dataset containing ONLY normal flows (label == 0),
    with the same feature engineering as the supervised models.

    Returns:
        X_train_scaled: np.ndarray
        X_val_scaled: np.ndarray
        scaler: fitted StandardScaler
    """
    # 1) Load full processed dataset and build X, y as usual
    df = load_processed_df("merged_placeholder.csv")
    X, y = build_X_y(df)

    # 2) Filter to normal traffic only (label == 0)
    normal_mask = (y == 0)
    X_normal = X[normal_mask]

    print(f"[build_normal_only_dataset] Full X shape: {X.shape}")
    print(f"[build_normal_only_dataset] Normal-only X shape: {X_normal.shape}")

    # 3) Train/validation split (still ONLY normal)
    X_train, X_val = train_test_split(
        X_normal,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    print(f"[build_normal_only_dataset] Normal train shape: {X_train.shape}")
    print(f"[build_normal_only_dataset] Normal val shape: {X_val.shape}")

    # 4) Scale features (fit on train, apply to both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, scaler


if __name__ == "__main__":
    X_train_scaled, X_val_scaled, scaler = build_normal_only_dataset()
    print("\n[build_normal_only_dataset] Done.")
    print(f"  X_train_scaled shape: {X_train_scaled.shape}")
    print(f"  X_val_scaled shape:   {X_val_scaled.shape}")
