# src/features/build_basic_features.py

import pandas as pd
from pathlib import Path
from typing import Tuple
from src.config import PROCESSED_DATA_DIR, LABEL_COLUMN

NUMERIC_COLS = [
    "packet_count",
    "byte_count",
    "flow_duration",
    "avg_packet_size",
    "pkt_rate",
    "byte_rate",
    "tcp_flag_syn",
    "tcp_flag_ack",
]

CATEGORICAL_COLS = ["proto"]


def load_processed_df(name: str = "merged_placeholder.csv") -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / name
    df = pd.read_csv(path)
    print(f"[build_basic_features] Loaded processed DF from {path} with shape {df.shape}")
    return df


def build_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Ensure label exists
    if LABEL_COLUMN not in df.columns:
        raise KeyError(f"Expected label column '{LABEL_COLUMN}' in dataframe")

    # Select numeric features (fill NaNs with 0 for now)
    X_num = df[NUMERIC_COLS].fillna(0)

    # One-hot encode categorical columns
    X_cat = pd.get_dummies(df[CATEGORICAL_COLS].astype(str), drop_first=False)

    # Concatenate
    X = pd.concat([X_num, X_cat], axis=1)

    y = df[LABEL_COLUMN].astype(int)

    print(f"[build_basic_features] Feature matrix shape: {X.shape}, label shape: {y.shape}")
    return X, y


if __name__ == "__main__":
    df = load_processed_df()
    X, y = build_X_y(df)
    print("First 5 rows of X:")
    print(X.head())
    print("Label distribution:")
    print(y.value_counts())
