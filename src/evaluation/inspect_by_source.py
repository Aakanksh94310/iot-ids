# src/evaluation/inspect_by_source.py

import pandas as pd
from pathlib import Path

from src.config import PROCESSED_DATA_DIR


def inspect_by_source(processed_name: str = "merged_placeholder.csv") -> None:
    path = PROCESSED_DATA_DIR / processed_name
    print(f"[inspect_by_source] Loading processed dataset from: {path}\n")

    df = pd.read_csv(path)

    # Basic shape
    print(f"[inspect_by_source] Shape: {df.shape}")
    print(f"[inspect_by_source] Columns: {list(df.columns)}\n")

    # Overall label distribution
    print("[inspect_by_source] Overall label counts:")
    print(df["label"].value_counts().sort_index())
    print("\n[inspect_by_source] Overall label proportions (%):")
    print((df["label"].value_counts(normalize=True).sort_index() * 100).round(4))
    print()

    # Dataset source distribution
    print("[inspect_by_source] dataset_source counts:")
    print(df["dataset_source"].value_counts())
    print()

    # Label distribution per dataset_source
    print("[inspect_by_source] Label counts by dataset_source:")
    by_src = df.groupby("dataset_source")["label"].value_counts().unstack(fill_value=0)
    print(by_src)
    print("\n[inspect_by_source] Label proportions (%) by dataset_source:")
    by_src_prop = df.groupby("dataset_source")["label"].value_counts(normalize=True).unstack(fill_value=0) * 100
    print(by_src_prop.round(4))

    # Simple sanity check: attack/normal ratio per source
    print("\n[inspect_by_source] Attack/Normal ratio (1 vs 0) per dataset_source:")
    for src, grp in df.groupby("dataset_source"):
        counts = grp["label"].value_counts()
        normal = counts.get(0, 0)
        attack = counts.get(1, 0)
        if normal == 0:
            ratio = float("inf")
        else:
            ratio = attack / normal
        print(f"  {src}: {attack} / {normal} = {ratio:.2f}x")


if __name__ == "__main__":
    inspect_by_source()
