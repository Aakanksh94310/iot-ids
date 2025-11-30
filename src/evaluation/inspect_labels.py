# src/evaluation/inspect_labels.py

from pathlib import Path
import pandas as pd
from src.config import PROCESSED_DATA_DIR, LABEL_COLUMN


def inspect_label_distribution(filename: str = "merged_placeholder.csv") -> None:
    path = PROCESSED_DATA_DIR / filename
    print(f"[inspect_labels] Loading processed dataset from: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}")

    df = pd.read_csv(path)
    if LABEL_COLUMN not in df.columns:
        raise KeyError(f"Expected label column '{LABEL_COLUMN}' in dataframe")

    y = df[LABEL_COLUMN]

    total = len(y)
    counts = y.value_counts().sort_index()
    proportions = counts / total * 100.0

    print("\n[inspect_labels] Label counts:")
    for label, cnt in counts.items():
        print(f"  label={label}: {cnt} samples")

    print("\n[inspect_labels] Label proportions (%):")
    for label, prop in proportions.items():
        print(f"  label={label}: {prop:.4f}%")

    if 0 in counts and 1 in counts:
        ratio = counts[1] / max(counts[0], 1)
        print(f"\n[inspect_labels] Attack/Normal ratio (1 vs 0): {ratio:.2f}x")
    else:
        print("\n[inspect_labels] Could not compute ratio (one of the labels is missing).")


if __name__ == "__main__":
    inspect_label_distribution()
