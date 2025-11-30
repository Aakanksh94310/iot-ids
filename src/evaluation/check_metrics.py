import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "data" / "metrics"

SUP = METRICS_DIR / "supervised_metrics.csv"
ANO = METRICS_DIR / "anomaly_metrics.csv"

def show_table(path, name):
    if path.exists():
        print(f"\n=== {name} FOUND ===")
        try:
            df = pd.read_csv(path)
            print(df)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    else:
        print(f"\n=== {name} NOT FOUND === {path}")

def main():
    print(f"METRICS DIR: {METRICS_DIR}")
    show_table(SUP, "supervised_metrics.csv")
    show_table(ANO, "anomaly_metrics.csv")

if __name__ == "__main__":
    main()
