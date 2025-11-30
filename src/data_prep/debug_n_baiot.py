# src/data_prep/debug_n_baiot.py

import pandas as pd
from src.config import N_BAIOT_CSV


def debug_n_baiot(nrows: int = 20) -> None:
    """
    Memory-safe inspector for the N-BaIoT CSV.

    Reads only the first `nrows` rows so you don't blow up RAM.
    Prints:
      - path
      - head(nrows)
      - column names
      - dtypes
    """
    print(f"[debug_n_baiot] Expecting file at: {N_BAIOT_CSV}")

    if not N_BAIOT_CSV.exists():
        print(
            "[debug_n_baiot] ERROR: N-BaIoT CSV not found at:",
            N_BAIOT_CSV,
            "\nPlace the file in data/raw/ as 'n_baiot.csv' or update config.py.",
        )
        return

    print(f"[debug_n_baiot] Reading only the first {nrows} rows for inspection...")
    df = pd.read_csv(N_BAIOT_CSV, nrows=nrows)

    print(f"\n[debug_n_baiot] Sample shape: {df.shape}")

    print(f"\n[debug_n_baiot] Head ({nrows} rows):")
    print(df.head(nrows))

    print("\n[debug_n_baiot] Columns:")
    print(list(df.columns))

    print("\n[debug_n_baiot] dtypes:")
    print(df.dtypes)


if __name__ == "__main__":
    debug_n_baiot()
