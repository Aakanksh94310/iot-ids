# src/data_prep/load_raw.py

import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR, BOT_IOT_CSV, N_BAIOT_CSV

# This file will eventually contain all functions that load the *raw* datasets.
# Right now it's mostly a skeleton so you have a place to plug real logic
# once your raw CSVs are ready.

def load_bot_iot():
    """
    Loads the Bot-IoT dataset from raw CSV.
    """
    path = BOT_IOT_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Bot-IoT CSV not found at: {path}. "
            "Place the file in data/raw/ or update config.py."
        )

    df = pd.read_csv(path, low_memory=False)  # 👈 added low_memory=False
    print(f"Loaded Bot-IoT with shape: {df.shape}")
    return df



def load_n_baiot(max_rows: int | None = 500_000) -> pd.DataFrame:
    """
    Load the N-BaIoT dataset from raw CSV.

    The full combined n_baiot.csv can be very large and may not fit
    into memory comfortably on a laptop. For this project, we load
    at most `max_rows` rows by default.

    Set max_rows=None if you ever want to try loading the full file.
    """
    path = N_BAIOT_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"N-BaIoT CSV not found at: {path}. "
            "Place the file in data/raw/ or adjust config.py."
        )

    read_kwargs = {"low_memory": False}
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows

    df = pd.read_csv(path, **read_kwargs)
    print(f"Loaded N-BaIoT with shape: {df.shape} (max_rows={max_rows})")
    return df




def debug_list_raw_files():
    """
    Tiny helper so you can quickly see what raw files are in the folder.
    Useful when you're not sure about filenames or extensions.
    """
    files = list(RAW_DATA_DIR.glob("*"))
    print("Files in data/raw/:")
    for f in files:
        print("  -", f.name)


if __name__ == "__main__":
    # Running this directly will just show the raw file list.
    debug_list_raw_files()
