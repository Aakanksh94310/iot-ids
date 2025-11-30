# src/data_prep/build_n_baiot_from_original.py

import re
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, N_BAIOT_CSV


def build_n_baiot_from_folder(source_dir: str | Path):
    """
    Combine the original N-BaIoT per-device/per-attack CSVs into a single CSV.

    Assumes source_dir has files like:
        1.benign.csv
        1.gafgyt.combo.csv
        1.mirai.scan.csv
        ...
        9.benign.csv
        9.mirai.syn.csv
        ...

    We:
      - Read each CSV
      - Add columns: [device_id, traffic_type, label]
      - Concatenate everything
      - Save to N_BAIOT_CSV (data/raw/n_baiot.csv)
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    all_files = sorted(source_dir.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    print(f"[build_n_baiot] Found {len(all_files)} CSV files in {source_dir}")

    dfs = []

    for csv_path in all_files:
        name = csv_path.name  # e.g. "1.benign.csv", "2.gafgyt.junk.csv"

        # Extract device id and traffic type from filename
        # Pattern: "{device_id}.{traffic}.csv"
        # Where traffic could contain dots (e.g. gafgyt.scan)
        m = re.match(r"^(\d+)\.(.+)\.csv$", name)
        if not m:
            print(f"[build_n_baiot] Skipping unexpected filename: {name}")
            continue

        device_id = int(m.group(1))
        traffic_type = m.group(2)  # e.g. "benign", "gafgyt.junk", "mirai.scan"

        # Define label: 0 = benign, 1 = attack
        label = 0 if "benign" in traffic_type.lower() else 1

        print(f"[build_n_baiot] Reading {name}: device_id={device_id}, traffic_type={traffic_type}, label={label}")

        df = pd.read_csv(csv_path)

        # Add metadata/label columns
        df["device_id"] = device_id
        df["traffic_type"] = traffic_type
        df["label"] = label

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid CSVs were processed; check filenames.")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"[build_n_baiot] Combined shape: {full_df.shape}")

    # Ensure raw dir exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save to canonical raw path
    full_df.to_csv(N_BAIOT_CSV, index=False)
    print(f"[build_n_baiot] Saved combined N-BaIoT to: {N_BAIOT_CSV}")


if __name__ == "__main__":
    # Your extracted dataset location:
    src_dir = r"D:\MASTERS\CIS600-IOT INTERNET\Project\iot-ids\data\N-BaIoT\dataset"
    build_n_baiot_from_folder(src_dir)
