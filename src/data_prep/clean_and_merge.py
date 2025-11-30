# src/data_prep/clean_and_merge.py

import pandas as pd
from pathlib import Path
from src.config import PROCESSED_DATA_DIR
from src.data_prep.load_raw import load_bot_iot, load_n_baiot

# This file will become the "heart" of your preprocessing layer.
# For now, we only outline the functions, the stages, and the
# general idea of a unified feature schema. We'll fill the real
# cleaning logic later (after we inspect the raw datasets).


# ---------------- UNIFIED FEATURE SCHEMA -----------------
# These are the FEATURES we want across ALL datasets.
# Later, when we load Bot-IoT or N-BaIoT, we will manually
# map their raw column names to these standardized names.

UNIFIED_SCHEMA = [
    "packet_count",
    "byte_count",
    "flow_duration",
    "avg_packet_size",
    "pkt_rate",
    "byte_rate",
    "proto",
    "tcp_flag_syn",
    "tcp_flag_ack",
    # label is technically target, but we keep it around
    "label",
]


# ------------------- PLACEHOLDER FUNCTIONS --------------------

def standardize_bot_iot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Bot-IoT raw dataframe into our UNIFIED_SCHEMA + label.
    """
    print("Standardizing Bot-IoT...")

    out = pd.DataFrame()

    # Ensure numeric types where needed
    df_num = df.copy()

    # Safe numeric conversions (errors='coerce' -> NaN -> fillna(0))
    for col in ["pkts", "bytes", "dur", "rate"]:
        if col in df_num.columns:
            df_num[col] = pd.to_numeric(df_num[col], errors="coerce")

    # 1) Core traffic stats
    out["packet_count"] = df_num["pkts"].fillna(0)
    out["byte_count"] = df_num["bytes"].fillna(0)
    out["flow_duration"] = df_num["dur"].fillna(0)

    # 2) Derived stats
    pc = out["packet_count"].replace(0, pd.NA)
    out["avg_packet_size"] = (out["byte_count"] / pc).fillna(0)

    out["pkt_rate"] = df_num["rate"].fillna(0)

    dur_safe = out["flow_duration"].replace(0, pd.NA)
    out["byte_rate"] = (out["byte_count"] / dur_safe).fillna(0)

    # 3) Protocol (keep as string/categorical)
    out["proto"] = df["proto"].astype(str)

    # 4) TCP flags -> simple SYN/ACK indicators
    flgs = df["flgs"].astype(str)
    out["tcp_flag_syn"] = flgs.str.contains("S").astype(int)
    out["tcp_flag_ack"] = flgs.str.contains("A").astype(int)

    # 5) Label: binary attack flag from 'attack' column
    if "attack" in df.columns:
        out["label"] = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(int)
    else:
        out["label"] = 0

    # 6) Dataset source tag
    out["dataset_source"] = "bot_iot"

    print(f"Bot-IoT standardized shape: {out.shape}")
    return out


def standardize_n_baiot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize N-BaIoT to the same UNIFIED_SCHEMA used for Bot-IoT.
    """
    print("Standardizing N-BaIoT...")

    out = pd.DataFrame(index=df.index)

    # Use mutual information / entropy style features as proxies
    out["packet_count"] = df["MI_dir_L5_weight"]
    out["byte_count"] = df["MI_dir_L5_mean"]
    out["flow_duration"] = df["MI_dir_L5_variance"]

    # Use other mid-level stats as "avg size" and "rates"
    out["avg_packet_size"] = df["MI_dir_L3_mean"]
    out["pkt_rate"] = df["HH_L1_mean"]
    out["byte_rate"] = df["HpHp_L1_mean"]

    # N-BaIoT has no protocol flags the way Bot-IoT does.
    out["proto"] = "n_baiot"
    out["tcp_flag_syn"] = 0
    out["tcp_flag_ack"] = 0

    # Label already created in build_n_baiot_from_original (0 benign / 1 attack)
    out["label"] = df["label"].astype(int)

    # Dataset source tag
    out["dataset_source"] = "n_baiot"

    print(f"N-BaIoT standardized shape: {out.shape}")
    return out


def merge_datasets(bot_iot_df: pd.DataFrame, n_baiot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines Bot-IoT, N-BaIoT, and later your own captured traffic
    into ONE consistent dataframe.

    This will make the ML pipeline much simpler.
    For now we only concatenate because we haven't standardized yet.
    """
    print("Merging datasets (placeholder)...")
    merged = pd.concat([bot_iot_df, n_baiot_df], ignore_index=True)
    print(f"Merged dataset shape: {merged.shape}")
    return merged


def save_processed(df: pd.DataFrame, name="processed_placeholder.csv"):
    """
    Saves the processed dataframe to data/processed/ for inspection.
    """
    path = PROCESSED_DATA_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved processed dataset at: {path}")


# ---------------------- DEBUG ENTRYPOINT ------------------------

if __name__ == "__main__":
    print("Running clean_and_merge placeholder...")

    # For now: just try loading the raw datasets.
    # They won't exist yet, so it's safe and expected to comment these out
    # until you place the actual CSV files in data/raw/.

    try:
        bot = load_bot_iot()
        bot_std = standardize_bot_iot(bot)
    except FileNotFoundError:
        bot_std = pd.DataFrame()
        print("Bot-IoT not found yet.")

    try:
        nb = load_n_baiot()
        nb_std = standardize_n_baiot(nb)
    except FileNotFoundError:
        nb_std = pd.DataFrame()
        print("N-BaIoT not found yet.")

    # Merge (empty for now)
    merged = merge_datasets(bot_std, nb_std)

    # Save placeholder
    save_processed(merged, "merged_placeholder.csv")
