# src/edge/edge_benchmark.py

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pandas as pd
import psutil

from .edge_inference import FEATURE_ORDER, predict_one

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "processed" / "merged_placeholder.csv"
METRICS_DIR = ROOT / "data" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
EDGE_METRICS_PATH = METRICS_DIR / "edge_benchmark.csv"

# How many samples to test for benchmarking
N_SAMPLES = 5000  # adjust if needed


def build_feature_dict(row) -> dict:
    """
    Build a feature_dict from a pandas Series row using FEATURE_ORDER.
    """
    return {f: float(row[f]) for f in FEATURE_ORDER}


def main() -> None:
    print(f"[edge_benchmark] Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # sample only labeled flows (0/1)
    if "label" in df.columns:
        df = df[df["label"].isin([0, 1])]

    df_sample = df.sample(n=min(N_SAMPLES, len(df)), random_state=42)

    print(f"[edge_benchmark] Using {len(df_sample)} samples for benchmarking.")
    process = psutil.Process()

    # Warm-up CPU measurement
    _ = process.cpu_percent(interval=None)

    latencies_ms = []

    t_start = time.perf_counter()

    for _, row in df_sample.iterrows():
        features = build_feature_dict(row)

        t0 = time.perf_counter()
        _ = predict_one(features)
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

    t_end = time.perf_counter()
    total_time_s = t_end - t_start

    # After workload, measure CPU% over 1 second
    cpu_percent = process.cpu_percent(interval=1.0)
    mem_mb = process.memory_info().rss / (1024 * 1024)

    # Stats
    avg_lat = statistics.mean(latencies_ms)
    p95_lat = statistics.quantiles(latencies_ms, n=20)[18]  # ~95th percentile
    max_lat = max(latencies_ms)
    throughput = len(latencies_ms) / total_time_s

    print("\n========== Edge Benchmark Results ==========")
    print(f"Total samples:        {len(latencies_ms)}")
    print(f"Total time:           {total_time_s:.3f} s")
    print(f"Throughput:           {throughput:.2f} predictions / second\n")

    print(f"Average latency:      {avg_lat:.4f} ms")
    print(f"95th percentile:      {p95_lat:.4f} ms")
    print(f"Max latency:          {max_lat:.4f} ms\n")

    print(f"Process CPU usage:    {cpu_percent:.2f} %")
    print(f"Process RAM usage:    {mem_mb:.2f} MB")
    print("============================================")

    # -----------------------------------------------------------------
    # Save to CSV for dashboard
    # -----------------------------------------------------------------
    row = {
        "environment": "Laptop (unconstrained RF edge, 8 features)",  # edit label if you want
        "total_samples": len(latencies_ms),
        "total_time_s": total_time_s,
        "throughput_preds_per_s": throughput,
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat,
        "max_latency_ms": max_lat,
        "cpu_percent": cpu_percent,
        "ram_mb": mem_mb,
    }

    df_metrics = pd.DataFrame([row])
    df_metrics.to_csv(EDGE_METRICS_PATH, index=False)
    print(f"[edge_benchmark] Saved edge metrics to: {EDGE_METRICS_PATH}")


if __name__ == "__main__":
    main()
