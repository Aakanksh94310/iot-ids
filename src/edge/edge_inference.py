# src/edge/edge_inference.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


def _find_model_path() -> Path:
    """
    Find a RandomForest model for edge inference.

    Priority:
    1. rf_edge_8f.joblib
    2. any joblib with 'rf' or 'random_forest' in the name
    3. first .joblib in models/
    """
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    joblib_files = sorted(MODELS_DIR.glob("*.joblib"))
    if not joblib_files:
        raise FileNotFoundError(
            f"No .joblib model files found in {MODELS_DIR}. "
            "Run src.training.train_rf_edge first."
        )

    # 1) Exact edge model
    for p in joblib_files:
        if p.stem.lower() == "rf_edge_8f":
            return p

    # 2) Any RF-like model
    rf_candidates = [
        p
        for p in joblib_files
        if "rf" in p.stem.lower() or "random_forest" in p.stem.lower()
    ]
    if rf_candidates:
        return rf_candidates[0]

    # 3) Fallback: first joblib
    return joblib_files[0]


def _find_scaler_path() -> Path:
    """
    Find scaler for the edge RF model.

    Priority:
    1. rf_edge_8f_scaler.joblib
    2. any joblib containing both 'rf' and 'scaler'
    3. any joblib containing 'scaler'
    """
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    joblib_files = sorted(MODELS_DIR.glob("*.joblib"))
    if not joblib_files:
        raise FileNotFoundError(
            f"No .joblib files found in {MODELS_DIR}. "
            "Expected a scaler like 'rf_edge_8f_scaler.joblib'."
        )

    # 1) Exact edge scaler
    for p in joblib_files:
        if p.stem.lower() == "rf_edge_8f_scaler":
            return p

    # 2) RF + scaler in name
    rf_scaler_candidates = [
        p
        for p in joblib_files
        if "rf" in p.stem.lower() and "scaler" in p.stem.lower()
    ]
    if rf_scaler_candidates:
        return rf_scaler_candidates[0]

    # 3) Any scaler
    scaler_candidates = [p for p in joblib_files if "scaler" in p.stem.lower()]
    if scaler_candidates:
        return scaler_candidates[0]

    raise FileNotFoundError(
        f"No scaler .joblib file found in {MODELS_DIR}. "
        "Expected something like 'rf_edge_8f_scaler.joblib'."
    )


RF_MODEL_PATH = _find_model_path()
SCALER_PATH = _find_scaler_path()

print(f"[edge_inference] Using model:  {RF_MODEL_PATH.name}")
print(f"[edge_inference] Using scaler: {SCALER_PATH.name}")

_rf_model = joblib.load(RF_MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)

print(f"[edge_inference] RF expects {_rf_model.n_features_in_} features.")

# ---------------------------------------------------------------------
# Feature order (must match training!)
# ---------------------------------------------------------------------

FEATURE_ORDER: List[str] = [
    "packet_count",
    "byte_count",
    "flow_duration",
    "avg_packet_size",
    "pkt_rate",
    "byte_rate",
    "tcp_flag_syn",
    "tcp_flag_ack",
]

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def preprocess_feature_dict(feature_dict: Dict[str, float]) -> np.ndarray:
    """
    Convert a Python dict of features into a scaled numpy array of shape (1, 8).

    Expects keys exactly matching FEATURE_ORDER.
    """
    try:
        x = np.array([[float(feature_dict[f]) for f in FEATURE_ORDER]], dtype=float)
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(f"Missing feature '{missing}' in feature_dict") from e

    x_scaled = _scaler.transform(x)
    return x_scaled


def predict_one(feature_dict: Dict[str, float]) -> int:
    """
    Predict a single flow as 0 (normal) or 1 (attack).

    Parameters
    ----------
    feature_dict : dict
        Dictionary with keys = FEATURE_ORDER, values = numeric.

    Returns
    -------
    int
        0 for normal, 1 for attack.
    """
    x_scaled = preprocess_feature_dict(feature_dict)
    y_pred = _rf_model.predict(x_scaled)[0]
    return int(y_pred)


def predict_batch(feature_dicts: List[Dict[str, float]]) -> np.ndarray:
    """
    Predict a batch of flows.

    Parameters
    ----------
    feature_dicts : list of dict
        Each dict has keys = FEATURE_ORDER.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) with 0/1 predictions.
    """
    if not feature_dicts:
        return np.array([], dtype=int)

    X = np.array(
        [[float(d[f]) for f in FEATURE_ORDER] for d in feature_dicts],
        dtype=float,
    )
    X_scaled = _scaler.transform(X)
    y_pred = _rf_model.predict(X_scaled)
    return y_pred.astype(int)
