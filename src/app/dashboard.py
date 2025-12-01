# src/app/dashboard.py

import sys
from pathlib import Path

# --- Make project root importable when running via "streamlit run" ---
ROOT = Path(__file__).resolve().parents[2]  # .../iot-ids
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------------------------

import pandas as pd
import streamlit as st
import joblib

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.features.build_basic_features import build_X_y

# ---------------------------------------------------------------------
# Feature subsets & helpers for different models
# ---------------------------------------------------------------------

# Edge model feature set (must match train_rf_edge.py)
EDGE_FEATURE_COLS = [
    "packet_count",
    "byte_count",
    "flow_duration",
    "avg_packet_size",
    "pkt_rate",
    "byte_rate",
    "proto",          # encoded as integer inside build_X_y
    "tcp_flag_syn",
]


def get_model_feature_cols(model_name: str, full_feature_cols):
    """
    Decide which feature columns to use for a given model.

    - Edge-style models (e.g. rf_edge_8f) use EDGE_FEATURE_COLS (8 features).
    - All other models default to the full feature set returned by build_X_y().
    """
    # You can refine this condition later if you add more edge-style models.
    if "edge" in model_name.lower():
        return EDGE_FEATURE_COLS

    # Default: use the full feature set
    return full_feature_cols


def get_model_scaler(model_name: str, scalers_dict):
    """
    Try to find a matching scaler object for the given model.

    Convention:
      rf_edge_8f        -> rf_edge_8f_scaler.joblib
      some_model_name   -> some_model_name_scaler.joblib

    If no exact match is found, we fall back to the first scaler whose
    key starts with the model name. If none is found, return None.
    """
    # Exact key with "_scaler" suffix
    key_exact = f"{model_name}_scaler"
    if key_exact in scalers_dict:
        return scalers_dict[key_exact]

    # Fallback: any scaler whose name starts with the model name
    for k, v in scalers_dict.items():
        if k.startswith(model_name):
            return v

    # No scaler found – caller should handle None
    return None

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROCESSED_CSV = PROCESSED_DATA_DIR / "merged_placeholder.csv"

METRICS_DIR = ROOT / "data" / "metrics"
SUPERVISED_METRICS_CSV = METRICS_DIR / "supervised_metrics.csv"
ANOMALY_METRICS_CSV = METRICS_DIR / "anomaly_metrics.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_processed():
    df = pd.read_csv(PROCESSED_CSV)
    return df


@st.cache_data(show_spinner=False)
def load_metrics():
    """
    Load all model metrics from:
      - supervised_metrics.csv  (RF, LightGBM, MLP, CNN, TCN)
      - anomaly_metrics.csv     (Autoencoder, IsolationForest)

    Returns a single DataFrame with an extra 'family' column:
      - 'supervised' or 'anomaly'
    """
    dfs = []

    if SUPERVISED_METRICS_CSV.exists():
        sup = pd.read_csv(SUPERVISED_METRICS_CSV)
        sup["family"] = "supervised"
        dfs.append(sup)

    if ANOMALY_METRICS_CSV.exists():
        ano = pd.read_csv(ANOMALY_METRICS_CSV)
        ano["family"] = "anomaly"
        dfs.append(ano)

    if not dfs:
        return None

    all_metrics = pd.concat(dfs, ignore_index=True)
    return all_metrics


@st.cache_data(show_spinner=False)
def get_feature_columns():
    """Get the exact feature columns used for training the RF models."""
    df = load_processed()
    X, y = build_X_y(df)
    return list(X.columns)


@st.cache_resource(show_spinner=False)
def load_all_models():
    """
    Load all usable .joblib models from MODELS_DIR.

    Only keeps objects that look like classifiers (must have .predict()).
    This automatically picks up RF / LGBM / MLP etc., and skips scalers.
    """
    models = {}
    for p in MODELS_DIR.glob("*.joblib"):
        obj = None
        try:
            obj = joblib.load(p)
        except Exception:
            continue  # skip unreadable files

        # Only keep objects that behave like models (have predict)
        if hasattr(obj, "predict"):
            models[p.stem] = obj
        else:
            # e.g. autoencoder_normal_scaler.joblib -> skipped
            continue

    return models

@st.cache_resource(show_spinner=False)
def load_all_scalers():
    """
    Load all scaler-like .joblib objects from MODELS_DIR.

    We treat anything that has `.transform` but does NOT have `.predict`
    as a scaler (e.g. StandardScaler, MinMaxScaler).

    Examples it will pick up:
      - rf_edge_8f_scaler.joblib
      - autoencoder_normal_scaler.joblib
    """
    scalers = {}
    for p in MODELS_DIR.glob("*.joblib"):
        obj = None
        try:
            obj = joblib.load(p)
        except Exception:
            # unreadable / incompatible object – skip
            continue

        # Heuristic: scaler-like (has transform, no predict)
        if hasattr(obj, "transform") and not hasattr(obj, "predict"):
            scalers[p.stem] = obj

    return scalers


def explain_models():
    st.markdown(
        """
### What models did we train?

**Supervised (classification IDS)**  
These directly predict **label = 0 (normal)** or **label = 1 (attack)**:

1. **Random Forest – SMOTE / Undersampled**  
   - Trained on the unified Bot-IoT + N-BaIoT dataset.  
   - SMOTE variant: synthetically upsamples the minority (normal) class.  
   - Undersampled variant: down-samples attacks to get a balanced training set.

2. **LightGBM – SMOTE / Undersampled**  
   - Gradient-boosted trees optimised for speed and tabular performance.  
   - Same resampling strategies as RF.

3. **MLP – SMOTE / Undersampled**  
   - A shallow feed-forward neural network on the 8 numeric features.  

4. **CNN1D_UNDER**  
   - Treats the 8 features as a 1D sequence.  
   - 1D convolutions learn local patterns across features.

5. **TCN_UNDER**  
   - Temporal Convolutional Network; stacked dilated 1D convolutions.  
   - Captures longer-range dependencies across the feature vector.  
   - In practice, this is your strongest deep classifier among the supervised models.

All supervised models use the **same 8 numeric features** and standard metrics:  
accuracy, precision/recall/F1 for both classes (0 = normal, 1 = attack).

---

**Anomaly-based IDS (unsupervised / semi-supervised)**  

6. **Autoencoder (reconstruction error)**  
   - Trained *only on normal traffic* (label = 0).  
   - At test time, flows with **high reconstruction error** are treated as anomalies.  
   - Threshold chosen as the 99th percentile of reconstruction error on normal-only flows.  
   - Interpreted as: “if the autoencoder struggles to reconstruct this flow, call it suspicious.”

7. **Isolation Forest (anomaly score)**  
   - Also trained on normal-only samples.  
   - Uses random partitioning to “isolate” points that look different from normal flows.  
   - We use the **decision function** as a normality score and invert it; scores above a
     threshold (again chosen from the 99th percentile of normal scores) are flagged as anomalies.
"""
    )


def explain_datasets():
    st.markdown(
        """
### What datasets are we using?

**1. Bot-IoT (reduced sample)**  
- Highly skewed: ~0.2% normal vs 99.8% attack.  
- Contains many raw network fields; we keep only a core subset.

**2. N-BaIoT (aggregated from many device logs)**  
- Still attack-heavy, but less extreme (~10% normal vs 90% attack).  
- Original dataset is split across 90+ CSVs; we build a single `n_baiot.csv` and then
  map it into the same feature schema as Bot-IoT.

**Unified processed dataset**

Both are standardised and merged into:

`data/processed/merged_placeholder.csv`

with columns:

- **Features we use for *all* models (8 numeric + 2 categorical):**
  - `packet_count` – number of packets in the flow.  
  - `byte_count` – total bytes transferred.  
  - `flow_duration` – duration of the flow in time units.  
  - `avg_packet_size` – `byte_count / packet_count` (average bytes per packet).  
  - `pkt_rate` – packets per unit time.  
  - `byte_rate` – bytes per unit time.  
  - `tcp_flag_syn` – 1 if SYN flag is set, else 0.  
  - `tcp_flag_ack` – 1 if ACK flag is set, else 0.  
  - `proto` – protocol (e.g., tcp/udp) as a categorical feature.  
  - `dataset_source` – which dataset the flow came from: `"bot_iot"` or `"n_baiot"`.

- **Columns used as labels / metadata:**
  - `label` – ground truth (0 = normal, 1 = attack).  
  - `dataset_source` – as above, used for filtering and explanation.

**Features we effectively discard**

The original raw datasets contain many additional low-level fields (e.g., device-specific stats,
per-direction counters, some higher-order statistics). For this project we:

- Focus on a **compact, interpretable feature set** that is common to both datasets.  
- Drop columns that:
  - Are not present in both Bot-IoT and N-BaIoT, or  
  - Are strongly redundant with the above rates/counts, or  
  - Make the feature space much higher-dimensional without clear gain for a student project.

This makes the pipeline easier to explain and allows us to compare several models fairly.

---

### How do we classify a flow as an attack?

**Supervised models (RF / LightGBM / MLP / CNN / TCN)**  

1. Each model is trained on labeled data (`label` = 0 or 1).  
2. At inference time, the model outputs either:
   - A hard class (0 or 1), or  
   - A probability `P(attack)` which we threshold at 0.5 by default.
3. If `P(attack) >= 0.5` → **attack (1)**, else → **normal (0)**.  

So supervised models learn a decision boundary in the 8-dimensional feature space that separates normal vs attack flows.

**Anomaly-based models (Autoencoder / Isolation Forest)**  

- They only see **normal (label = 0)** flows during training.  
- At test time, they compute an **anomaly score**:
  - Autoencoder: reconstruction error `‖x - f(x)‖²`.  
  - Isolation Forest: anomaly score from how “isolated” the point is in the forest.  
- For both, we pick a **threshold based on normal samples** (99th percentile).  
  - Score below threshold → treated as **normal**.  
  - Score above threshold → treated as **anomalous/attack**.

This gives you a nice story for **zero-day attacks**: you can still flag unknown behaviours
without having explicit attack labels in training.
"""
    )


def list_project_files():
    st.markdown(
        """
### Key project files (for report / viva)

**Configuration & paths**
- `src/config.py`

**Data preparation**
- `src/data_prep/load_raw.py` – lists / validates files in `data/raw/`.  
- `src/data_prep/clean_and_merge.py` – builds `merged_placeholder.csv` (unified schema).  
- `src/data_prep/build_n_baiot_from_original.py` – aggregates original N-BaIoT CSVs.  
- `src/data_prep/debug_n_baiot.py` – quick sanity checks for the N-BaIoT CSV.

**Feature engineering**
- `src/features/build_basic_features.py` – builds the 8 numeric features + proto/flags.  
- `src/features/normal_only_dataset.py` – extracts and scales normal-only flows for the autoencoder.

**Training scripts**
- `src/training/train_rf_baseline.py`  
- `src/training/train_rf_smote.py`  
- `src/training/train_rf_undersampled.py`  
- `src/training/train_autoencoder_normal.py` – trains the normal-only autoencoder.

**Evaluation & metrics**
- `src/evaluation/inspect_labels.py` – label distribution sanity checks.  
- `src/evaluation/inspect_by_source.py` – label distribution split by dataset_source.  
- `src/evaluation/eval_supervised.py` – RF + LightGBM + MLP (SMOTE / UNDER) metrics.  
- `src/evaluation/eval_autoencoder_anomaly.py` – autoencoder anomaly metrics.  
- `src/evaluation/eval_isolation_forest_anomaly.py` – Isolation Forest anomaly metrics.  
- `src/evaluation/eval_cnn_tcn.py` – CNN1D_UNDER and TCN_UNDER supervised deep models.  
- `src/evaluation/check_metrics.py` – quick view of current metrics files.  

**Dashboard**
- `src/app/dashboard.py`  ← this Streamlit app
"""
    )


# ---------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="IoT IDS – Bot-IoT + N-BaIoT Dashboard",
    layout="wide",
)

st.title("📡 IoT Intrusion Detection Dashboard")
st.caption(
    "Bot-IoT + N-BaIoT · RF / LightGBM / MLP / CNN / TCN (supervised) + "
    "Autoencoder / IsolationForest (anomaly-based)"
)

# Load data / models / feature template / metrics
if not PROCESSED_CSV.exists():
    st.error(f"Processed dataset not found at `{PROCESSED_CSV}`. Run `clean_and_merge.py` first.")
    st.stop()

df = load_processed()
all_metrics = load_metrics()
feature_cols = get_feature_columns()
models = load_all_models()
scalers = load_all_scalers()
FULL_FEATURE_COLS = feature_cols


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.header("Controls")

# Dataset filter (used in Overview, Datasets tab samples, Graphs)
dataset_filter = st.sidebar.selectbox(
    "Filter by dataset source (affects Overview, Dataset Samples, Graphs)",
    options=["All", "bot_iot", "n_baiot"],
    index=0,
    key="sidebar_dataset_filter",   
)

# Model selection for Live Detection (using models dict)
if models:
    model_choice = st.sidebar.selectbox(
        "Select model for Live Detection",
        options=list(models.keys()),
        key="sidebar_live_model_choice",   
    )
else:
    model_choice = st.sidebar.selectbox(
        "Select model for Live Detection",
        options=["(no models found in models/)"],
        index=0,
        key="sidebar_live_model_choice_empty", 
    )

show_raw_preview = st.sidebar.checkbox(
    "Show sample of processed dataset (Overview tab)",
    value=False,
    key="sidebar_show_raw_preview",  
)

# Apply dataset filter
if dataset_filter == "All":
    df_view = df
else:
    df_view = df[df["dataset_source"] == dataset_filter]

# ---------------------------------------------------------------------
# Top KPI cards (based on filtered view)
# ---------------------------------------------------------------------
total_rows = len(df_view)
total_features = df_view.shape[1] - 2  # roughly exclude label + dataset_source
label_counts_view = df_view["label"].value_counts().to_dict()
normal_view = label_counts_view.get(0, 0)
attack_view = label_counts_view.get(1, 0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total flows (filtered)", f"{total_rows:,}")
col2.metric("Normal flows (label=0)", f"{normal_view:,}")
col3.metric("Attack flows (label=1)", f"{attack_view:,}")
attack_ratio = (attack_view / normal_view) if normal_view > 0 else float("inf")
col4.metric("Attack : Normal ratio", f"{attack_ratio:,.2f}x" if normal_view > 0 else "∞")

st.divider()

# ---------------------------------------------------------------------
# Tabs for structure
# ---------------------------------------------------------------------
tab_overview, tab_datasets, tab_models, tab_graphs, tab_live, tab_interpret = st.tabs(
    ["📊 Overview", "🧾 Datasets & Files", "🤖 Model Performance", "📈 Graphs", "🚨 Live Detection", "🧠 Interpretation"]
)

# =========================== OVERVIEW TAB ============================
with tab_overview:
    st.subheader("Project summary")

    st.markdown(
        """
This project builds a **unified Intrusion Detection System (IDS)** for IoT traffic by
combining two popular research datasets:

- **Bot-IoT** – extremely imbalanced but very rich in attacks.  
- **N-BaIoT** – many device types, more moderate imbalance.

We:

1. **Clean and merge** both datasets into a single processed CSV with a shared feature schema.  
2. **Engineer a compact feature set** (counts, rates, TCP flags, protocol) that works for both.  
3. **Train multiple supervised classifiers** (Random Forest, LightGBM, MLP, CNN, TCN).  
4. **Train anomaly detectors** (Autoencoder, Isolation Forest) using only normal traffic.  
5. **Evaluate all models** using accuracy and per-class precision/recall/F1.  
6. Use this dashboard to **visualise the data, compare model metrics, and run live detection demos**.

---

### Label distribution (current filter)

The charts below show how imbalanced the dataset is, even after merging:

- Most flows are **attacks (label = 1)**.  
- Normal traffic (label = 0) is a small but crucial minority.

"""
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Overall label distribution (current filter)**")
        st.bar_chart(df_view["label"].value_counts().sort_index())

    with col_right:
        if "dataset_source" in df.columns:
            st.markdown("**Label counts by dataset_source**")
            counts_by_src = (
                df.groupby(["dataset_source", "label"])
                .size()
                .reset_index(name="count")
                .pivot(index="dataset_source", columns="label", values="count")
                .fillna(0)
            )
            st.dataframe(counts_by_src, use_container_width=True)

    if show_raw_preview:
        st.markdown("### Sample of processed dataset")
        st.dataframe(df_view.head(100), use_container_width=True)

    # Edge deployment metrics (RF edge model, 8 features)

    from pathlib import Path

    EDGE_METRICS_PATH = Path("data/metrics/edge_benchmark.csv")

    st.markdown("---")
    st.markdown("### Edge deployment metrics (RF edge model, 8 features)")

    if EDGE_METRICS_PATH.exists():
        edge_df = pd.read_csv(EDGE_METRICS_PATH)

        # Optional: rename columns for nicer display
        edge_df_display = edge_df.rename(
            columns={
                "environment": "Environment",
                "total_samples": "Total samples",
                "total_time_s": "Total time (s)",
                "throughput_preds_per_s": "Throughput (pred/s)",
                "avg_latency_ms": "Avg latency (ms)",
                "p95_latency_ms": "P95 latency (ms)",
                "max_latency_ms": "Max latency (ms)",
                "cpu_percent": "CPU usage (%)",
                "ram_mb": "RAM usage (MB)",
            }
        )

        st.dataframe(edge_df_display, use_container_width=True)
        st.caption(
            "Metrics computed by `src.edge.edge_benchmark` on the merged dataset. "
            "This simulates running the RF edge model on a constrained edge device."
        )
    else:
        st.info(
            "Edge deployment benchmark not found. "
            "Run `uv run python -m src.edge.edge_benchmark` to generate edge metrics."
        )

# ====================== DATASETS & FILES TAB =========================
with tab_datasets:
    st.subheader("Datasets overview")

    # --- Dataset source counts ---
    if "dataset_source" in df.columns:
        counts_by_src = (
            df.groupby(["dataset_source", "label"])
            .size()
            .reset_index(name="count")
            .pivot(index="dataset_source", columns="label", values="count")
            .fillna(0)
        )
        counts_by_src = counts_by_src.rename(
            columns={
                0: "normal_flows (label=0)",
                1: "attack_flows (label=1)",
            }
        )
        st.markdown("### **Flows per dataset and label**")
        st.dataframe(counts_by_src, use_container_width=True)
    else:
        st.info("`dataset_source` column not found in processed dataset.")

    # --- Show top 20 rows per dataset ---
    st.markdown("---")
    st.subheader("Sample rows from each dataset source (Top 20)")

    if "dataset_source" in df.columns:
        for src in sorted(df["dataset_source"].unique()):
            st.markdown(f"#### **{src} — Top 20 rows**")
            st.dataframe(
                df[df["dataset_source"] == src].head(20),
                use_container_width=True,
            )
    else:
        st.info("Cannot show per-dataset samples — `dataset_source` missing")

    # --- Feature schema + explanation ---
    st.markdown("---")
    st.subheader("Datasets and feature schema")
    explain_datasets()

    # --- Files section ---
    st.markdown("---")
    st.subheader("Important project files")
    list_project_files()

# ===================== MODEL PERFORMANCE TAB =========================
with tab_models:
    st.subheader("Model performance – combined metrics")

    if all_metrics is None:
        st.warning(
            "No metrics CSVs found in `data/metrics/`. "
            "Run the evaluation scripts (eval_supervised, eval_autoencoder_anomaly, "
            "eval_isolation_forest_anomaly, eval_cnn_tcn) first."
        )
    else:
        fam_filter = st.radio(
            "Filter by model family",
            options=["All", "supervised", "anomaly"],
            horizontal=True,
        )

        if fam_filter == "supervised":
            metrics_view = all_metrics[all_metrics["family"] == "supervised"]
        elif fam_filter == "anomaly":
            metrics_view = all_metrics[all_metrics["family"] == "anomaly"]
        else:
            metrics_view = all_metrics

        st.markdown("#### Metrics (per model, full dataset)")
        st.dataframe(metrics_view, use_container_width=True)

        st.caption(
            "Columns: accuracy, precision/recall/F1 for class 0 (normal) and class 1 (attack).  "
            "Family = 'supervised' (RF/LGBM/MLP/CNN/TCN) or 'anomaly' (Autoencoder/IsolationForest)."
        )

    st.markdown("---")
    st.subheader("Models summary")
    explain_models()

# ======================== GRAPHS TAB ================================
with tab_graphs:
    st.subheader("Per-model metrics, confusion & comparisons")

    if all_metrics is None:
        st.warning(
            "No metrics available. Run evaluation scripts to populate `data/metrics/` first."
        )
    else:
        model_list = sorted(all_metrics["model"].unique().tolist())

        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected_model = st.selectbox(
                "Select a primary model to inspect",
                model_list,
                key="graphs_primary_model",
            )
        with col_sel2:
            compare_metric = st.selectbox(
                "Metric for cross-model comparison",
                [
                    "accuracy",
                    "precision_0", "recall_0", "f1_0",
                    "precision_1", "recall_1", "f1_1",
                ],
                index=0,
                key="graphs_compare_metric",
            )

        row = all_metrics[all_metrics["model"] == selected_model].iloc[0]

        st.markdown(f"### Metrics for `{selected_model}`")

        # ---------------- KPI cards ----------------
        acc = float(row["accuracy"])
        f1_normal = float(row["f1_0"])
        f1_attack = float(row["f1_1"])
        macro_f1 = (f1_normal + f1_attack) / 2.0
        recall_attack = float(row["recall_1"])

        k1, k2, k3 = st.columns(3)
        k1.metric("Accuracy", f"{acc:.4f}")
        k2.metric("Macro F1 (0/1 avg)", f"{macro_f1:.4f}")
        k3.metric("Attack recall (label=1)", f"{recall_attack:.4f}")

        st.markdown("---")

        # ---------------- Class-wise metric charts ----------------
        metrics_cols = [
            "accuracy",
            "precision_0",
            "recall_0",
            "f1_0",
            "precision_1",
            "recall_1",
            "f1_1",
        ]
        metric_series = row[metrics_cols]

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Class 0 (Normal) – precision / recall / F1**")
            cls0 = metric_series[["precision_0", "recall_0", "f1_0"]]
            st.bar_chart(cls0.to_frame("value"))

        with col_m2:
            st.markdown("**Class 1 (Attack) – precision / recall / F1**")
            cls1 = metric_series[["precision_1", "recall_1", "f1_1"]]
            st.bar_chart(cls1.to_frame("value"))

        st.markdown("---")

        # ---------------- Approximate confusion matrix ----------------
        total_0 = int((df["label"] == 0).sum())
        total_1 = int((df["label"] == 1).sum())

        rec0 = float(row["recall_0"])
        rec1 = float(row["recall_1"])

        # TN = recall_0 * total_0, TP = recall_1 * total_1
        tn = rec0 * total_0
        tp = rec1 * total_1
        fp = total_0 - tn
        fn = total_1 - tp

        tn_i = int(round(tn))
        tp_i = int(round(tp))
        fp_i = int(round(fp))
        fn_i = int(round(fn))

        st.markdown("### Approximate confusion matrix (on full dataset)")

        # Long-form view (TP/TN/FP/FN)
        cm_df = pd.DataFrame(
            {"count": [tp_i, tn_i, fp_i, fn_i]},
            index=[
                "TP (attack correctly flagged)",
                "TN (normal correctly passed)",
                "FP (normal flagged as attack)",
                "FN (attack missed)",
            ],
        )

        # 2x2 matrix view
        cm_matrix = pd.DataFrame(
            [[tn_i, fp_i],
             [fn_i, tp_i]],
            index=["Actual 0 (normal)", "Actual 1 (attack)"],
            columns=["Pred 0 (normal)", "Pred 1 (attack)"],
        )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Confusion matrix (2×2 view)**")
            st.dataframe(cm_matrix, use_container_width=True)

        with col_c2:
            st.markdown("**TP / TN / FP / FN breakdown**")
            st.bar_chart(cm_df)

        st.caption(
            "Confusion counts are approximated from recall values and global label counts. "
            "Small discrepancies vs exact values may appear due to rounding."
        )

        st.markdown("---")

        # ---------------- Cross-model comparison chart ----------------
        st.markdown(f"### Cross-model comparison – `{compare_metric}`")

        compare_df = (
            all_metrics.set_index("model")[compare_metric]
            .sort_values(ascending=False)
        )
        st.bar_chart(compare_df.to_frame(compare_metric))

        st.caption(
            "This chart compares all models on the selected metric. "
            "Use it to quickly see which model has the best accuracy, F1, or recall."
        )
# ===================== LIVE DETECTION TAB ============================
with tab_live:
    st.subheader("Live Detection – try random values or upload CSV")

    if not models:
        st.error(
            "No usable models found in `models/`. "
            "Make sure you have trained and saved some `.joblib` models."
        )
    else:
        if model_choice not in models:
            st.error("Selected model is not available. Try reloading the page.")
        else:
            model = models[model_choice]
            st.success(f"Using model: **{model_choice}**")

            mode = st.radio(
                "Detection mode",
                ["Manual single flow", "Upload CSV (batch)"],
                horizontal=True,
            )

            proto_options = sorted(df["proto"].astype(str).unique().tolist())
            ds_options = sorted(df["dataset_source"].astype(str).unique().tolist())

            # Get template feature columns from training
            template_cols = feature_cols

            # ---------- Manual single flow ----------
            if mode == "Manual single flow":
                st.markdown(
                    """
            Use this to **play with random feature values** and see if the model flags it as normal or attack.  
            Values are in the **raw feature space** (counts / rates); the app internally applies the
            same preprocessing (encoding + scaling) that was used during training.
            """
                )

                # ------------------------------------------------------------------
                # RANDOM FLOW LOADERS (NORMAL / ATTACK)
                # ------------------------------------------------------------------
                b1, b2 = st.columns(2)

                with b1:
                    if st.button("🎲 Load random NORMAL flow from dataset"):
                        df_normals = df[df["label"] == 0]
                        if not df_normals.empty:
                            row = df_normals.sample(1).iloc[0]
                            st.session_state["manual_packet_count"] = float(row["packet_count"])
                            st.session_state["manual_byte_count"] = float(row["byte_count"])
                            st.session_state["manual_flow_duration"] = float(row["flow_duration"])
                            st.session_state["manual_avg_packet_size"] = float(row["avg_packet_size"])
                            st.session_state["manual_pkt_rate"] = float(row["pkt_rate"])
                            st.session_state["manual_byte_rate"] = float(row["byte_rate"])
                            st.session_state["manual_proto"] = str(row["proto"])
                            st.session_state["manual_tcp_flag_syn"] = bool(row["tcp_flag_syn"])
                            st.session_state["manual_tcp_flag_ack"] = bool(row["tcp_flag_ack"])
                            st.session_state["manual_dataset_source"] = str(row["dataset_source"])
                            st.session_state["manual_true_label"] = 0

                with b2:
                    if st.button("⚠️ Load random ATTACK flow from dataset"):
                        df_attacks = df[df["label"] == 1]
                        if not df_attacks.empty:
                            row = df_attacks.sample(1).iloc[0]
                            st.session_state["manual_packet_count"] = float(row["packet_count"])
                            st.session_state["manual_byte_count"] = float(row["byte_count"])
                            st.session_state["manual_flow_duration"] = float(row["flow_duration"])
                            st.session_state["manual_avg_packet_size"] = float(row["avg_packet_size"])
                            st.session_state["manual_pkt_rate"] = float(row["pkt_rate"])
                            st.session_state["manual_byte_rate"] = float(row["byte_rate"])
                            st.session_state["manual_proto"] = str(row["proto"])
                            st.session_state["manual_tcp_flag_syn"] = bool(row["tcp_flag_syn"])
                            st.session_state["manual_tcp_flag_ack"] = bool(row["tcp_flag_ack"])
                            st.session_state["manual_dataset_source"] = str(row["dataset_source"])
                            st.session_state["manual_true_label"] = 1

                # ------------------------------------------------------------------
                # MANUAL FORM (WIRED TO SESSION STATE)
                # ------------------------------------------------------------------
                with st.form("manual_flow_form"):
                    c1, c2, c3 = st.columns(3)
                    c4, c5, c6 = st.columns(3)
                    c7, c8, c9 = st.columns(3)

                    packet_count = c1.number_input(
                        "packet_count",
                        min_value=0.0,
                        value=float(st.session_state.get("manual_packet_count", df["packet_count"].median())),
                        key="manual_packet_count",
                        step=0.000001,
                        format="%.6f",
                    )
                    byte_count = c2.number_input(
                        "byte_count",
                        min_value=0.0,
                        value=float(st.session_state.get("manual_byte_count", df["byte_count"].median())),
                        key="manual_byte_count",
                        step=0.000001,
                        format="%.6f",
                    )
                    flow_duration = c3.number_input(
                        "flow_duration",
                        min_value=0.0,
                        value=float(st.session_state.get("manual_flow_duration", df["flow_duration"].median())),
                        key="manual_flow_duration",
                        step=0.000001,
                        format="%.6f",
                    )

                    avg_packet_size = c4.number_input(
                        "avg_packet_size",
                        min_value=0.0,
                        value=float(st.session_state.get("manual_avg_packet_size", df["avg_packet_size"].median())),
                        key="manual_avg_packet_size",
                        step=0.000001,
                        format="%.6f",
                    )
                    pkt_rate = c5.number_input(
                        "pkt_rate",
                        min_value=0.0,
                        value=float(st.session_state.get("manual_pkt_rate", df["pkt_rate"].median())),
                        key="manual_pkt_rate",
                        step=0.000001,
                        format="%.6f",
                    )
                    byte_rate = c6.number_input(
                        "byte_rate",
                        min_value=0.0,
                        value=float(st.session_state.get("manual_byte_rate", df["byte_rate"].median())),
                        key="manual_byte_rate",
                        step=0.000001,
                        format="%.6f",
                    )

                    proto = c7.selectbox(
                        "proto",
                        options=proto_options,
                        index=proto_options.index(st.session_state.get("manual_proto", "tcp"))
                        if st.session_state.get("manual_proto", "tcp") in proto_options else 0,
                        key="manual_proto",
                    )
                    tcp_flag_syn = c8.checkbox(
                        "tcp_flag_syn",
                        value=bool(st.session_state.get("manual_tcp_flag_syn", False)),
                        key="manual_tcp_flag_syn",
                    )
                    tcp_flag_ack = c9.checkbox(
                        "tcp_flag_ack",
                        value=bool(st.session_state.get("manual_tcp_flag_ack", False)),
                        key="manual_tcp_flag_ack",
                    )

                    dataset_source = st.selectbox(
                        "dataset_source (which dataset style does this look like?)",
                        options=ds_options,
                        index=ds_options.index(st.session_state.get("manual_dataset_source", "bot_iot"))
                        if st.session_state.get("manual_dataset_source", "bot_iot") in ds_options else 0,
                        key="manual_dataset_source",
                    )

                    submitted = st.form_submit_button("🔍 Detect")

                # ------------------------------------------------------------------
                # PREDICTION BLOCK
                # ------------------------------------------------------------------
                if submitted:
                    manual_row = {
                        "packet_count": packet_count,
                        "byte_count": byte_count,
                        "flow_duration": flow_duration,
                        "avg_packet_size": avg_packet_size,
                        "pkt_rate": pkt_rate,
                        "byte_rate": byte_rate,
                        "proto": str(proto),
                        "tcp_flag_syn": int(tcp_flag_syn),
                        "tcp_flag_ack": int(tcp_flag_ack),
                        "label": 0,
                        "dataset_source": dataset_source,
                    }

                    df_manual = pd.DataFrame([manual_row])

                    # Build X using training pipeline
                    X_manual, _ = build_X_y(df_manual)

                    # 1) Select feature subset
                    model_feature_cols = get_model_feature_cols(model_choice, FULL_FEATURE_COLS)
                    X_manual = X_manual.reindex(columns=model_feature_cols, fill_value=0)

                    # 2) Apply associated scaler
                    scaler = get_model_scaler(model_choice, scalers)
                    if scaler is not None:
                        X_in = scaler.transform(X_manual)
                    else:
                        X_in = X_manual

                    # 3) Predict
                    pred = int(model.predict(X_in)[0])
                    proba = None
                    try:
                        proba = float(model.predict_proba(X_in)[0][1])
                    except Exception:
                        proba = None

                    # Display prediction
                    if pred == 1:
                        st.error("🔴 **Model prediction: ATTACK (label=1)**")
                    else:
                        st.success("🟢 **Model prediction: NORMAL (label=0)**")

                    if proba is not None:
                        st.write(f"Estimated attack probability: **{proba:.4f}**")

                    # Optional: show ground truth if flow was sampled
                    if "manual_true_label" in st.session_state:
                        st.info(f"Ground truth of sampled flow: **{st.session_state['manual_true_label']}**")

                    st.markdown("**Feature vector used:**")
                    st.json(manual_row)


            # ---------- CSV batch mode ----------
            else:
                st.markdown(
                    """
Upload a CSV with rows in the **processed schema**:

Required columns (case-sensitive):
- `packet_count`, `byte_count`, `flow_duration`
- `avg_packet_size`, `pkt_rate`, `byte_rate`
- `proto`, `tcp_flag_syn`, `tcp_flag_ack`

Optional:
- `label` (for comparison)
- `dataset_source` (if missing, you can set a default).
"""
                )

                uploaded = st.file_uploader("Upload CSV", type=["csv"])
                max_rows = 50000

                if uploaded is not None:
                    try:
                        df_new = pd.read_csv(uploaded)
                    except Exception as e:
                        st.error(f"Could not read CSV: {e}")
                        df_new = None

                    if df_new is not None:
                        st.write(f"Uploaded rows: **{len(df_new):,}**")
                        if len(df_new) > max_rows:
                            st.warning(f"File is large; using only first {max_rows:,} rows for prediction.")
                            df_new = df_new.head(max_rows)

                        required_cols = [
                            "packet_count",
                            "byte_count",
                            "flow_duration",
                            "avg_packet_size",
                            "pkt_rate",
                            "byte_rate",
                            "proto",
                            "tcp_flag_syn",
                            "tcp_flag_ack",
                        ]
                        missing = [c for c in required_cols if c not in df_new.columns]

                        if missing:
                            st.error(f"Missing required columns in CSV: {missing}")
                        else:
                            # Ensure label exists (dummy if needed)
                            if "label" not in df_new.columns:
                                df_new["label"] = -1

                            # Ensure dataset_source exists
                            if "dataset_source" not in df_new.columns:
                                default_ds = st.selectbox(
                                    "CSV has no `dataset_source` column – apply this value to all rows:",
                                    options=ds_options,
                                    index=ds_options.index("bot_iot") if "bot_iot" in ds_options else 0,
                                )
                                df_new["dataset_source"] = default_ds

                            if st.button("🔍 Run batch detection"):
                                # Build features using same pipeline
                                X_new, _ = build_X_y(df_new)

                                # 1) Select correct feature subset for this model
                                model_feature_cols = get_model_feature_cols(model_choice, FULL_FEATURE_COLS)
                                X_new = X_new.reindex(columns=model_feature_cols, fill_value=0)

                                # 2) Apply scaler if available
                                scaler = get_model_scaler(model_choice, scalers)
                                if scaler is not None:
                                    X_in = scaler.transform(X_new)
                                else:
                                    X_in = X_new

                                # 3) Predict
                                preds = model.predict(X_in)
                                try:
                                    probas = model.predict_proba(X_in)[:, 1]
                                except Exception:
                                    probas = None

                                df_out = df_new.copy()
                                df_out["predicted_label"] = preds
                                if probas is not None:
                                    df_out["predicted_attack_proba"] = probas

                                counts = pd.Series(preds).value_counts().to_dict()
                                normal_pred = counts.get(0, 0)
                                attack_pred = counts.get(1, 0)

                                st.success(
                                    f"Detection complete on **{len(df_out):,}** rows – "
                                    f"Normal: {normal_pred:,}, Attack: {attack_pred:,}"
                                )

                                st.markdown("### Preview with predictions (first 200 rows)")
                                st.dataframe(df_out.head(200), use_container_width=True)

                                st.download_button(
                                    "⬇️ Download predictions as CSV",
                                    data=df_out.to_csv(index=False).encode("utf-8"),
                                    file_name="predictions_with_labels.csv",
                                    mime="text/csv",
                                )

# ===================== INTERPRETATION TAB ============================
with tab_interpret:
    st.subheader("High-level interpretation & narrative")

    st.markdown(
        """
### 1. Data imbalance story

- **Bot-IoT** is *extremely* skewed (≈0.2% normal vs 99.8% attack).  
- **N-BaIoT** is attack-heavy but less extreme (≈10% normal vs 90% attack).  
- Combined dataset: still very imbalanced (≈3–4% normal overall).

This motivates:

- A plain **baseline** (RF on imbalanced data),  
- **Resampling techniques** (SMOTE + undersampling), and  
- **Anomaly-based models** that only require normal data.

---

### 2. What do the metrics tell us?

- **Random Forest / LightGBM** with resampling reach **>99.9% accuracy** with very high
  precision and recall for both classes.  
- **MLP** performs worse than tree-based methods on this tabular data (especially in the SMOTE
  variant), which is a useful negative result to mention.  
- **CNN1D and TCN** show that even simple deep architectures on the 8 features can perform
  competitively, with TCN achieving strong accuracy and F1.  
- The **Autoencoder** and **Isolation Forest** have:
  - Very high **precision for attacks** (when they say “attack”, they are usually correct),  
  - But much lower **recall for attacks** – they miss many attack flows on this dataset.  
  This is expected for one-class anomaly setups when attacks dominate the test distribution.

So the **best detectors for this particular mixed dataset are the supervised models**
(RF / LightGBM / TCN), but the anomaly-based models remain important for telling
a **zero-day / unseen attack** story.

---

### 3. How to present this in your viva / report

- Emphasise that this is a **multi-dataset IoT IDS** with:
  - unified feature engineering,  
  - multiple supervised baselines, and  
  - anomaly-based detectors.

- Suggested flow for your oral explanation:
  1. Start with the **data imbalance problem** and why Bot-IoT alone is tricky.  
  2. Explain how merging with N-BaIoT and using a common feature schema gives a more robust setup.  
  3. Walk through the **pipeline**:
     - load & clean → feature engineering → train supervised + anomaly models → evaluate → dashboard.  
  4. Show the **Model Performance** and **Graphs** tabs:
     - compare RF/LGBM/MLP/CNN/TCN vs Autoencoder/Isolation Forest.  
     - point out trade-offs in precision vs recall.  
  5. End with the **Live Detection** tab as a demo:
     - manually tweak features to see how the model flips from normal → attack,  
     - or upload a CSV and show batch predictions.

This gives you a complete story: **data → features → models → metrics → live demo**.
"""
    )
