# IoT-IDS: Intrusion Detection System for IoT Networks  
**Using Bot-IoT + N-BaIoT Datasets**

This project implements a binary **Intrusion Detection System (IDS)** for IoT environments using:

- A merged and reduced dataset from **Bot-IoT** and **N-BaIoT**
- Supervised learning (Random Forest, with multiple class-balancing strategies)
- Unsupervised anomaly detection (Autoencoder)
- Edge-oriented variants of the model
- A **Streamlit dashboard** for visualizing metrics and dataset samples

> Course: CIS-600 – IoT Security & Privacy  
> Author: Aakanksh Singh

---

## Project Structure

```text
iot-ids/
├── .venv/                      # Local virtual environment (not pushed to Git)
│
├── data/
│   ├── metrics/
│   │   ├── anomaly_metrics.csv
│   │   ├── edge_benchmark.csv
│   │   └── supervised_metrics.csv
│   │
│   ├── N-BaIoT/
│   │   └── dataset/
│   │       ├── 1.benign.csv
│   │       ├── 1.gafgyt.combo.csv
│   │       ├── ...
│   │       ├── 9.mirai.udpplain.csv
│   │       ├── data_summary.csv
│   │       ├── device_info.csv
│   │       ├── features.csv
│   │       └── README.md
│   │
│   ├── processed/
│   │   ├── merged_placeholder.csv
│   │   ├── model_metrics_by_dataset.csv
│   │   └── model_metrics_overall.csv
│   │
│   └── raw/
│       ├── bot_iot_reduced.csv
│       └── n_baiot.csv
│
├── experiments/
│   ├── exp_ae_normal_bot_iot.md
│   ├── exp_rf_bot_iot_baseline.md
│   ├── exp_rf_bot_iot_smote.md
│   └── exp_rf_bot_iot_undersampled.md
│
├── models/
│   ├── autoencoder_normal_bot_iot.pth
│   ├── autoencoder_normal_scaler.joblib
│   ├── rf_bot_iot_baseline.joblib
│   ├── rf_bot_iot_smote.joblib
│   ├── rf_bot_iot_undersampled.joblib
│   ├── rf_edge_8f.joblib
│   └── rf_edge_8f_scaler.joblib
│
└── src/
    ├── config.py
    ├── __init__.py
    │
    ├── app/
    │   └── dashboard.py           # Streamlit UI
    │
    ├── data_prep/
    │   ├── build_n_baiot_from_original.py
    │   ├── clean_and_merge.py
    │   ├── debug_n_baiot.py
    │   └── load_raw.py
    │
    ├── edge/
    │   ├── edge_benchmark.py      # Compare edge models
    │   ├── edge_inference.py      # Edge inference using rf_edge_8f
    │   ├── edge_server.py         # Simple server for edge model
    │   └── __init__.py
    │
    ├── evaluation/
    │   ├── check_metrics.py
    │   ├── collect_model_metrics.py
    │   ├── eval_autoencoder_anomaly.py
    │   ├── eval_cnn_tcn.py
    │   ├── eval_isolation_forest_anomaly.py
    │   ├── eval_rf_by_source.py
    │   ├── eval_supervised.py
    │   ├── inspect_by_source.py
    │   └── inspect_labels.py
    │
    ├── features/
    │   ├── build_basic_features.py
    │   └── normal_only_dataset.py
    │
    └── training/
        ├── train_autoencoder_normal.py
        ├── train_rf_baseline.py
        ├── train_rf_edge.py
        ├── train_rf_smote.py
        └── train_rf_undersampled.py
