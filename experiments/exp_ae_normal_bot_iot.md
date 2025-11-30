# Experiment: Autoencoder Trained on Normal Bot-IoT Traffic (Unsupervised Anomaly Detection)

**Date:** 2025-11-27  
**Dataset:** `merged_placeholder.csv` (1,000,000 rows)  

**Approach:**  
Unsupervised anomaly detection using a feed-forward autoencoder trained **only on normal traffic (label = 0)**.  
At inference time, reconstruction error is used as an anomaly score. High error ⇒ likely attack.

---

## Data Preparation

- Source: `data/processed/merged_placeholder.csv`
- Features: same 13-dimensional feature vector as supervised RF models.
- Label: `label` (0 = Normal, 1 = Attack)

**Normal-only dataset:**

- Total normal samples: 1,993
- Train/validation split:
  - Train normals: 1,594
  - Val normals: 399

**Scaling:**

- `StandardScaler` fitted on normal train set.
- Applied to both normal validation set during training and to the full dataset at evaluation time.

---

## Model

**Type:** Simple feed-forward autoencoder (PyTorch)  
**Script:** `src/training/train_autoencoder_normal.py`

**Architecture (input_dim = 13):**

- Encoder:
  - Linear(13 → 32), ReLU
  - Linear(32 → 16), ReLU
- Decoder:
  - Linear(16 → 32), ReLU
  - Linear(32 → 13)

**Training:**

- Optimizer: Adam (lr = 1e-3)
- Loss: MSE (per-feature reconstruction loss)
- Epochs: 20
- Batch size: 64
- Device: CPU

Final losses after 20 epochs:
- Train loss ≈ 0.0149
- Val loss ≈ 0.0122

Model weights stored at:
- `models/autoencoder_normal_bot_iot.pth`  
Scaler stored at:
- `models/autoencoder_normal_scaler.joblib`

---

## Evaluation as Anomaly Detector

**Script:** `src/evaluation/eval_autoencoder_anomaly.py`  

**Scoring:**

- Compute reconstruction error for each sample in the full dataset (1M rows).
- Use MSE averaged over feature dimension as the error score.

**Threshold selection:**

- Threshold chosen as the **99th percentile of reconstruction error among normal samples**.
- Threshold value:
  - `0.236594`

**Error statistics (all samples):**

- min ≈ 0.000062  
- max ≈ 80.786736  
- mean ≈ 0.179770  

---

## Results (Evaluated on FULL dataset)

**Accuracy:** 0.1112

**Confusion Matrix (FULL set):**

[[ 1973 20]
[888828 109179]]


Rows = true labels, columns = predicted labels.

**Classification Report:**

- Class 0 (Normal)
  - Precision: 0.0022  
  - Recall:    0.9900  
  - F1-score:  0.0044  
  - Support:   1,993  

- Class 1 (Attack)
  - Precision: 0.9998  
  - Recall:    0.1094  
  - F1-score:  0.1972  
  - Support:   998,007  

---

## Interpretation

- The autoencoder **very accurately recognises normal flows** (recall ≈ 0.99) and almost never classifies normal traffic as attack.
- However, many attack flows exhibit feature patterns that are “close enough” to normal in the 13-dimensional feature space, resulting in **low reconstruction error** and being misclassified as normal.
- Consequently, attack recall is low (~0.11) on this dataset.
- Compared to the supervised RandomForest with SMOTE (which achieves near-perfect performance on both classes), the autoencoder acts more as a *conceptual baseline* for unsupervised anomaly detection rather than a competitive classifier on this particular processed Bot-IoT subset.
- This experiment still demonstrates the typical behavior of reconstruction-based anomaly detection on heavily imbalanced, structured network datasets:
  - Strong at modeling “normality”
  - Not sufficient alone to detect all attack types without richer features or model capacity.
