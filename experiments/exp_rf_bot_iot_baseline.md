# Experiment: RF Bot-IoT Baseline

**Date:** 2025-11-26  
**Dataset:** `data/processed/merged_placeholder.csv`  
**Rows:** 1,000,000  
**Features (X):** 13  
- packet_count  
- byte_count  
- flow_duration  
- avg_packet_size  
- pkt_rate  
- byte_rate  
- tcp_flag_syn  
- tcp_flag_ack  
- one-hot encoded proto (5 dummy columns)

**Label (y):** `label`  
- 0 = Normal  
- 1 = Attack  

---

## Model

**Type:** RandomForestClassifier (sklearn)  
**Code:** `src/training/train_rf_baseline.py`

**Hyperparameters:**

- `n_estimators=100`
- `max_depth=None`
- `n_jobs=-1`
- `random_state=42`
- `class_weight='balanced_subsample'`

---

## Train/Test Split

- `test_size=0.2`
- `random_state=42`
- `stratify=y`

Train shape: (800000, 13)  
Test shape: (200000, 13)

---

## Results

**Accuracy:** 0.9903

**Confusion Matrix** (rows = true, cols = predicted):

|           | Pred 0 | Pred 1 |
|-----------|--------|--------|
| **True 0**|   391  |    8   |
| **True 1**|  1928  | 197673 |

**Per-class metrics:**

- Class 0 (Normal)
  - Precision ≈ 0.1686  
  - Recall ≈ 0.9799  
  - F1-score ≈ 0.2877  
  - Support = 399  

- Class 1 (Attack)
  - Precision ≈ 1.0000  
  - Recall ≈ 0.9903  
  - F1-score ≈ 0.9951  
  - Support = 199601  

**Interpretation (short):**

- Dataset is highly imbalanced (attacks dominate).  
- Model is already very strong at detecting attacks (≈0.995 F1).  
- Performance on the Normal class is weaker due to very few normal samples.  
- This serves as the **baseline** for future improvements (better imbalance handling, more features, additional datasets).
