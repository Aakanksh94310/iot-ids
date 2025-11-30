# Experiment: RF Bot-IoT with SMOTE (Oversampling)

**Date:** 2025-11-27  
**Dataset:** `merged_placeholder.csv` (1,000,000 rows)  

**Balancing Strategy:**  
- Train/test split performed first on the original imbalanced data (80/20, stratified).  
- Applied SMOTE **only on the training set** with:
  - `sampling_strategy = 0.1`
  - This makes the minority class (normal, label=0) ≈ 10% of the majority class (attack, label=1) in the training data.

**Resampled Train Distribution:**

- Before SMOTE (train only):
  - label 0 (Normal): 1,594
  - label 1 (Attack): 798,406

- After SMOTE:
  - label 0 (Normal): 79,840
  - label 1 (Attack): 798,406
  - Total: 878,246 training samples

---

## Model

**Type:** RandomForestClassifier  
**Script:** `src/training/train_rf_smote.py`  

**Hyperparameters:**
- n_estimators = 100  
- max_depth = None  
- n_jobs = -1  
- random_state = 42  
- class_weight = None (not needed; SMOTE balances training set)

---

## Results (Evaluated on TEST set)

**Accuracy:** 0.9999

**Confusion Matrix (TEST set, 200,000 rows):**

[[ 391 8]
[ 13 199588]]


Rows = true, columns = predicted.

**Per-class metrics:**

### Normal (0)
- Precision: 0.9678  
- Recall:    0.9799  
- F1-score:  0.9738  
- Support:   399  

### Attack (1)
- Precision: 1.0000  
- Recall:    0.9999  
- F1-score:  0.9999  
- Support:   199,601  

---

## Interpretation

- SMOTE massively improves normal-class performance compared to both:
  - the original imbalanced baseline, and  
  - the simple undersampled baseline.
- The model now correctly detects almost all normal flows (recall ≈ 0.98) and rarely mislabels attacks as normal (very strong metrics for class 1).
- Overall accuracy and F1 for both classes are extremely high, suggesting that SMOTE is an effective strategy for this dataset.
- This configuration is a strong candidate for the **“best supervised baseline”** in the project, against which anomaly-based models and multi-dataset models (with N-BaIoT) can be compared.


| Model                    | Normal Precision | Normal Recall | Attack F1 | Accuracy   |
| ------------------------ | ---------------- | ------------- | --------- | ---------- |
| RF baseline (imbalanced) | ~0.17            | ~0.98         | ~0.995    | 0.9903     |
| RF undersampled          | ~0.15            | **1.00**      | ~0.994    | 0.9887     |
| **RF + SMOTE**           | **0.97**         | ~0.98         | **~1.00** | **0.9999** |

Among all three balancing strategies tested (no balancing, basic undersampling, and SMOTE), SMOTE produced the best overall trade-off, with ≈0.97 precision and ≈0.98 recall on normal traffic and nearly perfect performance on attack traffic.