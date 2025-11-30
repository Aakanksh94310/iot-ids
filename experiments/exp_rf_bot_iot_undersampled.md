# Experiment: RF Bot-IoT Undersampled Baseline

**Date:** 2025-11-27  
**Dataset:** `merged_placeholder.csv` (1,000,000 rows)  
**Balancing:** Undersampled attack class to match 1,993 normal samples.  
**Balanced Training Dataset Size:** 3,986 rows

---

## Model

**Type:** RandomForestClassifier  
**Training Script:** `src/training/train_rf_undersampled.py`  

**Hyperparameters:**
- n_estimators=100
- max_depth=None
- n_jobs=-1
- random_state=42
- class_weight=None (not needed; training set is balanced)

---

## Results (Evaluated on FULL dataset)

**Accuracy:** 0.9887

**Confusion Matrix (full 1M rows):**
[[ 1993 0]
[ 11284 986723]]


**Class Metrics:**

### Normal (0)
- Precision: 0.1501  
- Recall: 1.0000  
- F1-score: 0.2610  
- Support: 1,993  

### Attack (1)
- Precision: 1.0000  
- Recall: 0.9887  
- F1-score: 0.9943  
- Support: 998,007  

---

## Interpretation

- Model now detects *all* normal traffic correctly (recall = 1.0).
- Precision for normal is still low due to extreme imbalance.
- Attack detection remains extremely strong (F1 ≈ 0.994).
- Undersampling improves fairness but reduces precision.
- Additional imbalance techniques (SMOTE, class-weighting, anomaly models) are needed.
