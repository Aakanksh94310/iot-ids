import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from lightgbm import LGBMClassifier


# --------------------------------------------------
# Config
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "merged_placeholder.csv"
METRICS_CSV = PROJECT_ROOT / "data" / "metrics" / "supervised_metrics.csv"

# label for dashboard / plots – just a tag, change if you like
DATASET_SOURCE = "Bot-IoT+N-BaIoT_reduced"

# CHANGE THIS if your label column has a different name
LABEL_COLUMN = "label"


# --------------------------------------------------
# Helper: metric logging
# --------------------------------------------------
def evaluate_and_log(model_name: str,
                     dataset_source: str,
                     y_true,
                     y_pred,
                     rows_list: list):
    """
    Append one row of metrics into rows_list with a consistent schema
    that the Streamlit dashboard can read.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    row = {
        "model": model_name,
        "dataset": dataset_source,
        "accuracy": acc,
        "precision_0": prec[0],
        "recall_0": rec[0],
        "f1_0": f1[0],
        "precision_1": prec[1],
        "recall_1": rec[1],
        "f1_1": f1[1],
    }
    rows_list.append(row)


def main():
    # --------------------------------------------------
    # 1. Load processed dataset
    # --------------------------------------------------
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(f"Processed CSV not found at: {PROCESSED_CSV}")

    df = pd.read_csv(PROCESSED_CSV)

    if LABEL_COLUMN not in df.columns:
        raise KeyError(
            f"Expected label column '{LABEL_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop label, then keep only numeric features for modeling
    feature_df = df.drop(columns=[LABEL_COLUMN])

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns
    non_numeric_cols = [c for c in feature_df.columns if c not in numeric_cols]

    print(f"[eval_supervised] Using {len(numeric_cols)} numeric features: {list(numeric_cols)}")
    if non_numeric_cols:
        print(f"[eval_supervised] Dropping non-numeric columns: {non_numeric_cols}")

    X = feature_df[numeric_cols]
    y = df[LABEL_COLUMN].astype(int)

    print(f"[eval_supervised] Loaded {df.shape[0]} rows, {X.shape[1]} features")
    print("[eval_supervised] Label distribution:")
    print(y.value_counts())

    # --------------------------------------------------
    # 2. Train / test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # 3. Resampling (SMOTE + undersampling)
    # --------------------------------------------------
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    under = RandomUnderSampler(random_state=42)
    X_train_under, y_train_under = under.fit_resample(X_train, y_train)

    print(f"[eval_supervised] After SMOTE: {X_train_smote.shape}")
    print(pd.Series(y_train_smote).value_counts())
    print(f"[eval_supervised] After undersampling: {X_train_under.shape}")
    print(pd.Series(y_train_under).value_counts())

    # --------------------------------------------------
    # 4. Collect metrics for all models
    # --------------------------------------------------
    metrics_rows = []

    # --------------------------------------------------
    # 5. RandomForest baselines
    # --------------------------------------------------
    rf_smote = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    rf_smote.fit(X_train_smote, y_train_smote)
    y_pred_rf_smote = rf_smote.predict(X_test)

    evaluate_and_log(
        model_name="RandomForest_SMOTE",
        dataset_source=DATASET_SOURCE,
        y_true=y_test,
        y_pred=y_pred_rf_smote,
        rows_list=metrics_rows,
    )

    rf_under = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    rf_under.fit(X_train_under, y_train_under)
    y_pred_rf_under = rf_under.predict(X_test)

    evaluate_and_log(
        model_name="RandomForest_UNDER",
        dataset_source=DATASET_SOURCE,
        y_true=y_test,
        y_pred=y_pred_rf_under,
        rows_list=metrics_rows,
    )

    # --------------------------------------------------
    # 6. LightGBM models
    # --------------------------------------------------
    lgbm_smote = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    lgbm_smote.fit(X_train_smote, y_train_smote)
    y_pred_lgbm_smote = lgbm_smote.predict(X_test)

    evaluate_and_log(
        model_name="LightGBM_SMOTE",
        dataset_source=DATASET_SOURCE,
        y_true=y_test,
        y_pred=y_pred_lgbm_smote,
        rows_list=metrics_rows,
    )

    lgbm_under = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    lgbm_under.fit(X_train_under, y_train_under)
    y_pred_lgbm_under = lgbm_under.predict(X_test)

    evaluate_and_log(
        model_name="LightGBM_UNDER",
        dataset_source=DATASET_SOURCE,
        y_true=y_test,
        y_pred=y_pred_lgbm_under,
        rows_list=metrics_rows,
    )

    # --------------------------------------------------
    # 7. MLP models (with scaling)
    # --------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(X_train)  # fit on original training data

    X_train_smote_scaled = scaler.transform(X_train_smote)
    X_train_under_scaled = scaler.transform(X_train_under)
    X_test_scaled = scaler.transform(X_test)

    mlp_smote = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=20,   
        random_state=42,
        verbose=False,
    )
    mlp_smote.fit(X_train_smote_scaled, y_train_smote)
    y_pred_mlp_smote = mlp_smote.predict(X_test_scaled)

    evaluate_and_log(
        model_name="MLP_SMOTE",
        dataset_source=DATASET_SOURCE,
        y_true=y_test,
        y_pred=y_pred_mlp_smote,
        rows_list=metrics_rows,
    )

    mlp_under = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=20,
        random_state=42,
        verbose=False,
    )
    mlp_under.fit(X_train_under_scaled, y_train_under)
    y_pred_mlp_under = mlp_under.predict(X_test_scaled)

    evaluate_and_log(
        model_name="MLP_UNDER",
        dataset_source=DATASET_SOURCE,
        y_true=y_test,
        y_pred=y_pred_mlp_under,
        rows_list=metrics_rows,
    )

    # --------------------------------------------------
    # 8. Save metrics for Streamlit dashboard
    # --------------------------------------------------
    metrics_df = pd.DataFrame(metrics_rows)
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_CSV, index=False)

    print(f"[eval_supervised] Saved metrics to {METRICS_CSV}")
    print(metrics_df)


if __name__ == "__main__":
    main()
