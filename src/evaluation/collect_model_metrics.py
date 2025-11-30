import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.features.build_basic_features import load_processed_df, build_X_y


MODEL_FILES = {
    "rf_baseline": "rf_bot_iot_baseline.joblib",
    "rf_smote": "rf_bot_iot_smote.joblib",
    "rf_undersampled": "rf_bot_iot_undersampled.joblib",
}


def extract_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        "f1_0": report["0"]["f1-score"],
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_1": report["1"]["f1-score"],
    }


def main():
    df = load_processed_df("merged_placeholder.csv")
    X, y = build_X_y(df)

    overall_rows = []
    source_rows = []

    for name, file_name in MODEL_FILES.items():
        path = MODELS_DIR / file_name

        if not path.exists():
            print(f"Skipping {name}, not found: {path}")
            continue

        print(f"Loading model: {name}")
        model = joblib.load(path)

        # ---- Overall metrics ----
        y_pred = model.predict(X)
        metrics = extract_metrics(y, y_pred)
        metrics["model"] = name          # 👈 changed from "model_name"
        overall_rows.append(metrics)

        # ---- Per-dataset-source metrics ----
        for source, df_source in df.groupby("dataset_source"):
            idx = df_source.index
            X_s = X.loc[idx]
            y_s = y.loc[idx]
            y_pred_s = model.predict(X_s)
            m_s = extract_metrics(y_s, y_pred_s)
            m_s["model"] = name       
            m_s["dataset_source"] = source
            source_rows.append(m_s)

    # Save output CSVs
    out_overall = pd.DataFrame(overall_rows)
    out_by_source = pd.DataFrame(source_rows)

    out_overall_path = PROCESSED_DATA_DIR / "model_metrics_overall.csv"
    out_by_source_path = PROCESSED_DATA_DIR / "model_metrics_by_dataset.csv"

    out_overall.to_csv(out_overall_path, index=False)
    out_by_source.to_csv(out_by_source_path, index=False)

    print(f"Saved: {out_overall_path}")
    print(f"Saved: {out_by_source_path}")


if __name__ == "__main__":
    main()
