from datetime import datetime
import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

SCHEMA_VERSION = "1.0"

FEATURE_COLUMNS = [
    "zone2_power",
    "zone3_power",
    "power_sum_23",
    "hour_sin",
    "hour_cos",
    "temperature",
    "humidity",
    "hdd",
    "cdd",
]


def retraining_required():
    report_path = "monitoring/reports/drift_report.json"

    if not os.path.exists(report_path):
        print("No drift report found. Retraining will run.")
        return True

    with open(report_path, "r") as f:
        report = json.load(f)

    required = report.get("retraining_required", True)

    if required:
        print("Retraining required based on monitoring report.")
    else:
        print("No retraining required. Existing model will be kept.")

    return required


def main():
    print("=" * 70)
    print("Training Zone 1 Power Consumption Model")
    print("=" * 70)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    train_path = params["data"]["processed_train_path"]
    target_column = params["data"]["target_column"]
    model_path = params["outputs"]["model_path"]

    n_estimators = params["model"]["n_estimators"]
    random_state = params["model"]["random_state"]

    os.makedirs("models", exist_ok=True)

    if not retraining_required():
        if os.path.exists(model_path):
            print(f"Existing model found: {model_path}")
            print("Skipping model training.")
            return
        else:
            print("No existing model found. Training will run anyway.")

    print(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[target_column]

    print(f"Training rows: {len(X_train)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    print("Training Random Forest model")
    model.fit(X_train, y_train)

    quantiles = np.quantile(y_train, [0.25, 0.5, 0.75])

    category_thresholds = {
        "low": float(quantiles[0]),
        "medium": float(quantiles[1]),
        "high": float(quantiles[2]),
    }

    feature_ranges = {
        col: {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "mean": float(X_train[col].mean()),
        }
        for col in FEATURE_COLUMNS
    }

    deployment_bundle = {
        "schema_version": SCHEMA_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "model_name": "Random Forest",
        "feature_columns": FEATURE_COLUMNS,
        "feature_ranges": feature_ranges,
        "category_thresholds": category_thresholds,
        "training_metadata": {
            "n_train": len(X_train),
            "n_features": len(FEATURE_COLUMNS),
            "n_estimators": n_estimators,
            "random_state": random_state,
            "retrained": True,
        },
    }

    joblib.dump(deployment_bundle, model_path)

    print(f"Model saved to {model_path}")
    print("Training completed successfully")


if __name__ == "__main__":
    main()


#test
