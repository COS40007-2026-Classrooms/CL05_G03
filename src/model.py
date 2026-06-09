from datetime import datetime
import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

# Version number used to track the deployment bundle structure
SCHEMA_VERSION = "1.0"

# Features selected based on previous EDA and feature engineering
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
    # Path to the monitoring report generated earlier
    report_path = "monitoring/reports/drift_report.json"

    # If no monitoring report exists, retraining should proceed
    if not os.path.exists(report_path):
        print("No drift report found. Retraining will run.")
        return True

    # Load the monitoring report
    with open(report_path, "r") as f:
        report = json.load(f)

    # Check whether retraining is required
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

    # Load training settings from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Read file paths and model settings
    train_path = params["data"]["processed_train_path"]
    target_column = params["data"]["target_column"]
    model_path = params["outputs"]["model_path"]

    n_estimators = params["model"]["n_estimators"]
    random_state = params["model"]["random_state"]

    # Create the models folder if it does not exist
    os.makedirs("models", exist_ok=True)

    # Skip retraining if monitoring indicates it is unnecessary
    if not retraining_required():
        if os.path.exists(model_path):
            print(f"Existing model found: {model_path}")
            print("Skipping model training.")
            return
        else:
            print("No existing model found. Training will run anyway.")

    print(f"Loading training data from {train_path}")

    # Load the processed training dataset
    train_df = pd.read_csv(train_path)

    # Separate features and target variable
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[target_column]

    print(f"Training rows: {len(X_train)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    # Create the Random Forest model using settings from params.yaml
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    print("Training Random Forest model")

    # Train the model using the training dataset
    model.fit(X_train, y_train)

    # Calculate quartiles to define energy consumption categories
    quantiles = np.quantile(y_train, [0.25, 0.5, 0.75])

    category_thresholds = {
        "low": float(quantiles[0]),
        "medium": float(quantiles[1]),
        "high": float(quantiles[2]),
    }

    # Store feature statistics for future validation and monitoring
    feature_ranges = {
        col: {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "mean": float(X_train[col].mean()),
        }
        for col in FEATURE_COLUMNS
    }

    # Save everything needed for deployment in a single bundle
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

            # Indicates that this model was produced during retraining
            "retrained": True,
        },
    }

    # Save the deployment bundle as a joblib file
    joblib.dump(deployment_bundle, model_path)

    print(f"Model saved to {model_path}")
    print("Training completed successfully")


if __name__ == "__main__":
    main()


# test