import os
import json
from datetime import datetime

import yaml
import pandas as pd
from scipy.stats import ks_2samp


# Main function
def main():
    print("=" * 70)
    print("Running monitoring checks")
    print("=" * 70)

    # Open the YAML file and load the project settings
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Get the required file paths and target column from params.yaml
    train_path = params["data"]["train_path"]
    new_data_path = params["data"]["new_data_path"]
    metrics_path = params["outputs"]["metrics_path"]
    target_column = params["data"]["target_column"]

    # Create monitoring folders if they do not already exist
    os.makedirs("monitoring/reports", exist_ok=True)
    os.makedirs("monitoring/logs", exist_ok=True)
    os.makedirs("monitoring/alerts", exist_ok=True)

    # Load the original training dataset
    train_df = pd.read_csv(train_path)

    # Create the monitoring report structure
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_quality": {},
        "drift_detection": {},
        "performance": {},
        "alerts": []
    }

    # Load the new dataset if available
    if not os.path.exists(new_data_path):
        report["alerts"].append(
            "No new data file found. Monitoring used training data only."
        )

        # Use training data as a fallback
        new_df = train_df.copy()
    else:
        new_df = pd.read_csv(new_data_path)

    # ---------------------------------------------------
    # Data quality checks
    # ---------------------------------------------------

    # Check for missing values, duplicates, and column mismatches
    report["data_quality"] = {
        "train_rows": int(len(train_df)),
        "new_data_rows": int(len(new_df)),
        "new_data_missing_values": int(new_df.isnull().sum().sum()),
        "new_data_duplicate_rows": int(new_df.duplicated().sum()),
        "column_match": list(train_df.columns) == list(new_df.columns)
    }

    # Generate alerts if data quality problems are found
    if report["data_quality"]["new_data_missing_values"] > 0:
        report["alerts"].append(
            "Missing values detected in new data."
        )

    if report["data_quality"]["new_data_duplicate_rows"] > 0:
        report["alerts"].append(
            "Duplicate rows detected in new data."
        )

    if not report["data_quality"]["column_match"]:
        report["alerts"].append(
            "New data columns do not match training data columns."
        )

    # ---------------------------------------------------
    # Data drift checks
    # ---------------------------------------------------

    drift_results = {}
    drift_detected = False

    # Select numerical columns for drift detection
    # Exclude the target variable
    numeric_cols = [
        col for col in train_df.select_dtypes(include="number").columns
        if col != target_column and col in new_df.columns
    ]

    # Compare the distributions using the KS test
    for col in numeric_cols:

        stat, p_value = ks_2samp(
            train_df[col],
            new_df[col]
        )

        # Drift is considered present if p-value is below 0.05
        has_drift = p_value < 0.05

        drift_results[col] = {
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": bool(has_drift)
        }

        if has_drift:
            drift_detected = True

    # Save the drift detection results
    report["drift_detection"] = {
        "method": "Kolmogorov-Smirnov test",
        "threshold": 0.05,
        "drift_detected": bool(drift_detected),
        "feature_results": drift_results
    }

    # Add an alert if drift is detected
    if drift_detected:
        report["alerts"].append(
            "Data drift detected in one or more features."
        )

    # Retraining is required if drift or any alerts exist
    retraining_required = (
        drift_detected or len(report["alerts"]) > 0
    )

    report["retraining_required"] = bool(
        retraining_required
    )

    # ---------------------------------------------------
    # Performance monitoring
    # ---------------------------------------------------

    # Load previous evaluation metrics if available
    if os.path.exists(metrics_path):

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        report["performance"] = metrics.get(
            "metrics", {}
        )

    else:
        report["alerts"].append(
            "metrics.json not found. Performance monitoring skipped."
        )

    # ---------------------------------------------------
    # Save monitoring reports
    # ---------------------------------------------------

    # Save the complete monitoring report
    with open(
        "monitoring/reports/drift_report.json",
        "w"
    ) as f:
        json.dump(report, f, indent=4)

    # Append the report to the monitoring log
    with open(
        "monitoring/logs/monitoring.log",
        "a"
    ) as f:
        f.write(json.dumps(report) + "\n")

    # Save only the alerts into a separate file
    with open(
        "monitoring/alerts/alerts.json",
        "w"
    ) as f:
        json.dump(
            {"alerts": report["alerts"]},
            f,
            indent=4
        )

    print("Monitoring completed successfully.")
    print(f"Drift detected: {drift_detected}")
    print(f"Alerts: {len(report['alerts'])}")

    print("Saved: monitoring/reports/drift_report.json")
    print("Saved: monitoring/alerts/alerts.json")


if __name__ == "__main__":
    main()