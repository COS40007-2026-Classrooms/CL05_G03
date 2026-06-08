import os
import json
from datetime import datetime

import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    print("=" * 70)
    print("Evaluating Zone 1 Power Consumption Model")
    print("=" * 70)

    with open("params.yaml", "r") as f: # Load evaluation parameters from YAML file
        params = yaml.safe_load(f) # Load evaluation parameters from YAML file

    test_path = params["data"]["test_path"]
    target_column = params["data"]["target_column"]
    model_path = params["outputs"]["model_path"]
    metrics_path = params["outputs"]["metrics_path"]

    os.makedirs("artifacts", exist_ok=True) # Ensure the output directory for artifacts exists

    test_df = pd.read_csv(test_path) # Load the test dataset from the specified path in the parameters
    bundle = joblib.load(model_path) # Load the trained model and metadata from the specified path in the parameters using joblib for efficient deserialization

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred)) 
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    bundle["training_metadata"]["metrics"] = { # Add evaluation metrics to the training metadata in the deployment bundle for record-keeping and future reference
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }
    bundle["training_metadata"]["n_test"] = len(X_test)
    joblib.dump(bundle, model_path)

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"), # Record the timestamp of the evaluation for tracking and historical analysis
        "model_type": bundle["model_name"], 
        "metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        },
    }

    with open(metrics_path, "w") as f: # Save the evaluation metrics to a JSON file specified in the parameters for easy access and integration with other tools or dashboards
        json.dump(metrics, f, indent=4) # Save the evaluation metrics to a JSON file specified in the parameters for easy access and integration with other tools or dashboards

    with open("metrics.txt", "w") as f: 
        f.write("=" * 70 + "\n") 
        f.write("ZONE 1 POWER CONSUMPTION MODEL EVALUATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {bundle['model_name']}\n")
        f.write(f"Testing samples: {len(X_test)}\n")
        f.write(f"Number of features: {len(feature_columns)}\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"  MAE:  {mae:.2f} W\n")
        f.write(f"  RMSE: {rmse:.2f} W\n")
        f.write(f"  R2:   {r2:.4f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, y_pred, alpha=0.4, s=25)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)
    axes[0].set_xlabel("Actual Zone 1 Power (W)")
    axes[0].set_ylabel("Predicted Zone 1 Power (W)")
    axes[0].set_title(f"Predicted vs Actual\nR2 = {r2:.4f}, MAE = {mae:.0f} W")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    importances = model.feature_importances_
    order = np.argsort(importances)
    axes[1].barh([feature_columns[i] for i in order], importances[order])
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Feature Importance")
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("model_results.png", dpi=120, bbox_inches="tight")
    plt.savefig("artifacts/model_results.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")
    print("Evaluation completed successfully")


if __name__ == "__main__":
    main()