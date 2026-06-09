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

    # Load evaluation settings from the YAML configuration file
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Read required file paths and settings from the configuration
    test_path = params["data"]["test_path"]
    target_column = params["data"]["target_column"]
    model_path = params["outputs"]["model_path"]
    metrics_path = params["outputs"]["metrics_path"]

    # Create the artifacts folder if it does not already exist
    os.makedirs("artifacts", exist_ok=True)

    # Load the test dataset
    test_df = pd.read_csv(test_path)

    # Load the saved model bundle and metadata
    bundle = joblib.load(model_path)

    # Extract the trained model and the feature names used during training
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    # Separate predictors and target values from the test set
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    # Generate predictions using the trained model
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics to measure model performance
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    '''
    # Save the evaluation results back into the deployment bundle
    # This can be useful for future tracking of model performance

    bundle["training_metadata"]["metrics"] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    bundle["training_metadata"]["n_test"] = len(X_test)

    joblib.dump(bundle, model_path)
    '''

    # Store the evaluation results together with the evaluation timestamp
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_type": bundle["model_name"],
        "metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        },
    }

    # Save metrics into a JSON file for monitoring and reporting
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Create a text report containing the evaluation summary
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

    # Create two plots to visualise model performance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot actual values against predicted values
    axes[0].scatter(y_test, y_pred, alpha=0.4, s=25)

    # Draw the ideal prediction line
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)

    axes[0].set_xlabel("Actual Zone 1 Power (W)")
    axes[0].set_ylabel("Predicted Zone 1 Power (W)")

    axes[0].set_title(
        f"Predicted vs Actual\nR2 = {r2:.4f}, MAE = {mae:.0f} W"
    )

    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Display which features contributed the most to the predictions
    importances = model.feature_importances_

    # Sort features from lowest to highest importance
    order = np.argsort(importances)

    axes[1].barh(
        [feature_columns[i] for i in order],
        importances[order]
    )

    axes[1].set_xlabel("Importance")
    axes[1].set_title("Feature Importance")
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()

    # Save the evaluation plots
    plt.savefig("model_results.png", dpi=120, bbox_inches="tight")
    plt.savefig("artifacts/model_results.png", dpi=120, bbox_inches="tight")

    plt.close()

    # Print evaluation results to the GitHub Actions logs
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")
    print("Evaluation completed successfully")


if __name__ == "__main__":
    main()