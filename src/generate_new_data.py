import os
import yaml
import numpy as np
import pandas as pd


def main():
    print("=" * 70)
    print("Generating synthetic new data")
    print("=" * 70)

    # Load file paths and settings from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Get the training data path and output path for the new synthetic data
    train_path = params["data"]["train_path"]
    new_data_path = params["data"]["new_data_path"]

    # Use the same random state so the result can be repeated
    random_state = params["model"]["random_state"]

    # Get the number of new rows to create, or use 300 as the default value
    new_rows = params["retraining"].get("new_rows", 300)

    # Create the output folder if it does not already exist
    os.makedirs(os.path.dirname(new_data_path), exist_ok=True)

    # Load the existing training dataset
    train_df = pd.read_csv(train_path)

    # Stop the program if the training dataset has no data
    if train_df.empty:
        raise ValueError("Training data is empty. Cannot generate new data.")

    # Make sure the sample size is not bigger than the training dataset
    sample_size = min(new_rows, len(train_df))

    # Create new data by sampling rows from the training dataset
    # replace=True allows the same row to be selected more than once
    new_data = train_df.sample(
        n=sample_size,
        replace=True,
        random_state=random_state
    ).reset_index(drop=True)

    # Create a random number generator for adding small noise
    rng = np.random.default_rng(random_state)

    # Select only numerical columns because noise should be added only to numbers
    numeric_cols = new_data.select_dtypes(include=[np.number]).columns

    # Add small random noise to numerical columns
    # This makes the new rows slightly different from the original training rows
    for col in numeric_cols:
        std = train_df[col].std()

        # Skip columns that have no variation
        if pd.isna(std) or std == 0:
            continue

        # Create small noise based on 3% of the column standard deviation
        noise = rng.normal(
            loc=0,
            scale=std * 0.03,
            size=len(new_data)
        )

        # Add the noise to the selected column
        new_data[col] = new_data[col] + noise

    # Keep values positive because power consumption and sensor values should not be negative
    new_data = new_data.clip(lower=0)

    # Save the synthetic data as a CSV file
    new_data.to_csv(new_data_path, index=False)

    print("Synthetic new data created successfully.")
    print(f"Saved to: {new_data_path}")
    print(f"Rows created: {len(new_data)}")
    print(f"Columns: {len(new_data.columns)}")


if __name__ == "__main__":
    main()