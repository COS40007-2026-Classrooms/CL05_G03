import os
import yaml
import numpy as np
import pandas as pd


def main():
    print("=" * 70)
    print("Generating synthetic new data")
    print("=" * 70)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    train_path = params["data"]["train_path"]
    new_data_path = params["data"]["new_data_path"]
    random_state = params["model"]["random_state"]
    new_rows = params["retraining"].get("new_rows", 300)

    os.makedirs(os.path.dirname(new_data_path), exist_ok=True)

    train_df = pd.read_csv(train_path)

    if train_df.empty:
        raise ValueError("Training data is empty. Cannot generate new data.")

    sample_size = min(new_rows, len(train_df))

    new_data = train_df.sample(
        n=sample_size,
        replace=True,
        random_state=random_state
    ).reset_index(drop=True)

    rng = np.random.default_rng(random_state)

    numeric_cols = new_data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        std = train_df[col].std()

        if pd.isna(std) or std == 0:
            continue

        noise = rng.normal(
            loc=0,
            scale=std * 0.03,
            size=len(new_data)
        )

        new_data[col] = new_data[col] + noise

    new_data = new_data.clip(lower=0)

    new_data.to_csv(new_data_path, index=False)

    print("Synthetic new data created successfully.")
    print(f"Saved to: {new_data_path}")
    print(f"Rows created: {len(new_data)}")
    print(f"Columns: {len(new_data.columns)}")


if __name__ == "__main__":
    main()