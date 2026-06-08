import os
import yaml
import pandas as pd


def main():
    print("=" * 70)
    print("Preprocessing data for retraining")
    print("=" * 70)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    train_path = params["data"]["train_path"]
    new_data_path = params["data"]["new_data_path"]
    processed_train_path = params["data"]["processed_train_path"]
    min_new_rows = params["retraining"].get("min_new_rows", 0)

    os.makedirs(os.path.dirname(processed_train_path), exist_ok=True)

    train_df = pd.read_csv(train_path)

    print(f"Existing training rows: {len(train_df)}")

    if not os.path.exists(new_data_path):
        print("No new data file found.")
        print("Using existing training data only.")

        train_df.to_csv(processed_train_path, index=False)

        print(f"Saved processed training data to: {processed_train_path}")
        print("Preprocessing completed successfully.")
        return

    new_df = pd.read_csv(new_data_path)

    print(f"New data rows: {len(new_df)}")

    if len(new_df) == 0:
        print("New data file is empty.")
        print("Using existing training data only.")

        train_df.to_csv(processed_train_path, index=False)

        print(f"Saved processed training data to: {processed_train_path}")
        print("Preprocessing completed successfully.")
        return

    if len(new_df) < min_new_rows:
        print(
            f"New data has fewer rows than the retraining threshold. "
            f"Required: {min_new_rows}, found: {len(new_df)}"
        )
        print("Using existing training data only.")

        train_df.to_csv(processed_train_path, index=False)

        print(f"Saved processed training data to: {processed_train_path}")
        print("Preprocessing completed successfully.")
        return

    if list(train_df.columns) != list(new_df.columns):
        raise ValueError(
            "New data columns do not match training data columns. "
            "Please make sure data/new_data.csv has the same schema as train/train.csv."
        )

    combined_df = pd.concat([train_df, new_df], ignore_index=True)

    before = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    duplicates_removed = before - len(combined_df)

    combined_df.to_csv(processed_train_path, index=False)

    print("New data preprocessing completed successfully.")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Updated training rows: {len(combined_df)}")
    print(f"Saved processed training data to: {processed_train_path}")


if __name__ == "__main__":
    main()