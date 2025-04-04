import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from src.config_loader import config
from src.training.lgbm_multi_train import train_lgbm_multi
from src.training.lgbm_single_train import train_lgbm
from src.preprocessing.process_data import continuous_columns


def main():
    load_dotenv(override=True)

    # Mode selection: "single" or "multi"
    mode = "single"  # Change to "single" for single-step training
    steps = 12  # Number of steps for multi-step training

    # Paths for models
    single_model_path = os.path.join(
        config.model_dir, f"{config.model_type}_{config.symbol}_single.pkl"
    )
    multi_model_path = os.path.join(
        config.model_dir, f"{config.model_type}_{config.symbol}_multi.pkl"
    )

    # Load processed data
    df = pd.read_parquet(config.processed_file)

    if mode == "single":
        print("Training single-step LGBM model...")
        # Extract features and target
        X = df[continuous_columns]
        y = df["close"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train single-step model
        train_lgbm(X_train, y_train, X_test, y_test, save_path=single_model_path)

    elif mode == "multi":
        print("Training multi-step LGBM model...")

        # Train multi-step model
        train_lgbm_multi(df, steps=steps, save_path=multi_model_path)


if __name__ == "__main__":
    main()
