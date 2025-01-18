import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import lightgbm as lgb
from src.preprocessing.process_data import continuous_columns


def train_lgbm_multi(df, steps=12, save_path=None):
    """
    Train a single LightGBM model to predict multiple steps simultaneously.
    Each target column represents one future step.
    """

    # Prepare features and multi-step targets
    df, target_cols = prepare_multi_step_targets(df, steps=steps)

    # Validate continuous columns
    feature_columns = [col for col in continuous_columns if col in df.columns]
    if not feature_columns:
        raise ValueError("No valid feature columns found in the DataFrame.")

    # Select features and targets
    X = df[feature_columns]

    models = {}
    for step, target_col in enumerate(target_cols, start=1):
        print(f"Training model for step {step}...")

        # Extract target for the current step
        y = df[target_col]

        # Initial split: 90% training/validation, 10% test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        # Further split training/validation: 80% train, 10% validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1111, random_state=42
        )  # 10% of 90% is ~11.11%

        # - X_train and y_train: 80% of the total dataset
        # - X_val and y_val: 10% of the total dataset
        # - X_test and y_test: 10% of the total dataset

        # Train a single LightGBM model
        print("Training a single model for multi-step prediction...")
        model = lgb.LGBMRegressor(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
        )
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        step_rmse = root_mean_squared_error(y_test, y_pred)
        step_r2 = r2_score(y_test, y_pred)
        print(f"Step {step} RMSE: {step_rmse:.4f}")
        print(f"Step {step} R²: {step_r2:.4f}")

        step_save_path = f"{save_path}_step_{step}.pkl" if save_path else None
        if step_save_path:
            os.makedirs(os.path.dirname(step_save_path), exist_ok=True)
            joblib.dump(model, step_save_path)
            print(f"Model for step {step} saved to {step_save_path}")

        # Store the model for this step
        models[f"step_{step}"] = model

    return models


def prepare_multi_step_targets(df, steps=12):
    """
    Adds future close price columns as targets for multi-step prediction.
    Returns the updated DataFrame and a list of target column names.
    """
    if "close" not in df.columns:
        raise ValueError("'close' column is missing from the dataset.")

    target_cols = []
    for step in range(1, steps + 1):
        target_col = f"target_{step}"
        df[target_col] = df["close"].shift(-step)
        target_cols.append(target_col)

    # Drop rows with NaN targets (incomplete future data)
    df = df.dropna(subset=target_cols)
    return df, target_cols
