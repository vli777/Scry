import joblib
import os
import pandas as pd
from sklearn.base import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from ..config_loader import config


def train_lgbm_multi(df, steps=12, save_path=None):
    """
    Train a single LightGBM model to predict multiple steps simultaneously.
    Each target column represents one future step.
    """
    # Prepare features and multi-step targets
    target_cols = [f"target_{step}" for step in range(1, steps + 1)]
    X = df.drop(columns=target_cols + ["timestamp", "symbol", "close"])
    Y = df[target_cols]

    # Initial split: 90% training/validation, 10% test
    train_val_X, test_X, train_val_y, test_y = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )

    # Further split training/validation: 80% train, 10% validation
    train_X, val_X, train_y, val_y = train_test_split(
        train_val_X, train_val_y, test_size=0.1111, random_state=42
    )  # 10% of 90% is ~11.11%

    # - train_X and train_y: 80% of the total dataset
    # - val_X and val_y: 10% of the total dataset
    # - test_X and test_y: 10% of the total dataset

    # Train a single LightGBM model
    print("Training a single model for multi-step prediction...")
    model = lgb.LGBMRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
    )
    model.fit(train_X, train_y)

    # Evaluate the model on each step
    y_pred = model.predict(test_X)
    for step in range(steps):
        step_rmse = root_mean_squared_error(test_y.iloc[:, step], y_pred[:, step])
        step_r2 = r2_score(test_y.iloc[:, step], y_pred[:, step])
        print(f"Step {step + 1} RMSE: {step_rmse:.4f}")
        print(f"Step {step + 1} R²: {step_r2:.4f}")

    # Save the model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")

    return model


def prepare_multi_step_targets(df, steps=12):
    """
    Adds future close price columns as targets for multi-step prediction.
    """
    for step in range(1, steps + 1):
        df[f"target_{step}"] = df["close"].shift(-step)

    # Drop rows with NaN targets (incomplete future data)
    df = df.dropna(subset=[f"target_{step}" for step in range(1, steps + 1)])
    return df


if __name__ == "__main__":
    # Load processed data
    df = pd.read_parquet(config.processed_file)

    # Prepare multi-step targets
    df = prepare_multi_step_targets(df, steps=12)

    # Generate dynamic model path
    multi_model_path = os.path.join(
        config.model_dir, f"{config.model_type}_{config.symbol}_multi.pkl"
    )

    # Train a single LightGBM model for multi-step prediction
    model = train_lgbm_multi(df, steps=12, save_path=multi_model_path)

    print("Multi-step LightGBM training complete.")
