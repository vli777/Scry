from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import joblib
import pandas as pd

from src.downloaders.schwab import fetch_data_schwab, save_to_parquet
from src.preprocessing.process_data import prepare_features
from src.config_loader import Config, config


def predict_close_price_lgbm(
    config,
    bearer_token,
    steps=12,
    max_days=200,
):
    """
    Predict the next N close prices for a given ticker using LightGBM.

    :param config: Config object with all paths and symbol details.
    :param bearer_token: API bearer token for authentication.
    :param steps: Number of future steps to predict.
    :param max_days: Number of days to retain for rolling features.
    :return: DataFrame with predicted values for the next N steps.
    """
    # --- Load the last max_days of data ---
    df_features = pd.read_parquet(config.processed_file)
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=max_days)
    df_recent = df_features[df_features["timestamp"] >= cutoff_date]

    # --- Fetch today's data ---
    start_timestamp = df_recent["timestamp"].max()
    df_today = fetch_data_schwab(
        symbol=config.symbol,
        bearer_token=bearer_token,
        frequency_type=config.frequency_type,
        frequency=config.frequency,
        start_date=start_timestamp,
        end_date=datetime.now(timezone.utc),
    )
    if df_today:
        save_to_parquet(df_today, config.processed_file)  # Update processed file
        df_recent = pd.concat([df_recent, pd.DataFrame(df_today)]).reset_index(
            drop=True
        )

    # --- Prepare features for prediction ---
    df_prepared, _ = prepare_features(
        df_recent, scaler_path=config.scaler_file, fit_scaler=False
    )
    last_row = df_prepared.iloc[[-1]]  # Use the most recent row for prediction
    continuous_columns = [
        col for col in last_row.columns if col not in {"timestamp", "close", "symbol"}
    ]

    # --- Load model and make predictions ---
    model = joblib.load(config.model_file)
    predictions = model.predict(last_row[continuous_columns])

    # Create a DataFrame for the predictions
    prediction_df = pd.DataFrame(
        {f"step_{i+1}": [pred] for i, pred in enumerate(predictions.flatten())}
    )

    return prediction_df


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    bearer_token = os.getenv("SCHWAB_BEARER_TOKEN")
    if bearer_token is None:
        raise ValueError("SCHWAB_BEARER_TOKEN environment variable not set")

    # Load config
    config = Config()

    # Predict the next 12 steps (5-min intervals)
    predictions = predict_close_price_lgbm(
        config=config,
        bearer_token=bearer_token,
        steps=12,
        max_days=200,
    )
    print("Predictions for the next 12 steps (5-min intervals):")
    print(predictions)
