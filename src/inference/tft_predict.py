from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import (
    TemporalFusionTransformer,
)
from src.config_loader import config
from src.downloaders.utils.helpers import get_last_saved_timestamp
from src.downloaders.schwab import fetch_data_schwab, save_to_parquet
from src.preprocessing.process_data import prepare_features


def build_tft_dataset(
    df: pd.DataFrame,
    target: str = "close",
    time_idx_col: str = "time_idx",
    group_id_col: str = "symbol",
    max_encoder_length: int = 48,
    max_prediction_length: int = 12,  # default to 12 steps for a 1-hour forecast on 5-min data
    known_reals: list = None,
    unknown_reals: list = None,
) -> TimeSeriesDataSet:
    """
    Builds a TimeSeriesDataSet from a prepared DataFrame for PyTorch Forecasting.

    :param df: Already scaled, feature-complete DataFrame.
    :param target: Name of the column to predict (e.g., 'close').
    :param time_idx_col: Column name for the integer time index.
    :param group_id_col: Column name for the group ID (e.g., 'symbol').
    :param max_encoder_length: How many past timesteps the model sees.
    :param max_prediction_length: Forecast horizon.
    :param known_reals: List of columns known in the future.
    :param unknown_reals: List of columns only known up to the current timestep.
    """
    # Sort by time to ensure correct order
    df = df.sort_values(time_idx_col).reset_index(drop=True)

    # If time_idx_col doesn't exist, create it
    if time_idx_col not in df.columns:
        df[time_idx_col] = range(len(df))

    # If group_id_col doesn't exist, create it
    if group_id_col not in df.columns:
        df[group_id_col] = "symbol"

    # Identify reals
    known_reals = known_reals or []
    unknown_reals = unknown_reals or []

    # Create the dataset
    tft_dataset = TimeSeriesDataSet(
        df,
        time_idx=time_idx_col,
        group_ids=[group_id_col],
        target=target,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=None,  # Already scaled externally
    )
    return tft_dataset


def predict_with_tft(
    tft_dataset: TimeSeriesDataSet, model_path: str, batch_size: int = 1
) -> float:
    """
    1) Loads the TFT model from a checkpoint.
    2) Creates a dataloader from tft_dataset.
    3) Generates predictions and returns the final forecast step (the 12th prediction).
    """
    # Create dataloader
    inference_dataloader = tft_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=8
    )

    # Load TFT model
    tft_model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    tft_model.eval()

    # Generate predictions
    raw_predictions = tft_model.predict(inference_dataloader, mode="raw")

    # Convert to NumPy for easier manipulation (optional)
    all_predictions = raw_predictions.numpy()

    return all_predictions


def predict_close_price_tft(
    config, bearer_token, max_encoder_length=48, max_prediction_length=12
):
    """
    Unified pipeline for:
    - Fetching and appending data.
    - Preprocessing and computing features.
    - Building TFT dataset.
    - Making predictions.
    """
    # Paths from config
    file_path = config.raw_file
    scaler_path = config.scaler_file
    model_path = config.model_file

    # Load existing data and determine start date
    last_saved_timestamp = get_last_saved_timestamp(file_path)
    now = datetime.now(timezone.utc)
    market_close_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
    effective_end_date = min(now, market_close_time)

    start = (
        last_saved_timestamp + timedelta(minutes=config.frequency)
        if last_saved_timestamp
        else config.start_date
    )

    # Fetch new data
    chunked_candles = fetch_data_schwab(
        symbol=config.symbol,
        bearer_token=bearer_token,
        frequency_type=config.frequency_type,
        frequency=config.frequency,
        start_date=start,
        end_date=effective_end_date,
        max_chunk_days=10,
    )

    # Save new data if available
    if chunked_candles:
        save_to_parquet(chunked_candles, file_path)
    else:
        print("No new data fetched. Proceeding with existing data.")

    # Preprocess data
    df = pd.read_parquet(file_path)
    df, scaler = prepare_features(df, scaler_path=scaler_path, fit_scaler=False)

    # Ensure `time_idx` exists
    if "time_idx" not in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["time_idx"] = range(len(df))

    known_reals = ["time_idx"]
    all_cols = list(df.columns)
    exclude_cols = {"timestamp", "symbol", "time_idx", "close"}  # 'close' is target
    unknown_reals = [col for col in all_cols if col not in exclude_cols]

    tft_dataset = build_tft_dataset(
        df=df,
        target="close",
        time_idx_col="time_idx",
        group_id_col="symbol",
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        known_reals=known_reals,
        unknown_reals=unknown_reals,
    )

    pred_value = predict_with_tft(tft_dataset, model_path=model_path)
    return pred_value


if __name__ == "__main__":
    load_dotenv(override=True)
    bearer_token = os.getenv("SCHWAB_BEARER_TOKEN")
    if bearer_token is None:
        raise ValueError("SCHWAB_BEARER_TOKEN environment variable not set")

    # We want to forecast 1 hour out on 5-min data => 12 steps
    pred_close = predict_close_price_tft(
        config,
        bearer_token,
        max_encoder_length=48,  # see 48 past bars (4 hours) for context
        max_prediction_length=12,  # predict the next 12 bars => 1 hour total
    )

    # Format predictions for printing
    formatted_predictions = [
        [f"{value:.3f}" for value in prediction] for prediction in pred_close
    ]

    print(f"Predicted multi-step close for 1 hr:")
    for i, prediction in enumerate(formatted_predictions):
        formatted = ", ".join(prediction)
        print(f"Sample {i + 1}: {formatted}")
