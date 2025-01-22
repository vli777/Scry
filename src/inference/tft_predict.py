from datetime import datetime, timedelta, timezone
import os
from time import time
from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import (
    TemporalFusionTransformer,
)
import torch
from tqdm import tqdm
from src.config_loader import config
from src.downloaders.utils.helpers import get_last_saved_timestamp
from src.downloaders.schwab import fetch_data_schwab, save_to_parquet
from src.preprocessing.process_data import prepare_features

torch.set_float32_matmul_precision("medium")


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
) -> np.ndarray:
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataloader
    inference_dataloader = tft_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=16
    )

    # Load TFT model
    tft_model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    tft_model.eval()

    predictions = []
    # Wrap the dataloader with tqdm for progress indication
    for batch in tqdm(inference_dataloader, desc="Predicting"):
        # Unpack batch: typically (x, y)
        if isinstance(batch, (list, tuple)):
            x, _ = batch
        else:
            x = batch

        # Move each tensor in the input dict to the target device
        if isinstance(x, dict):
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
        elif torch.is_tensor(x):
            x = x.to(device)

        # Call model’s forward directly without 'mode'
        out = tft_model(x)
        # Collect predictions from output dictionary
        predictions.append(out["prediction"].detach().cpu().numpy())

    # Concatenate all batch predictions
    all_predictions = np.concatenate(predictions, axis=0)
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
    processed_file_path = config.processed_file
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

    # Save the updated data
    save_to_parquet(df, processed_file_path)
    print(f"Updated DataFrame saved to {processed_file_path}.")
    
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

    start_time = time()
    predictions = predict_with_tft(tft_dataset, model_path=model_path)
    end_time = time()
    print(f"Inference took {end_time - start_time:.2f} seconds.")
    return predictions


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

    # Ensure pred_close is 2D with shape (samples, steps)
    pred_close = np.array(pred_close).squeeze()

    print(type(pred_close), pred_close.shape)
    print(
        type(pred_close[0]),
        pred_close[0].shape if hasattr(pred_close[0], "shape") else None,
    )

    # Get the last prediction sequence (assuming shape: [num_samples, forecast_steps])
    latest_prediction = pred_close[-1]

    # load scaler to restore original price scale
    scaler = joblib.load(config.scaler_file)

    scaled_prediction = latest_prediction.reshape(-1, 1)  # shape becomes (12, 1)
    original_prediction = scaler.inverse_transform(scaled_prediction)
    original_prediction = original_prediction.flatten()  # shape (12,)

    formatted_original = [f"{price:.3f}" for price in original_prediction]
    print("Predicted multi-step close for next 1 hr (original scale):")
    print(", ".join(formatted_original))
