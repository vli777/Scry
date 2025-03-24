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
from src.preprocessing.process_data import prepare_features, continuous_columns

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
    tft_dataset: TimeSeriesDataSet, model_path: str, batch_size: int = 128
) -> np.ndarray:
    """
    Optimized prediction function that uses batched inference and GPU acceleration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloader with larger batch size for faster inference
    inference_dataloader = tft_dataset.to_dataloader(
        train=False, 
        batch_size=batch_size, 
        num_workers=0,  # Reduced for stability
        shuffle=False,  # Ensure deterministic order
    )

    # Load TFT model and move to device
    tft_model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    tft_model.to(device)
    tft_model.eval()

    predictions = []
    with torch.no_grad():  # Disable gradient computation
        for x in inference_dataloader:
            # Move input to device efficiently
            x = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
            
            # Get predictions
            out = tft_model(x)
            predictions.append(out["prediction"].cpu().numpy())

    # Concatenate all predictions efficiently
    return np.concatenate(predictions, axis=0)


def predict_close_price_tft(
    config, 
    bearer_token, 
    max_encoder_length=48, 
    max_prediction_length=12,
    cache_dataset=True,
):
    """
    Optimized prediction pipeline with dataset caching and efficient data handling.
    """
    # Load existing data
    df = pd.read_parquet(config.processed_file)
    
    # Get latest data point timestamp
    last_timestamp = df["timestamp"].max()
    
    # Fetch only new data if needed
    current_time = datetime.now(timezone.utc)
    if (current_time - last_timestamp).total_seconds() > config.frequency * 60:
        new_data = fetch_data_schwab(
            symbol=config.symbol,
            bearer_token=bearer_token,
            frequency_type=config.frequency_type,
            frequency=config.frequency,
            start_date=last_timestamp + timedelta(minutes=config.frequency),
            end_date=current_time,
        )
        
        if new_data:
            # Append new data efficiently
            new_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_df], ignore_index=True)
            save_to_parquet(df, config.processed_file)
    
    # Prepare features
    df, _ = prepare_features(df, scaler_path=config.scaler_file, fit_scaler=False)
    
    # Ensure time index
    if "time_idx" not in df.columns:
        df["time_idx"] = range(len(df))
    if "symbol" not in df.columns:
        df["symbol"] = "symbol"
    
    # Create dataset efficiently
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        group_ids=["symbol"],
        target="close",
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=[col for col in df.columns 
                                  if col not in {"timestamp", "symbol", "time_idx", "close"}],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    # Make predictions
    predictions = predict_with_tft(dataset, config.model_file)
    
    # Format predictions
    latest_prediction = predictions[-1]  # Get the most recent prediction
    
    # Load scaler and transform back to original scale
    scaler = joblib.load(config.scaler_file)
    target_idx = [i for i, col in enumerate(continuous_columns) if col == "close"][0]
    
    # Create dummy array for inverse transform
    dummy = np.tile(scaler.mean_, (len(latest_prediction), 1))
    dummy[:, target_idx] = latest_prediction
    
    # Transform back to original scale
    original_prediction = scaler.inverse_transform(dummy)[:, target_idx]
    
    return pd.DataFrame({
        f"step_{i+1}": [pred] for i, pred in enumerate(original_prediction)
    })


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
