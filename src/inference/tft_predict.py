import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import (
    TemporalFusionTransformer,
)

from data_utils.incremental_features import calculate_features_incrementally
from data_utils.incremental_data import fetch_current_data
from training.process_data import prepare_features


def fetch_and_incremental_update(
    feature_path: str,
    scaler_path: str,
    ticker: str = "SPY",
    bearer_token: str = "your_bearer_token",
    period_type: str = "day",
    frequency_type: str = "minute",
    frequency: int = 5,
):
    """
    1) Load existing data from feature_path.
    2) Identify last timestamp in the dataset.
    3) Fetch new data from that timestamp to now, using the default 5-min bars
       or custom intervals if provided.
    4) Incrementally calculate rolling features.
    5) Apply existing scaler (StandardScaler).

    Returns a DataFrame (df_prepared) ready for modeling.
    """
    try:
        df_features = pd.read_parquet(feature_path)
    except FileNotFoundError:
        df_features = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    if not df_features.empty:
        last_timestamp = df_features["timestamp"].max()
    else:
        # default to 2 years ago if no data
        last_timestamp = pd.Timestamp.now() - pd.Timedelta(days=730)

    df_new = fetch_current_data(
        ticker=ticker,
        start=last_timestamp,
        bearer_token=bearer_token,
        period_type=period_type,
        frequency_type=frequency_type,
        frequency=frequency,
    )

    if df_new.empty:
        print("No new data available from the API.")
        df_prepared, _ = prepare_features(
            df_features, scaler_path=scaler_path, fit_scaler=False
        )
        return df_prepared

    # Incrementally update features
    df_combined = calculate_features_incrementally(df_features, df_new)

    # Prepare features & apply scaling
    df_prepared, _ = prepare_features(
        df_combined, scaler_path=scaler_path, fit_scaler=False
    )

    return df_prepared


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
        df[group_id_col] = "SPY"

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
        train=False, batch_size=batch_size, shuffle=False
    )

    # Load TFT model
    tft_model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    tft_model.eval()

    with torch.no_grad():
        raw_predictions = tft_model.predict(inference_dataloader)
        # shape is (batch_size, max_prediction_length) => (batch_size, 12)

        # We want the 12th step (index 11) from the *last* batch entry
        # raw_predictions[-1, 11] => final step of the last sample in the batch
        pred_value = raw_predictions[-1, -1].item()

    return pred_value


def predict_close_price_tft(
    ticker="SPY",
    interval="5min",
    model_path="models/tft_model.pth",
    feature_path="data/processed/spy_features.parquet",
    scaler_path="data/processed/scaler.pkl",
    max_encoder_length=48,
    max_prediction_length=12,
):
    """
    High-level orchestrator that:
    1) Fetches & increments updates
    2) Builds TFT dataset
    3) Predicts using TFT model (12 steps ahead).
    """
    df_prepared = fetch_and_incremental_update(
        feature_path=feature_path,
        scaler_path=scaler_path,
        ticker=ticker,
        interval=interval,
        days=200,
    )

    known_reals = ["time_idx"]
    all_cols = list(df_prepared.columns)
    exclude_cols = {"timestamp", "symbol", "time_idx", "close"}  # 'close' is target
    unknown_reals = [col for col in all_cols if col not in exclude_cols]

    tft_dataset = build_tft_dataset(
        df=df_prepared,
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
    # We want to forecast 1 hour out on 5-min data => 12 steps
    pred_close = predict_close_price_tft(
        ticker="SPY",
        interval="5min",
        model_path="models/tft_model.pth",
        feature_path="data/processed/spy_features.parquet",
        scaler_path="data/processed/scaler.pkl",
        max_encoder_length=48,  # see 48 past bars (4 hours) for context
        max_prediction_length=12,  # predict the next 12 bars => 1 hour total
    )
    print(f"Predicted close in 1 hour: {pred_close:.4f}")
