import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import joblib
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config_loader import config

from compute_features import compute_indicators


stoch_intervals = [5, 10, 15, 20, 25, 30]
ma_periods = [20, 50, 60, 120, 200, 300]
continuous_columns = (
    [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "macd",
        "macd_signal",
        "macd_hist",
        "adx",
        "bollinger_upper",
        "bollinger_lower",
        "acceleration_band",
        "atr",
        "tsi",
    ]
    + [f"stoch_k_{i}" for i in stoch_intervals]
    + [f"stoch_d_{i}" for i in stoch_intervals]
    + [f"kdj_k_{i}" for i in stoch_intervals]
    + [f"kdj_d_{i}" for i in stoch_intervals]
    + [f"kdj_j_{i}" for i in stoch_intervals]
    + [f"ma_{i}" for i in ma_periods]
)


def prepare_features(df, scaler_path=None, fit_scaler=False):
    """
    1) Sort by timestamp
    2) Compute all indicators
    3) Drop NaNs
    4) Scale columns
    5) Return (df, scaler)
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)

    # Compute indicators using the base function
    df = compute_indicators(df, stoch_intervals, ma_periods)

    # Drop rows with NaNs (from rolling periods, etc.)
    df.dropna(inplace=True)

    # Confirm all columns exist, warn if missing
    missing_columns = [col for col in continuous_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: missing columns: {missing_columns}")

    # Fit or load scaler
    if fit_scaler or not (scaler_path and os.path.exists(scaler_path)):
        # If we need to fit a scaler either because fit_scaler is True
        # or the scaler file doesn't exist, then do so.
        scaler = StandardScaler()
        scaler.fit(df[continuous_columns])
        if scaler_path:
            joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)

    # Transform
    df[continuous_columns] = scaler.transform(df[continuous_columns])

    return df, scaler


if __name__ == "__main__":
    # Check if processed file already exists
    if os.path.exists(config.processed_file):
        print(
            f"Processed file {config.processed_file} already exists. Skipping processing."
        )
    else:
        df = pd.read_parquet(config.raw_file)
        df, scaler = prepare_features(
            df, scaler_path=config.scaler_file, fit_scaler=False
        )
        df.to_parquet(
            config.processed_file, engine="pyarrow", compression="snappy", index=False
        )
        print(f"Processing complete. File saved to {config.processed_file}")

    # Check loaded data regardless
    df = pd.read_parquet(config.processed_file)
    print(f"Total rows loaded: {len(df)}")
    print(df.tail())
