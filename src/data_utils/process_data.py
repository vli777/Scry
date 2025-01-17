import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import joblib

from data_utils.compute_features import compute_indicators

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
    if fit_scaler:
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
    # Load the dataset
    FILE_PATH = "data/raw/SPY_5min_data.parquet"
    SCALER_PATH = "data/processed/scaler.pkl"
    PROCESSED_FILE_PATH = "data/processed/SPY_5min_processed.parquet"

    df = pd.read_parquet(FILE_PATH)
    df = prepare_features(df, scaler_path=SCALER_PATH)
    df.to_parquet(
        PROCESSED_FILE_PATH, engine="pyarrow", compression="snappy", index=False
    )
