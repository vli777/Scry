import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import joblib

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
    Prepare features (technical indicators, scaling, etc.) for a given DataFrame.

    :param df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'vwap', 'timestamp']
    :param scaler_path: Path to load or save the fitted scaler, depending on `fit_scaler`.
    :param fit_scaler: If True, fit a new scaler to `df`. Otherwise, load from `scaler_path`.
    :return: df, scaler
    """
    # --- 1) Sort by timestamp and ensure datetime ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)

    # --- 2) Compute the same technical indicators as in training ---

    # MACD
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    # ADX
    adx = ta.adx(df["high"], df["low"], df["close"])
    df["adx"] = adx["ADX_14"]

    # Bollinger Bands
    bollinger = ta.bbands(df["close"], length=20, std=2.0)
    df["bollinger_upper"] = bollinger["BBU_20_2.0"]
    df["bollinger_lower"] = bollinger["BBL_20_2.0"]
    df["acceleration_band"] = bollinger["BBU_20_2.0"] - bollinger["BBL_20_2.0"]

    # Stochastic, KDJ, etc. (same intervals as in training)
    for interval in stoch_intervals:
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=interval, d=3)
        df[f"stoch_k_{interval}"] = stoch[f"STOCHk_{interval}_3_3"]
        df[f"stoch_d_{interval}"] = stoch[f"STOCHd_{interval}_3_3"]

        kdj_df = ta.kdj(
            high=df["high"], low=df["low"], close=df["close"], length=interval, signal=3
        )
        df[f"kdj_k_{interval}"] = kdj_df[f"K_{interval}_3"]
        df[f"kdj_d_{interval}"] = kdj_df[f"D_{interval}_3"]
        df[f"kdj_j_{interval}"] = kdj_df[f"J_{interval}_3"]

    # Moving Averages (same periods)
    for period in ma_periods:
        df[f"ma_{period}"] = df["close"].rolling(period).mean()

    # ATR
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # TSI
    tsi_df = ta.tsi(df["close"], fast=13, slow=25, signal=13)
    df["tsi"] = tsi_df["TSI_13_25_13"]
    df["tsi_signal"] = tsi_df["TSIs_13_25_13"]

    # --- 3) Drop rows with NaNs introduced by rolling calculations ---
    df.dropna(inplace=True)

    # Ensure all columns exist; fill or drop if needed
    missing_columns = [col for col in continuous_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: missing columns: {missing_columns}")
        # Either fill with 0 or drop them, depending on your preference

    # If fitting a new scaler (during initial training)...
    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(df[continuous_columns])
        if scaler_path:
            joblib.dump(scaler, scaler_path)
    else:
        # Otherwise load the existing scaler
        scaler = joblib.load(scaler_path)

    # Apply the transformation
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
