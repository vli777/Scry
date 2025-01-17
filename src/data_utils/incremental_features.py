import pandas as pd
import pandas_ta as ta


def calculate_features_incrementally(
    df_recent,
    df_new,
    stoch_intervals=[5, 10, 15, 20, 25, 30],
    ma_periods=[20, 50, 60, 120, 200],
):
    """
    Incrementally update rolling features for new data.

    :param df_recent: DataFrame containing the most recent data with precomputed features.
    :param df_new: DataFrame containing new data to append.
    :param stoch_intervals: List of intervals for stochastic and KDJ calculations.
    :param ma_periods: List of moving average periods.
    :return: Updated DataFrame with all features recalculated incrementally.
    """
    # Append new data to recent data
    df = pd.concat([df_recent, df_new]).reset_index(drop=True)

    # Ensure only the required rolling window (longest period) is retained
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=200)
    df = df[df["timestamp"] >= cutoff_date].reset_index(drop=True)

    # --- MACD ---
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    # --- ADX ---
    adx = ta.adx(df["high"], df["low"], df["close"])
    df["adx"] = adx["ADX_14"]

    # --- Bollinger Bands ---
    bollinger = ta.bbands(df["close"], length=20, std=2.0)
    df["bollinger_upper"] = bollinger["BBU_20_2.0"]
    df["bollinger_lower"] = bollinger["BBL_20_2.0"]
    df["acceleration_band"] = bollinger["BBU_20_2.0"] - bollinger["BBL_20_2.0"]

    # --- Stochastic and KDJ ---
    for interval in stoch_intervals:
        # Stochastic
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=interval, d=3)
        df[f"stoch_k_{interval}"] = stoch[f"STOCHk_{interval}_3_3"]
        df[f"stoch_d_{interval}"] = stoch[f"STOCHd_{interval}_3_3"]

        # KDJ
        kdj = ta.kdj(df["high"], df["low"], df["close"], length=interval, signal=3)
        df[f"kdj_k_{interval}"] = kdj[f"K_{interval}_3"]
        df[f"kdj_d_{interval}"] = kdj[f"D_{interval}_3"]
        df[f"kdj_j_{interval}"] = kdj[f"J_{interval}_3"]

    # --- Moving Averages ---
    for period in ma_periods:
        df[f"ma_{period}"] = df["close"].rolling(period).mean()

    # --- ATR ---
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # --- TSI ---
    tsi = ta.tsi(df["close"], fast=13, slow=25, signal=13)
    df["tsi"] = tsi["TSI_13_25_13"]
    df["tsi_signal"] = tsi["TSIs_13_25_13"]

    # Return the full dataset with updated features
    return df
