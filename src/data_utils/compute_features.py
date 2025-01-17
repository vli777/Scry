import pandas as pd
import pandas_ta as ta


def compute_indicators(
    df: pd.DataFrame,
    stoch_intervals=[5, 10, 15, 20, 25, 30],
    ma_periods=[20, 50, 60, 120, 200],
) -> pd.DataFrame:
    """
    Compute all technical indicators on the given DataFrame.

    :param df: DataFrame with at least columns ['open', 'high', 'low', 'close', 'volume'].
    :param stoch_intervals: List of intervals for stoch/KDJ.
    :param ma_periods: List of MA periods.
    :return: DataFrame with new indicator columns added.
    """

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

    # --- Stochastic & KDJ ---
    for interval in stoch_intervals:
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=interval, d=3)
        df[f"stoch_k_{interval}"] = stoch[f"STOCHk_{interval}_3_3"]
        df[f"stoch_d_{interval}"] = stoch[f"STOCHd_{interval}_3_3"]

        kdj = ta.kdj(
            high=df["high"], low=df["low"], close=df["close"], length=interval, signal=3
        )
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

    return df
