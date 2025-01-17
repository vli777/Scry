from datetime import datetime
import pandas as pd

from downloaders.schwab import fetch_data_schwab


def fetch_current_data(
    ticker: str = "SPY",
    start: datetime = None,
    bearer_token: str = "your_bearer_token",
    period_type: str = "day",
    frequency_type: str = "minute",
    frequency: int = 5,
):
    """
    Fetch recent data from the Schwab API, starting from the last timestamp in the dataset.

    :param ticker: The stock ticker symbol.
    :param start: The last timestamp in the existing dataset (datetime).
    :param bearer_token: The API bearer token for authentication.
    :param period_type: Data period granularity e.g. 'day'
    :param frequency_type: Tick interval granularity, default 'minute'
    :param frequency: Tick interval number value, default 5 for 5 minute bars
    :return: DataFrame with the new data.
    """
    # --- Define the date range ---
    from_date = (
        pd.Timestamp(start) if start else pd.Timestamp.now() - pd.Timedelta(days=1)
    )
    to_date = pd.Timestamp.now()

    # --- Fetch data ---
    candles = fetch_data_schwab(
        symbol=ticker,
        from_date=from_date.to_pydatetime(),
        to_date=to_date.to_pydatetime(),
        bearer_token=bearer_token,
        period_type=period_type,
        frequency_type=frequency_type,
        frequency=frequency,
    )

    # --- Convert to DataFrame ---
    df = pd.DataFrame(candles)
    if not df.empty:
        # Convert 'datetime' (API response) to pandas Timestamp
        df["timestamp"] = pd.to_datetime(df["datetime"], unit="ms")
        df = df.rename(columns={"datetime": "timestamp"})  # Ensure consistency
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    return df
