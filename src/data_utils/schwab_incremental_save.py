import pandas as pd

from downloaders.schwab import fetch_data_schwab


def fetch_current_data(
    ticker, interval="1h", start=None, bearer_token="your_bearer_token"
):
    """
    Fetch recent data from the Schwab API, starting from the last timestamp in the dataset.

    :param ticker: The stock ticker symbol.
    :param interval: Data interval (e.g., '1h', '1d').
    :param start: The last timestamp in the existing dataset (datetime).
    :param bearer_token: The API bearer token for authentication.
    :return: DataFrame with the new data.
    """
    # --- Interval mapping ---
    interval_map = {
        "1h": ("day", "minute", 60),  # Daily period, hourly frequency
        "1d": ("month", "daily", 1),  # Monthly period, daily frequency
    }

    if interval not in interval_map:
        raise ValueError(f"Unsupported interval: {interval}")

    period_type, frequency_type, frequency = interval_map[interval]

    # --- Define the date range ---
    from_date = (
        pd.Timestamp(start) if start else pd.Timestamp.now() - pd.Timedelta(days=1)
    )
    to_date = pd.Timestamp.now()

    # --- Fetch data ---
    candles = fetch_data_schwab(
        symbol=ticker,
        period_type=period_type,
        frequency_type=frequency_type,
        frequency=frequency,
        from_date=from_date,
        to_date=to_date,
        bearer_token=bearer_token,
    )

    # --- Convert to DataFrame ---
    df = pd.DataFrame(candles)
    if not df.empty:
        # Convert 'datetime' (API response) to pandas Timestamp
        df["timestamp"] = pd.to_datetime(df["datetime"], unit="ms")
        df = df.rename(columns={"datetime": "timestamp"})  # Ensure consistency
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    return df
