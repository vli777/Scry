import os
from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

load_dotenv()

API_TOKEN = os.environ.get("SCHWAB_BEARER_TOKEN")
if API_TOKEN is None:
    raise ValueError("SCHWAB_BEARER_TOKEN environment variable not set")

BASE_URL = "https://api.schwabapi.com/marketdata/v1/pricehistory"
SYMBOL = "SPY"
START_DATE = datetime.now(timezone.utc) - timedelta(days=10)
FREQUENCY_TYPE = "minute"
FREQUENCY = 5
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = f"{SYMBOL}_{FREQUENCY}min_data.parquet"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_data_schwab(
    symbol: str,
    bearer_token: str,
    period_type: str = None,
    period: int = None,
    frequency_type: str = "minute",
    frequency: int = 5,
    start_date: datetime = None,
    end_date: datetime = None,
    max_chunk_days: int = 10,
):
    """
    Unified Schwab fetch function.
    - If no large date range is specified, or period/periodType is specified,
      it does a single request.
    - If start_date/end_date is given (and it's longer than the max chunk size),
      it chunks by max_chunk_days internally.
    - Returns a combined list of candles.
    """
    if period_type and period and not start_date and not end_date:
        # === Single fetch based on period ===
        params = {
            "symbol": symbol,
            "periodType": period_type,
            "period": period,
            "frequencyType": frequency_type,
            "frequency": frequency,
        }
        return _fetch_once_schwab(bearer_token, params)
    elif start_date and end_date:
        # === Potentially chunked fetch based on date range ===
        return _fetch_chunked_schwab(
            symbol=symbol,
            bearer_token=bearer_token,
            frequency_type=frequency_type,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            max_chunk_days=max_chunk_days,
        )
    else:
        # === Fallback: no explicit period/date => default to 10 days on day periodType ===
        params = {
            "symbol": symbol,
            "periodType": "day",
            "period": 10,
            "frequencyType": frequency_type,
            "frequency": frequency,
        }
        return _fetch_once_schwab(bearer_token, params)


def _fetch_once_schwab(bearer_token: str, params: dict):
    """
    Makes a single GET request to the Schwab pricehistory endpoint with the provided parameters.
    Returns a list of candle objects.
    """
    url = BASE_URL
    headers = {"Authorization": f"Bearer {bearer_token}"}
    print(f"Requesting single chunk => {url}, params={params}")

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    candles = data.get("candles", [])

    if candles:
        print(
            f"Received {len(candles)} candles. Last datetime: {candles[-1]['datetime']}"
        )
    else:
        print("No candles received.")

    return candles


def _fetch_chunked_schwab(
    symbol: str,
    bearer_token: str,
    frequency_type: str,
    frequency: int,
    start_date: datetime,
    end_date: datetime,
    max_chunk_days: int = 10,
):
    """
    Automatically fetches data in multiple 10-day chunks for 5-min bars.
    Accumulates candle lists and returns them as one combined list.
    """
    all_candles = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=max_chunk_days), end_date)
        print(
            f"Fetching data from {current_start.isoformat()} to {current_end.isoformat()}..."
        )

        params = {
            "symbol": symbol,
            "periodType": "day",
            "frequencyType": frequency_type,
            "frequency": frequency,
            "startDate": int(current_start.timestamp() * 1000),
            "endDate": int(current_end.timestamp() * 1000),
        }
        chunk_candles = _fetch_once_schwab(bearer_token, params)

        if chunk_candles:
            all_candles.extend(chunk_candles)
            # Move current_start to last bar + 5 minutes
            last_bar_time = pd.to_datetime(
                chunk_candles[-1]["datetime"], unit="ms", utc=True
            )
            print(f"Last bar time in this chunk: {last_bar_time.isoformat()}")
            current_start = last_bar_time + timedelta(minutes=5)
            print(f"Next chunk starts at: {current_start.isoformat()}")
        else:
            # If no data, skip to next chunk
            print(
                f"No data returned for {current_start.isoformat()} to {current_end.isoformat()}."
            )
            current_start = current_end + timedelta(minutes=5)
            print(f"Skipping to next chunk, new start: {current_start.isoformat()}")

    return all_candles


def get_last_saved_timestamp(filename):
    if not os.path.exists(filename):
        return None
    df = pd.read_parquet(filename)
    if df.empty:
        return None
    ts = df["timestamp"].max()
    # Localize to UTC if timestamp is naive
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def fill_missing_gaps(df, freq_minutes=5):
    # Ensure 'timestamp' is a UTC-aware datetime and set it as the DataFrame index
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    # Create a complete date range from the minimum to maximum timestamp at the expected frequency
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=pd.Timedelta(minutes=freq_minutes),
        tz="UTC",
    )

    # Reindex the DataFrame to the full range, inserting NaN for missing timestamps
    df = df.reindex(full_range)
    df.index.name = "timestamp"
    df.reset_index(inplace=True)

    return df


def save_to_parquet(candles, filename, freq_minutes=5):
    if not candles:
        print("No data to save.")
        return

    # Convert the list of candles to a DataFrame and ensure UTC-aware timestamps
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    if os.path.exists(filename):
        existing_data = pd.read_parquet(filename)
        # Ensure existing data timestamps are UTC-aware
        existing_data["timestamp"] = pd.to_datetime(
            existing_data["timestamp"], utc=True
        )
        existing_data = existing_data[
            ["timestamp", "open", "high", "low", "close", "volume"]
        ]

        # Combine old and new data
        df = pd.concat([existing_data, df], ignore_index=True)

    # Remove duplicates based on timestamp
    df.drop_duplicates(subset=["timestamp"], inplace=True)

    # Fill gaps by inserting missing timestamps
    df = fill_missing_gaps(df, freq_minutes=freq_minutes)

    # Save the complete DataFrame to Parquet
    df.to_parquet(filename, engine="pyarrow", compression="snappy", index=False)
    print(f"Data saved to {filename}.")


def main():
    file_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    last_saved_timestamp = get_last_saved_timestamp(file_path)

    if last_saved_timestamp:
        start = last_saved_timestamp + timedelta(minutes=5)
        print(f"Resuming from {start}...")
    else:
        start = START_DATE
        print(f"Starting from the default start date: {start}...")

    chunked_candles = fetch_data_schwab(
        symbol=SYMBOL,
        bearer_token=API_TOKEN,
        frequency_type=FREQUENCY_TYPE,
        frequency=FREQUENCY,
        start_date=start,
        end_date=datetime.now(timezone.utc),
        max_chunk_days=10,  # chunk by api limit of 10 days
    )
    save_to_parquet(chunked_candles, file_path)


if __name__ == "__main__":
    main()
