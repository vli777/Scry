import os
import requests
import pandas as pd
import pyarrow
from datetime import datetime, timedelta
import time

# Env variable or direct string (not recommended for production)
API_TOKEN = os.environ.get("SCHWAB_BEARER_TOKEN")
if API_TOKEN is None:
    raise ValueError("SCHWAB_BEARER_TOKEN environment variable not set")

BASE_URL = "https://api.schwabapi.com/marketdata/v1/pricehistory"
SYMBOL = "SPY"
PERIOD_TYPE = "day"  # from your example
FREQUENCY_TYPE = "minute"  # from your example
FREQUENCY = 5  # 5-minute bars
START_DATE = datetime(2023, 1, 10)
END_DATE = datetime(2025, 1, 10)
CHUNK_DAYS = 90
RATE_LIMIT_CALLS = 5
RATE_LIMIT_SLEEP = 12
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = "SPY_5min_data.parquet"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_data_schwab(
    symbol, period_type, frequency_type, frequency, from_date, to_date, bearer_token
):
    """
    Fetch data from the Schwab Market Data API
    """
    url = BASE_URL
    headers = {"Authorization": f"Bearer {bearer_token}"}

    params = {
        "symbol": symbol,
        "periodType": period_type,
        "frequencyType": frequency_type,
        "frequency": frequency,
        "startDate": int(from_date.timestamp() * 1000),
        "endDate": int(to_date.timestamp() * 1000),
    }

    print(f"Requesting data: {url} params={params}")
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    # Schwab’s response format:
    # { candles: [ { "open", "high", "low", "close", "volume", "datetime" } ] }
    candles = data.get("candles", [])
    return candles


def save_chunk_to_parquet(data, filename):
    """
    Convert the candle list to a DataFrame and save or append to Parquet.
    """
    if not data:
        print("No data to save.")
        return

    df = pd.DataFrame(data)

    # The example Schwab structure:
    # { "open", "high", "low", "close", "volume", "datetime" }

    # Convert datetime to a Python datetime if it's in ms or iso format
    # If it's an epoch in milliseconds:
    # df["timestamp"] = pd.to_datetime(df["datetime"], unit="ms")
    # Otherwise, if it's a standard ISO8601:
    # df["timestamp"] = pd.to_datetime(df["datetime"])
    #
    df["timestamp"] = pd.to_datetime(df["datetime"], unit="ms")

    # Rename columns to match existing schema
    df.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
        inplace=True,
    )

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # If a file already exists, append new data (and deduplicate)
    if os.path.exists(filename):
        existing_data = pd.read_parquet(filename)
        df = pd.concat([existing_data, df], ignore_index=True)

    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.to_parquet(filename, engine="pyarrow", compression="snappy", index=False)
    print(f"Data saved to {filename}.")


def get_last_saved_timestamp(filename):
    """
    Return the most recent timestamp in the existing parquet file, or None if empty.
    """
    if not os.path.exists(filename):
        return None
    df = pd.read_parquet(filename)
    if df.empty:
        return None
    return df["timestamp"].max()


def main():
    last_saved_timestamp = get_last_saved_timestamp(
        os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    )
    if last_saved_timestamp:
        current_start = last_saved_timestamp + timedelta(minutes=5)
        print(f"Resuming from {current_start}...")
    else:
        current_start = START_DATE
        print(f"Starting from the default start date: {current_start}...")

    call_count = 0

    while current_start < END_DATE:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS), END_DATE)
        print(f"Fetching data from {current_start} to {current_end}...")

        try:
            candles = fetch_data_schwab(
                SYMBOL,
                PERIOD_TYPE,
                FREQUENCY_TYPE,
                FREQUENCY,
                current_start,
                current_end,
                bearer_token=API_TOKEN,
            )

            if candles:
                # Save chunk to parquet
                save_chunk_to_parquet(candles, os.path.join(OUTPUT_DIR, OUTPUT_FILE))
                print(f"Saved data for {current_start} to {current_end}.")
                # Move current_start to last bar
                last_bar_time = pd.to_datetime(candles[-1]["datetime"], unit="ms")
                current_start = last_bar_time + timedelta(minutes=5)
            else:
                print(f"No data returned for {current_start} to {current_end}.")
                current_start = current_end + timedelta(days=1)

            # Rate limiting
            call_count += 1
            if call_count % RATE_LIMIT_CALLS == 0:
                print(f"Rate limit reached. Sleeping for {RATE_LIMIT_SLEEP} seconds...")
                time.sleep(RATE_LIMIT_SLEEP)

        except Exception as e:
            print(f"Error fetching data for {current_start} to {current_end}: {e}")
            break

    print(f"All data saved to {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}.")


if __name__ == "__main__":
    main()
