import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import time


api_token = os.environ.get("POLYGON_API_TOKEN")

if api_token is None:
    raise ValueError("POLYGON_API_TOKEN environment variable not set")

API_KEY = api_token
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"
TICKER = "SPY"
MULTIPLIER = 5  # 5-minute bars
TIMESPAN = "minute"
START_DATE = datetime(2023, 1, 10)  # Original start date
END_DATE = datetime(2025, 1, 10)  # Final end date
CHUNK_DAYS = 90  # 3-month chunks
RATE_LIMIT_CALLS = 5  # Max calls per minute
RATE_LIMIT_SLEEP = 12  # Seconds to sleep after RATE_LIMIT_CALLS
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = "SPY_5min_data.parquet"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_data(ticker, multiplier, timespan, from_date, to_date, adjusted=True):
    url = f"{BASE_URL}/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": "asc",
        "limit": 50000,  # Fetch the maximum number of rows
        "apiKey": API_KEY,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("results", [])


def save_chunk_to_parquet(data, filename):
    df = pd.DataFrame(data)
    if df.empty:
        print("No data to save.")
        return
    # Convert and rename columns
    df["timestamp"] = pd.to_datetime(
        df["t"], unit="ms"
    )  # Convert timestamp to datetime

    # Include VWAP in the dataset
    df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "transactions",
        },
        inplace=True,
    )

    # Retain all relevant columns
    df = df[
        ["timestamp", "open", "high", "low", "close", "volume", "vwap", "transactions"]
    ]

    if os.path.exists(filename):
        existing_data = pd.read_parquet(filename)
        df = pd.concat([existing_data, df], ignore_index=True)
    df.drop_duplicates(subset=["timestamp"], inplace=True)  # Ensure no duplicates
    df.to_parquet(filename, engine="pyarrow", compression="snappy", index=False)
    print(f"Data saved to {filename}.")


def get_last_saved_timestamp(filename):
    if not os.path.exists(filename):
        return None
    df = pd.read_parquet(filename)
    if df.empty:
        return None
    return df["timestamp"].max()


def main():
    # Get the last saved timestamp or start from the default start date
    last_saved_timestamp = get_last_saved_timestamp(
        os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    )
    if last_saved_timestamp:
        current_start = last_saved_timestamp + timedelta(
            minutes=5
        )  # Start 5 minutes after the last saved timestamp
        print(f"Resuming from {current_start}...")
    else:
        current_start = START_DATE
        print(f"Starting from the default start date: {current_start}...")

    call_count = 0

    while current_start < END_DATE:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS), END_DATE)
        print(f"Fetching data from {current_start} to {current_end}...")
        try:
            data = fetch_data(
                TICKER, MULTIPLIER, TIMESPAN, current_start.date(), current_end.date()
            )
            if data:
                save_chunk_to_parquet(data, os.path.join(OUTPUT_DIR, OUTPUT_FILE))
                print(f"Saved data for {current_start.date()} to {current_end.date()}.")
                # Update current_start based on the last data point fetched
                current_start = pd.to_datetime(data[-1]["t"], unit="ms") + timedelta(
                    minutes=5
                )
            else:
                print(f"No data for {current_start.date()} to {current_end.date()}.")
                current_start = current_end + timedelta(days=1)  # Skip to next chunk

            # Rate-limiting
            call_count += 1
            if call_count % RATE_LIMIT_CALLS == 0:
                print(f"Rate limit reached. Sleeping for {RATE_LIMIT_SLEEP} seconds...")
                time.sleep(RATE_LIMIT_SLEEP)
        except Exception as e:
            print(
                f"Error fetching data for {current_start.date()} to {current_end.date()}: {e}"
            )
            break  # Exit loop on error to avoid overwriting or duplicating data

    print(f"All data saved to {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}.")


if __name__ == "__main__":
    main()
