from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import time

from src.downloaders.utils.helpers import get_last_saved_timestamp
from src.config_loader import config

load_dotenv(override=True)

api_token = os.environ.get("POLYGON_API_TOKEN")

if api_token is None:
    raise ValueError("POLYGON_API_TOKEN environment variable not set")

API_KEY = api_token
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=2 * 365)  # 2 Years back
CHUNK_DAYS = 90  # 3-month chunks
RATE_LIMIT_CALLS = 5  # Max calls per minute
RATE_LIMIT_SLEEP = 60  # Seconds to sleep after RATE_LIMIT_CALLS


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


def main():
    # Get the last saved timestamp or start from the default start date
    last_saved_timestamp = get_last_saved_timestamp(config.raw_file)
    if last_saved_timestamp:
        current_start = last_saved_timestamp + timedelta(
            minutes=config.frequency
        )  # Start x minutes after the last saved timestamp
        print(f"Resuming from {current_start}...")
    else:
        current_start = START_DATE
        print(f"Starting from the default start date: {current_start}...")

    call_count = 0
    start_time = time.time()
    previous_start = None

    while current_start < END_DATE:
        # Break if current_start hasn't moved since last iteration
        if current_start == previous_start:
            print("No progress in fetching; breaking loop.")
            break
        previous_start = current_start

        current_end = min(current_start + timedelta(days=CHUNK_DAYS), END_DATE)
        print(f"Fetching data from {current_start} to {current_end}...")
        try:
            data = fetch_data(
                config.symbol,
                config.frequency,
                config.frequency_type,
                current_start.date(),
                current_end.date(),
            )
            if data:
                save_chunk_to_parquet(data, config.raw_file)
                print(f"Saved data for {current_start.date()} to {current_end.date()}.")
                # Update current_start based on the last data point fetched
                current_start = pd.to_datetime(data[-1]["t"], unit="ms") + timedelta(
                    minutes=config.frequency
                )
            else:
                print(f"No data for {current_start.date()} to {current_end.date()}.")
                current_start = current_end + timedelta(days=1)  # Skip to next chunk

            # Rate-limiting
            call_count += 1
            if call_count % RATE_LIMIT_CALLS == 0:
                elapsed_time = time.time() - start_time
                if elapsed_time < RATE_LIMIT_SLEEP:
                    sleep_time = RATE_LIMIT_SLEEP - elapsed_time
                    print(
                        f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds..."
                    )
                    time.sleep(sleep_time)
                # Reset start_time for the next batch of 5 calls
                start_time = time.time()

        except Exception as e:
            print(
                f"Error fetching data for {current_start.date()} to {current_end.date()}: {e}"
            )
            break  # Exit loop on error to avoid overwriting or duplicating data

    print(f"All data saved to {config.raw_file}.")


if __name__ == "__main__":
    main()
