import pandas as pd
from datetime import datetime, timedelta
import pandas_market_calendars as mcal

OUTPUT_FILE = "data/raw/SPY_5min_data.parquet"
START_DATE = datetime(2023, 1, 10)  # Start of data
END_DATE = datetime(2025, 1, 10, 23, 55)  # End of data (last valid 5-minute interval)


def generate_market_hours_timestamps(start_date, end_date, interval_minutes=5):
    # Get the NYSE trading calendar
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    # Generate timestamps for each trading day
    all_timestamps = []
    for index, row in schedule.iterrows():
        market_open = row["market_open"].tz_localize(None)  # Remove timezone info
        market_close = row["market_close"].tz_localize(None)
        # Generate timestamps excluding the exact close time
        day_timestamps = pd.date_range(
            start=market_open,
            end=market_close - timedelta(minutes=1),
            freq=f"{interval_minutes}T",
        )
        all_timestamps.extend(day_timestamps)

    return pd.Series(all_timestamps)


def check_timestamp_order(file_path):
    import pandas as pd

    df = pd.read_parquet(file_path)
    if not df["timestamp"].is_monotonic_increasing:
        print("Timestamps are not in order!")
        # Optionally, reorder the data
        df.sort_values(by="timestamp", inplace=True)
        df.to_parquet(file_path, engine="pyarrow", compression="snappy", index=False)
        print("Timestamps have been reordered and saved.")
    else:
        print("Timestamps are in order.")


def verify_data(filename, start_date, end_date, interval_minutes=5):
    # Load the data
    df = pd.read_parquet(filename)
    print(f"Loaded {len(df)} rows from {filename}.")

    check_timestamp_order(filename)

    # Ensure the timestamp column is a datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Generate expected timestamps for market hours only
    expected_timestamps = generate_market_hours_timestamps(
        start_date, end_date, interval_minutes
    )

    # Find missing and duplicate timestamps
    actual_timestamps = pd.Series(df["timestamp"].sort_values())
    missing_timestamps = set(expected_timestamps) - set(actual_timestamps)
    duplicate_timestamps = actual_timestamps[actual_timestamps.duplicated()]

    # Print results
    print(f"Expected timestamps: {len(expected_timestamps)}")
    print(f"Actual timestamps: {len(actual_timestamps)}")
    print(f"Missing timestamps: {len(missing_timestamps)}")
    print(f"Duplicate timestamps: {len(duplicate_timestamps)}")

    if missing_timestamps:
        print("Missing timestamps:")
        print(sorted(missing_timestamps)[:10])  # Print first 10 missing for brevity

    if not duplicate_timestamps.empty:
        print("Duplicate timestamps:")
        print(duplicate_timestamps)

    # Return validation status
    if not missing_timestamps and duplicate_timestamps.empty:
        print("Data verification passed: No missing or duplicate timestamps.")
    else:
        print("Data verification failed: Check missing or duplicate timestamps.")


if __name__ == "__main__":
    verify_data(OUTPUT_FILE, START_DATE, END_DATE)
