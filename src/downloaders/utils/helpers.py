import os
import pandas as pd


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


def inspect_last_data(filename):
    # Load the Parquet file into a DataFrame
    df = pd.read_parquet(filename)

    # Convert the 'timestamp' column from epoch milliseconds to datetime, if it exists
    if "timestamp" in df.columns:
        # Create a new column with converted timestamps
        df["converted_timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Print the last 5 rows of the DataFrame
    print("Last 5 rows of the data:")
    print(df.tail(5))

    # Specifically print the original and converted timestamps for clarity
    if "timestamp" in df.columns and "converted_timestamp" in df.columns:
        print("\nOriginal and converted timestamps for the last 5 rows:")
        print(df[["timestamp", "converted_timestamp"]].tail(5))
