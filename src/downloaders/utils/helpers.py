import os
import pandas as pd
from pytz import timezone as pytz_timezone 
from datetime import datetime, time, timedelta, timezone


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


def is_market_open(current_time, market_timezone='US/Eastern'):
    """
    Check if the current time is within market hours.
    
    Args:
        current_time (datetime): Current datetime with timezone.
        market_timezone (str): Timezone of the market (default: 'US/Eastern').
        
    Returns:
        bool: True if market is open, False otherwise.
    """
    # Define market open and close times in market's local timezone
    market_tz = pytz_timezone(market_timezone)
    local_time = current_time.astimezone(market_tz)
    
    # Market hours: 9:30 AM to 4:00 PM (NYSE as an example)
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    # Check if today is a weekday (Monday=0, Sunday=6)
    if local_time.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Check if current time is within market hours
    return market_open <= local_time.time() <= market_close
