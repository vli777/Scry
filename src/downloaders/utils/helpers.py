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
