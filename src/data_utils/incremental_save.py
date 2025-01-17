from dotenv import load_dotenv
import os
import pandas as pd

from inference.lgbm_predict import calculate_features_incrementally, fetch_current_data

# Load environment variables
load_dotenv()


def save_incremental_data(
    ticker="SPY", interval="5min", feature_path="data/processed/spy_features.parquet"
):
    """
    Fetch new 5-minute data and update saved features incrementally, skipping API calls if data is up-to-date.
    """
    # Load existing features
    df_recent = pd.read_parquet(feature_path)
    last_timestamp = pd.to_datetime(df_recent["timestamp"].max())

    now = pd.Timestamp.now()

    # If you want a "no-op" check for the last 5 minutes of trading (4:00 PM close),
    # you can do something like:
    market_close = pd.Timestamp(now.date()) + pd.Timedelta(hours=16)  # 4:00 PM
    if last_timestamp >= market_close - pd.Timedelta(minutes=5):
        print(f"Data is already up-to-date for {ticker} at {interval} interval.")
        return

    # Fetch new data using the bearer token from .env
    bearer_token = os.getenv("SCHWAB_BEARER_TOKEN")
    df_today = fetch_current_data(
        ticker, interval=interval, start=last_timestamp, bearer_token=bearer_token
    )

    if df_today.empty:
        print(f"No new data to fetch for {ticker} at {interval} interval.")
        return

    # Incrementally update features
    df_combined = calculate_features_incrementally(df_recent, df_today)

    # Save updated features back to the file
    df_combined.to_parquet(feature_path, index=False)
    print(f"Features updated and saved to {feature_path}")


if __name__ == "__main__":
    save_incremental_data(ticker="SPY", interval="5min")
