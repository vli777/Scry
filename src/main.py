from datetime import datetime, timedelta, timezone
import os
import argparse
from dotenv import load_dotenv


from .config_loader import config

from downloaders.schwab import fetch_data_schwab, save_to_parquet
from downloaders.utils.helpers import get_last_saved_timestamp
from inference.tft_predict import predict_close_price_tft
from training.tft_train import train_tft


def main():
    load_dotenv()

    # CLI Parser
    parser = argparse.ArgumentParser(description="CLI for ML tasks")
    parser.add_argument(
        "command",
        choices=[
            "data",
            "train",
            "predict",
        ],
        help="Task to perform",
    )
    parser.add_argument(
        "--max-encoder-length", type=int, default=48, help="Max encoder length for TFT"
    )
    parser.add_argument(
        "--max-prediction-length",
        type=int,
        default=12,
        help="Max prediction length for TFT",
    )

    args = parser.parse_args()

    bearer_token = os.getenv("SCHWAB_BEARER_TOKEN")
    if bearer_token is None:
        raise ValueError("SCHWAB_BEARER_TOKEN environment variable not set")

    try:
        if args.command == "data":
            handle_data(args, bearer_token)
        elif args.command == "train":
            handle_train(args)
        elif args.command == "predict":
            handle_predict(args, bearer_token)
        else:
            parser.print_help()
    except Exception as e:
        print(f"An error occurred: {e}")


def handle_data(args, bearer_token):
    """Handle data fetching logic."""
    print(f"Fetching and updating data for: {config.symbol}")

    # Determine start and end dates
    last_saved_timestamp = get_last_saved_timestamp(config.raw_file)
    now = datetime.now(timezone.utc)

    # Assume market closes at 21:00 UTC
    market_close_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
    effective_end_date = min(now, market_close_time)

    start_date = (
        last_saved_timestamp + timedelta(minutes=config.frequency)
        if last_saved_timestamp
        else now - timedelta(days=10)  # Default to the last 10 days
    )

    # Fetch new data using `fetch_data_schwab`
    candles = fetch_data_schwab(
        symbol=config.symbol,
        bearer_token=bearer_token,
        frequency_type=config.frequency_type,
        frequency=config.frequency,
        start_date=start_date,
        end_date=effective_end_date,
        max_chunk_days=10,  # You can retain chunking just in case
    )

    if candles:
        save_to_parquet(candles, config.raw_file)
        print(f"New data appended to {config.raw_file}")
    else:
        print("No new data fetched.")


def handle_train(args):
    """Handle training logic."""
    print(f"Starting training with processed data: {config.processed_file}")
    train_tft(data_path=config.processed_file, model_path=config.model_path)


def handle_predict(args, bearer_token):
    """Handle batch prediction logic."""
    print(f"Running batch prediction for ticker: {config.symbol}")
    pred_close = predict_close_price_tft(
        config=config,
        bearer_token=bearer_token,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
    )
    print(f"Predicted close price: {pred_close:.4f}")


if __name__ == "__main__":
    main()
