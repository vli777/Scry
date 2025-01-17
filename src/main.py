import argparse
import os
from dotenv import load_dotenv

from training.tft import train_tft
from inference.tft_predict import fetch_and_incremental_update, predict_close_price_tft


def main():
    load_dotenv()  # Load .env variables if you are using them

    # Grab defaults from environment variables or use fallback
    default_data_path = os.getenv("DATA_PATH", "data/processed/spy_features.parquet")
    default_model_path = os.getenv("MODEL_PATH", "models/tft_model.pth")

    parser = argparse.ArgumentParser(description="CLI for ML tasks")
    parser.add_argument("command", choices=["train", "batch_predict", "fetch_data"])
    parser.add_argument("--data-path", default=default_data_path)
    parser.add_argument("--model-path", default=default_model_path)

    # If you want optional arguments to override intervals or other model params
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--interval", default="5min")
    parser.add_argument("--max-encoder-length", type=int, default=48)
    parser.add_argument("--max-prediction-length", type=int, default=12)

    args = parser.parse_args()

    if args.command == "train":
        # Train the TFT model offline
        train_tft(data_path=args.data_path, model_path=args.model_path)
    elif args.command == "batch_predict":
        # Use your existing predict function
        pred_close = predict_close_price_tft(
            ticker=args.ticker,
            interval=args.interval,
            model_path=args.model_path,
            feature_path=args.data_path,
            scaler_path="data/processed/scaler.pkl",
            max_encoder_length=args.max_encoder_length,
            max_prediction_length=args.max_prediction_length,
        )
        print(f"Predicted close: {pred_close:.4f}")
    elif args.command == "fetch_data":
        # Incrementally update your dataset
        fetch_and_incremental_update(feature_path=args.data_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
