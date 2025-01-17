import joblib
import pandas as pd

from data_utils.incremental_features import calculate_features_incrementally
from data_utils.incremental_data import fetch_current_data
from training.process_data import prepare_features


def predict_close_price(
    ticker="SPY",
    interval="1h",
    model_path="models/lgbm_model.pkl",
    feature_path="data/processed/spy_features.parquet",
    scaler_path="data/processed/scaler.pkl",
):
    """
    Predict the next close price for a given ticker and interval.
    Focuses only on inference, without saving new data.
    """
    # --- Load the last 200 days of data ---
    df_features = pd.read_parquet(feature_path)
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=200)
    df_recent = df_features[df_features["timestamp"] >= cutoff_date]

    # --- Fetch today's data ---
    df_today = fetch_current_data(
        ticker, interval=interval, start=df_recent["timestamp"].max()
    )

    # --- Update rolling features ---
    df_combined = calculate_features_incrementally(df_recent, df_today)

    # --- Prepare features for prediction ---
    df_prepared, _ = prepare_features(
        df_combined, scaler_path=scaler_path, fit_scaler=False
    )
    last_row = df_prepared.iloc[[-1]]
    continuous_columns = [
        col for col in last_row.columns if col not in {"timestamp", "close"}
    ]

    # --- Load model and make prediction ---
    model = joblib.load(model_path)
    pred = model.predict(last_row[continuous_columns])[0]

    return pred


if __name__ == "__main__":
    # Predict the next 1-hour close for SPY
    next_hour_close = predict_close_price(ticker="SPY", interval="1h")
    print(f"Predicted next 1-hour close: {next_hour_close:.2f}")

    # Predict the next daily close for SPY
    next_day_close = predict_close_price(ticker="SPY", interval="1d")
    print(f"Predicted next daily close: {next_day_close:.2f}")
