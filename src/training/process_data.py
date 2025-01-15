import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
FILE_PATH = "data/raw/SPY_5min_data.parquet"
df = pd.read_parquet(FILE_PATH)

# Ensure timestamp is a datetime object
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort data by timestamp
df.sort_values(by="timestamp", inplace=True)

# Add Technical Indicators
# MACD
macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
df["macd"] = macd["MACD_12_26_9"]
df["macd_signal"] = macd["MACDs_12_26_9"]
df["macd_hist"] = macd["MACDh_12_26_9"]

# ADX
adx = ta.adx(df["high"], df["low"], df["close"])
df["adx"] = adx["ADX_14"]

# Bollinger Bands
bollinger = ta.bbands(df["close"], length=20, std=2.0)
df["bollinger_upper"] = bollinger["BBU_20_2.0"]
df["bollinger_lower"] = bollinger["BBL_20_2.0"]
df["acceleration_band"] = bollinger["BBU_20_2.0"] - bollinger["BBL_20_2.0"]

# Stochastic Oscillators and KDJ for 5, 10, 15, ..., 30 intervals
stoch_intervals = [5, 10, 15, 20, 25, 30]
for interval in stoch_intervals:
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=interval, d=3)
    df[f"stoch_k_{interval}"] = stoch[f"STOCHk_{interval}_3_3"]
    df[f"stoch_d_{interval}"] = stoch[f"STOCHd_{interval}_3_3"]

    kdj_df = ta.kdj(
        high=df["high"], low=df["low"], close=df["close"], length=interval, signal=3
    )
    df[f"kdj_k_{interval}"] = kdj_df[f"K_{interval}_3"]
    df[f"kdj_d_{interval}"] = kdj_df[f"D_{interval}_3"]
    df[f"kdj_j_{interval}"] = kdj_df[f"J_{interval}_3"]

# Moving Averages
ma_periods = [20, 50, 60, 120, 200, 300]
for period in ma_periods:
    df[f"ma_{period}"] = df["close"].rolling(period).mean()

# adv true range
df["atr"] = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=14)

# Calculate True Strength Index (TSI)
tsi_df = ta.tsi(df["close"], fast=13, slow=25, signal=13)
df["tsi"] = tsi_df[f"TSI_13_25_13"]
df["tsi_signal"] = tsi_df[f"TSIs_13_25_13"]

# Normalize Continuous Features
scaler = StandardScaler()
continuous_columns = (
    [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "macd",
        "macd_signal",
        "macd_hist",
        "adx",
        "bollinger_upper",
        "bollinger_lower",
        "acceleration_band",
        "atr",
        "tsi",
    ]
    + [f"stoch_k_{i}" for i in stoch_intervals]
    + [f"stoch_d_{i}" for i in stoch_intervals]
    + [f"kdj_k_{i}" for i in stoch_intervals]
    + [f"kdj_d_{i}" for i in stoch_intervals]
    + [f"kdj_j_{i}" for i in stoch_intervals]
    + [f"ma_{i}" for i in ma_periods]
)

# Check for missing columns
missing_columns = [col for col in continuous_columns if col not in df.columns]
if missing_columns:
    print(f"Warning: Missing columns - {missing_columns}")

# Apply scaling only to rows where all rolling calculations are complete
df.dropna(
    inplace=True
)  # Drop rows with NaN values from rolling or indicator calculations
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

# Save the cleaned and processed dataset
PROCESSED_FILE_PATH = "data/processed/SPY_5min_processed.parquet"
df.to_parquet(PROCESSED_FILE_PATH, engine="pyarrow", compression="snappy", index=False)

# Save scaler
SCALER_FILE_PATH = "data/processed/scaler.pkl"
joblib.dump(scaler, SCALER_FILE_PATH)
print(f"Scaler saved to {SCALER_FILE_PATH}")

print(f"Processed data saved to {PROCESSED_FILE_PATH}")
