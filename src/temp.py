import numpy as np
import joblib
from src.downloaders.utils.helpers import inspect_last_data
from src.config_loader import config
from src.preprocessing.process_data import continuous_columns

# 1. Get the last prediction sequence as a numeric array
latest_prediction = np.array(
    [1.638, 1.653, 1.644, 1.630, 1.614, 1.596, 1.575, 1.550, 1.523, 1.494, 1.463, 1.430]
)

# 2. Load the scaler
scaler = joblib.load(config.scaler_file)
n_features = len(scaler.mean_)
target_idx = continuous_columns.index("close")

# Create a dummy array filled with the scaler's mean values
dummy = np.tile(scaler.mean_, (12, 1))  # shape (12, n_features)

# Replace the 'close' column in the dummy array with your scaled predictions
dummy[:, target_idx] = latest_prediction

# Inverse transform the dummy array
original_all_features = scaler.inverse_transform(dummy)

# Extract the original-scale 'close' predictions from the transformed array
original_prediction = original_all_features[:, target_idx]  # shape (12,)

# 5. Format and print the original scale predictions
formatted_original = [f"{price:.3f}" for price in original_prediction]
print("Predicted multi-step close for next 1 hr (original scale):")
print(", ".join(formatted_original))

inspect_last_data(config.processed_file)
