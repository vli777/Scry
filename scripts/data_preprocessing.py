import h5py
from sklearn.model_selection import train_test_split
import numpy as np

# Load processed data
with h5py.File("data/processed/normalized_data.h5", "r") as hf:
    features = hf["features"][:]
    labels = hf["labels"][:]

# Split into train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save splits
with h5py.File("data/training/train_data.h5", "w") as hf:
    hf.create_dataset("features", data=X_train)
    hf.create_dataset("labels", data=y_train)
with h5py.File("data/training/val_data.h5", "w") as hf:
    hf.create_dataset("features", data=X_val)
    hf.create_dataset("labels", data=y_val)
with h5py.File("data/training/test_data.h5", "w") as hf:
    hf.create_dataset("features", data=X_test)
    hf.create_dataset("labels", data=y_test)
