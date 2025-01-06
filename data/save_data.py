import h5py
import numpy as np
from typing import string

# Example downloaded data
tickers = ["AAPL", "GOOG", "MSFT"]
prices = [150.34, 2801.12, 299.36]
volumes = [2000000, 1500000, 3000000]

# Save to HDF5
def save_data(file_path: string, data: )-> void:
    try:
        with h5py.File("data/downloaded_data.h5", "w") as hf:
            hf.create_dataset("tickers", data=np.array(tickers, dtype="S"))  # Store strings as bytes
            hf.create_dataset("prices", data=prices)
            hf.create_dataset("volumes", data=volumes)
    except err as ExceptionError:
        logging.error('file not saved', err)
        
def load_data(file_path: string)->list:
    try:
        with h5py.File("data/downloaded_data.h5", "r") as hf:
            tickers = [t.decode("utf-8") for t in hf["tickers"][:]]
            prices = hf["prices"][:]
            volumes = hf["volumes"][:]
    except err as Exception:
        logging.error('error loading file', err)
        
        return { tickers, prices, volumes }
