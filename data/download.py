import os
import logging
from typing import List, Optional
from datetime import datetime
from pandas.tseries.offsets import BDay
import h5py
import yfinance as yf
import pandas as pd
from pandas_market_calendars import get_calendar


logger = logging.getLogger(__name__)

def download_prices(tickers: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetches 'Adj Close' prices for specified tickers within the given date range.
    """
    if not tickers:
        logger.error("The tickers list is empty.")
        raise ValueError("The tickers list is empty.")

    try:
        logger.info(f"Fetching prices for tickers: {tickers}")
        data = yf.download(tickers, start=start_date, end=end_date)
        
        if 'Adj Close' not in data:
            raise ValueError("'Adj Close' prices not found in the fetched data.")
        
        close_data = data['Adj Close'].fillna(method='ffill').dropna()
        logger.info("Prices fetched successfully.")
        return close_data
    except Exception as e:
        logger.error(f"An error occurred while fetching data: {e}")
        return None

def is_trading_day(date: str) -> bool:
    """
    Check if a given date is a trading day.
    """
    nyse = get_calendar("NYSE")
    schedule = nyse.valid_days(start_date=date, end_date=date)
    return pd.to_datetime(date) in schedule

def get_last_date(hdf5_file: str, dataset_name: str) -> Optional[str]:
    """
    Fetch the last available date from the HDF5 dataset.
    """
    with h5py.File(hdf5_file, "r") as hf:
        if dataset_name in hf:
            # Assuming 'Date' is the first column
            last_date = hf[dataset_name][-1, 0].decode("utf-8")
            return last_date
    return None

def append_new_data(tickers: List[str], hdf5_file: str, dataset_name: str, start_date: str, end_date: str):
    """
    Download and append new data to the HDF5 file, using the given date range.
    """
    # Fetch data from yfinance
    data = download_prices(tickers, start_date, end_date)
    if data is None or data.empty:
        logger.info("No new data available.")
        return
    
    # Reset index for HDF5 storage
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    
    # Append data to HDF5
    with h5py.File(hdf5_file, "a") as hf:
        if dataset_name in hf:
            # Append to existing dataset
            existing_data = hf[dataset_name]
            existing_data.resize(existing_data.shape[0] + data.shape[0], axis=0)
            existing_data[-data.shape[0]:] = data.to_numpy()
        else:
            # Create dataset
            hf.create_dataset(
                dataset_name,
                data=data.to_numpy(),
                maxshape=(None, data.shape[1]),
                compression="gzip",
            )
    logger.info("New data appended successfully.")

def get_data(tickers: List[str], hdf5_file: str, dataset_name: str, start_date: str, end_date: str):
    """
    Manage data by either downloading from scratch or appending new data.
    """
    if not os.path.exists(hdf5_file):
        logger.info("HDF5 file not found. Downloading data from scratch.")
        append_new_data(tickers, hdf5_file, dataset_name, start_date, end_date)
    else:
        last_date = get_last_date(hdf5_file, dataset_name)
        if last_date is None:
            logger.info("Dataset not found in HDF5 file. Downloading data from scratch.")
            append_new_data(tickers, hdf5_file, dataset_name, start_date, end_date)
        else:
            next_date = (pd.to_datetime(last_date) + BDay(1)).strftime("%Y-%m-%d")
            logger.info(f"Last available date: {last_date}. Fetching new data from {next_date} to {end_date}.")
            append_new_data(tickers, hdf5_file, dataset_name, next_date, end_date)


if __name__ == "__main__":
    tickers = ["AAPL", "GOOGL", "MSFT"]
    hdf5_file = "data/stock_data.h5"
    dataset_name = "prices"
    start_date = "2013-01-01"  
    end_date = datetime.now().strftime("%Y-%m-%d") 
    
    get_data(tickers, hdf5_file, dataset_name, start_date, end_date)
