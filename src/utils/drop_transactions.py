import pandas as pd
import os

PROCESSED_FILE_PATH = "data/processed/SPY_5min_processed.parquet"
file_path = PROCESSED_FILE_PATH

# Ensure the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        # Load the Parquet file
        df = pd.read_parquet(file_path)
        print(f"Type of df: {type(df)}")  # Debug the type

        # Check if df is a tuple
        if isinstance(df, tuple):
            print("df is a tuple. Unpacking the first element.")
            df = df[0]  # Assuming the DataFrame is the first element of the tuple

        if isinstance(df, pd.DataFrame):  # Ensure it's a DataFrame
            if "transactions" in df.columns:
                df.drop(columns=["transactions"], inplace=True)
                print("'transactions' column removed.")

            # Save the modified DataFrame back to a Parquet file
            df.to_parquet(
                PROCESSED_FILE_PATH, engine="pyarrow", compression="snappy", index=False
            )
            print(f"Modified Parquet file saved to {PROCESSED_FILE_PATH}")
        else:
            print(f"Unexpected type for df: {type(df)}")

    except Exception as e:
        print(f"An error occurred: {e}")
