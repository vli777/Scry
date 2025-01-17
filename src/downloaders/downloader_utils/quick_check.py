import pandas as pd
import sys

def check_parquet(file_path, freq_minutes=5):
    # Load the Parquet file
    df = pd.read_parquet(file_path)
    
    if "timestamp" not in df.columns:
        print("No 'timestamp' column found in the file.")
        return
    
    # Ensure timestamps are timezone-aware and sorted
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Check for duplicate timestamps
    duplicates = df[df.duplicated(subset=["timestamp"], keep=False)]
    if not duplicates.empty:
        print("Found duplicate timestamps:")
        print(duplicates)
    else:
        print("No duplicate timestamps found.")
    
    # Check for missing intervals
    expected_delta = pd.Timedelta(minutes=freq_minutes)
    df["timedelta"] = df["timestamp"].diff()
    
    # Identify rows where the gap is not equal to the expected frequency
    missing = df[df["timedelta"] != expected_delta]
    
    if not missing.empty:
        print("\nMissing or irregular timestamp intervals detected:")
        for i, row in missing.iterrows():
            # Skip the first row since its timedelta is NaT
            if pd.isnull(row["timedelta"]):
                continue
            prev_time = df.at[i-1, "timestamp"]
            curr_time = row["timestamp"]
            gap = row["timedelta"]
            print(f"Gap detected: from {prev_time} to {curr_time} (delta: {gap})")
    else:
        print("All timestamp intervals are consistent.")
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_parquet.py <path_to_parquet> [freq_minutes]")
    else:
        file_path = sys.argv[1]
        freq = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        check_parquet(file_path, freq)
