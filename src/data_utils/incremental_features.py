import pandas as pd

from data_utils.compute_features import compute_indicators


def calculate_features_incrementally(
    df_recent: pd.DataFrame,
    df_new: pd.DataFrame,
    days: int = 200,
):
    """
    1) Concatenate df_recent + df_new
    2) Retain only last X days
    3) Compute technical indicators
    4) Return updated DataFrame
    """
    # Merge
    df = pd.concat([df_recent, df_new]).reset_index(drop=True)

    # Retain only last `days` days
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cutoff_date].reset_index(drop=True)

    # Compute indicators
    df = compute_indicators(df)

    return df
