import pandas as pd
import pytorch_forecasting
import pytorch_lightning as pl
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import (
    TemporalFusionTransformer,
)
from pytorch_forecasting.data import DataLoader


def train_tft(
    data_path="data/processed/SPY_5min_processed.parquet",
    model_path="models/tft_model.pth",
    max_encoder_length=48,
    max_prediction_length=12,
    batch_size=64,
    max_epochs=10,
):
    """
    Train a TFT model on SPY 5-minute data to forecast 12 steps (1 hour) into the future.
    Saves the trained model checkpoint to model_path.
    """
    # 1) Load and inspect data
    df = pd.read_parquet(data_path).copy()
    # Data is already scaled. Example columns: timestamp, open, high, ..., close, volume, etc.

    # 2) Ensure chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 3) Create a 'time_idx' for PyTorch Forecasting if not present
    if "time_idx" not in df.columns:
        df["time_idx"] = range(len(df))  # each row increments by 1

    # 4) Create a 'symbol' column (since we have only SPY, make it constant)
    if "symbol" not in df.columns:
        df["symbol"] = "SPY"

    # 5) Identify which columns to treat as unknown reals
    #    We'll treat 'close' as our target, so exclude it from unknown reals.
    #    Also exclude 'timestamp', 'symbol', 'time_idx'.
    all_cols = list(df.columns)
    exclude_cols = {"timestamp", "symbol", "time_idx", "close"}
    unknown_reals = [c for c in all_cols if c not in exclude_cols]

    # 6) Split into training and validation sets
    #    We'll do a simple index-based split.
    #    Adjust as needed (e.g., time-based cutoff).
    max_time_idx = df["time_idx"].max()
    training_cutoff = (
        max_time_idx - 5000
    )  # keep ~ 5000 rows for validation, for example

    train_df = df[df["time_idx"] <= training_cutoff]
    val_df = df[df["time_idx"] > training_cutoff]

    # 7) Build TimeSeriesDataSet for training
    #    We want to forecast 'close' 12 steps (i.e., 1 hour for 5-min intervals).
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        group_ids=["symbol"],
        target="close",
        max_encoder_length=max_encoder_length,  # how many past 5-min bars the model sees
        max_prediction_length=max_prediction_length,  # 12 future bars => 1 hour
        time_varying_unknown_reals=unknown_reals,  # e.g., open, high, low, volume, indicators, etc.
        target_normalizer=None,  # we skip additional normalization since data is scaled externally
    )

    # 8) Build validation dataset from the same config
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, stop_randomization=True
    )

    # 9) Create dataloaders
    train_loader = training.to_dataloader(
        train=True, batch_size=batch_size, shuffle=True
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=batch_size, shuffle=False
    )

    # 10) Define the TFT model from the training dataset metadata
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=32,  # number of hidden units in each layer
        attention_head_size=4,  # number of attention heads
        dropout=0.1,
        loss=pytorch_forecasting.metrics.MAE(),  # use MAE or MSE, your choice
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # 11) Train the model with PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=0 if not torch.cuda.is_available() else 1,  # use GPU if available
        gradient_clip_val=0.1,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 12) Save the best checkpoint
    trainer.save_checkpoint(model_path)
    print(f"Training complete. Model saved to {model_path}")


if __name__ == "__main__":
    train_tft(
        data_path="data/processed/SPY_5min_processed.parquet",
        model_path="models/tft_model.pth",
        max_encoder_length=48,
        max_prediction_length=12,
        batch_size=64,
        max_epochs=10,
    )
