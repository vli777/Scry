import os
import pandas as pd
import pytorch_forecasting
import pytorch_lightning as pl
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import (
    TemporalFusionTransformer,
)
from pytorch_forecasting.data import DataLoader

from src.config_loader import config


def train_tft(
    data_path,
    model_path,
    max_encoder_length=48,
    max_prediction_length=12,
    batch_size=64,
    max_epochs=10,
    val_split=0.1,
    learning_rate=1e-3,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
):
    """
    Train a TFT model on time series data to forecast future steps.
    Saves the trained model checkpoint to model_path.
    """
    print("Loading and preparing data...")
    # 1) Load and inspect data
    df = pd.read_parquet(data_path).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Ensure required columns exist
    if "time_idx" not in df.columns:
        df["time_idx"] = range(len(df))
    if "symbol" not in df.columns:
        df["symbol"] = "symbol"

    # 2) Identify unknown reals
    all_cols = list(df.columns)
    exclude_cols = {"timestamp", "symbol", "time_idx", "close"}
    unknown_reals = [c for c in all_cols if c not in exclude_cols]

    # 3) Split into training and validation sets
    max_time_idx = df["time_idx"].max()
    training_cutoff = int(max_time_idx * (1 - val_split))  # dynamic split
    train_df = df[df["time_idx"] <= training_cutoff]
    val_df = df[df["time_idx"] > training_cutoff]

    print(f"Training data: {len(train_df)}, Validation data: {len(val_df)}")

    # 4) Build TimeSeriesDataSet
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        group_ids=["symbol"],
        target="close",
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=None,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, stop_randomization=True
    )

    # 5) Create dataloaders
    train_loader = training.to_dataloader(
        train=True, batch_size=batch_size, shuffle=True
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=batch_size, shuffle=False
    )

    # 6) Define TFT model
    print("Initializing Temporal Fusion Transformer...")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        loss=pytorch_forecasting.metrics.MAE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    print(f"Number of parameters in the model: {tft.size()/1e3:.1f}k")

    # 7) Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=1 if device.type == "cuda" else 0,
        gradient_clip_val=0.1,
    )
    print("Starting training...")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 8) Save the best model
    trainer.save_checkpoint(model_path)
    print(f"Training complete. Model saved to {model_path}")


if __name__ == "__main__":
    train_tft(
        data_path=config.processed_file,
        model_path=config.model_file,
        max_encoder_length=48,
        max_prediction_length=12,
        batch_size=64,
        max_epochs=10,
        val_split=0.1,
    )
