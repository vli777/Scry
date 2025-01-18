import pandas as pd
import pytorch_forecasting
import lightning.pytorch as pl
import torch
torch.set_float32_matmul_precision('medium')
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import MAE, TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
import pickle
from lightning.pytorch.tuner import Tuner

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

    # 4) Build training and validation datasets
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        group_ids=["symbol"],
        target="close",
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=None,
        add_relative_time_idx = True
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, stop_randomization=True
    )

    # 5) Create dataloaders
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

    # 6) Configure network and trainer
    pl.seed_everything(42)
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        accelerator=(
            "gpu" if torch.cuda.is_available() else "cpu"
        ),  # Automatically selects GPU or CPU
        max_epochs=max_epochs,  # Your desired maximum epochs
        gradient_clip_val=0.1,  # Gradient clipping
        logger=logger,  # Set True to use the default logger
        enable_checkpointing=True,  # Enable checkpointing
        callbacks=[
            lr_logger,
            early_stop_callback,
        ],  
    )
    print("Starting training...")

    # 7) Define TFT model
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

    # 8) Fit model network
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print(f"Training complete. Model saved to {model_path}")

    # 9) Hyperparameter tuning with Optuna
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna",
        n_trials=100,
        max_epochs=max_epochs,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )

    # save study results - also we can resume tuning at a later point in time
    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    print(study.best_trial.params)

    # 10) Evaluate performance
    # load the best model according to the validation loss
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
    MAE()(predictions.output, predictions.y)
    
    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    predictions = best_tft.predict(val_dataloader, return_x=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output)
    best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)  
    
    interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
    best_tft.plot_interpretation(interpretation)
    
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
