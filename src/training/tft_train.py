import pandas as pd
import pytorch_forecasting
import lightning.pytorch as pl
import torch

torch.set_float32_matmul_precision("medium")
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_forecasting import MAE, RMSE, TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
import pickle

from src.config_loader import config


def train_tft(
    data_path,
    model_path,
    max_encoder_length=48,
    max_prediction_length=12,
    batch_size=128,  # Increased batch size for faster training
    max_epochs=50,   # Increased epochs with early stopping
    val_split=0.1,
    learning_rate=3e-3,  # Slightly increased learning rate
    hidden_size=16,      # Reduced complexity
    attention_head_size=2,  # Reduced number of attention heads
    dropout=0.1,
    hidden_continuous_size=8,  # Added explicit continuous size
    gradient_clip_val=0.1
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
    training_cutoff = int(max_time_idx * (1 - val_split))
    train_df = df[df["time_idx"] <= training_cutoff]
    val_df = df[df["time_idx"] > training_cutoff]

    print(f"Training data: {len(train_df)}, Validation data: {len(val_df)}")

    # 4) Build training and validation datasets with optimized settings
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        group_ids=["symbol"],
        target="close",
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=None,  # We handle normalization externally
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,  # More robust to missing data
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, stop_randomization=True
    )

    # 5) Create dataloaders with optimized settings
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0,  # Reduced for more stable training
        persistent_workers=False,
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size * 2,  # Larger batch size for validation
        num_workers=0,
        persistent_workers=False,
    )

    # 6) Configure network and trainer with optimized settings
    pl.seed_everything(42)
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,  # Reduced patience for faster training
        verbose=False,
        mode="min"
    )
    
    lr_logger = LearningRateMonitor()

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.model_dir,
        filename=f"tft_{config.symbol}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip_val,
        logger=True,
        enable_checkpointing=True,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        deterministic=True,  # Added for reproducibility
        precision=16 if torch.cuda.is_available() else 32,  # Use mixed precision if GPU available
    )

    # 7) Define TFT model with optimized architecture
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=pytorch_forecasting.metrics.MAE(),
        reduce_on_plateau_patience=3,
        logging_metrics=torch.nn.ModuleList([MAE(), RMSE()]),  # Track both metrics
    )

    print(f"Number of parameters in the model: {tft.size()/1e3:.1f}k")

    # 8) Fit model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    return trainer, tft, train_dataloader, val_dataloader


def tune_hyperparameters(train_dataloader, val_dataloader, model_path, max_epochs=10):
    """
    Perform hyperparameter tuning using Optuna.
    """
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path=model_path,
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
        use_learning_rate_finder=True,
    )

    with open(f"{model_path}_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    print("Best Hyperparameters:", study.best_trial.params)
    return study


def evaluate_model(trainer, val_dataloader):
    """
    Evaluate the trained model on validation data.
    """
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    predictions = best_tft.predict(
        val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
    )
    print("Validation MAE:", MAE()(predictions.output, predictions.y))

    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(
        raw_predictions.x, raw_predictions.output
    )
    best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)


if __name__ == "__main__":
    trainer, tft, train_loader, val_loader = train_tft(
        data_path=config.processed_file,
        model_path=config.model_file,
        max_encoder_length=48,
        max_prediction_length=12,
        batch_size=128,
        max_epochs=50,
        val_split=0.1,
    )

    # Uncomment the following lines for hyperparameter tuning and evaluation:
    study = tune_hyperparameters(train_loader, val_loader, model_path="optuna")
    evaluate_model(trainer, val_loader)
