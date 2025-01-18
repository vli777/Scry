import os
import yaml


class Config:
    def __init__(self, config_path="config.yaml"):
        # Load the YAML configuration file
        with open(config_path, "r") as file:
            self._config = yaml.safe_load(file)

        # Extract values
        self.symbol = self._config["symbol"]
        self.frequency = self._config["frequency"]
        self.frequency_type = self._config["frequency_type"]
        self.model_type = self._config["model_type"]

        # Extract paths
        paths = self._config["paths"]
        self.raw_dir = paths["raw_dir"]
        self.processed_dir = paths["processed_dir"]
        self.scaler_dir = paths["scaler_dir"]
        self.model_dir = paths["model_dir"]

        # Create directories if not exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.scaler_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Dynamic paths based on symbol and frequency
        self.raw_file = os.path.join(
            self.raw_dir, f"{self.symbol}_{self.frequency}min_data.parquet"
        )
        self.scaler_file = os.path.join(self.scaler_dir, f"scaler_{self.symbol}.pkl")
        self.processed_file = os.path.join(
            self.processed_dir, f"{self.symbol}_{self.frequency}min_processed.parquet"
        )
        self.model_file = os.path.join(
            self.model_dir, f"{self.model_type}_{self.symbol}.pkl"
        )


# Create a singleton instance of Config
config = Config()
