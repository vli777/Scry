# Scry
A time-series forecasting pipeline using a Temporal Fusion Transformer (TFT) to predict the next 1 hour of price movement on 5-minute interval data. This project demonstrates:

- Data fetching & incremental updates
- Feature engineering (technical indicators)
- Training a TFT model
- Batch predictions
- Real-time inference via FastAPI

## Table of Contents
- Installation
- Setup & Configuration
- Usage
- Offline CLI: main.py
- REST API: server.py
- References

## Installation

Clone the repository:
```

git clone https://github.com/username/my_project.git
cd my_project
```
Create a virtual environment (optional but recommended):
```

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
Install dependencies:
```

pip install -r requirements.txt
```
Make sure you have PyTorch, PyTorch Lightning, pytorch-forecasting, and FastAPI as needed.

## Setup & Configuration
Environment Variables:
Create a file named .env in the project root (added to .gitignore so it’s not committed):

```

DATA_PATH=data/processed/spy_features.parquet
MODEL_PATH=models/tft_model.pth
SCHWAB_BEARER_TOKEN=YourSecretToken
```
Or any other environment variables you need.

## Paths & Constants:

DATA_PATH: Path to the Parquet file storing your 5-minute processed data.
MODEL_PATH: Path to save or load your TFT model checkpoints.
Additional paths or tokens can be added as needed.

## Data Requirements:

The project expects 5-minute interval data in a Parquet file, e.g. data/processed/spy_features.parquet.
The DataFrame columns might include: timestamp, open, high, low, close, volume, ....

## Usage

## Offline CLI: main.py
The main.py script provides a command-line interface (CLI) for tasks like training, batch predictions, and data fetching. Run:

```
python main.py <command> [options...]
Train the TFT model:
```

```
python main.py train --data-path data/processed/spy_features.parquet --model-path models/tft_model.pth
```
Trains a multi-step TFT model (e.g., 12-step horizon for 1 hour of 5-min bars).

## Batch Predict:

```
python main.py batch_predict --data-path data/processed/spy_features.parquet --model-path models/tft_model.pth
```
Loads the saved TFT model and performs a batch prediction (using predict_close_price_tft under the hood).
Fetch Data (Incremental Update):

```
python main.py fetch_data --data-path data/processed/spy_features.parquet
```
Fetches new 5-minute data from the last timestamp to now, updates rolling features, and saves the updated Parquet.
Arguments (example):

--data-path: path to your processed parquet file.
--model-path: path to your TFT model checkpoint.
--ticker: symbol like SPY (default).
--interval: e.g. 5min.
(These arguments can be extended or customized as needed.)

## REST API: server.py

```
python server.py
```
This will start a server (default: http://localhost:8000). Key endpoints:

Root: GET /

Returns a simple welcome message.
Predict: POST /predict

Example request:
```
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"symbol": "SPY"}'
```
The server calls predict_close_price_tft internally, loads the latest dataset, and returns the predicted close price.
You might want to schedule or run incremental updates separately (via main.py fetch_data) so the data is always fresh.
Adding More Endpoints
You can add more routes to server.py for other symbols, intervals, or any custom logic—like streaming predictions, or retrieving historical data.

## References
- PyTorch Forecasting: https://pytorch-forecasting.readthedocs.io/
- FastAPI: https://fastapi.tiangolo.com/
- PyTorch Lightning: https://www.pytorchlightning.ai/
