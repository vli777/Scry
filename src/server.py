from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from inference.tft_predict import predict_close_price_tft


class PredictRequest(BaseModel):
    symbol: str = "SPY"


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, this is your TFT inference API."}


@app.post("/predict")
def predict(req: PredictRequest):
    pred_value = predict_close_price_tft(
        ticker=req.symbol,
    )
    return {"symbol": req.symbol, "prediction": pred_value}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
