import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from src.inference.tft_predict import predict_close_price_tft

load_dotenv(override=True)
BEARER_TOKEN = os.getenv("BEARER_TOKEN")


class PredictRequest(BaseModel):
    symbol: str


# Initialize app and rate limiter
app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.get("/")
def read_root():
    return {"message": "Hello, this is your TFT inference API."}


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


@app.post("/predict/")
@limiter.limit("5/minute")
async def predict(symbol: str):
    if not BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="Bearer token not configured")

    try:
        prediction = predict_close_price_tft(
            config={"symbol": symbol},
            bearer_token=BEARER_TOKEN,
            max_encoder_length=48,
            max_prediction_length=12,
        )
        return {"symbol": symbol, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
