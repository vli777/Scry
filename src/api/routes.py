from fastapi import FastAPI, HTTPException, Query
from typing import List

app = FastAPI(title="Scry API", version="1.0")

@app.get("/close_prices")
def get_close_prices(
    tickers: List[str] = Query(..., description="List of ticker symbols"),
    start_date: str = Query('2013-01-01', description="Start date in YYYY-MM-DD"),
    end_date: str = Query('2023-12-31', description="End date in YYYY-MM-DD")
):
    """
    Fetches the 'Close' prices for the specified tickers within the given date range.

    - **tickers**: List of ticker symbols (e.g., SPY, AAPL)
    - **start_date**: Start date in 'YYYY-MM-DD' format
    - **end_date**: End date in 'YYYY-MM-DD' format
    """
    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided.")
    
    close_prices = fetch_close_prices(tickers, start_date, end_date)
    
    if close_prices.empty:
        raise HTTPException(status_code=404, detail="No data found for the provided tickers and date range.")
    
    # Convert DataFrame to JSON
    close_prices_json = close_prices.to_json(orient="split")
    return json.loads(close_prices_json)
