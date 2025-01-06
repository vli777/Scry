import pytest
from src.data.download import download_prices

def test_download_prices():
    tickers = ["AAPL", "GOOG"]
    data = download_prices(tickers)
    assert data is not None, "Failed to fetch prices"
    assert not data.empty, "Data is empty"
    assert "AAPL" in data.columns, "Ticker data missing"

def test_empty_tickers():
    with pytest.raises(ValueError, match="The tickers list is empty."):
        download_prices([])
