# Calculate technical indicators
for ticker in tickers:
    close_data[f'{ticker}_RSI'] = ta.rsi(close_data[ticker], length=14)
    macd = ta.macd(close_data[ticker])
    close_data[f'{ticker}_MACD'] = macd['MACD']
    close_data[f'{ticker}_MACD_SIGNAL'] = macd['MACDs']
    # Add more indicators as needed

# Handle missing values
close_data.fillna(method='ffill', inplace=True)
close_data.fillna(method='bfill', inplace=True)
