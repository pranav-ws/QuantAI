import yfinance as yf
import pandas as pd

# ---- Download stock data ----
# Using Reliance Industries (Indian stock on NSE)
# For US stocks, just use: 'AAPL', 'TSLA', 'GOOGL' etc.

ticker = 'RELIANCE.NS'

data = yf.download(ticker, start='2023-01-01', end='2024-12-31')

# ---- See what the data looks like ----
print("=== First 5 rows ===")
print(data.head())

print("\n=== Last 5 rows ===")
print(data.tail())

print(f"\n=== Total trading days fetched: {len(data)} ===")
print(f"=== Columns: {data.columns.tolist()} ===")
