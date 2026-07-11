import yfinance as yf
import mplfinance as mpf

# Download Reliance data for 2024
ticker = 'RELIANCE.NS'
data = yf.download(ticker, start='2024-01-01', end='2024-12-31')

# Fix column names for mplfinance (it needs simple column names)
data.columns = [col[0] for col in data.columns]

# Plot candlestick chart with 20 & 50 day moving averages
mpf.plot(
    data,
    type='candle',
    style='charles',
    title=f'\n{ticker} — Daily Candlestick Chart (2024)',
    ylabel='Price (INR)',
    mav=(20, 50),
    volume=True,
    figsize=(14, 8),
    show_nontrading=False
)
