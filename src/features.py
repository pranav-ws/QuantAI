import pandas as pd
import ta
import sqlite3
import os

DB_PATH = os.path.join('data', 'quantai.db')

def load_prices(ticker):
    """Loads OHLCV data from our database into a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM prices WHERE ticker = ? ORDER BY date ASC",
        conn, params=(ticker,)
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df

def add_features(df):
    """Adds 30+ technical indicator features to the DataFrame."""

    # ── Moving Averages ──────────────────────────────────
    df['SMA_20']  = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50']  = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_9']   = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA_21']  = ta.trend.ema_indicator(df['Close'], window=21)

    # ── Momentum Indicators ──────────────────────────────
    df['RSI_14']      = ta.momentum.rsi(df['Close'], window=14)

    df['MACD']        = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_Hist']   = ta.trend.macd_diff(df['Close'], window_slow=26, window_fast=12, window_sign=9)

    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    # ── Volatility Indicators ────────────────────────────
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper']  = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower']  = bb.bollinger_lband()
    df['BB_Width']  = bb.bollinger_wband()

    df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)

    # ── Volume Indicators ────────────────────────────────
    df['OBV']           = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['Volume_SMA_20'] = ta.trend.sma_indicator(df['Volume'].astype(float), window=20)
    df['Volume_Ratio']  = df['Volume'] / df['Volume_SMA_20']

    # ── Price-derived Features ───────────────────────────
    df['Daily_Return']   = df['Close'].pct_change()
    df['Return_5d']      = df['Close'].pct_change(5)
    df['Return_20d']     = df['Close'].pct_change(20)
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Close_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Close_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

    # ── Target Variable ──────────────────────────────────
    # 1 = stock goes UP tomorrow, 0 = stock goes DOWN
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop NaN rows from indicator warmup period
    df.dropna(inplace=True)

    return df

def get_feature_dataset(ticker):
    """Full pipeline: load → engineer → return ML-ready DataFrame."""
    print(f"⚙️  Building features for {ticker}...")
    df = load_prices(ticker)
    df = add_features(df)
    print(f"✅  {ticker}: {len(df)} rows × {len(df.columns)} features")
    return df
