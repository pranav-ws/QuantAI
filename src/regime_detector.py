"""
src/regime_detector.py

Market Regime Detection for QuantAI.

Detects 4 market regimes using 5 signals:
  1. Trend     — is Nifty 50 above its 200-day MA?
  2. Momentum  — is short-term trend above long-term?
  3. Volatility— is VIX / daily range elevated?
  4. Breadth   — how many stocks are in uptrend?
  5. RSI       — is the market overbought or oversold?

Regime affects trading behaviour:
  BULL     → lower confidence threshold (58% → 55%) — trade more
  BEAR     → higher threshold (58% → 68%) — very selective
  SIDEWAYS → normal threshold (58%)
  VOLATILE → higher threshold (58% → 65%) — avoid whipsaws
"""

import numpy as np# type: ignore
import pandas as pd# type: ignore
import sqlite3
import os
import json
from datetime import datetime, date

DB_PATH      = os.path.join('data', 'quantai.db')
REGIME_CACHE = os.path.join('data', 'regime_cache.json')

# ── Regime definitions ────────────────────────────────────
REGIMES = {
    'BULL'     : {
        'emoji'      : '🐂',
        'color'      : 'green',
        'description': 'Market in uptrend — conditions favour buying',
        'threshold'  : 0.55,   # lower bar — more trades
        'max_positions': 6,
    },
    'BEAR'     : {
        'emoji'      : '🐻',
        'color'      : 'red',
        'description': 'Market in downtrend — stay defensive',
        'threshold'  : 0.68,   # higher bar — very selective
        'max_positions': 2,
    },
    'SIDEWAYS' : {
        'emoji'      : '↔️',
        'color'      : 'yellow',
        'description': 'No clear trend — trade selectively',
        'threshold'  : 0.60,
        'max_positions': 4,
    },
    'VOLATILE' : {
        'emoji'      : '⚡',
        'color'      : 'orange',
        'description': 'High volatility — reduce position sizes',
        'threshold'  : 0.65,
        'max_positions': 3,
    },
}

# ── Load market data ──────────────────────────────────────
def load_market_prices(lookback_days=300):
    """
    Loads closing prices for all stocks.
    Uses our existing SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        """SELECT ticker, date, close FROM prices
           WHERE date >= date('now', ?)
           ORDER BY date ASC""",
        conn, params=(f'-{lookback_days} days',)
    )
    conn.close()

    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot(
        index='date', columns='ticker', values='close'
    )
    pivot.index = pd.to_datetime(pivot.index)
    pivot.ffill(inplace=True)
    return pivot

# ── Signal 1: Trend ───────────────────────────────────────
def calculate_trend_score(prices_df):
    """
    Checks what % of stocks are above their 50-day and 200-day MA.
    >70% above 200d MA = strong bull
    <30% above 200d MA = bear
    """
    if len(prices_df) < 205:
        return 0.5, {}

    ma50_above  = 0
    ma200_above = 0
    total       = 0

    details = {}
    for ticker in prices_df.columns:
        series = prices_df[ticker].dropna()
        if len(series) < 205:
            continue

        current  = float(series.iloc[-1])
        ma50     = float(series.rolling(50).mean().iloc[-1])
        ma200    = float(series.rolling(200).mean().iloc[-1])

        above_50  = current > ma50
        above_200 = current > ma200

        if above_50:  ma50_above  += 1
        if above_200: ma200_above += 1
        total += 1

        details[ticker] = {
            'above_50' : above_50,
            'above_200': above_200
        }

    if total == 0:
        return 0.5, {}

    pct_above_50  = ma50_above  / total
    pct_above_200 = ma200_above / total

    # Score: 0 = full bear, 1 = full bull
    trend_score = (pct_above_50 * 0.4 + pct_above_200 * 0.6)

    return round(trend_score, 3), {
        'pct_above_ma50' : round(pct_above_50 * 100, 1),
        'pct_above_ma200': round(pct_above_200 * 100, 1),
        'stocks_checked' : total
    }

# ── Signal 2: Momentum ────────────────────────────────────
def calculate_momentum_score(prices_df):
    """
    Compares average 20-day return vs 60-day return.
    Positive = accelerating uptrend
    Negative = momentum fading
    """
    if len(prices_df) < 65:
        return 0.5, {}

    returns_20 = []
    returns_60 = []

    for ticker in prices_df.columns:
        series = prices_df[ticker].dropna()
        if len(series) < 65:
            continue
        ret_20 = (float(series.iloc[-1]) -
                  float(series.iloc[-20])) / float(series.iloc[-20])
        ret_60 = (float(series.iloc[-1]) -
                  float(series.iloc[-60])) / float(series.iloc[-60])
        returns_20.append(ret_20)
        returns_60.append(ret_60)

    if not returns_20:
        return 0.5, {}

    avg_20 = np.mean(returns_20)
    avg_60 = np.mean(returns_60)

    # Normalize to 0-1
    momentum_score = 0.5 + (avg_20 - avg_60) * 5
    momentum_score = max(0.0, min(1.0, momentum_score))

    return round(momentum_score, 3), {
        'avg_return_20d': round(avg_20 * 100, 2),
        'avg_return_60d': round(avg_60 * 100, 2),
        'momentum_direction': 'UP' if avg_20 > avg_60 else 'DOWN'
    }

# ── Signal 3: Volatility ──────────────────────────────────
def calculate_volatility_score(prices_df, window=20):
    """
    Measures average daily range across all stocks.
    High volatility = score closer to 0 (bad for trading)
    Low volatility  = score closer to 1 (good for trading)
    """
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        """SELECT ticker, date, high, low, close
           FROM prices
           WHERE date >= date('now', '-60 days')
           ORDER BY date ASC""",
        conn
    )
    conn.close()

    if df.empty:
        return 0.5, {}

    daily_ranges = []
    for ticker in df['ticker'].unique():
        t_df = df[df['ticker'] == ticker].tail(window)
        if len(t_df) < 10:
            continue
        t_df = t_df.copy()
        t_df['range_pct'] = ((t_df['high'] - t_df['low']) /
                              t_df['close'])
        daily_ranges.append(float(t_df['range_pct'].mean()))

    if not daily_ranges:
        return 0.5, {}

    avg_range = np.mean(daily_ranges)

    # Historical normal range ~1.5%, elevated >2.5%
    normal_range = 0.015
    high_range   = 0.030

    if avg_range <= normal_range:
        vol_score = 1.0
    elif avg_range >= high_range:
        vol_score = 0.0
    else:
        vol_score = 1 - (avg_range - normal_range) / \
                    (high_range - normal_range)

    return round(vol_score, 3), {
        'avg_daily_range_pct': round(avg_range * 100, 2),
        'volatility_level'   : 'LOW' if vol_score > 0.6 else
                               'HIGH' if vol_score < 0.3 else 'MEDIUM'
    }

# ── Signal 4: Breadth ─────────────────────────────────────
def calculate_breadth_score(prices_df, window=10):
    """
    Advance-Decline ratio: how many stocks went UP today?
    >60% advancing = bullish breadth
    <40% advancing = bearish breadth
    """
    if len(prices_df) < window + 1:
        return 0.5, {}

    recent_returns = prices_df.pct_change(window).iloc[-1]
    advancing = (recent_returns > 0).sum()
    declining = (recent_returns < 0).sum()
    total     = advancing + declining

    if total == 0:
        return 0.5, {}

    ad_ratio     = advancing / total
    breadth_score = ad_ratio

    return round(breadth_score, 3), {
        'advancing'     : int(advancing),
        'declining'     : int(declining),
        'ad_ratio'      : round(ad_ratio * 100, 1),
        'breadth_signal': 'BULLISH'  if ad_ratio > 0.6 else
                          'BEARISH'  if ad_ratio < 0.4 else 'NEUTRAL'
    }

# ── Signal 5: RSI of market ───────────────────────────────
def calculate_market_rsi(prices_df, period=14):
    """
    Average RSI across all stocks.
    >65 = market overbought (be careful)
    <35 = market oversold (opportunity)
    40-60 = neutral
    """
    rsi_values = []

    for ticker in prices_df.columns:
        series = prices_df[ticker].dropna()
        if len(series) < period + 5:
            continue

        delta  = series.diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / (loss + 1e-10)
        rsi    = 100 - (100 / (1 + rs))
        rsi_values.append(float(rsi.iloc[-1]))

    if not rsi_values:
        return 0.5, {}

    avg_rsi = np.mean(rsi_values)

    # Score: overbought (>65) → 0.3, oversold (<35) → 0.7
    # Normal range → 0.5
    if avg_rsi > 70:
        rsi_score = 0.2   # very overbought
    elif avg_rsi > 60:
        rsi_score = 0.4   # mildly overbought
    elif avg_rsi < 30:
        rsi_score = 0.9   # very oversold (buy opportunity)
    elif avg_rsi < 40:
        rsi_score = 0.7   # mildly oversold
    else:
        rsi_score = 0.5   # neutral

    return round(rsi_score, 3), {
        'avg_market_rsi'  : round(avg_rsi, 1),
        'market_condition': 'OVERBOUGHT' if avg_rsi > 65 else
                            'OVERSOLD'   if avg_rsi < 35 else 'NEUTRAL'
    }

# ── Master regime classifier ──────────────────────────────
def detect_regime(use_cache_minutes=60):
    """
    Combines all 5 signals into a regime classification.

    Weights:
      Trend      : 35% (most important)
      Breadth    : 25%
      Momentum   : 20%
      Volatility : 15%
      RSI        : 5%

    Returns full regime dict with all sub-scores.
    """
    # Check cache first
    if os.path.exists(REGIME_CACHE):
        with open(REGIME_CACHE) as f:
            cached = json.load(f)
        cached_at = datetime.fromisoformat(
            cached.get('generated_at', '2000-01-01')
        )
        age_mins  = (datetime.now() - cached_at).seconds / 60
        if age_mins < use_cache_minutes:
            return cached

    print("  🔍 Detecting market regime...")

    prices = load_market_prices(lookback_days=300)
    if prices.empty:
        return _default_regime()

    # Calculate all 5 signals
    trend_score,  trend_data  = calculate_trend_score(prices)
    mom_score,    mom_data    = calculate_momentum_score(prices)
    vol_score,    vol_data    = calculate_volatility_score(prices)
    breadth_score,breadth_data= calculate_breadth_score(prices)
    rsi_score,    rsi_data    = calculate_market_rsi(prices)

    # Weighted composite score
    composite = (
        trend_score   * 0.35 +
        breadth_score * 0.25 +
        mom_score     * 0.20 +
        vol_score     * 0.15 +
        rsi_score     * 0.05
    )

    # ── Classify regime ───────────────────────────────────
    # High volatility overrides everything
    if vol_score < 0.25:
        regime = 'VOLATILE'

    # Bear: low trend + low breadth
    elif composite < 0.35:
        regime = 'BEAR'

    # Bull: high trend + high breadth
    elif composite >= 0.60 and trend_score >= 0.60:
        regime = 'BULL'

    # Sideways: everything else
    else:
        regime = 'SIDEWAYS'

    result = {
        'regime'       : regime,
        'composite'    : round(composite, 3),
        'generated_at' : datetime.now().isoformat(),
        'date'         : str(date.today()),

        # Individual scores
        'scores' : {
            'trend'     : trend_score,
            'momentum'  : mom_score,
            'volatility': vol_score,
            'breadth'   : breadth_score,
            'rsi'       : rsi_score,
        },

        # Detailed data
        'details': {
            'trend'     : trend_data,
            'momentum'  : mom_data,
            'volatility': vol_data,
            'breadth'   : breadth_data,
            'rsi'       : rsi_data,
        },

        # Trading parameters for this regime
        'trading' : REGIMES[regime],
    }

    # Save to cache
    os.makedirs('data', exist_ok=True)
    with open(REGIME_CACHE, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result

def _default_regime():
    """Fallback regime when data is unavailable."""
    return {
        'regime'    : 'SIDEWAYS',
        'composite' : 0.5,
        'scores'    : {},
        'details'   : {},
        'trading'   : REGIMES['SIDEWAYS'],
        'generated_at': datetime.now().isoformat(),
    }

def get_regime_threshold(regime_result):
    """Returns confidence threshold for current regime."""
    return regime_result.get('trading', {}).get(
        'threshold', 0.58
    )

def get_regime_summary(regime_result):
    """Returns a one-line summary string."""
    r = regime_result
    regime  = r.get('regime', 'SIDEWAYS')
    emoji   = REGIMES[regime]['emoji']
    desc    = REGIMES[regime]['description']
    score   = r.get('composite', 0.5)
    return f"{emoji} {regime} (score: {score:.2f}) — {desc}"