"""
src/sector_rotation.py

Sector Rotation Strategy for QuantAI.

How it works:
  1. Groups all 50 stocks into sectors
  2. Calculates sector momentum (1m, 3m, 6m returns)
  3. Ranks sectors from strongest to weakest
  4. Generates BUY signals for stocks in top sectors
  5. Generates EXIT signals for stocks in bottom sectors

Sector scoring uses 3 factors:
  - Momentum  : recent price performance (40%)
  - Breadth   : % of stocks in sector above MA50 (35%)
  - Relative  : sector vs market performance (25%)
"""

import numpy as np # type: ignore
import pandas as pd # type: ignore
import sqlite3
import os
import json
from datetime import datetime, date

DB_PATH          = os.path.join('data', 'quantai.db')
ROTATION_CACHE   = os.path.join('data', 'sector_rotation_cache.json')

# ── Sector definitions ────────────────────────────────────
SECTOR_MAP = {
    'Technology' : [
        'TCS.NS', 'INFY.NS', 'WIPRO.NS',
        'HCLTECH.NS', 'TECHM.NS'
    ],
    'Banking'    : [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS',
        'AXISBANK.NS', 'KOTAKBANK.NS', 'INDUSINDBK.NS'
    ],
    'Financial Services': [
        'BAJFINANCE.NS', 'BAJAJFINSV.NS',
        'SBILIFE.NS', 'HDFCLIFE.NS'
    ],
    'Automobile' : [
        'MARUTI.NS', 'TATAMOTORS.NS', 'EICHERMOT.NS',
        'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'M&M.NS'
    ],
    'Energy'     : [
        'RELIANCE.NS', 'ONGC.NS', 'BPCL.NS'
    ],
    'Utilities'  : [
        'NTPC.NS', 'POWERGRID.NS'
    ],
    'Infrastructure': [
        'LT.NS', 'ADANIPORTS.NS', 'ADANIENT.NS', 'BEL.NS'
    ],
    'Pharma'     : [
        'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS',
        'DIVISLAB.NS', 'APOLLOHOSP.NS'
    ],
    'Metals'     : [
        'TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS'
    ],
    'Cement'     : [
        'GRASIM.NS', 'ULTRACEMCO.NS', 'SHREECEM.NS'
    ],
    'FMCG'       : [
        'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS',
        'BRITANNIA.NS', 'TATACONSUM.NS'
    ],
    'Consumer'   : [
        'ASIANPAINT.NS', 'TITAN.NS'
    ],
    'Telecom'    : [
        'BHARTIARTL.NS'
    ],
    'Mining'     : [
        'COALINDIA.NS'
    ],
}

# ── Load price data ───────────────────────────────────────
def load_sector_prices(lookback_days=200):
    """Loads closing prices for all stocks grouped by sector."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        """SELECT ticker, date, close FROM prices
           WHERE date >= date('now', ?)
           ORDER BY date ASC""",
        conn, params=(f'-{lookback_days} days',)
    )
    conn.close()

    if df.empty:
        return {}

    pivot = df.pivot(
        index='date', columns='ticker', values='close'
    )
    pivot.index = pd.to_datetime(pivot.index)
    pivot.ffill(inplace=True)

    # Group by sector
    sector_prices = {}
    for sector, tickers in SECTOR_MAP.items():
        available = [t for t in tickers if t in pivot.columns]
        if available:
            sector_prices[sector] = pivot[available]

    return sector_prices

# ── Sector performance ────────────────────────────────────
def calculate_sector_performance(sector_prices):
    """
    Calculates performance metrics for each sector.

    Metrics:
      return_1m  : 1-month return
      return_3m  : 3-month return
      return_6m  : 6-month return
      volatility : annualised volatility
      sharpe     : risk-adjusted return
      breadth    : % of stocks above MA50
      rs_score   : relative strength vs all stocks
    """
    if not sector_prices:
        return {}

    # Build market average for relative strength
    all_series = []
    for sector, df in sector_prices.items():
        for col in df.columns:
            all_series.append(df[col])
    if all_series:
        market_avg = pd.concat(all_series, axis=1).mean(axis=1)
    else:
        market_avg = None

    results = {}

    for sector, df in sector_prices.items():
        if df.empty or len(df) < 22:
            continue

        # Equal-weight sector index
        sector_index = df.mean(axis=1)
        returns      = sector_index.pct_change().dropna()

        # Returns
        ret_1m = (float(sector_index.iloc[-1]) -
                  float(sector_index.iloc[-22])) / \
                  float(sector_index.iloc[-22]) \
                  if len(sector_index) >= 22 else 0

        ret_3m = (float(sector_index.iloc[-1]) -
                  float(sector_index.iloc[-66])) / \
                  float(sector_index.iloc[-66]) \
                  if len(sector_index) >= 66 else ret_1m

        ret_6m = (float(sector_index.iloc[-1]) -
                  float(sector_index.iloc[-132])) / \
                  float(sector_index.iloc[-132]) \
                  if len(sector_index) >= 132 else ret_3m

        # Volatility (annualised)
        vol = float(returns.std()) * np.sqrt(252)

        # Sharpe (simplified, 0% risk-free rate)
        sharpe = (ret_1m * 12) / (vol + 1e-6)

        # Breadth: % above MA50
        above_ma50 = 0
        for col in df.columns:
            s = df[col].dropna()
            if len(s) >= 50:
                ma50 = float(s.rolling(50).mean().iloc[-1])
                if float(s.iloc[-1]) > ma50:
                    above_ma50 += 1
        breadth = above_ma50 / len(df.columns) \
                  if df.columns.size > 0 else 0

        # Relative strength vs market
        if market_avg is not None and len(market_avg) >= 22:
            mkt_ret_1m = (float(market_avg.iloc[-1]) -
                          float(market_avg.iloc[-22])) / \
                          float(market_avg.iloc[-22])
            rs_score   = ret_1m - mkt_ret_1m
        else:
            rs_score = 0

        # Individual stock details
        stock_details = {}
        for ticker in df.columns:
            s = df[ticker].dropna()
            if len(s) < 2:
                continue
            s_ret_1m = (float(s.iloc[-1]) -
                        float(s.iloc[min(-22, -len(s))])) / \
                        float(s.iloc[min(-22, -len(s))])
            stock_details[ticker] = round(s_ret_1m * 100, 2)

        results[sector] = {
            'sector'       : sector,
            'n_stocks'     : len(df.columns),
            'stocks'       : list(df.columns),
            'return_1m'    : round(ret_1m * 100, 2),
            'return_3m'    : round(ret_3m * 100, 2),
            'return_6m'    : round(ret_6m * 100, 2),
            'volatility'   : round(vol * 100, 2),
            'sharpe'       : round(sharpe, 3),
            'breadth'      : round(breadth * 100, 1),
            'rs_score'     : round(rs_score * 100, 2),
            'stock_returns': stock_details,
        }

    return results

# ── Sector ranking ────────────────────────────────────────
def rank_sectors(sector_performance):
    """
    Ranks sectors from strongest to weakest using composite score.

    Composite = Momentum(40%) + Breadth(35%) + RS(25%)
    """
    if not sector_performance:
        return []

    # Normalize each metric to 0-1
    perf_list = list(sector_performance.values())
    if not perf_list:
        return []

    def normalize(values):
        mn, mx = min(values), max(values)
        if mx == mn:
            return [0.5] * len(values)
        return [(v - mn) / (mx - mn) for v in values]

    ret_1m_vals = [p['return_1m'] for p in perf_list]
    breadth_vals= [p['breadth']   for p in perf_list]
    rs_vals     = [p['rs_score']  for p in perf_list]
    sharpe_vals = [p['sharpe']    for p in perf_list]

    ret_norm     = normalize(ret_1m_vals)
    breadth_norm = normalize(breadth_vals)
    rs_norm      = normalize(rs_vals)
    sharpe_norm  = normalize(sharpe_vals)

    for i, perf in enumerate(perf_list):
        perf['momentum_score'] = round(ret_norm[i], 3)
        perf['breadth_score']  = round(breadth_norm[i], 3)
        perf['rs_score_norm']  = round(rs_norm[i], 3)
        perf['sharpe_norm']    = round(sharpe_norm[i], 3)
        perf['composite_score'] = round(
            ret_norm[i]     * 0.40 +
            breadth_norm[i] * 0.35 +
            rs_norm[i]      * 0.25,
            3
        )

    ranked = sorted(
        perf_list,
        key=lambda x: x['composite_score'],
        reverse=True
    )

    for i, r in enumerate(ranked):
        r['rank']   = i + 1
        r['tier']   = ('TOP'    if i < len(ranked) // 3 else
                        'MIDDLE' if i < 2 * len(ranked) // 3
                        else 'BOTTOM')
        r['signal'] = ('BUY'    if r['tier'] == 'TOP'    else
                        'HOLD'   if r['tier'] == 'MIDDLE'
                        else 'AVOID')

    return ranked

# ── Generate rotation signals ─────────────────────────────
def get_rotation_signals(top_n_sectors=3):
    """
    Main function: returns rotation signals for all stocks.

    Returns:
      ranked_sectors: list of sectors ranked best to worst
      buy_stocks    : stocks in top sectors (BUY candidates)
      avoid_stocks  : stocks in bottom sectors (AVOID)
      sector_etf    : best stock per sector (sector proxy)
    """
    sector_prices = load_sector_prices()
    if not sector_prices:
        return [], [], [], {}

    perf    = calculate_sector_performance(sector_prices)
    ranked  = rank_sectors(perf)

    if not ranked:
        return [], [], [], {}

    top_sectors    = [r['sector'] for r in ranked[:top_n_sectors]]
    bottom_sectors = [r['sector']
                      for r in ranked[-top_n_sectors:]]

    buy_stocks   = []
    avoid_stocks = []
    best_per_sector = {}

    for r in ranked:
        sector  = r['sector']
        stocks  = r['stocks']
        returns = r.get('stock_returns', {})

        if sector in top_sectors:
            # Best stock in sector = highest 1m return
            if returns:
                best = max(returns.items(),
                           key=lambda x: x[1])
                best_per_sector[sector] = {
                    'ticker'    : best[0],
                    'return_1m' : best[1],
                    'sector'    : sector,
                    'rank'      : r['rank'],
                    'score'     : r['composite_score'],
                }
            buy_stocks.extend(stocks)

        elif sector in bottom_sectors:
            avoid_stocks.extend(stocks)

    # Save to cache
    cache = {
        'generated_at'   : datetime.now().isoformat(),
        'date'           : str(date.today()),
        'ranked_sectors' : ranked,
        'buy_stocks'     : buy_stocks,
        'avoid_stocks'   : avoid_stocks,
        'best_per_sector': best_per_sector,
        'top_sectors'    : top_sectors,
        'bottom_sectors' : bottom_sectors,
    }
    os.makedirs('data', exist_ok=True)
    with open(ROTATION_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)

    return ranked, buy_stocks, avoid_stocks, best_per_sector

def load_rotation_cache():
    """Loads cached rotation signals."""
    if not os.path.exists(ROTATION_CACHE):
        return None
    with open(ROTATION_CACHE) as f:
        return json.load(f)

def is_rotation_buy(ticker):
    """
    Returns True if ticker is in a top-ranked sector.
    Used by paper_trader.py to filter signals.
    """
    cache = load_rotation_cache()
    if not cache:
        return True   # if no cache, allow all
    return ticker in cache.get('buy_stocks', [])

def is_rotation_avoid(ticker):
    """Returns True if ticker should be avoided."""
    cache = load_rotation_cache()
    if not cache:
        return False
    return ticker in cache.get('avoid_stocks', [])

def get_sector_for_ticker(ticker):
    """Returns sector name for a given ticker."""
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return 'Unknown'