"""
src/factor_model.py

Factor Investing Model for QuantAI.

Implements 3 classic factors proven by decades of academic research:

1. VALUE FACTOR (Fama-French, 1992)
   Stocks trading cheap relative to fundamentals outperform.
   Metrics: P/E ratio, P/B ratio, P/S ratio
   Signal: LOW P/E = cheap = positive factor score

2. MOMENTUM FACTOR (Jegadeesh & Titman, 1993)
   Stocks that performed well in past 12 months (skip last 1 month)
   continue to outperform over next 3-12 months.
   Metric: 12-1 month return (exclude most recent month to avoid reversal)
   Signal: HIGH past return = positive factor score

3. QUALITY FACTOR (Novy-Marx, 2013)
   Profitable, stable companies with strong balance sheets outperform.
   Metrics: ROE, ROA, gross margin, debt/equity, earnings consistency
   Signal: HIGH profitability + LOW leverage = positive factor score

Combination: Equal-weight 3 factors → composite factor score
High score stocks → get confidence boost in ensemble
Low score stocks  → get confidence penalty
"""

import numpy as np# type: ignore
import pandas as pd# type: ignore
import yfinance as yf# type: ignore
import sqlite3
import json
import os
from datetime import datetime, date

FACTOR_CACHE   = os.path.join('data', 'factor_scores.json')
DB_PATH        = os.path.join('data', 'quantai.db')

# ── Stock universe ────────────────────────────────────────
from src.data_collector import STOCK_UNIVERSE

# ── Helper: safe fetch ────────────────────────────────────
def safe_get(info, key, default=None):
    """Safely gets a value from yfinance info dict."""
    val = info.get(key, default)
    if val is None or val == 'N/A':
        return default
    try:
        return float(val)
    except Exception:
        return default

# ── Factor 1: VALUE ───────────────────────────────────────
def calculate_value_score(ticker, info):
    """
    Value factor: lower valuation multiples = higher score.

    Metrics used:
      P/E  ratio (trailingPE)  — price vs earnings
      P/B  ratio (priceToBook) — price vs book value
      P/S  ratio (priceToSalesTrailing12Months) — price vs sales

    Lower is better for value → we invert each metric.
    """
    pe  = safe_get(info, 'trailingPE')
    pb  = safe_get(info, 'priceToBook')
    ps  = safe_get(info, 'priceToSalesTrailing12Months')
    ev_ebitda = safe_get(info, 'enterpriseToEbitda')

    scores = []

    # P/E: reasonable range 5-50, lower = better value
    if pe and 0 < pe < 200:
        # Invert and normalize: P/E of 10 is better than 40
        pe_score = max(0, min(1, (50 - pe) / 45))
        scores.append(pe_score)

    # P/B: range 0.5-15, lower = better value
    if pb and 0 < pb < 50:
        pb_score = max(0, min(1, (15 - pb) / 14.5))
        scores.append(pb_score)

    # P/S: range 0.1-20, lower = better value
    if ps and 0 < ps < 50:
        ps_score = max(0, min(1, (20 - ps) / 19.9))
        scores.append(ps_score)

    # EV/EBITDA: range 3-30, lower = better
    if ev_ebitda and 0 < ev_ebitda < 100:
        ev_score = max(0, min(1, (30 - ev_ebitda) / 27))
        scores.append(ev_score)

    if not scores:
        return None, {}

    return round(np.mean(scores), 4), {
        'pe'       : round(pe, 2)  if pe else None,
        'pb'       : round(pb, 2)  if pb else None,
        'ps'       : round(ps, 2)  if ps else None,
        'ev_ebitda': round(ev_ebitda, 2) if ev_ebitda else None,
    }

# ── Factor 2: MOMENTUM ────────────────────────────────────
def calculate_momentum_score(ticker, prices_df=None):
    """
    Momentum factor: 12-1 month return.

    Classic momentum skips the most recent month to avoid
    short-term reversal (stocks mean-revert over 1 month
    but continue over 2-12 months).

    Returns score 0-1 based on percentile rank.
    """
    try:
        if prices_df is None or ticker not in prices_df.columns:
            conn = sqlite3.connect(DB_PATH)
            df   = pd.read_sql_query(
                "SELECT date, close FROM prices "
                "WHERE ticker=? ORDER BY date ASC",
                conn, params=(ticker,)
            )
            conn.close()
            if df.empty or len(df) < 66:
                return None, {}
            prices = df['close']
        else:
            prices = prices_df[ticker].dropna()

        if len(prices) < 66:
            return None, {}

        current    = float(prices.iloc[-1])
        month_ago  = float(prices.iloc[-22])   # 1 month ago
        year_ago   = float(prices.iloc[max(-252, -len(prices))])

        # 12-1 month momentum (skip last month)
        ret_12_1   = (month_ago - year_ago) / year_ago \
                     if len(prices) >= 252 else \
                     (current - year_ago) / year_ago

        # 3-month momentum
        ret_3m     = float(prices.iloc[-1]) - float(prices.iloc[-66])
        ret_3m    /= float(prices.iloc[-66])

        # 1-month momentum (for reversal signal)
        ret_1m     = (current - month_ago) / month_ago

        return round(ret_12_1, 4), {
            'return_12_1m': round(ret_12_1 * 100, 2),
            'return_3m'   : round(ret_3m    * 100, 2),
            'return_1m'   : round(ret_1m    * 100, 2),
        }

    except Exception as e:
        return None, {}

# ── Factor 3: QUALITY ─────────────────────────────────────
def calculate_quality_score(ticker, info):
    """
    Quality factor: profitable, stable, low-leverage companies.

    Metrics:
      ROE (returnOnEquity)   — how efficiently equity is used
      ROA (returnOnAssets)   — how efficiently assets are used
      Gross Margin           — pricing power
      Debt/Equity            — financial stability
      Current Ratio          — liquidity

    Higher ROE/ROA/margin + lower debt = higher quality score.
    """
    roe           = safe_get(info, 'returnOnEquity')
    roa           = safe_get(info, 'returnOnAssets')
    gross_margin  = safe_get(info, 'grossMargins')
    de_ratio      = safe_get(info, 'debtToEquity')
    current_ratio = safe_get(info, 'currentRatio')
    profit_margin = safe_get(info, 'profitMargins')

    scores = []

    # ROE: 0-40% range, higher = better
    if roe is not None and -0.5 < roe < 2:
        roe_score = max(0, min(1, roe / 0.40))
        scores.append(roe_score)

    # ROA: 0-20% range, higher = better
    if roa is not None and -0.5 < roa < 1:
        roa_score = max(0, min(1, roa / 0.20))
        scores.append(roa_score)

    # Gross margin: 0-80%, higher = better
    if gross_margin and 0 < gross_margin < 1:
        gm_score = max(0, min(1, gross_margin / 0.80))
        scores.append(gm_score)

    # Debt/Equity: lower = better (0 = no debt)
    if de_ratio is not None and de_ratio >= 0:
        de_score = max(0, min(1, 1 - de_ratio / 300))
        scores.append(de_score)

    # Current ratio: >1.5 is healthy
    if current_ratio and current_ratio > 0:
        cr_score = max(0, min(1, (current_ratio - 0.5) / 3.0))
        scores.append(cr_score)

    # Profit margin
    if profit_margin and -0.5 < profit_margin < 1:
        pm_score = max(0, min(1, profit_margin / 0.30))
        scores.append(pm_score)

    if not scores:
        return None, {}

    return round(np.mean(scores), 4), {
        'roe'          : round(roe * 100, 2)         if roe          else None,
        'roa'          : round(roa * 100, 2)         if roa          else None,
        'gross_margin' : round(gross_margin * 100, 2)if gross_margin else None,
        'debt_equity'  : round(de_ratio, 2)          if de_ratio is not None else None,
        'current_ratio': round(current_ratio, 2)     if current_ratio else None,
        'profit_margin': round(profit_margin * 100, 2)if profit_margin else None,
    }

# ── Composite factor score ────────────────────────────────
def calculate_factor_score(ticker, prices_df=None,
                            weights=None):
    """
    Combines all 3 factors into one composite score.

    Default weights:
      Value    : 35%
      Momentum : 35%
      Quality  : 30%
    """
    if weights is None:
        weights = {'value': 0.35, 'momentum': 0.35,
                   'quality': 0.30}

    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    value_raw,  value_det  = calculate_value_score(ticker, info)
    mom_raw,    mom_det    = calculate_momentum_score(ticker, prices_df)
    quality_raw,qual_det   = calculate_quality_score(ticker, info)

    # Normalize momentum to 0-1 using sigmoid-like transform
    if mom_raw is not None:
        mom_score = 1 / (1 + np.exp(-mom_raw * 8))
        mom_score = round(float(mom_score), 4)
    else:
        mom_score = None

    # Weighted composite
    components   = {}
    total_weight = 0
    composite    = 0

    if value_raw is not None:
        components['value']    = value_raw
        composite             += value_raw * weights['value']
        total_weight          += weights['value']

    if mom_score is not None:
        components['momentum'] = mom_score
        composite             += mom_score * weights['momentum']
        total_weight          += weights['momentum']

    if quality_raw is not None:
        components['quality']  = quality_raw
        composite             += quality_raw * weights['quality']
        total_weight          += weights['quality']

    if total_weight == 0:
        return None

    composite = composite / total_weight

    return {
        'ticker'         : ticker,
        'name'           : STOCK_UNIVERSE.get(ticker, ('', ''))[0],
        'sector'         : STOCK_UNIVERSE.get(ticker, ('', 'Unknown'))[1],
        'composite_score': round(composite, 4),
        'value_score'    : round(value_raw, 4) if value_raw else None,
        'momentum_score' : round(mom_score, 4) if mom_score else None,
        'quality_score'  : round(quality_raw, 4) if quality_raw else None,
        'value_details'  : value_det,
        'momentum_details': mom_det,
        'quality_details': qual_det,
        'tier'           : None,  # set after ranking
        'updated_at'     : datetime.now().isoformat(),
    }

# ── Rank all stocks ───────────────────────────────────────
def rank_all_stocks(use_cache_hours=24):
    """
    Calculates and ranks all 50 stocks by factor score.
    Caches results for 24 hours to avoid excessive API calls.
    """
    # Check cache
    if os.path.exists(FACTOR_CACHE):
        with open(FACTOR_CACHE) as f:
            cached = json.load(f)
        age = (datetime.now() -
               datetime.fromisoformat(
                   cached.get('generated_at', '2000-01-01')
               )).seconds / 3600
        if age < use_cache_hours:
            print(f"  Using cached factor scores "
                  f"({age:.1f}h old)")
            return cached.get('rankings', [])

    print(f"\n  Calculating factor scores for "
          f"{len(STOCK_UNIVERSE)} stocks...")
    print(f"  This fetches fundamental data — takes ~3 mins\n")

    # Load price data for momentum
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        "SELECT ticker, date, close FROM prices "
        "WHERE date >= date('now', '-400 days') "
        "ORDER BY date ASC",
        conn
    )
    conn.close()

    prices_df = None
    if not df.empty:
        prices_df = df.pivot(
            index='date', columns='ticker', values='close'
        )
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df.ffill(inplace=True)

    results = []
    for i, ticker in enumerate(STOCK_UNIVERSE, 1):
        print(f"  [{i:>2}/{len(STOCK_UNIVERSE)}] "
              f"{ticker:<20}", end='\r')
        score = calculate_factor_score(ticker, prices_df)
        if score:
            results.append(score)

    # Sort by composite score
    results.sort(
        key=lambda x: x['composite_score'], reverse=True
    )

    # Assign tiers
    n = len(results)
    for i, r in enumerate(results):
        r['rank'] = i + 1
        if i < n // 3:
            r['tier']   = 'TOP'
            r['signal'] = 'STRONG BUY'
        elif i < 2 * n // 3:
            r['tier']   = 'MIDDLE'
            r['signal'] = 'NEUTRAL'
        else:
            r['tier']   = 'BOTTOM'
            r['signal'] = 'AVOID'

    # Save cache
    os.makedirs('data', exist_ok=True)
    with open(FACTOR_CACHE, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'n_stocks'    : len(results),
            'rankings'    : results
        }, f, indent=2, default=str)

    print(f"\n  Done — {len(results)} stocks ranked")
    return results

# ── Get single stock factor score ─────────────────────────
def get_factor_score(ticker):
    """Gets factor score for one stock from cache."""
    if not os.path.exists(FACTOR_CACHE):
        return None, None
    with open(FACTOR_CACHE) as f:
        cached = json.load(f)
    for r in cached.get('rankings', []):
        if r['ticker'] == ticker:
            return r['composite_score'], r['tier']
    return None, None

def get_factor_confidence_boost(ticker):
    """
    Returns confidence adjustment based on factor score.
      TOP tier    → +0.05 boost
      MIDDLE tier → no change
      BOTTOM tier → -0.08 penalty
    """
    score, tier = get_factor_score(ticker)
    if tier == 'TOP':
        return 0.05
    elif tier == 'BOTTOM':
        return -0.08
    return 0.0