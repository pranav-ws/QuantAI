"""
src/pairs_finder.py

Finds the best correlated + cointegrated stock pairs from our universe.

Two-step filtering:
  Step 1 — Correlation  : pairs must be >70% correlated (same sector usually)
  Step 2 — Cointegration: pairs must share a long-run equilibrium price
                          (correlation alone is not enough — cointegration
                           means the SPREAD between them is mean-reverting)

Best pairs are same-sector stocks:
  Banking  : HDFCBANK vs ICICIBANK vs AXISBANK vs KOTAKBANK
  Tech     : TCS vs INFY vs WIPRO vs HCLTECH
  Auto     : MARUTI vs TATAMOTORS vs BAJAJ-AUTO
  FMCG     : HINDUNILVR vs ITC vs BRITANNIA
"""

import os
import sqlite3
import numpy as np# type: ignore
import pandas as pd# type: ignore
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller# type: ignore
import warnings
warnings.filterwarnings('ignore')

DB_PATH      = os.path.join('data', 'quantai.db')
PAIRS_CACHE  = os.path.join('data', 'pairs_cache.json')

# ── Load price data ───────────────────────────────────────
def load_close_prices(lookback_days=365):
    """
    Loads closing prices for all stocks from our database.
    Returns a DataFrame: index=date, columns=tickers
    """
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        """SELECT ticker, date, close FROM prices
           WHERE date >= date('now', ?)
           ORDER BY date ASC""",
        conn,
        params=(f'-{lookback_days} days',)
    )
    conn.close()

    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot(index='date', columns='ticker', values='close')
    pivot.index = pd.to_datetime(pivot.index)
    pivot.dropna(axis=1, thresh=int(len(pivot) * 0.9), inplace=True)
    pivot.ffill(inplace=True)                      # ← NEW (works on all versions)
    return pivot

# ── Correlation matrix ────────────────────────────────────
def get_correlation_matrix(prices_df):
    """Returns correlation matrix of returns."""
    returns = prices_df.pct_change().dropna()
    return returns.corr()

# ── Cointegration test ────────────────────────────────────
def test_cointegration(series1, series2):
    """
    Engle-Granger cointegration test.
    Returns (is_cointegrated, p_value, hedge_ratio)
    p_value < 0.05 means the pair IS cointegrated (spread is mean-reverting).
    """
    try:
        # Cointegration test
        score, pvalue, _ = coint(series1, series2)

        # Calculate hedge ratio using OLS
        from numpy.linalg import lstsq # type: ignore
        X = np.column_stack([series2.values, np.ones(len(series2))])
        hedge_ratio, _, _, _ = lstsq(X, series1.values, rcond=None)

        is_coint = pvalue < 0.05
        return is_coint, round(pvalue, 4), round(hedge_ratio[0], 4)

    except Exception:
        return False, 1.0, 1.0

# ── Half-life of mean reversion ───────────────────────────
def calculate_half_life(spread):
    """
    How many days does it take for the spread to revert halfway?
    Shorter half-life = faster mean reversion = better pair.
    Target: 5-30 days (too short = noise, too long = slow profit)
    """
    try:
        spread_lag  = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        spread_lag  = spread_lag.iloc[:len(spread_diff)]

        model       = np.polyfit(spread_lag, spread_diff, 1)
        half_life   = -np.log(2) / model[0]
        return round(half_life, 1)
    except Exception:
        return 999

# ── Main pair scanner ─────────────────────────────────────
def find_best_pairs(min_correlation=0.70,
                    max_pvalue=0.05,
                    max_half_life=30,
                    min_half_life=5,
                    top_n=20):
    """
    Scans all stock combinations and returns the best pairs.

    Filters:
      1. Correlation  > min_correlation (default 70%)
      2. Cointegration p-value < max_pvalue (default 5%)
      3. Half-life between min_half_life and max_half_life days
    """
    print("\n" + "="*58)
    print("  QuantAI Pairs Finder — Scanning for best pairs")
    print("="*58)

    prices = load_close_prices(lookback_days=500)
    if prices.empty:
        print("  ❌ No price data found. Run pipeline.py first.")
        return []

    tickers = list(prices.columns)
    print(f"  Stocks available : {len(tickers)}")

    # Step 1 — Correlation filter (fast)
    print(f"  Step 1: Calculating correlations...")
    corr_matrix  = get_correlation_matrix(prices)
    candidate_pairs = []

    for t1, t2 in combinations(tickers, 2):
        if t1 not in corr_matrix.columns or t2 not in corr_matrix.columns:
            continue
        corr = corr_matrix.loc[t1, t2]
        if abs(corr) >= min_correlation:
            candidate_pairs.append((t1, t2, round(corr, 4)))

    print(f"  Candidates after correlation filter: {len(candidate_pairs)}")

    # Step 2 — Cointegration filter (slower but more robust)
    print(f"  Step 2: Running cointegration tests...")
    valid_pairs = []

    for i, (t1, t2, corr) in enumerate(candidate_pairs):
        print(f"  Testing {i+1}/{len(candidate_pairs)}: "
              f"{t1.replace('.NS','')} ↔ {t2.replace('.NS','')}",
              end='\r')

        s1 = prices[t1].dropna()
        s2 = prices[t2].dropna()

        # Align series
        idx = s1.index.intersection(s2.index)
        if len(idx) < 100:
            continue

        s1, s2 = s1[idx], s2[idx]

        is_coint, pvalue, hedge = test_cointegration(s1, s2)
        if not is_coint:
            continue

        # Calculate spread and half-life
        spread     = s1 - hedge * s2
        half_life  = calculate_half_life(spread)

        if not (min_half_life <= half_life <= max_half_life):
            continue

        # Spread statistics
        spread_mean  = float(spread.mean())
        spread_std   = float(spread.std())
        current_z    = float((spread.iloc[-1] - spread_mean) / spread_std)

        valid_pairs.append({
            'stock1'      : t1,
            'stock2'      : t2,
            'name1'       : t1.replace('.NS', ''),
            'name2'       : t2.replace('.NS', ''),
            'correlation' : corr,
            'pvalue'      : pvalue,
            'hedge_ratio' : hedge,
            'half_life'   : half_life,
            'spread_mean' : round(spread_mean, 4),
            'spread_std'  : round(spread_std,  4),
            'current_zscore': round(current_z, 3),
            'signal'      : _get_pair_signal(current_z),
        })

    # Sort by half-life (shorter = better)
    valid_pairs.sort(key=lambda x: x['half_life'])

    print(f"\n  Valid pairs found: {len(valid_pairs)}")
    print("="*58)

    # Save to cache
    import json
    os.makedirs('data', exist_ok=True)
    with open(PAIRS_CACHE, 'w') as f:
        json.dump(valid_pairs[:top_n], f, indent=2)
    print(f"  Saved to {PAIRS_CACHE}")

    return valid_pairs[:top_n]

def _get_pair_signal(zscore, entry=2.0, exit_z=0.5):
    """
    Trading signal based on z-score:
      z < -2.0 : Stock1 cheap vs Stock2 → BUY Stock1, SHORT Stock2
      z > +2.0 : Stock1 expensive vs Stock2 → SHORT Stock1, BUY Stock2
      |z| < 0.5: Near mean → EXIT (close positions)
    """
    if zscore <= -entry:
        return 'BUY_1_SHORT_2'
    elif zscore >= entry:
        return 'SHORT_1_BUY_2'
    elif abs(zscore) <= exit_z:
        return 'EXIT'
    else:
        return 'HOLD'

def load_cached_pairs():
    """Loads previously found pairs from cache."""
    import json
    if not os.path.exists(PAIRS_CACHE):
        return []
    with open(PAIRS_CACHE) as f:
        return json.load(f)