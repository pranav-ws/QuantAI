"""
src/pairs_trader.py

Live pairs trading engine.
Monitors active pairs and generates BUY/SHORT signals.
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, date
from src.pairs_finder import (load_cached_pairs, load_close_prices,
                               _get_pair_signal)

PAIRS_TRADES_PATH = os.path.join('data', 'pairs_trades.json')

def load_pairs_trades():
    if os.path.exists(PAIRS_TRADES_PATH):
        with open(PAIRS_TRADES_PATH) as f:
            return json.load(f)
    return []

def save_pairs_trades(trades):
    os.makedirs('data', exist_ok=True)
    with open(PAIRS_TRADES_PATH, 'w') as f:
        json.dump(trades, f, indent=2, default=str)

def get_live_zscore(pair, prices_df):
    """
    Calculates the current live z-score for a pair.
    """
    t1 = pair['stock1']
    t2 = pair['stock2']
    h  = pair['hedge_ratio']

    if t1 not in prices_df.columns or t2 not in prices_df.columns:
        return None, None, None

    s1 = prices_df[t1].dropna()
    s2 = prices_df[t2].dropna()

    idx    = s1.index.intersection(s2.index)
    s1, s2 = s1[idx], s2[idx]

    spread       = s1 - h * s2
    spread_mean  = float(spread.mean())
    spread_std   = float(spread.std())

    if spread_std == 0:
        return None, None, None

    current_spread = float(spread.iloc[-1])
    zscore         = (current_spread - spread_mean) / spread_std

    return round(zscore, 3), round(spread_mean, 4), round(spread_std, 4)

def scan_pairs_signals(entry_z=2.0, exit_z=0.5):
    """
    Scans all cached pairs and returns live trading signals.
    """
    pairs = load_cached_pairs()
    if not pairs:
        return []

    prices = load_close_prices(lookback_days=365)
    if prices.empty:
        return []

    signals = []

    for pair in pairs:
        zscore, s_mean, s_std = get_live_zscore(pair, prices)
        if zscore is None:
            continue

        signal = _get_pair_signal(zscore, entry_z, exit_z)

        t1     = pair['stock1']
        t2     = pair['stock2']
        p1     = float(prices[t1].iloc[-1]) if t1 in prices.columns else 0
        p2     = float(prices[t2].iloc[-1]) if t2 in prices.columns else 0

        signals.append({
            'stock1'      : t1,
            'stock2'      : t2,
            'name1'       : t1.replace('.NS', ''),
            'name2'       : t2.replace('.NS', ''),
            'correlation' : pair['correlation'],
            'half_life'   : pair['half_life'],
            'hedge_ratio' : pair['hedge_ratio'],
            'zscore'      : zscore,
            'signal'      : signal,
            'price1'      : round(p1, 2),
            'price2'      : round(p2, 2),
            'spread_mean' : s_mean,
            'spread_std'  : s_std,
            'date'        : str(date.today()),
        })

    # Sort by absolute z-score (strongest signals first)
    signals.sort(key=lambda x: abs(x['zscore']), reverse=True)
    return signals

def calculate_pairs_position(signal, capital, max_pct=0.15):
    """
    Calculates position sizes for both legs of the pair.
    Uses equal dollar value for both legs (market neutral).
    """
    leg_capital = capital * max_pct

    if signal['signal'] in ('BUY_1_SHORT_2', 'SHORT_1_BUY_2'):
        shares1 = int(leg_capital / signal['price1']) if signal['price1'] > 0 else 0
        shares2 = int(leg_capital / signal['price2']) if signal['price2'] > 0 else 0
        return shares1, shares2

    return 0, 0