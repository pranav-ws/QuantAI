"""
src/pairs_backtest.py
Backtests a pairs trading strategy on historical data.
"""

import numpy as np
import pandas as pd
from src.pairs_finder import load_close_prices, _get_pair_signal

def backtest_pair(stock1, stock2, hedge_ratio,
                  initial_capital=100000,
                  entry_z=2.0, exit_z=0.5,
                  lookback=365):
    """
    Simulates pairs trading for two stocks.
    Returns equity curve and trade statistics.
    """
    prices    = load_close_prices(lookback_days=lookback + 60)
    if stock1 not in prices.columns or stock2 not in prices.columns:
        return None, None

    s1 = prices[stock1]
    s2 = prices[stock2]
    idx = s1.index.intersection(s2.index)
    s1, s2 = s1[idx], s2[idx]

    # Use first 60 days to establish spread statistics
    spread       = s1 - hedge_ratio * s2
    spread_mean  = spread.iloc[:60].mean()
    spread_std   = spread.iloc[:60].std()

    capital      = initial_capital
    position     = None   # None, 'long1_short2', 'short1_long2'
    shares1      = 0
    shares2      = 0
    entry_price1 = 0
    entry_price2 = 0
    trades       = []
    equity       = []

    for i in range(60, len(s1)):
        p1 = float(s1.iloc[i])
        p2 = float(s2.iloc[i])
        dt = s1.index[i]

        # Rolling spread stats (60-day window)
        window_spread = spread.iloc[max(0, i-60):i]
        spread_mean   = float(window_spread.mean())
        spread_std    = float(window_spread.std())

        if spread_std == 0:
            equity.append({'date': dt, 'value': capital})
            continue

        current_spread = float(spread.iloc[i])
        zscore = (current_spread - spread_mean) / spread_std
        signal = _get_pair_signal(zscore, entry_z, exit_z)

        # Exit logic
        if position and signal == 'EXIT':
            if position == 'long1_short2':
                pnl = (p1 - entry_price1) * shares1 + \
                      (entry_price2 - p2) * shares2
            else:
                pnl = (entry_price1 - p1) * shares1 + \
                      (p2 - entry_price2) * shares2

            capital += pnl
            trades.append({
                'exit_date' : str(dt.date()),
                'pnl'       : round(pnl, 2),
                'result'    : '✅ WIN' if pnl > 0 else '❌ LOSS',
                'zscore'    : round(zscore, 3),
            })
            position = None

        # Entry logic
        elif not position and signal == 'BUY_1_SHORT_2':
            leg_cap  = capital * 0.15
            shares1  = int(leg_cap / p1)
            shares2  = int(leg_cap / p2)
            entry_price1 = p1
            entry_price2 = p2
            position = 'long1_short2'

        elif not position and signal == 'SHORT_1_BUY_2':
            leg_cap  = capital * 0.15
            shares1  = int(leg_cap / p1)
            shares2  = int(leg_cap / p2)
            entry_price1 = p1
            entry_price2 = p2
            position = 'short1_long2'

        equity.append({'date': dt, 'value': capital})

    equity_df = pd.DataFrame(equity).set_index('date')
    closed    = [t for t in trades if 'pnl' in t]
    wins      = sum(1 for t in closed if t['pnl'] > 0)
    total_ret = (capital - initial_capital) / initial_capital * 100

    return equity_df, {
        'total_return' : round(total_ret, 2),
        'total_trades' : len(closed),
        'wins'         : wins,
        'losses'       : len(closed) - wins,
        'win_rate'     : round(wins/len(closed)*100 if closed else 0, 1),
        'final_capital': round(capital, 0),
        'trades'       : closed[-10:]
    }