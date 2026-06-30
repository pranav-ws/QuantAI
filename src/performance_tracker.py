"""
src/performance_tracker.py

Full trade performance tracking for QuantAI.

Tracks every paper trade from entry to exit:
  - P&L per trade (rupees + percentage)
  - Win rate, profit factor, Sharpe ratio
  - Best/worst trades, current streak
  - Performance by sector, model, confidence band
  - Drawdown analysis
  - Rolling returns over time
"""

import json
import os
import numpy as np # type: ignore
import pandas as pd# type: ignore
from datetime import datetime, date, timedelta
import sqlite3

TRADES_PATH   = os.path.join('data', 'paper_trades.json')
CAPITAL_PATH  = os.path.join('data', 'paper_capital.json')
PERF_PATH     = os.path.join('data', 'performance_report.json')
DB_PATH       = os.path.join('data', 'quantai.db')

# ── Load trades ───────────────────────────────────────────
def load_all_trades():
    """Loads all trades from paper_trades.json."""
    if not os.path.exists(TRADES_PATH):
        return []
    with open(TRADES_PATH) as f:
        return json.load(f)

def load_capital():
    """Loads capital state."""
    if not os.path.exists(CAPITAL_PATH):
        return {'capital': 100000.0, 'peak': 100000.0,
                'start': str(date.today())}
    with open(CAPITAL_PATH) as f:
        return json.load(f)

# ── Close open trades ─────────────────────────────────────
def close_open_trades(trades):
    """
    Closes open trades using latest prices from database.
    This lets us calculate unrealised P&L.
    """
    conn = sqlite3.connect(DB_PATH)
    for t in trades:
        if t.get('status') != 'OPEN':
            continue
        try:
            result = pd.read_sql_query(
                "SELECT close FROM prices "
                "WHERE ticker=? "
                "ORDER BY date DESC LIMIT 1",
                conn, params=(t['ticker'],)
            )
            if not result.empty:
                current  = float(result['close'].iloc[0])
                shares   = t.get('shares', 0)
                entry    = t.get('price', current)
                pnl      = (current - entry) * shares
                pnl_pct  = (current - entry) / entry * 100
                t['current_price'] = round(current, 2)
                t['unrealised_pnl']= round(pnl, 2)
                t['unrealised_pct']= round(pnl_pct, 2)
        except Exception:
            continue
    conn.close()
    return trades

# ── Core metrics ──────────────────────────────────────────
def calculate_metrics(trades):
    """
    Calculates all performance metrics from trade list.
    """
    if not trades:
        return {}

    closed = [t for t in trades
              if t.get('status') == 'CLOSED'
              and t.get('pnl') is not None]
    open_t = [t for t in trades
              if t.get('status') == 'OPEN']

    cap    = load_capital()
    start  = cap.get('start', str(date.today()))
    capital= cap.get('capital', 100000)

    if not closed:
        return {
            'total_trades'  : len(trades),
            'open_trades'   : len(open_t),
            'closed_trades' : 0,
            'capital'       : capital,
            'start_date'    : start,
            'message'       : 'No closed trades yet — keep paper trading!'
        }

    pnls     = [t['pnl'] for t in closed]
    pnl_pcts = [t.get('pnl_pct', 0) for t in closed
                if t.get('pnl_pct') is not None]

    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(closed) if closed else 0
    avg_win  = np.mean(wins)   if wins   else 0
    avg_loss = np.mean(losses) if losses else 0

    # Profit factor: gross profit / gross loss
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    profit_factor= (gross_profit / gross_loss
                    if gross_loss > 0 else float('inf'))

    # Total P&L
    total_pnl  = sum(pnls)
    total_return = total_pnl / 100000 * 100

    # Best and worst trades
    best_trade  = max(closed, key=lambda x: x.get('pnl', 0))
    worst_trade = min(closed, key=lambda x: x.get('pnl', 0))

    # Expectancy: avg P&L per trade
    expectancy = np.mean(pnls)

    # Current streak
    streak, streak_type = _calculate_streak(closed)

    # Sharpe ratio (annualised from daily returns)
    if pnl_pcts:
        daily_ret    = np.array(pnl_pcts) / 100
        sharpe       = (np.mean(daily_ret) /
                        (np.std(daily_ret) + 1e-10) *
                        np.sqrt(252))
    else:
        sharpe = 0

    # Sortino ratio (only downside deviation)
    if pnl_pcts:
        down_ret     = [r/100 for r in pnl_pcts if r < 0]
        down_std     = np.std(down_ret) if down_ret else 1e-10
        sortino      = (np.mean(daily_ret) /
                        down_std * np.sqrt(252))
    else:
        sortino = 0

    # Max drawdown
    equity_curve = _build_equity_curve(closed, 100000)
    max_dd       = _calculate_max_drawdown(equity_curve)

    # Calmar ratio: return / max drawdown
    calmar = (total_return / abs(max_dd)
              if max_dd != 0 else 0)

    # Avg hold duration
    durations = []
    for t in closed:
        if t.get('date') and t.get('exit_date'):
            try:
                d1 = datetime.strptime(t['date'], '%Y-%m-%d')
                d2 = datetime.strptime(
                    t['exit_date'], '%Y-%m-%d'
                )
                durations.append((d2 - d1).days)
            except Exception:
                pass
    avg_hold = np.mean(durations) if durations else 0

    return {
        # Summary
        'total_trades'   : len(trades),
        'open_trades'    : len(open_t),
        'closed_trades'  : len(closed),
        'start_date'     : start,
        'capital'        : round(capital, 2),

        # P&L
        'total_pnl'      : round(total_pnl, 2),
        'total_return_pct': round(total_return, 2),
        'gross_profit'   : round(gross_profit, 2),
        'gross_loss'     : round(gross_loss, 2),

        # Win/Loss
        'win_rate'       : round(win_rate * 100, 2),
        'n_wins'         : len(wins),
        'n_losses'       : len(losses),
        'avg_win'        : round(avg_win, 2),
        'avg_loss'       : round(avg_loss, 2),
        'avg_win_pct'    : round(np.mean([t.get('pnl_pct',0)
                                          for t in closed
                                          if t.get('pnl',0) > 0
                                          ] or [0]), 2),
        'avg_loss_pct'   : round(np.mean([t.get('pnl_pct',0)
                                          for t in closed
                                          if t.get('pnl',0) <= 0
                                          ] or [0]), 2),

        # Quality metrics
        'profit_factor'  : round(profit_factor, 3),
        'expectancy'     : round(expectancy, 2),
        'sharpe_ratio'   : round(sharpe, 3),
        'sortino_ratio'  : round(sortino, 3),
        'max_drawdown'   : round(max_dd, 2),
        'calmar_ratio'   : round(calmar, 3),
        'avg_hold_days'  : round(avg_hold, 1),

        # Best/Worst
        'best_trade'     : {
            'ticker': best_trade.get('ticker', ''),
            'pnl'   : best_trade.get('pnl', 0),
            'pnl_pct': best_trade.get('pnl_pct', 0),
            'date'  : best_trade.get('date', ''),
        },
        'worst_trade'    : {
            'ticker': worst_trade.get('ticker', ''),
            'pnl'   : worst_trade.get('pnl', 0),
            'pnl_pct': worst_trade.get('pnl_pct', 0),
            'date'  : worst_trade.get('date', ''),
        },

        # Streak
        'current_streak' : streak,
        'streak_type'    : streak_type,

        # Equity curve
        'equity_curve'   : equity_curve,
    }

# ── Breakdown analysis ────────────────────────────────────
def sector_breakdown(trades):
    """Performance broken down by sector."""
    from src.data_collector import STOCK_UNIVERSE
    closed  = [t for t in trades
               if t.get('status') == 'CLOSED'
               and t.get('pnl') is not None]
    sectors = {}
    for t in closed:
        ticker = t.get('ticker', '')
        sector = STOCK_UNIVERSE.get(
            ticker, ('', 'Unknown')
        )[1]
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(t['pnl'])

    result = {}
    for sector, pnls in sectors.items():
        wins = [p for p in pnls if p > 0]
        result[sector] = {
            'n_trades'    : len(pnls),
            'total_pnl'   : round(sum(pnls), 2),
            'win_rate'    : round(len(wins)/len(pnls)*100
                                  if pnls else 0, 1),
            'avg_pnl'     : round(np.mean(pnls), 2),
        }
    return dict(sorted(result.items(),
                        key=lambda x: x[1]['total_pnl'],
                        reverse=True))

def confidence_breakdown(trades):
    """Performance broken down by model confidence bands."""
    closed  = [t for t in trades
               if t.get('status') == 'CLOSED'
               and t.get('pnl') is not None
               and t.get('confidence') is not None]

    bands   = {
        '58-62%': (0.58, 0.62),
        '62-66%': (0.62, 0.66),
        '66-70%': (0.66, 0.70),
        '70-75%': (0.70, 0.75),
        '75%+':   (0.75, 1.00),
    }
    result  = {}
    for label, (lo, hi) in bands.items():
        subset  = [t for t in closed
                   if lo <= t['confidence'] < hi]
        if subset:
            pnls = [t['pnl'] for t in subset]
            wins = [p for p in pnls if p > 0]
            result[label] = {
                'n_trades': len(subset),
                'win_rate': round(len(wins)/len(subset)*100, 1),
                'avg_pnl' : round(np.mean(pnls), 2),
                'total_pnl': round(sum(pnls), 2),
            }
    return result

def monthly_breakdown(trades):
    """Performance broken down by month."""
    closed = [t for t in trades
              if t.get('status') == 'CLOSED'
              and t.get('pnl') is not None
              and t.get('date')]
    months = {}
    for t in closed:
        try:
            month = t['date'][:7]   # YYYY-MM
        except Exception:
            continue
        if month not in months:
            months[month] = []
        months[month].append(t['pnl'])

    result = {}
    for month, pnls in sorted(months.items()):
        wins = [p for p in pnls if p > 0]
        result[month] = {
            'n_trades' : len(pnls),
            'total_pnl': round(sum(pnls), 2),
            'win_rate' : round(len(wins)/len(pnls)*100, 1),
        }
    return result

# ── Helpers ───────────────────────────────────────────────
def _calculate_streak(closed):
    """Calculates current win/loss streak."""
    if not closed:
        return 0, 'NONE'
    streak = 0
    last   = 'WIN' if closed[-1].get('pnl', 0) > 0 else 'LOSS'
    for t in reversed(closed):
        is_win = t.get('pnl', 0) > 0
        if (is_win  and last == 'WIN') or \
           (not is_win and last == 'LOSS'):
            streak += 1
        else:
            break
    return streak, last

def _build_equity_curve(closed, initial=100000):
    """Builds equity curve from closed trades."""
    curve  = [initial]
    equity = initial
    for t in sorted(closed,
                    key=lambda x: x.get('date', '')):
        equity += t.get('pnl', 0)
        curve.append(round(equity, 2))
    return curve

def _calculate_max_drawdown(equity_curve):
    """Calculates max drawdown from equity curve."""
    if not equity_curve:
        return 0
    peak  = equity_curve[0]
    max_dd = 0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (val - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
    return round(max_dd, 2)

# ── Close a trade manually ────────────────────────────────
def close_trade(ticker, exit_price=None, exit_date=None):
    """
    Closes an open trade.
    Used when you want to mark a paper trade as closed.
    If exit_price not given, uses latest DB price.
    """
    trades = load_all_trades()

    if exit_price is None:
        conn   = sqlite3.connect(DB_PATH)
        result = pd.read_sql_query(
            "SELECT close FROM prices WHERE ticker=? "
            "ORDER BY date DESC LIMIT 1",
            conn, params=(ticker,)
        )
        conn.close()
        if result.empty:
            print(f"  No price found for {ticker}")
            return False
        exit_price = float(result['close'].iloc[0])

    if exit_date is None:
        exit_date = str(date.today())

    closed_count = 0
    for t in trades:
        if (t.get('ticker') == ticker and
                t.get('status') == 'OPEN'):
            entry    = t.get('price', exit_price)
            shares   = t.get('shares', 0)
            pnl      = (exit_price - entry) * shares
            pnl_pct  = (exit_price - entry) / entry * 100

            t['status']      = 'CLOSED'
            t['exit_price']  = round(exit_price, 2)
            t['exit_date']   = exit_date
            t['pnl']         = round(pnl, 2)
            t['pnl_pct']     = round(pnl_pct, 2)
            t['hold_days']   = 1
            closed_count    += 1

    if closed_count:
        cap = load_capital()
        os.makedirs('data', exist_ok=True)
        with open(TRADES_PATH, 'w') as f:
            json.dump(trades, f, indent=2, default=str)
        print(f"  Closed {closed_count} trade(s) for "
              f"{ticker} @ ₹{exit_price:.1f}")

    return closed_count > 0