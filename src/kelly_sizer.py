"""
src/kelly_sizer.py

Kelly Criterion position sizing for QuantAI.

The Kelly formula tells you the mathematically optimal
fraction of capital to bet given your edge.

Full Kelly: f* = (p*b - q) / b
  p = win probability (model confidence)
  b = win/loss ratio (avg win % / avg loss %)
  q = 1 - p

We use Fractional Kelly (0.5x) in practice because:
  - Full Kelly assumes perfect probability estimates
  - Our model confidence is noisy
  - Half Kelly reduces drawdowns significantly
  - Still captures most of the edge

Three modes:
  1. Static Kelly   — uses historical win/loss from trades
  2. Dynamic Kelly  — updates win/loss ratio in real time
  3. Ensemble Kelly — weights confidence from all 4 models
"""

import numpy as np# type: ignore
import pandas as pd# type: ignore
import sqlite3
import json
import os
from datetime import datetime, date

TRADES_PATH  = os.path.join('data', 'paper_trades.json')
CAPITAL_PATH = os.path.join('data', 'paper_capital.json')
KELLY_PATH   = os.path.join('data', 'kelly_stats.json')

# ── Kelly defaults (used before enough trade history) ─────
DEFAULT_WIN_RATE  = 0.55     # 55% — conservative estimate
DEFAULT_WIN_LOSS  = 1.5      # avg win is 1.5x avg loss
KELLY_FRACTION    = 0.5      # half Kelly — safer in practice
MAX_KELLY_BET     = 0.25     # never bet more than 25% on one trade
MIN_KELLY_BET     = 0.01     # never bet less than 1%

# ── Core Kelly formula ────────────────────────────────────
def kelly_fraction(win_prob, win_loss_ratio):
    """
    Calculates the optimal Kelly fraction.

    Args:
      win_prob      : probability of winning (0 to 1)
      win_loss_ratio: (avg win %) / (avg loss %)

    Returns:
      f* : fraction of capital to bet (0 to 1)

    Example:
      win_prob = 0.60, win_loss_ratio = 1.5
      f* = (0.60 * 1.5 - 0.40) / 1.5 = 0.333 → bet 33% of capital
      Half Kelly → 0.167 → bet 16.7% of capital
    """
    if win_prob <= 0 or win_loss_ratio <= 0:
        return 0.0

    p = win_prob
    q = 1 - win_prob
    b = win_loss_ratio

    f_star = (p * b - q) / b

    # Negative Kelly = no edge = don't bet
    if f_star <= 0:
        return 0.0

    return round(f_star, 4)

def half_kelly(win_prob, win_loss_ratio):
    """Returns half Kelly fraction (safer in practice)."""
    return kelly_fraction(win_prob, win_loss_ratio) * KELLY_FRACTION

def fractional_kelly(win_prob, win_loss_ratio, fraction=0.5):
    """Returns any fractional Kelly."""
    return kelly_fraction(win_prob, win_loss_ratio) * fraction

# ── Calculate win/loss stats from trade history ───────────
def get_kelly_stats_from_trades():
    """
    Analyses closed trades to extract real win/loss stats.
    Falls back to defaults if not enough history.

    Returns:
      win_rate      : actual win rate from closed trades
      win_loss_ratio: actual avg win / avg loss ratio
      n_trades      : number of closed trades analysed
    """
    if not os.path.exists(TRADES_PATH):
        return DEFAULT_WIN_RATE, DEFAULT_WIN_LOSS, 0

    with open(TRADES_PATH) as f:
        trades = json.load(f)

    closed = [t for t in trades if t.get('status') == 'CLOSED'
              and t.get('pnl') is not None]

    if len(closed) < 10:
        # Not enough history — use defaults
        return DEFAULT_WIN_RATE, DEFAULT_WIN_LOSS, len(closed)

    wins   = [t['pnl'] for t in closed if t['pnl'] > 0]
    losses = [abs(t['pnl']) for t in closed if t['pnl'] <= 0]

    win_rate = len(wins) / len(closed)

    avg_win  = np.mean(wins)   if wins   else 0
    avg_loss = np.mean(losses) if losses else 1

    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else DEFAULT_WIN_LOSS

    # Save stats for reference
    stats = {
        'win_rate'       : round(win_rate, 4),
        'win_loss_ratio' : round(win_loss_ratio, 4),
        'n_trades'       : len(closed),
        'n_wins'         : len(wins),
        'n_losses'       : len(losses),
        'avg_win'        : round(avg_win, 2),
        'avg_loss'       : round(avg_loss, 2),
        'updated_at'     : datetime.now().isoformat(),
    }
    os.makedirs('data', exist_ok=True)
    with open(KELLY_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    return win_rate, win_loss_ratio, len(closed)

# ── Main position sizer ───────────────────────────────────
def kelly_position_size(
    capital,
    price,
    model_confidence,
    ticker=None,
    regime='SIDEWAYS',
    use_ensemble=True,
    verbose=False
):
    """
    Calculates optimal position size using Kelly Criterion.

    Args:
      capital          : available capital in rupees
      price            : current stock price
      model_confidence : ensemble model confidence (0 to 1)
      ticker           : stock ticker (for per-stock stats)
      regime           : market regime affects Kelly fraction
      use_ensemble     : blend model confidence with historical rates
      verbose          : print detailed breakdown

    Returns:
      shares      : number of shares to buy
      kelly_f     : kelly fraction used
      position_val: total position value in rupees
      details     : dict with full breakdown
    """
    # Step 1: Get win/loss stats
    win_rate, win_loss_ratio, n_trades = get_kelly_stats_from_trades()

    # Step 2: Blend model confidence with historical win rate
    # As we get more trades, trust history more than model
    if n_trades >= 30:
        # Enough history — blend 60% historical, 40% model
        blended_win_prob = (win_rate * 0.60 +
                            model_confidence * 0.40)
    elif n_trades >= 10:
        # Some history — blend 40% historical, 60% model
        blended_win_prob = (win_rate * 0.40 +
                            model_confidence * 0.60)
    else:
        # No history — trust model fully
        blended_win_prob = model_confidence

    # Step 3: Calculate base Kelly fraction
    base_kelly = half_kelly(blended_win_prob, win_loss_ratio)

    # Step 4: Regime adjustment
    # In bear market, be more conservative
    regime_multipliers = {
        'BULL'    : 1.10,   # +10% in bull market
        'SIDEWAYS': 1.00,   # normal
        'VOLATILE': 0.75,   # -25% in volatile market
        'BEAR'    : 0.50,   # -50% in bear market
    }
    regime_mult = regime_multipliers.get(regime, 1.0)

    # Step 5: Confidence scaling
    # Scale further based on how confident the model is
    if model_confidence >= 0.75:
        conf_mult = 1.20   # very high confidence → bigger
    elif model_confidence >= 0.65:
        conf_mult = 1.10
    elif model_confidence >= 0.58:
        conf_mult = 1.00   # normal
    else:
        conf_mult = 0.80   # low confidence → smaller

    # Step 6: Final Kelly fraction
    final_kelly = base_kelly * regime_mult * conf_mult

    # Step 7: Apply hard limits
    final_kelly = max(MIN_KELLY_BET, min(MAX_KELLY_BET, final_kelly))

    # Step 8: Calculate shares
    position_value = capital * final_kelly
    shares         = int(position_value / price)

    if shares <= 0:
        return 0, 0.0, 0.0, {}

    actual_value = shares * price

    details = {
        'win_rate'        : round(win_rate, 4),
        'win_loss_ratio'  : round(win_loss_ratio, 4),
        'n_trades_history': n_trades,
        'blended_win_prob': round(blended_win_prob, 4),
        'base_kelly'      : round(base_kelly, 4),
        'regime'          : regime,
        'regime_mult'     : regime_mult,
        'conf_mult'       : conf_mult,
        'final_kelly'     : round(final_kelly, 4),
        'position_pct'    : round(final_kelly * 100, 2),
        'shares'          : shares,
        'price'           : price,
        'position_value'  : round(actual_value, 2),
        'capital_used_pct': round(actual_value / capital * 100, 2),
        'stop_loss'       : round(price * 0.97, 2),
        'target'          : round(price * (1 + win_loss_ratio * 0.03), 2),
    }

    if verbose:
        print(f"\n  Kelly Position Sizing — {ticker or 'Stock'}")
        print(f"  {'─'*45}")
        print(f"  Win Rate (historical)  : {win_rate:.1%}")
        print(f"  Win/Loss Ratio         : {win_loss_ratio:.2f}x")
        print(f"  Model Confidence       : {model_confidence:.1%}")
        print(f"  Blended Win Probability: {blended_win_prob:.1%}")
        print(f"  Base Kelly (half)      : {base_kelly:.1%}")
        print(f"  Regime ({regime:<10})   : {regime_mult:.2f}x")
        print(f"  Confidence mult        : {conf_mult:.2f}x")
        print(f"  Final Kelly Fraction   : {final_kelly:.1%}")
        print(f"  ──────────────────────────────────────")
        print(f"  Capital                : ₹{capital:,.0f}")
        print(f"  Position Value         : ₹{actual_value:,.0f} "
              f"({final_kelly:.1%} of capital)")
        print(f"  Shares                 : {shares} @ ₹{price:.1f}")
        print(f"  Stop Loss              : ₹{details['stop_loss']:.1f} (-3%)")
        print(f"  Target                 : ₹{details['target']:.1f} "
              f"(+{win_loss_ratio*3:.1f}%)")
        print(f"  {'─'*45}")

    return shares, final_kelly, actual_value, details

# ── Kelly comparison table ────────────────────────────────
def kelly_comparison_table(capital=100000):
    """
    Shows how Kelly sizing changes across different
    confidence levels and win/loss ratios.
    Useful for understanding your system's edge.
    """
    print(f"\n{'='*65}")
    print(f"  Kelly Criterion — Position Size Table")
    print(f"  Capital: ₹{capital:,.0f}")
    print(f"  (Using Half Kelly, Win/Loss Ratio = 1.5x)")
    print(f"{'='*65}")
    print(f"  {'Confidence':>12} {'Kelly%':>8} "
          f"{'Position ₹':>12} {'Shares @₹1000':>14}")
    print(f"  {'─'*52}")

    for conf in [0.55, 0.58, 0.60, 0.63, 0.65,
                 0.68, 0.70, 0.73, 0.75, 0.80]:
        f      = half_kelly(conf, 1.5)
        f      = max(MIN_KELLY_BET, min(MAX_KELLY_BET, f))
        pos    = capital * f
        shares = int(pos / 1000)
        bar    = '█' * int(f * 100)
        print(f"  {conf:>11.0%} {f:>7.1%} "
              f"₹{pos:>11,.0f}  {shares:>8} sh  {bar}")

    print(f"{'='*65}\n")

# ── Growth simulation ─────────────────────────────────────
def simulate_kelly_growth(
    initial_capital=100000,
    win_rate=0.55,
    win_loss_ratio=1.5,
    n_trades=100,
    kelly_fraction_used=0.5
):
    """
    Simulates portfolio growth using Kelly sizing
    vs fixed 2% risk over n_trades.
    Shows the compounding effect of Kelly criterion.
    """
    np.random.seed(42)

    kelly_capital = initial_capital
    fixed_capital = initial_capital

    kelly_curve = [kelly_capital]
    fixed_curve = [fixed_capital]

    for _ in range(n_trades):
        win = np.random.random() < win_rate

        # Kelly sizing
        f_k = half_kelly(win_rate, win_loss_ratio) * kelly_fraction_used
        f_k = max(MIN_KELLY_BET, min(MAX_KELLY_BET, f_k))

        if win:
            kelly_capital *= (1 + f_k * win_loss_ratio * 0.03)
            fixed_capital *= (1 + 0.02 * win_loss_ratio * 0.03)
        else:
            kelly_capital *= (1 - f_k * 0.03)
            fixed_capital *= (1 - 0.02 * 0.03)

        kelly_curve.append(kelly_capital)
        fixed_curve.append(fixed_capital)

    return kelly_curve, fixed_curve