"""
run_monte_carlo.py — QuantAI Monte Carlo Simulation Runner

Runs two Monte Carlo simulations for a chosen ticker:

  1. Trade-resampling MC  — bootstraps the historical backtest's trade
     returns to show the realistic RANGE of future outcomes (not just
     the one path that happened historically).
  2. Price-path MC (GBM)  — simulates thousands of possible future price
     paths to estimate probability of profit, Value-at-Risk, and the
     odds of hitting a target price within a chosen horizon.

Usage:
    python run_monte_carlo.py

Edit the CONFIG section below to change ticker / capital / horizon.
Charts are saved to models/monte_carlo_trades.png and
models/monte_carlo_price.png — same convention as run_backtest.py
saving to models/backtest_results.png.
"""
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

from src.backtest import run_backtest
from src.features import load_prices
from src.monte_carlo import (
    run_trade_monte_carlo, print_trade_mc_report,
    run_price_monte_carlo, print_price_mc_report,
    probability_of_target,
)

# ── CONFIG ────────────────────────────────────────────────
TICKER          = 'RELIANCE.NS'
INITIAL_CAPITAL = 100000
N_SIMULATIONS   = 2000          # number of random paths to generate
N_TRADES        = None          # None = same number as historical backtest trades
PRICE_HORIZON_DAYS = 30         # how many trading days ahead to simulate prices
TARGET_PRICE_PCT   = 0.10       # show probability of price moving +10% from today
RANDOM_SEED      = 42           # set to None for a different result every run
USE_SLIPPAGE     = True         # realistic costs ON — Monte Carlo then reflects real-world drag
SLIPPAGE_PROFILE = 'delivery'   # 'delivery' (overnight) or 'intraday' (same-day square-off)

print("\n" + "=" * 55)
print("  QuantAI — Monte Carlo Simulation Runner")
print("=" * 55)

# ════════════════════════════════════════════════════════════
#  STEP 1 — Trade-resampling Monte Carlo
#  (reuses the existing backtest to get real historical trades)
# ════════════════════════════════════════════════════════════
print(f"\n[1/2] Running base backtest for {TICKER} to source trade history...")
equity_df, trade_df, metrics = run_backtest(
    ticker=TICKER,
    initial_capital=INITIAL_CAPITAL,
    confidence_threshold=0.58,
    use_slippage=USE_SLIPPAGE,
    slippage_profile=SLIPPAGE_PROFILE
)

if trade_df.empty or 'pnl_pct' not in trade_df.columns or trade_df['pnl_pct'].dropna().empty:
    print("  ⚠️  No completed trades available — skipping trade-resampling Monte Carlo.")
    trade_paths_df, trade_summary = None, None
else:
    trade_paths_df, trade_summary = run_trade_monte_carlo(
        trade_df,
        initial_capital=INITIAL_CAPITAL,
        n_simulations=N_SIMULATIONS,
        n_trades=N_TRADES,
        random_seed=RANDOM_SEED
    )
    print_trade_mc_report(trade_summary, ticker=TICKER)

# ════════════════════════════════════════════════════════════
#  STEP 2 — Price-path Monte Carlo (GBM)
# ════════════════════════════════════════════════════════════
print(f"\n[2/2] Running price-path Monte Carlo for {TICKER}...")
price_df = load_prices(TICKER)
price_paths_df, price_summary = run_price_monte_carlo(
    price_df,
    n_days=PRICE_HORIZON_DAYS,
    n_simulations=N_SIMULATIONS,
    random_seed=RANDOM_SEED
)
target_price = price_summary['last_price'] * (1 + TARGET_PRICE_PCT)
print_price_mc_report(price_summary, ticker=TICKER,
                       target_price=target_price, price_paths=price_paths_df)

# ── Plot 1: Trade-resampling fan chart ──────────────────────
if trade_paths_df is not None:
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig1.patch.set_facecolor('#0d0d1a')
    ax1.set_facecolor('#1a1a2e')

    # Plot a sample of individual paths faintly (avoid rendering thousands)
    sample_n = min(200, trade_paths_df.shape[1])
    sample_cols = np.random.choice(trade_paths_df.shape[1], sample_n, replace=False)
    ax1.plot(trade_paths_df.index, trade_paths_df.iloc[:, sample_cols],
              color='#4ecdc4', alpha=0.06, linewidth=0.8)

    # Percentile bands across ALL simulations
    p5  = trade_paths_df.quantile(0.05, axis=1)
    p25 = trade_paths_df.quantile(0.25, axis=1)
    p50 = trade_paths_df.quantile(0.50, axis=1)
    p75 = trade_paths_df.quantile(0.75, axis=1)
    p95 = trade_paths_df.quantile(0.95, axis=1)

    ax1.fill_between(trade_paths_df.index, p5, p95, color='#3b82f6', alpha=0.15, label='5th–95th percentile')
    ax1.fill_between(trade_paths_df.index, p25, p75, color='#3b82f6', alpha=0.30, label='25th–75th percentile')
    ax1.plot(trade_paths_df.index, p50, color='#f59e0b', linewidth=2, label='Median path')
    ax1.axhline(INITIAL_CAPITAL, color='white', linewidth=0.8, linestyle=':', alpha=0.6, label='Starting capital')

    ax1.set_title(f'Monte Carlo — Trade Resampling — {TICKER}\n'
                   f'({N_SIMULATIONS:,} simulations × {trade_summary["n_trades_per_path"]} trades)',
                   color='white', fontsize=12)
    ax1.set_xlabel('Trade number', color='white')
    ax1.set_ylabel('Portfolio Value (₹)', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig('models/monte_carlo_trades.png', dpi=150,
                bbox_inches='tight', facecolor='#0d0d1a')
    print("📊 Chart saved → models/monte_carlo_trades.png")
    plt.close(fig1)

# ── Plot 2: Price-path fan chart ────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor('#0d0d1a')
ax2.set_facecolor('#1a1a2e')

sample_n2 = min(200, price_paths_df.shape[1])
sample_cols2 = np.random.choice(price_paths_df.shape[1], sample_n2, replace=False)
ax2.plot(price_paths_df.index, price_paths_df.iloc[:, sample_cols2],
          color='#ff6b6b', alpha=0.06, linewidth=0.8)

pp5  = price_paths_df.quantile(0.05, axis=1)
pp25 = price_paths_df.quantile(0.25, axis=1)
pp50 = price_paths_df.quantile(0.50, axis=1)
pp75 = price_paths_df.quantile(0.75, axis=1)
pp95 = price_paths_df.quantile(0.95, axis=1)

ax2.fill_between(price_paths_df.index, pp5, pp95, color='#4ecdc4', alpha=0.15, label='5th–95th percentile')
ax2.fill_between(price_paths_df.index, pp25, pp75, color='#4ecdc4', alpha=0.30, label='25th–75th percentile')
ax2.plot(price_paths_df.index, pp50, color='#f59e0b', linewidth=2, label='Median path')
ax2.axhline(price_summary['last_price'], color='white', linewidth=0.8,
            linestyle=':', alpha=0.6, label="Today's price")
ax2.axhline(target_price, color='#10b981', linewidth=1, linestyle='--',
            alpha=0.8, label=f'Target +{TARGET_PRICE_PCT*100:.0f}%')

ax2.set_title(f'Monte Carlo — Price Simulation (GBM) — {TICKER}\n'
               f'({N_SIMULATIONS:,} simulations × {PRICE_HORIZON_DAYS} trading days)',
               color='white', fontsize=12)
ax2.set_xlabel('Trading day', color='white')
ax2.set_ylabel('Price (₹)', color='white')
ax2.tick_params(colors='white')
ax2.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig('models/monte_carlo_price.png', dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
print("📊 Chart saved → models/monte_carlo_price.png")
plt.show()