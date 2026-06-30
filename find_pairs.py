"""
find_pairs.py — Scan all 50 stocks and find the best pairs.
Run this once a week to refresh your pairs universe.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.pairs_finder import find_best_pairs, load_close_prices
from src.pairs_backtest import backtest_pair

# ── Step 1: Find best pairs ───────────────────────────────
pairs = find_best_pairs(
    min_correlation = 0.70,
    max_pvalue      = 0.05,
    max_half_life   = 30,
    min_half_life   = 3,
    top_n           = 15
)

if not pairs:
    print("\n  No pairs found. Make sure pipeline.py has been run.")
    exit()

# ── Step 2: Print pairs table ─────────────────────────────
print(f"\n{'='*70}")
print(f"  TOP PAIRS — Ranked by Half-Life (fastest mean reversion first)")
print(f"{'='*70}")
print(f"  {'Pair':<25} {'Corr':>6} {'p-val':>7} "
      f"{'HalfLife':>9} {'Z-Score':>8} {'Signal':<18}")
print(f"  {'-'*65}")

for p in pairs:
    z   = p['current_zscore']
    sig = p['signal']
    if   sig == 'BUY_1_SHORT_2' : emoji = '🟢 BUY 1 / SHORT 2'
    elif sig == 'SHORT_1_BUY_2' : emoji = '🔴 SHORT 1 / BUY 2'
    elif sig == 'EXIT'           : emoji = '⚪ EXIT'
    else                         : emoji = '⏸  HOLD'
    print(f"  {p['name1']:<12} ↔ {p['name2']:<12} "
          f"{p['correlation']:>5.2f} {p['pvalue']:>7.4f} "
          f"{p['half_life']:>8.1f}d {z:>+8.2f}  {emoji}")

print(f"{'='*70}\n")

# ── Step 3: Backtest the best pair ────────────────────────
if pairs:
    best = pairs[0]
    print(f"\n  Backtesting best pair: "
          f"{best['name1']} ↔ {best['name2']}")

    equity_df, stats = backtest_pair(
        best['stock1'], best['stock2'],
        best['hedge_ratio'], initial_capital=100000
    )

    if stats:
        print(f"\n  {'='*45}")
        print(f"  Backtest Results — "
              f"{best['name1']} ↔ {best['name2']}")
        print(f"  {'='*45}")
        print(f"  Total Return  : {stats['total_return']:>+.2f}%")
        print(f"  Total Trades  : {stats['total_trades']}")
        print(f"  Win Rate      : {stats['win_rate']:.1f}%")
        print(f"  Wins / Losses : "
              f"{stats['wins']} / {stats['losses']}")
        print(f"  Final Capital : "
              f"₹{stats['final_capital']:,.0f}")
        print(f"  {'='*45}\n")

        # ── Plot ─────────────────────────────────────────
        if equity_df is not None:
            prices  = load_close_prices()
            t1, t2  = best['stock1'], best['stock2']

            fig = plt.figure(figsize=(16, 10))
            fig.patch.set_facecolor('#0d0d1a')
            gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

            # Panel 1: Price history of both stocks (normalised)
            ax1 = fig.add_subplot(gs[0, :])
            if t1 in prices.columns and t2 in prices.columns:
                p1n = prices[t1] / prices[t1].iloc[0] * 100
                p2n = prices[t2] / prices[t2].iloc[0] * 100
                ax1.plot(p1n.index, p1n.values,
                         color='#4ecdc4', linewidth=1.2,
                         label=best['name1'])
                ax1.plot(p2n.index, p2n.values,
                         color='#ff6b6b', linewidth=1.2,
                         label=best['name2'])
                ax1.fill_between(p1n.index, p1n.values, p2n.values,
                                 alpha=0.1, color='yellow',
                                 label='Spread')
            ax1.set_title(
                f"Price History — {best['name1']} vs {best['name2']} "
                f"(Corr: {best['correlation']:.2f}  "
                f"Half-life: {best['half_life']}d)",
                color='white', fontsize=11
            )
            ax1.legend(fontsize=9)
            ax1.set_facecolor('#1a1a2e')
            ax1.tick_params(colors='white')
            ax1.set_ylabel('Normalised Price (100 = start)',
                           color='white')

            # Panel 2: Equity curve
            ax2 = fig.add_subplot(gs[1, 0])
            if equity_df is not None:
                eq_n = equity_df['value'] / equity_df['value'].iloc[0] * 100
                ax2.plot(equity_df.index, eq_n,
                         color='#ffd700', linewidth=1.5,
                         label='Pairs Strategy')
                ax2.axhline(100, color='white', linestyle='--',
                            linewidth=0.8, alpha=0.5)
                ax2.fill_between(equity_df.index, eq_n, 100,
                                 where=(eq_n >= 100),
                                 alpha=0.15, color='#ffd700')
            ax2.set_title('Pairs Strategy Equity Curve',
                          color='white', fontsize=11)
            ax2.set_facecolor('#1a1a2e')
            ax2.tick_params(colors='white')
            ax2.set_ylabel('Portfolio (100 = start)',
                           color='white')
            ax2.legend(fontsize=9)

            # Panel 3: All pairs half-life bar chart
            ax3 = fig.add_subplot(gs[1, 1])
            labels = [f"{p['name1']}↔{p['name2']}"
                      for p in pairs[:8]]
            hls    = [p['half_life'] for p in pairs[:8]]
            colors = ['#4ecdc4' if hl <= 10 else
                      '#ffd700' if hl <= 20 else
                      '#ff6b6b' for hl in hls]
            bars   = ax3.barh(labels[::-1], hls[::-1],
                              color=colors[::-1], edgecolor='none')
            ax3.axvline(10, color='#4ecdc4', linestyle='--',
                        linewidth=0.8, alpha=0.7,
                        label='Fast (<10d)')
            ax3.axvline(20, color='#ffd700', linestyle='--',
                        linewidth=0.8, alpha=0.7,
                        label='Medium (<20d)')
            ax3.set_title('Top Pairs — Half-Life (days)',
                          color='white', fontsize=11)
            ax3.set_facecolor('#1a1a2e')
            ax3.tick_params(colors='white', labelsize=8)
            ax3.set_xlabel('Days to mean reversion',
                           color='white')
            ax3.legend(fontsize=8)

            plt.suptitle(
                'QuantAI — Pairs Trading Analysis',
                color='white', fontsize=14, y=1.01
            )
            plt.savefig('models/pairs_analysis.png',
                        dpi=150, bbox_inches='tight',
                        facecolor='#0d0d1a')
            plt.show()
            print("  Chart saved → models/pairs_analysis.png")