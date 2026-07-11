"""
alpha_attribution_analyser.py

Runs the full alpha attribution and produces a 5-panel chart showing
WHERE each unit of alpha came from.

Modes:

  --ticker RELIANCE.NS  (default)
    Deep single-stock attribution using backtest data.
    Runs the test split of the stock's full history through the
    trained model, captures feature values at every trade entry,
    and attributes alpha across all 6 layers.
    Best for understanding how the model works on a specific stock.

  --portfolio
    Portfolio-level attribution across your live paper_trades.json.
    Skips feature attribution (would need per-trade feature snapshots)
    but does full sector, confidence, timing, rule-based, and model
    attribution. Best for reviewing how the live system has performed.

  --compare RELIANCE.NS TCS.NS HDFCBANK.NS ...
    Runs backtest attribution on multiple tickers and ranks them by
    alpha, Sharpe, and feature group dominance. Shows which stocks
    your model has the most genuine edge on.

Charts:

  Panel 1: Feature group attribution waterfall
    Horizontal bars showing each signal family's contribution to alpha.
    Coloured by group (Trend=teal, Momentum=amber, Vol=purple, etc.)

  Panel 2: Top individual features contribution
    The 10 features with highest |importance × PnL-correlation|.

  Panel 3: Confidence vs performance
    Line plot of avg PnL per confidence bucket.
    If the line slopes up → model confidence is well-calibrated.

  Panel 4: Monthly P&L heatmap / bar
    Is alpha spread across months or concentrated in 1-2 lucky runs?

  Panel 5: Sector attribution
    BHB decomposition: selection effect + allocation effect per sector.

Run:
    python alpha_attribution_analyser.py
    python alpha_attribution_analyser.py --ticker TCS.NS
    python alpha_attribution_analyser.py --portfolio
    python alpha_attribution_analyser.py --compare RELIANCE.NS TCS.NS INFY.NS
    python alpha_attribution_analyser.py --save
"""
import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as mpatches # type: ignore
from matplotlib.gridspec import GridSpec # type: ignore

from src.alpha_attribution import (
    AlphaAttributor, AlphaReport,
    FEATURE_GROUPS, GROUP_COLORS
)


TRADES_PATH = os.path.join('data', 'paper_trades.json')


def _load_trade_log() -> list:
    if not os.path.exists(TRADES_PATH):
        return []
    with open(TRADES_PATH) as f:
        return json.load(f)


# ── Single-stock 5-panel chart ────────────────────────────

def _plot_single(report: AlphaReport, save: bool = False):
    if not report.group_contributions and not report.monthly_pnl:
        print("  ⚠️  Not enough data for chart — need trained model + DB data.")
        return

    fig = plt.figure(figsize=(18, 14), facecolor='#0d0d1a')
    gs  = GridSpec(3, 2, hspace=0.42, wspace=0.32,
                   left=0.07, right=0.97, top=0.93, bottom=0.05)

    alpha_str = (f"  |  Alpha: {report.total_alpha_pct:+.2f}%"
                 if report.total_alpha_pct else '')
    fig.suptitle(
        f"QuantAI Alpha Attribution — {report.ticker}"
        f"  |  Return: {report.total_return_pct:+.2f}%"
        f"{alpha_str}  |  Sharpe: {report.sharpe:.2f}",
        color='white', fontsize=13, fontweight='bold', y=0.97
    )

    # ── Panel 1: Feature group contribution ──────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')

    if report.group_contributions:
        groups  = list(report.group_contributions.keys())
        contribs = list(report.group_contributions.values())
        colors   = [GROUP_COLORS.get(g, '#64748b') for g in groups]
        bars     = ax1.barh(groups, contribs, color=colors, alpha=0.85, height=0.55)

        for bar, val in zip(bars, contribs):
            x_pos = val + 0.0005 if val >= 0 else val - 0.0005
            ha    = 'left' if val >= 0 else 'right'
            ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
                      f'{val:+.4f}', va='center', ha=ha,
                      fontsize=8, color='white', fontweight='bold')

        ax1.axvline(0, color='white', linewidth=0.8, alpha=0.4)
        ax1.set_xlabel('Attribution Score (importance × PnL-corr)', color='white', fontsize=8)
        ax1.set_title('Layer 1: Feature Group Alpha Attribution',
                       color='white', fontsize=10, fontweight='bold', pad=8)
        ax1.tick_params(colors='#94a3b8', labelsize=8, length=0)
        ax1.invert_yaxis()

        legend_patches = [mpatches.Patch(color=GROUP_COLORS.get(g, '#64748b'), label=g)
                          for g in groups]
        ax1.legend(handles=legend_patches, fontsize=7, facecolor='#12121f',
                    labelcolor='white', edgecolor='#262640', loc='lower right')

    # ── Panel 2: Top individual features ─────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1a1a2e')

    if report.top_features:
        top_names   = [f[0] for f in report.top_features[:10]]
        top_vals    = [f[1] for f in report.top_features[:10]]
        feat_colors = [GROUP_COLORS.get(FEATURE_GROUPS.get(n, 'Other'), '#64748b')
                        for n in top_names]
        ax2.barh(top_names, top_vals, color=feat_colors, alpha=0.85, height=0.6)
        ax2.axvline(0, color='white', linewidth=0.8, alpha=0.4)
        for i, val in enumerate(top_vals):
            ax2.text(val + 0.0002 if val >= 0 else val - 0.0002, i,
                      f'{val:+.4f}', va='center',
                      ha='left' if val >= 0 else 'right',
                      fontsize=7.5, color='white')
        ax2.set_title('Layer 1b: Top 10 Individual Feature Contributions',
                       color='white', fontsize=10, fontweight='bold', pad=8)
        ax2.tick_params(colors='#94a3b8', labelsize=7.5, length=0)
        ax2.invert_yaxis()

    # ── Panel 3: Confidence vs performance ────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#1a1a2e')

    if report.confidence_buckets:
        labels   = list(report.confidence_buckets.keys())
        avg_pnls = [report.confidence_buckets[l]['avg_pnl_pct'] for l in labels]
        win_rates= [report.confidence_buckets[l]['win_rate'] for l in labels]
        n_trades = [report.confidence_buckets[l]['n_trades'] for l in labels]
        x        = range(len(labels))

        bar_cols = ['#22c55e' if p > 0 else '#ef4444' for p in avg_pnls]
        bars     = ax3.bar(x, avg_pnls, color=bar_cols, alpha=0.80, width=0.55)
        for bar, val, n in zip(bars, avg_pnls, n_trades):
            ax3.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.15,
                      f'{val:+.2f}%\n(n={n})',
                      ha='center', va='bottom', fontsize=7.5, color='white')

        ax3_twin = ax3.twinx()
        ax3_twin.plot(x, win_rates, color='#fbbf24', linewidth=1.8,
                       marker='o', markersize=5, label='Win Rate %')
        ax3_twin.set_ylabel('Win Rate (%)', color='#fbbf24', fontsize=8)
        ax3_twin.tick_params(colors='#fbbf24', labelsize=7)
        ax3_twin.set_ylim(0, 100)

        ax3.axhline(0, color='white', linewidth=0.6, alpha=0.4)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=8, color='#e2e8f0')
        ax3.set_ylabel('Avg PnL per Trade (%)', color='white', fontsize=8)
        ax3.set_title('Layer 3: Confidence Bucket Attribution\n'
                       '(does higher confidence → better trades?)',
                       color='white', fontsize=10, fontweight='bold', pad=8)
        ax3.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ── Panel 4: Monthly P&L ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#1a1a2e')

    if report.monthly_pnl:
        months   = list(report.monthly_pnl.keys())
        monthly_pnl = [report.monthly_pnl[m]['total_pnl_pct'] for m in months]
        m_colors = ['#22c55e' if p >= 0 else '#ef4444' for p in monthly_pnl]
        bars     = ax4.bar(months, monthly_pnl, color=m_colors, alpha=0.82, width=0.7)
        for bar, val in zip(bars, monthly_pnl):
            if abs(val) > 0.1:
                ax4.text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.2,
                          f'{val:+.1f}%', ha='center', fontsize=7, color='white')
        ax4.axhline(0, color='white', linewidth=0.6, alpha=0.4)
        ax4.set_xticklabels(months, rotation=45, ha='right',
                              fontsize=7, color='#e2e8f0')
        ax4.set_ylabel('Total P&L (%)', color='white', fontsize=8)
        ax4.set_title('Layer 4: Timing Attribution\n'
                       f'Best: {report.best_month}  |  Worst: {report.worst_month}',
                       color='white', fontsize=10, fontweight='bold', pad=8)
        ax4.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ── Panel 5: Sector attribution ───────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_facecolor('#1a1a2e')

    if report.sector_attribution:
        sectors  = list(report.sector_attribution.keys())
        sel_effs = [report.sector_attribution[s]['selection_effect'] for s in sectors]
        alloc_effs = [report.sector_attribution[s]['allocation_effect'] for s in sectors]
        avg_pnls = [report.sector_attribution[s]['avg_pnl_pct'] for s in sectors]
        x        = np.arange(len(sectors))
        w        = 0.3

        ax5.bar(x - w/2, sel_effs,   width=w, color='#4ecdc4', alpha=0.85,
                 label='Selection Effect (stock picking within sector)')
        ax5.bar(x + w/2, alloc_effs, width=w, color='#fbbf24', alpha=0.85,
                 label='Allocation Effect (over/underweight sector)')
        ax5.plot(x, avg_pnls, color='#a78bfa', linewidth=1.5,
                  marker='D', markersize=5, label='Avg Trade PnL%')

        ax5.axhline(0, color='white', linewidth=0.6, alpha=0.4)
        ax5.set_xticks(x)
        ax5.set_xticklabels(sectors, rotation=30, ha='right',
                              fontsize=8, color='#e2e8f0')
        ax5.set_ylabel('Attribution Effect (%)', color='white', fontsize=8)
        ax5.set_title('Layer 2: Sector Attribution (Brinson-Hood-Beebower Decomposition)',
                       color='white', fontsize=10, fontweight='bold', pad=8)
        ax5.tick_params(colors='#94a3b8', labelsize=7, length=0)
        ax5.legend(fontsize=8, facecolor='#12121f', labelcolor='white',
                    edgecolor='#262640', loc='upper right')

    os.makedirs('models', exist_ok=True)
    suffix   = report.ticker.replace('.NS', '').replace('/', '_')
    out_path = f'models/alpha_attribution_{suffix}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"📊 Chart saved → {out_path}")
    if not save:
        plt.show()


# ── Compare mode chart ────────────────────────────────────

def _plot_compare(reports: list[AlphaReport], save: bool = False):
    labels  = [r.ticker.replace('.NS', '') for r in reports]
    alphas  = [r.total_alpha_pct for r in reports]
    sharpes = [r.sharpe for r in reports]
    win_rates = [r.win_rate for r in reports]

    # Which feature group dominates each stock
    dominant_groups = []
    for r in reports:
        if r.group_contributions:
            dom = max(r.group_contributions, key=lambda g: r.group_contributions[g])
            dominant_groups.append(dom)
        else:
            dominant_groups.append('Unknown')

    fig = plt.figure(figsize=(16, 10), facecolor='#0d0d1a')
    gs  = GridSpec(2, 2, hspace=0.38, wspace=0.32,
                   left=0.07, right=0.97, top=0.92, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')
    bar_cols = ['#22c55e' if a > 0 else '#ef4444' for a in alphas]
    bars = ax1.bar(labels, alphas, color=bar_cols, alpha=0.82, width=0.55)
    for bar, val in zip(bars, alphas):
        ax1.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.1 if val >= 0 else bar.get_height() - 0.3,
                  f'{val:+.2f}%', ha='center', fontsize=8, color='white')
    ax1.axhline(0, color='white', linewidth=0.6, alpha=0.4)
    ax1.set_title('Alpha vs Buy & Hold per Ticker',
                   color='white', fontsize=10, fontweight='bold', pad=8)
    ax1.set_ylabel('Alpha (%)', color='white', fontsize=9)
    ax1.tick_params(colors='#94a3b8', labelsize=8, length=0)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1a1a2e')
    ax2.bar(labels, sharpes, color='#4ecdc4', alpha=0.82, width=0.55)
    ax2.axhline(1.0, color='#fbbf24', linewidth=1.0, linestyle='--',
                 alpha=0.7, label='Sharpe = 1.0 target')
    ax2.axhline(0, color='white', linewidth=0.6, alpha=0.3)
    ax2.set_title('Sharpe Ratio per Ticker',
                   color='white', fontsize=10, fontweight='bold', pad=8)
    ax2.set_ylabel('Sharpe Ratio', color='white', fontsize=9)
    ax2.legend(fontsize=7.5, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640')
    ax2.tick_params(colors='#94a3b8', labelsize=8, length=0)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#1a1a2e')
    ax3.bar(labels, win_rates, color='#a78bfa', alpha=0.82, width=0.55)
    ax3.axhline(50, color='white', linewidth=0.7, linestyle='--', alpha=0.4,
                 label='50% baseline')
    ax3.set_ylim(0, 100)
    ax3.set_title('Win Rate per Ticker',
                   color='white', fontsize=10, fontweight='bold', pad=8)
    ax3.set_ylabel('Win Rate (%)', color='white', fontsize=9)
    ax3.legend(fontsize=7.5, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640')
    ax3.tick_params(colors='#94a3b8', labelsize=8, length=0)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#1a1a2e')
    dom_colors = [GROUP_COLORS.get(g, '#64748b') for g in dominant_groups]
    ax4.scatter(alphas, sharpes, c=dom_colors, s=150, zorder=4, alpha=0.9)
    for label, a, s_r, dom in zip(labels, alphas, sharpes, dominant_groups):
        ax4.annotate(f'{label}\n({dom[:4]})', (a, s_r),
                      fontsize=7, color='white', ha='center', va='bottom',
                      xytext=(0, 8), textcoords='offset points')
    ax4.axvline(0, color='white', linewidth=0.6, alpha=0.4)
    ax4.axhline(1.0, color='#fbbf24', linewidth=0.7, linestyle='--', alpha=0.5)
    ax4.set_xlabel('Alpha (%)', color='white', fontsize=9)
    ax4.set_ylabel('Sharpe Ratio', color='white', fontsize=9)
    ax4.set_title('Alpha vs Sharpe  (colour = dominant feature group)',
                   color='white', fontsize=10, fontweight='bold', pad=8)
    ax4.tick_params(colors='#94a3b8', labelsize=7, length=0)

    legend_patches = [mpatches.Patch(color=c, label=g)
                      for g, c in GROUP_COLORS.items()]
    ax4.legend(handles=legend_patches, fontsize=7, facecolor='#12121f',
                labelcolor='white', edgecolor='#262640', loc='lower right')

    fig.suptitle('QuantAI Alpha Attribution — Multi-Ticker Comparison',
                  color='white', fontsize=13, fontweight='bold', y=0.97)

    os.makedirs('models', exist_ok=True)
    out_path = 'models/alpha_attribution_compare.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"📊 Chart saved → {out_path}")
    if not save:
        plt.show()


# ── Entry point ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='QuantAI Alpha Attribution — where does the edge come from?'
    )
    parser.add_argument('--ticker',    default='RELIANCE.NS',
                        help='Single ticker for deep attribution (default: RELIANCE.NS)')
    parser.add_argument('--portfolio', action='store_true',
                        help='Run portfolio-level attribution on paper_trades.json')
    parser.add_argument('--compare',   nargs='+', metavar='TICKER',
                        help='Compare attribution across multiple tickers')
    parser.add_argument('--save',      action='store_true',
                        help='Save PNG without showing window')
    args = parser.parse_args()

    attributor = AlphaAttributor()

    if args.portfolio:
        print("\n📊 Running portfolio-level attribution on paper_trades.json...\n")
        trade_log = _load_trade_log()
        if not trade_log:
            print("  ❌  No trades found. Run paper_trade.py first.")
            return
        report = attributor.run_portfolio_attribution(trade_log)
        attributor.print_report(report)
        _plot_single(report, save=args.save)

    elif args.compare:
        print(f"\n📊 Comparing attribution across {len(args.compare)} tickers...\n")
        reports = []
        for ticker in args.compare:
            print(f"  Running backtest attribution for {ticker}...")
            r = attributor.run_backtest_attribution(ticker)
            attributor.print_report(r)
            reports.append(r)
        _plot_compare(reports, save=args.save)

    else:
        ticker = args.ticker
        print(f"\n📊 Running backtest attribution for {ticker}...\n")
        report = attributor.run_backtest_attribution(ticker)
        attributor.print_report(report)
        _plot_single(report, save=args.save)


if __name__ == '__main__':
    main()