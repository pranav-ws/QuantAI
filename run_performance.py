"""
run_performance.py
Full performance report with charts.
Run anytime to see how your paper portfolio is doing.
"""
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
import matplotlib.patches as patches # type: ignore
import numpy as np # type: ignore
import json, os

from src.performance_tracker import (
    load_all_trades, load_capital,
    close_open_trades, calculate_metrics,
    sector_breakdown, confidence_breakdown,
    monthly_breakdown
)

# ── Load and enrich trades ────────────────────────────────
print("\n" + "="*62)
print("  QuantAI — Performance Report")
print("="*62)

trades  = load_all_trades()
trades  = close_open_trades(trades)
metrics = calculate_metrics(trades)
cap     = load_capital()

if not trades:
    print("  No trades yet. Run paper_trade.py first.")
    exit()

# ── Print full report ─────────────────────────────────────
print(f"\n  Portfolio Snapshot")
print(f"  {'─'*50}")
print(f"  Trading since  : {metrics.get('start_date')}")
print(f"  Total capital  : ₹{metrics.get('capital',0):>12,.2f}")
print(f"  Total trades   : {metrics.get('total_trades',0)}")
print(f"  Open positions : {metrics.get('open_trades',0)}")
print(f"  Closed trades  : {metrics.get('closed_trades',0)}")

if metrics.get('closed_trades', 0) == 0:
    print(f"\n  {metrics.get('message','Keep trading!')}")
    exit()

total_pnl = metrics.get('total_pnl', 0)
total_ret = metrics.get('total_return_pct', 0)
pnl_color = '🟢' if total_pnl >= 0 else '🔴'

print(f"\n  P&L Summary")
print(f"  {'─'*50}")
print(f"  Total P&L      : {pnl_color} "
      f"₹{total_pnl:>+10,.2f} ({total_ret:+.2f}%)")
print(f"  Gross profit   : ₹{metrics.get('gross_profit',0):>10,.2f}")
print(f"  Gross loss     : ₹{metrics.get('gross_loss',0):>10,.2f}")
print(f"  Expectancy     : ₹{metrics.get('expectancy',0):>+10,.2f} per trade")

print(f"\n  Win / Loss Stats")
print(f"  {'─'*50}")
wr   = metrics.get('win_rate', 0)
wr_b = '█' * int(wr / 5)
wr_s = '░' * (20 - int(wr / 5))
print(f"  Win rate       : [{wr_b}{wr_s}] {wr:.1f}%")
print(f"  Wins / Losses  : "
      f"{metrics.get('n_wins',0)} / "
      f"{metrics.get('n_losses',0)}")
print(f"  Avg win        : ₹{metrics.get('avg_win',0):>+8,.2f}"
      f"  ({metrics.get('avg_win_pct',0):+.2f}%)")
print(f"  Avg loss       : ₹{metrics.get('avg_loss',0):>+8,.2f}"
      f"  ({metrics.get('avg_loss_pct',0):+.2f}%)")
print(f"  Profit factor  : {metrics.get('profit_factor',0):.3f}"
      f"  (>1.5 is good)")

streak      = metrics.get('current_streak', 0)
streak_type = metrics.get('streak_type', '')
streak_emoji= '🟢' if streak_type == 'WIN' else '🔴'
print(f"  Current streak : "
      f"{streak_emoji} {streak} {streak_type}{'s' if streak>1 else ''}")

print(f"\n  Risk-Adjusted Metrics")
print(f"  {'─'*50}")
print(f"  Sharpe ratio   : {metrics.get('sharpe_ratio',0):>7.3f}"
      f"  (>1.0 is good)")
print(f"  Sortino ratio  : {metrics.get('sortino_ratio',0):>7.3f}")
print(f"  Max drawdown   : {metrics.get('max_drawdown',0):>7.2f}%"
      f"  (target > -20%)")
print(f"  Calmar ratio   : {metrics.get('calmar_ratio',0):>7.3f}")
print(f"  Avg hold       : {metrics.get('avg_hold_days',0):>7.1f} days")

# Best and worst trades
bt = metrics.get('best_trade', {})
wt = metrics.get('worst_trade', {})
print(f"\n  Best Trade     : "
      f"🏆 {bt.get('ticker','').replace('.NS','')} "
      f"₹{bt.get('pnl',0):+,.0f} "
      f"({bt.get('pnl_pct',0):+.1f}%) "
      f"on {bt.get('date','')}")
print(f"  Worst Trade    : "
      f"💔 {wt.get('ticker','').replace('.NS','')} "
      f"₹{wt.get('pnl',0):+,.0f} "
      f"({wt.get('pnl_pct',0):+.1f}%) "
      f"on {wt.get('date','')}")

# Sector breakdown
print(f"\n  Performance by Sector")
print(f"  {'─'*50}")
sbd = sector_breakdown(trades)
for sector, s in list(sbd.items())[:8]:
    bar   = '█' * int(abs(s['total_pnl']) / 500)
    color = '🟢' if s['total_pnl'] >= 0 else '🔴'
    print(f"  {color} {sector:<18} "
          f"₹{s['total_pnl']:>+8,.0f}  "
          f"WR:{s['win_rate']:>5.1f}%  "
          f"N:{s['n_trades']}")

# Confidence breakdown
print(f"\n  Performance by Model Confidence")
print(f"  {'─'*50}")
cbd = confidence_breakdown(trades)
for band, c in cbd.items():
    bar   = '█' * int(abs(c['total_pnl']) / 300)
    color = '🟢' if c['total_pnl'] >= 0 else '🔴'
    print(f"  {color} {band:<10} "
          f"N:{c['n_trades']:>3}  "
          f"WR:{c['win_rate']:>5.1f}%  "
          f"Avg:₹{c['avg_pnl']:>+7,.0f}  "
          f"Total:₹{c['total_pnl']:>+8,.0f}")

# Monthly breakdown
print(f"\n  Monthly P&L")
print(f"  {'─'*50}")
mbd = monthly_breakdown(trades)
for month, m in mbd.items():
    bar   = '█' * min(int(abs(m['total_pnl']) / 1000), 20)
    color = '🟢' if m['total_pnl'] >= 0 else '🔴'
    print(f"  {color} {month}  "
          f"₹{m['total_pnl']:>+8,.0f}  "
          f"WR:{m['win_rate']:>5.1f}%  "
          f"N:{m['n_trades']:>3}  "
          f"{bar}")

print(f"\n{'='*62}\n")

# ── Trade log ─────────────────────────────────────────────
closed = [t for t in trades
          if t.get('status') == 'CLOSED'
          and t.get('pnl') is not None]
open_t = [t for t in trades
          if t.get('status') == 'OPEN']

if closed:
    print(f"  Last 10 Closed Trades")
    print(f"  {'Date':<12} {'Ticker':<16} "
          f"{'Entry':>8} {'Exit':>8} "
          f"{'P&L':>10} {'%':>7}")
    print(f"  {'─'*62}")
    for t in sorted(closed,
                    key=lambda x: x.get('exit_date',''),
                    reverse=True)[:10]:
        pnl   = t.get('pnl', 0)
        pnl_p = t.get('pnl_pct', 0)
        color = '✅' if pnl > 0 else '❌'
        print(f"  {color} "
              f"{t.get('exit_date','N/A'):<11} "
              f"{t.get('ticker','').replace('.NS',''):<15} "
              f"₹{t.get('price',0):>7.1f} "
              f"₹{t.get('exit_price',0):>7.1f} "
              f"₹{pnl:>+9,.0f} "
              f"{pnl_p:>+6.1f}%")

if open_t:
    print(f"\n  Open Positions")
    print(f"  {'Ticker':<16} {'Entry':>8} "
          f"{'Current':>8} {'Unreal P&L':>12}")
    print(f"  {'─'*48}")
    for t in open_t:
        cur  = t.get('current_price', t.get('price', 0))
        unr  = t.get('unrealised_pnl', 0)
        color= '🟢' if unr >= 0 else '🔴'
        print(f"  {color} "
              f"{t.get('ticker','').replace('.NS',''):<15} "
              f"₹{t.get('price',0):>7.1f} "
              f"₹{cur:>7.1f} "
              f"₹{unr:>+10,.0f}")

# ── Chart ─────────────────────────────────────────────────
equity_curve = metrics.get('equity_curve', [])

if len(equity_curve) >= 3:
    fig  = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0d0d1a')
    gs   = gridspec.GridSpec(3, 3,
                              hspace=0.5, wspace=0.35)

    # Panel 1: Equity curve (spans top 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    n   = len(equity_curve)
    col = '#4ecdc4' if equity_curve[-1] >= equity_curve[0] \
          else '#ff6b6b'
    ax1.plot(range(n), equity_curve,
             color=col, linewidth=2.0,
             label='Portfolio value')
    ax1.fill_between(range(n), equity_curve,
                     equity_curve[0],
                     where=[e >= equity_curve[0]
                            for e in equity_curve],
                     alpha=0.15, color='#4ecdc4')
    ax1.fill_between(range(n), equity_curve,
                     equity_curve[0],
                     where=[e < equity_curve[0]
                            for e in equity_curve],
                     alpha=0.15, color='#ff6b6b')
    ax1.axhline(equity_curve[0], color='white',
                linewidth=0.8, linestyle='--',
                alpha=0.5, label='Starting capital')
    ax1.set_title(
        f'Equity Curve  |  '
        f'Total: ₹{equity_curve[-1]:,.0f} '
        f'({(equity_curve[-1]-equity_curve[0])/equity_curve[0]*100:+.2f}%)',
        color='white', fontsize=11
    )
    ax1.set_xlabel('Trade Number', color='white')
    ax1.set_ylabel('Portfolio Value (₹)', color='white')
    ax1.legend(fontsize=9)
    ax1.set_facecolor('#1a1a2e')
    ax1.tick_params(colors='white')
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f'₹{x/1000:.0f}k'
        )
    )

    # Panel 2: KPI card
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#1a1a2e')
    ax2.axis('off')
    kpis = [
        ('Win Rate',       f"{metrics.get('win_rate',0):.1f}%",
         '#4ecdc4' if metrics.get('win_rate',0) > 55
         else '#ffd700'),
        ('Profit Factor',
         f"{metrics.get('profit_factor',0):.2f}x",
         '#4ecdc4' if metrics.get('profit_factor',0) > 1.5
         else '#ff6b6b'),
        ('Sharpe Ratio',
         f"{metrics.get('sharpe_ratio',0):.3f}",
         '#4ecdc4' if metrics.get('sharpe_ratio',0) > 1.0
         else '#ffd700'),
        ('Max Drawdown',
         f"{metrics.get('max_drawdown',0):.1f}%",
         '#4ecdc4' if metrics.get('max_drawdown',0) > -15
         else '#ff6b6b'),
        ('Expectancy',
         f"₹{metrics.get('expectancy',0):+,.0f}",
         '#4ecdc4' if metrics.get('expectancy',0) > 0
         else '#ff6b6b'),
        ('Avg Hold',
         f"{metrics.get('avg_hold_days',0):.0f}d",
         '#ffd700'),
    ]
    ax2.text(0.5, 0.97, 'Key Metrics',
             ha='center', va='top',
             transform=ax2.transAxes,
             color='white', fontsize=11,
             fontweight='bold')
    y = 0.85
    for label, val, color in kpis:
        ax2.add_patch(patches.FancyBboxPatch(
            (0.03, y-0.10), 0.94, 0.11,
            boxstyle="round,pad=0.01",
            facecolor='#252545',
            edgecolor=color,
            linewidth=1.2,
            transform=ax2.transAxes
        ))
        ax2.text(0.08, y-0.02, label,
                 transform=ax2.transAxes,
                 color='#aaaaaa', fontsize=8.5,
                 va='top')
        ax2.text(0.92, y-0.02, val,
                 transform=ax2.transAxes,
                 color=color, fontsize=10,
                 fontweight='bold',
                 va='top', ha='right')
        y -= 0.138

    # Panel 3: P&L per trade waterfall
    ax3 = fig.add_subplot(gs[1, :2])
    pnls_sorted = sorted(closed,
                          key=lambda x: x.get('date',''))
    pnl_vals    = [t['pnl'] for t in pnls_sorted]
    bar_colors  = ['#4ecdc4' if p >= 0 else '#ff6b6b'
                   for p in pnl_vals]
    ax3.bar(range(len(pnl_vals)), pnl_vals,
            color=bar_colors, edgecolor='none',
            width=0.8)
    ax3.axhline(0, color='white',
                linewidth=0.8, alpha=0.5)
    ax3.axhline(metrics.get('avg_win', 0),
                color='#4ecdc4', linewidth=1.0,
                linestyle=':', alpha=0.7,
                label=f"Avg win ₹{metrics.get('avg_win',0):,.0f}")
    ax3.axhline(metrics.get('avg_loss', 0),
                color='#ff6b6b', linewidth=1.0,
                linestyle=':', alpha=0.7,
                label=f"Avg loss ₹{metrics.get('avg_loss',0):,.0f}")
    ax3.set_title('P&L per Trade (chronological)',
                  color='white', fontsize=11)
    ax3.set_xlabel('Trade Number', color='white')
    ax3.set_ylabel('P&L (₹)', color='white')
    ax3.legend(fontsize=8)
    ax3.set_facecolor('#1a1a2e')
    ax3.tick_params(colors='white')

    # Panel 4: Drawdown chart
    ax4 = fig.add_subplot(gs[1, 2])
    dds = []
    peak = equity_curve[0]
    for val in equity_curve:
        if val > peak:
            peak = val
        dds.append((val - peak) / peak * 100)
    ax4.fill_between(range(len(dds)), dds, 0,
                     color='#ff6b6b', alpha=0.5)
    ax4.plot(range(len(dds)), dds,
             color='#ff6b6b', linewidth=1.0)
    ax4.axhline(-15, color='#ffd700',
                linewidth=0.8, linestyle='--',
                label='-15% warning')
    ax4.axhline(-20, color='#ff0000',
                linewidth=0.8, linestyle='--',
                label='-20% danger')
    ax4.set_title('Drawdown History',
                  color='white', fontsize=11)
    ax4.set_ylabel('Drawdown (%)', color='white')
    ax4.legend(fontsize=8)
    ax4.set_facecolor('#1a1a2e')
    ax4.tick_params(colors='white')

    # Panel 5: Sector P&L
    ax5 = fig.add_subplot(gs[2, 0])
    sbd = sector_breakdown(trades)
    sec_names = list(sbd.keys())[:8]
    sec_pnls  = [sbd[s]['total_pnl'] for s in sec_names]
    sec_cols  = ['#4ecdc4' if p >= 0 else '#ff6b6b'
                 for p in sec_pnls]
    ax5.barh(sec_names[::-1], sec_pnls[::-1],
             color=sec_cols[::-1], edgecolor='none')
    ax5.axvline(0, color='white',
                linewidth=0.8, alpha=0.5)
    ax5.set_title('P&L by Sector',
                  color='white', fontsize=11)
    ax5.set_facecolor('#1a1a2e')
    ax5.tick_params(colors='white', labelsize=8)
    ax5.set_xlabel('Total P&L (₹)', color='white')

    # Panel 6: Win rate by confidence
    ax6 = fig.add_subplot(gs[2, 1])
    cbd  = confidence_breakdown(trades)
    if cbd:
        bands  = list(cbd.keys())
        wrs    = [cbd[b]['win_rate'] for b in bands]
        bar6   = ax6.bar(range(len(bands)), wrs,
                         color='#ffd700',
                         edgecolor='none', width=0.6)
        ax6.axhline(50, color='white',
                    linewidth=0.8, linestyle='--',
                    alpha=0.5, label='50% baseline')
        ax6.axhline(55, color='#4ecdc4',
                    linewidth=0.8, linestyle=':',
                    alpha=0.7, label='55% target')
        for bar, wr in zip(bar6, wrs):
            ax6.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{wr:.0f}%',
                ha='center', va='bottom',
                color='white', fontsize=8
            )
        ax6.set_xticks(range(len(bands)))
        ax6.set_xticklabels(bands, rotation=30,
                            ha='right', fontsize=7.5,
                            color='white')
        ax6.set_ylim(0, 100)
        ax6.set_title('Win Rate by Confidence Band',
                      color='white', fontsize=11)
        ax6.set_ylabel('Win Rate (%)', color='white')
        ax6.legend(fontsize=8)
        ax6.set_facecolor('#1a1a2e')
        ax6.tick_params(colors='white')

    # Panel 7: Monthly P&L
    ax7 = fig.add_subplot(gs[2, 2])
    mbd = monthly_breakdown(trades)
    if mbd:
        months    = list(mbd.keys())
        mpnls     = [mbd[m]['total_pnl'] for m in months]
        mcols     = ['#4ecdc4' if p >= 0 else '#ff6b6b'
                     for p in mpnls]
        ax7.bar(range(len(months)), mpnls,
                color=mcols, edgecolor='none', width=0.6)
        ax7.axhline(0, color='white',
                    linewidth=0.8, alpha=0.5)
        ax7.set_xticks(range(len(months)))
        ax7.set_xticklabels(months, rotation=30,
                            ha='right', fontsize=7.5,
                            color='white')
        ax7.set_title('Monthly P&L',
                      color='white', fontsize=11)
        ax7.set_ylabel('P&L (₹)', color='white')
        ax7.set_facecolor('#1a1a2e')
        ax7.tick_params(colors='white')

    plt.suptitle(
        'QuantAI — Full Performance Report',
        color='white', fontsize=14, y=1.01
    )
    plt.savefig('models/performance_report.png',
                dpi=150, bbox_inches='tight',
                facecolor='#0d0d1a')
    plt.show()
    print("  Chart saved → models/performance_report.png")