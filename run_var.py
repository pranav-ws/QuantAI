"""
run_var.py
Generates a full VaR report for your current paper portfolio.
Run this anytime to see your risk exposure.
"""
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec# type: ignore
import matplotlib.patches as patches# type: ignore
import numpy as np# type: ignore
import json, os

from src.var_calculator import (
    generate_var_report,
    calculate_stock_var,
    load_returns,
    calculate_cvar
)

# ── Generate report ───────────────────────────────────────
print("\n" + "="*62)
print("  QuantAI — Value at Risk Report")
print("="*62)

report = generate_var_report()

capital       = report['capital']
total_inv     = report.get('total_invested', 0)
port_var      = report.get('portfolio_var')
positions     = report.get('position_details', [])
stress        = report.get('stress_tests', {})

# ── Print summary ─────────────────────────────────────────
print(f"\n  Portfolio Snapshot")
print(f"  {'─'*40}")
print(f"  Total Capital     : ₹{capital:>12,.0f}")
print(f"  Cash (uninvested) : ₹{capital:>12,.0f}")
print(f"  Open Positions    : {report['open_positions']}")

if total_inv > 0:
    print(f"  Total Invested    : ₹{total_inv:>12,.0f}")

# ── Portfolio VaR ─────────────────────────────────────────
if port_var:
    print(f"\n  Portfolio VaR (95% confidence)")
    print(f"  {'─'*40}")
    print(f"  Diversified VaR   : "
          f"₹{port_var['diversified_var_inr']:>10,.0f}  "
          f"({port_var['diversified_var_pct']:.2f}%)")
    print(f"  Monte Carlo VaR   : "
          f"₹{port_var['monte_carlo_var_inr']:>10,.0f}")
    print(f"  Conservative VaR  : "
          f"₹{port_var['conservative_var_inr']:>10,.0f}  "
          f"({port_var['conservative_var_pct']:.2f}%)")
    print(f"  Diversif. Benefit : "
          f"₹{port_var['diversification_benefit']:>10,.0f}  "
          f"(saved by holding multiple stocks)")

# ── Individual positions ──────────────────────────────────
if positions:
    print(f"\n  Individual Position VaR")
    print(f"  {'─'*58}")
    print(f"  {'Ticker':<18} {'Value':>10} "
          f"{'VaR%':>7} {'VaR ₹':>10} {'CVaR ₹':>10}")
    print(f"  {'─'*58}")
    for p in sorted(positions,
                     key=lambda x: x['var_rupees'],
                     reverse=True):
        cvar_inr = p.get('cvar_inr') or 0
        print(f"  {p['ticker'].replace('.NS',''):<18} "
              f"₹{p['position_value']:>9,.0f} "
              f"{p['conservative_var']:>6.2f}% "
              f"₹{p['var_rupees']:>9,.0f} "
              f"₹{cvar_inr:>9,.0f}")
else:
    print(f"\n  No open positions — "
          f"showing VaR for full capital if deployed")

    # Show VaR estimates for top stocks
    from src.data_collector import STOCK_UNIVERSE
    print(f"\n  Hypothetical VaR (if ₹10,000 invested in each):")
    print(f"  {'─'*50}")
    print(f"  {'Stock':<18} {'Daily VaR':>10} "
          f"{'Annual Vol':>12} {'Worst Day':>10}")
    print(f"  {'─'*50}")
    for ticker in list(STOCK_UNIVERSE.keys())[:10]:
        sv = calculate_stock_var(ticker, 10000)
        if sv:
            print(f"  {ticker.replace('.NS',''):<18} "
                  f"₹{sv['var_rupees']:>9,.0f} "
                  f"{sv['annual_volatility']:>10.1f}% "
                  f"{sv['worst_day_pct']:>9.1f}%")

# ── Stress tests ──────────────────────────────────────────
if stress:
    print(f"\n  Stress Test Scenarios")
    print(f"  {'─'*50}")
    print(f"  {'Scenario':<28} {'Shock':>7} {'Loss ₹':>10}")
    print(f"  {'─'*50}")
    for scenario, data in stress.items():
        print(f"  {scenario:<28} "
              f"{data['shock_pct']:>6.1f}% "
              f"₹{data['loss_inr']:>9,.0f}")

print(f"\n{'='*62}")

# ── Chart ─────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0d0d1a')
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

# Panel 1: Return distribution for best stock
ax1 = fig.add_subplot(gs[0, :2])
from src.data_collector import STOCK_UNIVERSE
sample_ticker = list(STOCK_UNIVERSE.keys())[0]
returns       = load_returns(sample_ticker, 252)

if returns is not None:
    var_95   = np.percentile(returns, 5)
    var_99   = np.percentile(returns, 1)
    cvar_val = float(returns[returns <= var_95].mean())

    n, bins, _ = ax1.hist(
        returns * 100, bins=50,
        color='#4ecdc4', alpha=0.6, edgecolor='none'
    )
    ax1.axvline(var_95 * 100, color='#ffd700',
                linewidth=2, linestyle='--',
                label=f'VaR 95%: {abs(var_95)*100:.2f}%')
    ax1.axvline(var_99 * 100, color='#ff6b6b',
                linewidth=2, linestyle='--',
                label=f'VaR 99%: {abs(var_99)*100:.2f}%')
    ax1.axvline(cvar_val * 100, color='#ff0000',
                linewidth=1.5, linestyle=':',
                label=f'CVaR 95%: {abs(cvar_val)*100:.2f}%')

    ax1.fill_betweenx(
        [0, max(n)],
        min(returns*100), var_95*100,
        alpha=0.2, color='#ff6b6b', label='Loss tail'
    )
    ax1.set_title(
        f"Daily Return Distribution — "
        f"{sample_ticker.replace('.NS','')} (1 year)",
        color='white', fontsize=11
    )
    ax1.set_xlabel('Daily Return (%)', color='white')
    ax1.set_ylabel('Frequency', color='white')
    ax1.legend(fontsize=9)
    ax1.set_facecolor('#1a1a2e')
    ax1.tick_params(colors='white')

# Panel 2: VaR gauge
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#1a1a2e')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

if port_var:
    var_pct = port_var['conservative_var_pct']
    var_inr = port_var['conservative_var_inr']
else:
    var_pct = 2.5
    var_inr = capital * 0.025

# Color based on risk level
if var_pct < 2:
    risk_color  = '#4ecdc4'
    risk_label  = 'LOW RISK'
elif var_pct < 4:
    risk_color  = '#ffd700'
    risk_label  = 'MEDIUM RISK'
else:
    risk_color  = '#ff6b6b'
    risk_label  = 'HIGH RISK'

ax2.add_patch(patches.FancyBboxPatch(
    (0.5, 4), 9, 5,
    boxstyle="round,pad=0.3",
    facecolor='#252545',
    edgecolor=risk_color,
    linewidth=2
))
ax2.text(5, 8.2, 'Portfolio VaR (95%)',
         ha='center', color='white',
         fontsize=10, fontweight='bold')
ax2.text(5, 6.8, f'₹{var_inr:,.0f}',
         ha='center', color=risk_color,
         fontsize=20, fontweight='bold')
ax2.text(5, 5.8, f'{var_pct:.2f}% of portfolio',
         ha='center', color='white', fontsize=10)
ax2.text(5, 5.0, risk_label,
         ha='center', color=risk_color,
         fontsize=11, fontweight='bold')
ax2.text(5, 4.3,
         'Max loss on 95% of days',
         ha='center', color='#888888', fontsize=8)
ax2.set_title('Risk Gauge', color='white', fontsize=11)

# Panel 3: Monte Carlo simulation paths
ax3 = fig.add_subplot(gs[1, :2])
ret = load_returns(sample_ticker, 252)
if ret is not None:
    mu    = float(ret.mean())
    sigma = float(ret.std())
    days  = 30
    n_paths = 200
    np.random.seed(42)

    for _ in range(n_paths):
        path   = [100]
        shocks = np.random.normal(mu, sigma, days)
        for shock in shocks:
            path.append(path[-1] * (1 + shock))
        color = '#4ecdc4' if path[-1] >= 100 else '#ff6b6b'
        ax3.plot(path, color=color,
                 alpha=0.15, linewidth=0.6)

    ax3.axhline(100, color='white', linewidth=1.5,
                linestyle='--', alpha=0.8,
                label='Starting value')
    var_line = 100 * (1 - abs(var_95))
    ax3.axhline(var_line, color='#ffd700',
                linewidth=1.5, linestyle=':',
                label=f'VaR 95% floor')
    ax3.set_title(
        f'Monte Carlo — 200 Scenarios, 30 Days '
        f'({sample_ticker.replace(".NS","")})',
        color='white', fontsize=11
    )
    ax3.set_xlabel('Trading Days', color='white')
    ax3.set_ylabel('Portfolio Value (₹100 = start)',
                   color='white')
    ax3.legend(fontsize=9)
    ax3.set_facecolor('#1a1a2e')
    ax3.tick_params(colors='white')

# Panel 4: Stress test bar chart
ax4 = fig.add_subplot(gs[1, 2])
if stress:
    scenarios = list(stress.keys())
    losses    = [stress[s]['loss_inr'] for s in scenarios]
    short_labels = [s.split('(')[0].strip()[:15]
                    for s in scenarios]
    colors_s  = ['#ff6b6b', '#ffd700', '#ff9f43', '#ee5a24']

    bars = ax4.barh(short_labels, losses,
                    color=colors_s, edgecolor='none')
    for bar, loss in zip(bars, losses):
        ax4.text(bar.get_width() + max(losses)*0.02,
                 bar.get_y() + bar.get_height()/2,
                 f'₹{loss:,.0f}',
                 va='center', color='white', fontsize=8)
    ax4.set_title('Stress Test Losses',
                  color='white', fontsize=11)
    ax4.set_facecolor('#1a1a2e')
    ax4.tick_params(colors='white', labelsize=8)
    ax4.set_xlabel('Loss (₹)', color='white')

plt.suptitle(
    'QuantAI — Value at Risk Dashboard',
    color='white', fontsize=14, y=1.01
)
plt.savefig('models/var_report.png', dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.show()
print("  Chart saved → models/var_report.png")