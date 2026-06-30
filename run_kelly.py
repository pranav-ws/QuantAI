"""
run_kelly.py
Shows Kelly sizing analysis and compares with
your current fixed sizing system.
"""
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
import numpy as np # type: ignore

from src.kelly_sizer import (
    kelly_position_size,
    kelly_comparison_table,
    simulate_kelly_growth,
    get_kelly_stats_from_trades,
    half_kelly,
    kelly_fraction,
    MIN_KELLY_BET, MAX_KELLY_BET
)

# ── Print comparison table ────────────────────────────────
kelly_comparison_table(capital=100000)

# ── Get historical stats ──────────────────────────────────
win_rate, win_loss, n = get_kelly_stats_from_trades()
print(f"  Historical Stats from your trades:")
print(f"  Win Rate      : {win_rate:.1%}")
print(f"  Win/Loss Ratio: {win_loss:.2f}x")
print(f"  Trade History : {n} closed trades\n")

# ── Live example for top signals ─────────────────────────
print(f"{'='*58}")
print(f"  Kelly Sizing — Example for Today's Signals")
print(f"{'='*58}")

example_signals = [
    ('TCS.NS',        2161.4, 0.70, 'SIDEWAYS'),
    ('BHARTIARTL.NS', 1822.5, 0.72, 'BULL'),
    ('INFY.NS',       1116.4, 0.60, 'SIDEWAYS'),
    ('HDFCBANK.NS',    772.5, 0.63, 'BEAR'),
    ('SUNPHARMA.NS',   920.0, 0.58, 'SIDEWAYS'),
]

capital = 100000
print(f"  Capital: ₹{capital:,.0f}\n")
print(f"  {'Stock':<16} {'Conf':>6} {'Kelly%':>7} "
      f"{'Shares':>7} {'Value':>10} {'Old Size':>10}")
print(f"  {'─'*62}")

total_kelly = 0
total_old   = 0

for ticker, price, conf, regime in example_signals:
    shares, f, val, det = kelly_position_size(
        capital=capital,
        price=price,
        model_confidence=conf,
        ticker=ticker,
        regime=regime,
        verbose=False
    )
    # Old system: fixed 2% risk
    old_shares = int(capital * 0.02 / (price * 0.03))
    old_val    = old_shares * price
    total_kelly += val
    total_old   += old_val

    diff = val - old_val
    diff_str = f"{'+'if diff>=0 else ''}{diff:,.0f}"

    print(f"  {ticker.replace('.NS',''):<16} "
          f"{conf:>5.0%} {f:>6.1%}  "
          f"{shares:>6}sh  ₹{val:>8,.0f}  "
          f"₹{old_val:>8,.0f}")

print(f"  {'─'*62}")
print(f"  {'TOTAL':<16} {'':>6} {'':>7} "
      f"{'':>7} ₹{total_kelly:>8,.0f}  ₹{total_old:>8,.0f}")
print(f"  {'':>50} ({'higher' if total_kelly>total_old else 'lower'} "
      f"by ₹{abs(total_kelly-total_old):,.0f})")
print(f"{'='*58}\n")

# ── Simulate growth ───────────────────────────────────────
print(f"  Simulating 200 trades...")
kelly_curve, fixed_curve = simulate_kelly_growth(
    initial_capital=100000,
    win_rate=win_rate,
    win_loss_ratio=win_loss,
    n_trades=200,
    kelly_fraction_used=0.5
)

# ── Chart ─────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0d0d1a')
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

# Panel 1: Growth simulation
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(kelly_curve, color='#ffd700',
         linewidth=2.0, label='Kelly Criterion')
ax1.plot(fixed_curve, color='#4ecdc4',
         linewidth=1.5, linestyle='--', label='Fixed 2% Risk')
ax1.axhline(100000, color='white',
            linewidth=0.8, linestyle=':', alpha=0.5)
ax1.fill_between(range(len(kelly_curve)),
                 kelly_curve, fixed_curve,
                 where=[k > f for k, f in
                        zip(kelly_curve, fixed_curve)],
                 alpha=0.15, color='#ffd700')
ax1.set_title(
    f'Kelly vs Fixed Sizing — 200 Simulated Trades\n'
    f'Win Rate: {win_rate:.0%}  |  '
    f'Win/Loss: {win_loss:.1f}x',
    color='white', fontsize=11
)
ax1.set_xlabel('Trade Number', color='white')
ax1.set_ylabel('Portfolio Value (₹)', color='white')
ax1.legend(fontsize=10)
ax1.set_facecolor('#1a1a2e')
ax1.tick_params(colors='white')
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}k')
)

# Annotate final values
final_kelly = kelly_curve[-1]
final_fixed = fixed_curve[-1]
ax1.annotate(f'₹{final_kelly:,.0f}',
             xy=(200, final_kelly),
             color='#ffd700', fontsize=9,
             fontweight='bold')
ax1.annotate(f'₹{final_fixed:,.0f}',
             xy=(200, final_fixed),
             color='#4ecdc4', fontsize=9)

# Panel 2: Kelly fraction vs confidence
ax2 = fig.add_subplot(gs[0, 2])
confs      = np.linspace(0.50, 0.90, 100)
kelly_fracs= [max(MIN_KELLY_BET,
               min(MAX_KELLY_BET,
               half_kelly(c, win_loss)))
              for c in confs]
full_kelly  = [max(MIN_KELLY_BET,
               min(MAX_KELLY_BET,
               kelly_fraction(c, win_loss)))
               for c in confs]

ax2.plot(confs * 100, [f * 100 for f in kelly_fracs],
         color='#ffd700', linewidth=2,
         label='Half Kelly (used)')
ax2.plot(confs * 100, [f * 100 for f in full_kelly],
         color='#ff6b6b', linewidth=1.5,
         linestyle='--', label='Full Kelly (ref)')
ax2.axhline(MAX_KELLY_BET * 100, color='white',
            linewidth=0.8, linestyle=':',
            alpha=0.5, label='Max cap (25%)')
ax2.axvline(58, color='#4ecdc4', linewidth=0.8,
            linestyle=':', alpha=0.7,
            label='Min threshold (58%)')

# Shade the "active zone"
ax2.fill_between(confs * 100,
                 [f * 100 for f in kelly_fracs],
                 0, alpha=0.15, color='#ffd700')
ax2.set_title('Position Size vs Confidence',
              color='white', fontsize=11)
ax2.set_xlabel('Model Confidence (%)', color='white')
ax2.set_ylabel('Kelly Fraction (%)', color='white')
ax2.legend(fontsize=7)
ax2.set_facecolor('#1a1a2e')
ax2.tick_params(colors='white')
ax2.set_xlim(50, 90)
ax2.set_ylim(0, 30)

# Panel 3: Win/loss ratio impact
ax3 = fig.add_subplot(gs[1, 0])
ratios = np.linspace(0.5, 3.0, 100)
for conf, col, lbl in [(0.58, '#4ecdc4', '58%'),
                        (0.65, '#ffd700', '65%'),
                        (0.75, '#ff6b6b', '75%')]:
    fracs = [max(0, min(MAX_KELLY_BET,
                         half_kelly(conf, r)))
             for r in ratios]
    ax3.plot(ratios, [f * 100 for f in fracs],
             color=col, linewidth=1.5,
             label=f'Conf {lbl}')
ax3.axhline(MAX_KELLY_BET * 100, color='white',
            linewidth=0.8, linestyle=':',
            alpha=0.5)
ax3.axvline(win_loss, color='white',
            linewidth=0.8, linestyle='--',
            alpha=0.5,
            label=f'Your ratio ({win_loss:.1f}x)')
ax3.set_title('Kelly % vs Win/Loss Ratio',
              color='white', fontsize=11)
ax3.set_xlabel('Win/Loss Ratio', color='white')
ax3.set_ylabel('Kelly Fraction (%)', color='white')
ax3.legend(fontsize=8)
ax3.set_facecolor('#1a1a2e')
ax3.tick_params(colors='white')

# Panel 4: Today's signals comparison
ax4 = fig.add_subplot(gs[1, 1])
stocks    = [s[0].replace('.NS', '')
             for s in example_signals]
kelly_vals= []
old_vals  = []
for _, price, conf, regime in example_signals:
    _, f, val, _ = kelly_position_size(
        100000, price, conf, regime=regime
    )
    old_v = int(100000 * 0.02 / (price * 0.03)) * price
    kelly_vals.append(val / 1000)
    old_vals.append(old_v / 1000)

x = np.arange(len(stocks))
w = 0.35
ax4.bar(x - w/2, kelly_vals, w,
        color='#ffd700', label='Kelly', edgecolor='none')
ax4.bar(x + w/2, old_vals,   w,
        color='#4ecdc4', label='Fixed 2%',
        alpha=0.7, edgecolor='none')
ax4.set_xticks(x)
ax4.set_xticklabels(stocks, rotation=30,
                    ha='right', fontsize=8,
                    color='white')
ax4.set_title("Today's Signals — Kelly vs Fixed",
              color='white', fontsize=11)
ax4.set_ylabel('Position Value (₹000)', color='white')
ax4.legend(fontsize=9)
ax4.set_facecolor('#1a1a2e')
ax4.tick_params(colors='white')

# Panel 5: Kelly formula explainer
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor('#1a1a2e')
ax5.axis('off')
ax5.text(0.5, 0.95, 'Kelly Formula',
         ha='center', va='top',
         transform=ax5.transAxes,
         color='white', fontsize=12,
         fontweight='bold')
ax5.text(0.5, 0.82,
         'f* = (p × b - q) / b',
         ha='center', va='top',
         transform=ax5.transAxes,
         color='#ffd700', fontsize=14,
         fontweight='bold',
         family='monospace')
ax5.text(0.5, 0.70,
         'p = win probability\n'
         'b = win/loss ratio\n'
         'q = 1 - p (loss probability)',
         ha='center', va='top',
         transform=ax5.transAxes,
         color='#cccccc', fontsize=10,
         linespacing=1.8)

# Live calculation box
f_ex  = half_kelly(win_rate, win_loss)
f_ex  = max(MIN_KELLY_BET, min(MAX_KELLY_BET, f_ex))
pos_ex = 100000 * f_ex
ax5.add_patch(
    plt.Rectangle((0.05, 0.25), 0.9, 0.25,
                  facecolor='#252545',
                  edgecolor='#ffd700',
                  linewidth=1.5,
                  transform=ax5.transAxes)
)
ax5.text(0.5, 0.47,
         'Your current edge:',
         ha='center', va='top',
         transform=ax5.transAxes,
         color='#aaaaaa', fontsize=9)
ax5.text(0.5, 0.39,
         f'p={win_rate:.0%}  b={win_loss:.1f}x  '
         f'→ f*={f_ex:.1%}',
         ha='center', va='top',
         transform=ax5.transAxes,
         color='#ffd700', fontsize=11,
         fontweight='bold', family='monospace')
ax5.text(0.5, 0.29,
         f'Bet ₹{pos_ex:,.0f} per trade',
         ha='center', va='top',
         transform=ax5.transAxes,
         color='white', fontsize=10)
ax5.text(0.5, 0.15,
         f'Half Kelly reduces risk\n'
         f'Full Kelly maximises growth\n'
         f'but is too aggressive',
         ha='center', va='top',
         transform=ax5.transAxes,
         color='#888888', fontsize=8.5,
         linespacing=1.7)

plt.suptitle(
    'QuantAI — Kelly Criterion Position Sizing',
    color='white', fontsize=14, y=1.01
)
plt.savefig('models/kelly_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.show()
print("  Chart saved → models/kelly_analysis.png")