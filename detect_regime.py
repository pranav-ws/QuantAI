"""
detect_regime.py
Run this to see the current market regime with full analysis.
"""
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
import matplotlib.patches as patches # type: ignore
import numpy as np # type: ignore

from src.regime_detector import detect_regime, REGIMES, load_market_prices

# ── Detect regime ─────────────────────────────────────────
print("\n" + "="*62)
print("  QuantAI — Market Regime Detector")
print("="*62)

regime_data = detect_regime(use_cache_minutes=0)  # Force fresh

regime   = regime_data['regime']
comp     = regime_data['composite']
scores   = regime_data['scores']
details  = regime_data['details']
trading  = regime_data['trading']
emoji    = REGIMES[regime]['emoji']

# ── Print full report ─────────────────────────────────────
print(f"\n  Current Market Regime: {emoji} {regime}")
print(f"  Composite Score      : {comp:.3f} "
      f"(0=Bear, 0.5=Neutral, 1=Bull)")
print(f"  Description          : {trading['description']}")
print(f"  Confidence Threshold : {trading['threshold']:.0%} "
      f"(auto-adjusted for this regime)")
print(f"  Max Positions        : {trading['max_positions']}")

print(f"\n  {'─'*50}")
print(f"  5-Factor Breakdown")
print(f"  {'─'*50}")

factor_data = [
    ('Trend',      scores.get('trend', 0),
     details.get('trend', {}),     35),
    ('Breadth',    scores.get('breadth', 0),
     details.get('breadth', {}),   25),
    ('Momentum',   scores.get('momentum', 0),
     details.get('momentum', {}),  20),
    ('Volatility', scores.get('volatility', 0),
     details.get('volatility', {}),15),
    ('RSI',        scores.get('rsi', 0),
     details.get('rsi', {}),        5),
]

for name, score, detail, weight in factor_data:
    bar   = '█' * int(score * 20)
    space = '░' * (20 - int(score * 20))
    signal = '🟢' if score > 0.6 else '🔴' if score < 0.4 else '🟡'
    print(f"\n  {signal} {name:<12} "
          f"[{bar}{space}] {score:.3f} (weight: {weight}%)")

    # Print key detail
    if name == 'Trend' and detail:
        print(f"     Stocks above MA50 : "
              f"{detail.get('pct_above_ma50', 0):.1f}%")
        print(f"     Stocks above MA200: "
              f"{detail.get('pct_above_ma200', 0):.1f}%")
    elif name == 'Breadth' and detail:
        print(f"     Advancing : {detail.get('advancing', 0)} stocks")
        print(f"     Declining : {detail.get('declining', 0)} stocks")
        print(f"     A/D Ratio : {detail.get('ad_ratio', 0):.1f}%")
    elif name == 'Momentum' and detail:
        print(f"     Avg 20d return: "
              f"{detail.get('avg_return_20d', 0):+.2f}%")
        print(f"     Avg 60d return: "
              f"{detail.get('avg_return_60d', 0):+.2f}%")
    elif name == 'Volatility' and detail:
        print(f"     Avg daily range: "
              f"{detail.get('avg_daily_range_pct', 0):.2f}%")
        print(f"     Level: "
              f"{detail.get('volatility_level', 'N/A')}")
    elif name == 'RSI' and detail:
        print(f"     Avg market RSI: "
              f"{detail.get('avg_market_rsi', 0):.1f}")
        print(f"     Condition: "
              f"{detail.get('market_condition', 'N/A')}")

print(f"\n  {'─'*50}")
print(f"  Trading Strategy for {emoji} {regime} Regime:")
print(f"  • Confidence threshold : {trading['threshold']:.0%}")
print(f"  • Max open positions   : {trading['max_positions']}")
print(f"  • Strategy             : {trading['description']}")
print(f"{'='*62}\n")

# ── Chart ─────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0d0d1a')
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

# Regime colors
REGIME_COLORS = {
    'BULL'    : '#4ecdc4',
    'BEAR'    : '#ff6b6b',
    'SIDEWAYS': '#ffd700',
    'VOLATILE': '#ff9f43',
}
reg_color = REGIME_COLORS[regime]

# Panel 1: Regime gauge (big)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#1a1a2e')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

ax1.add_patch(patches.FancyBboxPatch(
    (0.3, 2), 9.4, 7.5,
    boxstyle="round,pad=0.3",
    facecolor='#252545',
    edgecolor=reg_color,
    linewidth=3
))
ax1.text(5, 8.8, 'Market Regime',
         ha='center', color='white',
         fontsize=11, fontweight='bold')
ax1.text(5, 7.5, f'{emoji} {regime}',
         ha='center', color=reg_color,
         fontsize=28, fontweight='bold')
ax1.text(5, 6.3,
         f'Composite Score: {comp:.3f}',
         ha='center', color='white', fontsize=11)
ax1.text(5, 5.4,
         trading['description'],
         ha='center', color='#aaaaaa',
         fontsize=8.5, wrap=True)
ax1.text(5, 4.3,
         f"Threshold: {trading['threshold']:.0%}  |  "
         f"Max positions: {trading['max_positions']}",
         ha='center', color=reg_color,
         fontsize=9, fontweight='bold')

# Score bar
bar_w = 8.5 * comp
ax1.add_patch(patches.Rectangle(
    (0.75, 2.8), 8.5, 0.6,
    facecolor='#333355', edgecolor='none'
))
ax1.add_patch(patches.Rectangle(
    (0.75, 2.8), bar_w, 0.6,
    facecolor=reg_color, edgecolor='none'
))
ax1.text(0.75, 2.4, '0\nBear',
         ha='center', color='#ff6b6b', fontsize=7)
ax1.text(9.25, 2.4, '1\nBull',
         ha='center', color='#4ecdc4', fontsize=7)
ax1.set_title('Regime Classification',
              color='white', fontsize=11, pad=10)

# Panel 2: Radar / spider chart of 5 factors
ax2 = fig.add_subplot(gs[0, 1], polar=True)
ax2.set_facecolor('#1a1a2e')

categories = ['Trend', 'Breadth', 'Momentum',
               'Volatility', 'RSI']
values = [
    scores.get('trend', 0),
    scores.get('breadth', 0),
    scores.get('momentum', 0),
    scores.get('volatility', 0),
    scores.get('rsi', 0),
]
values += values[:1]  # close the loop

angles = np.linspace(0, 2*np.pi, len(categories),
                     endpoint=False).tolist()
angles += angles[:1]

ax2.plot(angles, values,
         color=reg_color, linewidth=2)
ax2.fill(angles, values,
         color=reg_color, alpha=0.25)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories,
                    color='white', fontsize=9)
ax2.set_ylim(0, 1)
ax2.set_yticks([0.25, 0.5, 0.75])
ax2.set_yticklabels(['0.25', '0.50', '0.75'],
                    color='#666688', fontsize=7)
ax2.grid(color='#333355', linewidth=0.5)
ax2.set_title('5-Factor Radar',
              color='white', fontsize=11, pad=15)

# Panel 3: Regime history simulation
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('#1a1a2e')

prices = load_market_prices(300)
if not prices.empty:
    # Calculate rolling composite (simplified)
    sample    = list(prices.columns)[:10]
    avg_price = prices[sample].mean(axis=1)
    ma50      = avg_price.rolling(50).mean()
    ma200     = avg_price.rolling(200).mean()

    ax3.plot(avg_price.index, avg_price.values,
             color='white', linewidth=1.0,
             label='Market', alpha=0.8)
    ax3.plot(ma50.index, ma50.values,
             color='#ffd700', linewidth=1.2,
             linestyle='--', label='MA50')
    ax3.plot(ma200.index, ma200.values,
             color='#ff6b6b', linewidth=1.2,
             linestyle='--', label='MA200')

    # Shade regime zones
    cross_up   = (ma50 > ma200)
    prev_cross = cross_up.shift(1)
    for i in range(1, len(cross_up)):
        if cross_up.iloc[i]:
            ax3.axvspan(avg_price.index[i-1],
                        avg_price.index[i],
                        alpha=0.05, color='#4ecdc4')
        else:
            ax3.axvspan(avg_price.index[i-1],
                        avg_price.index[i],
                        alpha=0.05, color='#ff6b6b')

ax3.set_title('Market Trend History\n'
              '(Green=Uptrend / Red=Downtrend)',
              color='white', fontsize=10)
ax3.legend(fontsize=8)
ax3.set_facecolor('#1a1a2e')
ax3.tick_params(colors='white', labelsize=7)
ax3.set_ylabel('Avg Price', color='white')

# Panel 4: Factor scores bar chart
ax4 = fig.add_subplot(gs[1, 0])
factor_names   = ['Trend\n(35%)', 'Breadth\n(25%)',
                  'Momentum\n(20%)', 'Volatility\n(15%)',
                  'RSI\n(5%)']
factor_scores  = [
    scores.get('trend', 0),
    scores.get('breadth', 0),
    scores.get('momentum', 0),
    scores.get('volatility', 0),
    scores.get('rsi', 0),
]
factor_colors  = [reg_color if s > 0.5 else '#ff6b6b'
                  for s in factor_scores]

bars = ax4.bar(factor_names, factor_scores,
               color=factor_colors, edgecolor='none',
               width=0.6)
ax4.axhline(0.5, color='white', linewidth=0.8,
            linestyle='--', alpha=0.5,
            label='Neutral (0.5)')
for bar, score in zip(bars, factor_scores):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f'{score:.2f}',
             ha='center', va='bottom',
             color='white', fontsize=9)
ax4.set_ylim(0, 1.15)
ax4.set_title('Factor Scores (0=Bear, 1=Bull)',
              color='white', fontsize=11)
ax4.set_facecolor('#1a1a2e')
ax4.tick_params(colors='white', labelsize=8)
ax4.legend(fontsize=8)

# Panel 5: Breadth A/D chart
ax5 = fig.add_subplot(gs[1, 1])
b_data = details.get('breadth', {})
adv    = b_data.get('advancing', 25)
dec    = b_data.get('declining', 25)

ax5.barh(['Advancing', 'Declining'],
         [adv, dec],
         color=['#4ecdc4', '#ff6b6b'],
         edgecolor='none')
ax5.text(adv + 0.5, 0, str(adv),
         va='center', color='white', fontsize=12,
         fontweight='bold')
ax5.text(dec + 0.5, 1, str(dec),
         va='center', color='white', fontsize=12,
         fontweight='bold')
ax5.set_title(f'Market Breadth\n'
              f'A/D Ratio: '
              f'{b_data.get("ad_ratio", 50):.1f}%',
              color='white', fontsize=11)
ax5.set_facecolor('#1a1a2e')
ax5.tick_params(colors='white')
ax5.set_xlabel('Number of Stocks', color='white')

# Panel 6: Regime action guide
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#1a1a2e')
ax6.axis('off')

ax6.text(0.5, 0.97, 'Regime Playbook',
         ha='center', va='top',
         transform=ax6.transAxes,
         color='white', fontsize=11,
         fontweight='bold')

playbook = [
    ('🐂 BULL',     '#4ecdc4',
     '• Threshold: 55%\n• Max 6 positions\n• Trade aggressively'),
    ('🐻 BEAR',     '#ff6b6b',
     '• Threshold: 68%\n• Max 2 positions\n• Stay mostly cash'),
    ('↔️  SIDEWAYS', '#ffd700',
     '• Threshold: 60%\n• Max 4 positions\n• Selective trades'),
    ('⚡ VOLATILE', '#ff9f43',
     '• Threshold: 65%\n• Max 3 positions\n• Smaller sizes'),
]

y = 0.85
for reg_name, color, desc in playbook:
    bg_color = '#252545' if reg_name.split()[1].strip() \
                            != regime else '#2a2a5a'
    border   = color if reg_name.split()[1].strip() \
                        == regime else '#333355'
    lw       = 2 if reg_name.split()[1].strip() \
                    == regime else 0.5

    ax6.add_patch(patches.FancyBboxPatch(
        (0.02, y-0.19), 0.96, 0.20,
        boxstyle="round,pad=0.01",
        facecolor=bg_color,
        edgecolor=border,
        linewidth=lw,
        transform=ax6.transAxes
    ))
    ax6.text(0.06, y-0.02, reg_name,
             transform=ax6.transAxes,
             color=color, fontsize=9,
             fontweight='bold', va='top')
    ax6.text(0.06, y-0.08, desc,
             transform=ax6.transAxes,
             color='#cccccc', fontsize=7.5,
             va='top', family='monospace')
    y -= 0.23

plt.suptitle(
    f'QuantAI — Market Regime Detection  '
    f'{emoji} {regime}  (Score: {comp:.3f})',
    color='white', fontsize=13, y=1.01
)
plt.savefig('models/regime_detection.png', dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.show()
print("  Chart saved → models/regime_detection.png")