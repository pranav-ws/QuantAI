"""
sector_rotation.py
Runs sector rotation analysis and shows full rankings.
"""
import matplotlib.pyplot as plt# type: ignore
import matplotlib.gridspec as gridspec# type: ignore
import matplotlib.patches as patches# type: ignore
import numpy as np# type: ignore

from src.sector_rotation import (
    get_rotation_signals, SECTOR_MAP
)

print("\n" + "="*65)
print("  QuantAI — Sector Rotation Strategy")
print("="*65)
print("  Analysing all sectors...")

ranked, buy_stocks, avoid_stocks, best_per_sector = \
    get_rotation_signals(top_n_sectors=3)

if not ranked:
    print("  No data found. Run pipeline.py first.")
    exit()

# ── Print full ranking ────────────────────────────────────
print(f"\n  {'RANK':<5} {'SECTOR':<22} {'1M%':>6} "
      f"{'3M%':>6} {'Breadth':>8} {'RS':>6} "
      f"{'Score':>7} {'Signal':<8}")
print(f"  {'─'*70}")

TIER_COLORS = {
    'TOP'   : '🟢',
    'MIDDLE': '🟡',
    'BOTTOM': '🔴',
}

for r in ranked:
    emoji = TIER_COLORS[r['tier']]
    print(f"  {r['rank']:<4} {emoji} {r['sector']:<20} "
          f"{r['return_1m']:>+5.1f}% "
          f"{r['return_3m']:>+5.1f}% "
          f"{r['breadth']:>7.1f}% "
          f"{r['rs_score']:>+5.1f}% "
          f"{r['composite_score']:>6.3f}  "
          f"{r['signal']:<8}")

# ── Top sectors detail ────────────────────────────────────
print(f"\n{'='*65}")
print(f"  🟢 TOP SECTORS — BUY these stocks")
print(f"{'='*65}")

for r in ranked:
    if r['tier'] != 'TOP':
        continue
    print(f"\n  #{r['rank']} {r['sector']} "
          f"(Score: {r['composite_score']:.3f})")
    print(f"  {'─'*45}")
    returns = r.get('stock_returns', {})
    if returns:
        sorted_stocks = sorted(
            returns.items(), key=lambda x: x[1], reverse=True
        )
        for ticker, ret in sorted_stocks:
            bar   = '█' * max(0, int(abs(ret) * 2))
            color = '↑' if ret >= 0 else '↓'
            print(f"    {ticker.replace('.NS',''):<15} "
                  f"{color} {ret:>+6.1f}%  {bar}")

# ── Bottom sectors ────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  🔴 BOTTOM SECTORS — AVOID these stocks")
print(f"{'='*65}")
for r in ranked:
    if r['tier'] != 'BOTTOM':
        continue
    print(f"  #{r['rank']} {r['sector']}  "
          f"1M: {r['return_1m']:+.1f}%  "
          f"Score: {r['composite_score']:.3f}")

# ── Best stock per sector ─────────────────────────────────
if best_per_sector:
    print(f"\n{'='*65}")
    print(f"  ⭐ BEST STOCK PER TOP SECTOR")
    print(f"{'='*65}")
    for sector, data in best_per_sector.items():
        print(f"  {sector:<22} → "
              f"{data['ticker'].replace('.NS',''):<12} "
              f"({data['return_1m']:+.1f}% in 1M)")

print(f"\n{'='*65}\n")

# ── Chart ─────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0d0d1a')
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

# Sector colors
def sector_color(tier):
    return ('#4ecdc4' if tier == 'TOP' else
            '#ffd700' if tier == 'MIDDLE' else '#ff6b6b')

# Panel 1: Composite score bar chart (horizontal)
ax1 = fig.add_subplot(gs[0, :2])
sectors = [r['sector'][:12] for r in ranked]
scores  = [r['composite_score'] for r in ranked]
colors  = [sector_color(r['tier']) for r in ranked]

bars = ax1.barh(sectors[::-1], scores[::-1],
                color=colors[::-1], edgecolor='none',
                height=0.6)
ax1.axvline(0.5, color='white', linewidth=0.8,
            linestyle='--', alpha=0.5,
            label='Neutral (0.5)')
for bar, score in zip(bars, scores[::-1]):
    ax1.text(bar.get_width() + 0.01,
             bar.get_y() + bar.get_height()/2,
             f'{score:.3f}',
             va='center', color='white', fontsize=8)
ax1.set_title(
    'Sector Ranking — Composite Score '
    '(🟢TOP  🟡MIDDLE  🔴BOTTOM)',
    color='white', fontsize=12
)
ax1.set_facecolor('#1a1a2e')
ax1.tick_params(colors='white', labelsize=9)
ax1.set_xlabel('Composite Score', color='white')
ax1.legend(fontsize=8)

# Panel 2: Rotation wheel
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#1a1a2e')
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.axis('off')
ax2.set_title('Sector Rotation Wheel',
              color='white', fontsize=11)

n       = len(ranked)
angles  = np.linspace(0, 2*np.pi, n, endpoint=False)
for i, (r, angle) in enumerate(zip(ranked, angles)):
    x = np.cos(angle) * 1.1
    y = np.sin(angle) * 1.1
    color = sector_color(r['tier'])
    size  = 1200 if r['tier'] == 'TOP' else 800
    ax2.scatter(x, y, s=size, color=color,
                alpha=0.8, zorder=3)
    ax2.text(x * 1.0, y * 1.0,
             r['sector'][:8],
             ha='center', va='center',
             color='black' if r['tier'] == 'TOP' else 'white',
             fontsize=6.5, fontweight='bold', zorder=4)

ax2.text(0, 0, 'NIFTY\n50',
         ha='center', va='center',
         color='white', fontsize=9, fontweight='bold')
ax2.add_patch(plt.Circle(
    (0, 0), 0.25,
    color='#252545', zorder=2,
    linewidth=2, fill=True
))

# Panel 3: 1M returns heatmap by sector
ax3 = fig.add_subplot(gs[1, 0])
sector_names = [r['sector'][:12] for r in ranked]
returns_1m   = [r['return_1m'] for r in ranked]
colors_ret   = ['#4ecdc4' if v >= 0 else '#ff6b6b'
                for v in returns_1m]

bars3 = ax3.bar(range(len(sector_names)), returns_1m,
                color=colors_ret, edgecolor='none',
                width=0.7)
ax3.axhline(0, color='white', linewidth=0.8, alpha=0.5)
ax3.set_xticks(range(len(sector_names)))
ax3.set_xticklabels(sector_names, rotation=45,
                    ha='right', fontsize=7,
                    color='white')
ax3.set_title('1-Month Returns by Sector',
              color='white', fontsize=11)
ax3.set_ylabel('Return (%)', color='white')
ax3.set_facecolor('#1a1a2e')
ax3.tick_params(colors='white')

# Panel 4: Breadth by sector
ax4 = fig.add_subplot(gs[1, 1])
breadths = [r['breadth'] for r in ranked]
ax4.bar(range(len(sector_names)), breadths,
        color=colors, edgecolor='none', width=0.7)
ax4.axhline(50, color='white', linewidth=0.8,
            linestyle='--', alpha=0.5,
            label='50% (neutral)')
ax4.set_xticks(range(len(sector_names)))
ax4.set_xticklabels(sector_names, rotation=45,
                    ha='right', fontsize=7,
                    color='white')
ax4.set_title('Sector Breadth (% Stocks Above MA50)',
              color='white', fontsize=11)
ax4.set_ylabel('Breadth (%)', color='white')
ax4.set_ylim(0, 110)
ax4.legend(fontsize=8)
ax4.set_facecolor('#1a1a2e')
ax4.tick_params(colors='white')

# Panel 5: Action guide
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor('#1a1a2e')
ax5.axis('off')
ax5.text(0.5, 0.97, 'Rotation Action Guide',
         ha='center', va='top',
         transform=ax5.transAxes,
         color='white', fontsize=11,
         fontweight='bold')

y = 0.85
for r in ranked[:6]:
    color = sector_color(r['tier'])
    sig   = r['signal']
    ax5.add_patch(patches.FancyBboxPatch(
        (0.02, y-0.12), 0.96, 0.13,
        boxstyle="round,pad=0.01",
        facecolor='#252545',
        edgecolor=color,
        linewidth=1.2,
        transform=ax5.transAxes
    ))
    ax5.text(0.06, y-0.03,
             f"#{r['rank']} {r['sector'][:14]}",
             transform=ax5.transAxes,
             color=color, fontsize=8,
             fontweight='bold', va='top')
    ax5.text(0.06, y-0.08,
             f"1M: {r['return_1m']:+.1f}%  "
             f"Score: {r['composite_score']:.3f}  "
             f"→ {sig}",
             transform=ax5.transAxes,
             color='#cccccc', fontsize=7.5,
             va='top')
    y -= 0.155

plt.suptitle(
    'QuantAI — Sector Rotation Strategy',
    color='white', fontsize=14, y=1.01
)
plt.savefig('models/sector_rotation.png',
            dpi=150, bbox_inches='tight',
            facecolor='#0d0d1a')
plt.show()
print("  Chart saved → models/sector_rotation.png")