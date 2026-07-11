"""
run_factors.py
Runs the full factor model on all 50 stocks.
Run this once a week — fundamental data doesn't change daily.
"""
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
import numpy as np # type: ignore

from src.factor_model import rank_all_stocks

print("\n" + "="*65)
print("  QuantAI — Factor Investing Model")
print("  Value + Momentum + Quality")
print("="*65)

# Run with fresh data
rankings = rank_all_stocks(use_cache_hours=0)

if not rankings:
    print("  No data. Run pipeline.py first.")
    exit()

# ── Print full rankings ───────────────────────────────────
print(f"\n  {'RK':<4} {'TICKER':<18} {'SECTOR':<16} "
      f"{'VALUE':>7} {'MOM':>7} {'QUALITY':>8} "
      f"{'SCORE':>7} {'TIER':<10}")
print(f"  {'─'*80}")

TIER_EMOJI = {
    'TOP'   : '🟢',
    'MIDDLE': '🟡',
    'BOTTOM': '🔴',
}

for r in rankings:
    e   = TIER_EMOJI.get(r['tier'], '⚪')
    v   = f"{r['value_score']:.3f}"    if r['value_score']    else ' N/A '
    m   = f"{r['momentum_score']:.3f}" if r['momentum_score'] else ' N/A '
    q   = f"{r['quality_score']:.3f}"  if r['quality_score']  else ' N/A '
    print(f"  {r['rank']:<3} {e} "
          f"{r['ticker'].replace('.NS',''):<16} "
          f"{r['sector']:<16} "
          f"{v:>7} {m:>7} {q:>8} "
          f"{r['composite_score']:>7.3f} "
          f"{r['signal']:<10}")

# ── Print top 10 ──────────────────────────────────────────
top10 = [r for r in rankings if r['rank'] <= 10]
print(f"\n{'='*65}")
print(f"  🏆 TOP 10 FACTOR STOCKS")
print(f"{'='*65}")
for r in top10:
    print(f"\n  #{r['rank']} {r['ticker'].replace('.NS','')} "
          f"— {r['name']}")
    print(f"  Composite Score : {r['composite_score']:.3f}")
    if r.get('value_details'):
        vd = r['value_details']
        print(f"  Value   ({r['value_score']:.3f}): "
              f"P/E={vd.get('pe','N/A')}  "
              f"P/B={vd.get('pb','N/A')}  "
              f"P/S={vd.get('ps','N/A')}")
    if r.get('momentum_details'):
        md = r['momentum_details']
        print(f"  Momentum({r['momentum_score']:.3f}): "
              f"12M={md.get('return_12_1m','N/A')}%  "
              f"3M={md.get('return_3m','N/A')}%")
    if r.get('quality_details'):
        qd = r['quality_details']
        print(f"  Quality ({r['quality_score']:.3f}): "
              f"ROE={qd.get('roe','N/A')}%  "
              f"D/E={qd.get('debt_equity','N/A')}  "
              f"GM={qd.get('gross_margin','N/A')}%")

print(f"\n{'='*65}\n")

# ── Chart ─────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0d0d1a')
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

TIER_COLORS = {
    'TOP'   : '#4ecdc4',
    'MIDDLE': '#ffd700',
    'BOTTOM': '#ff6b6b',
}

valid = [r for r in rankings
         if r['composite_score'] is not None]

# Panel 1: Composite score bar chart
ax1 = fig.add_subplot(gs[0, :])
labels  = [r['ticker'].replace('.NS', '')
           for r in valid]
scores  = [r['composite_score'] for r in valid]
colors  = [TIER_COLORS[r['tier']] for r in valid]
bars    = ax1.bar(range(len(labels)), scores,
                  color=colors, edgecolor='none', width=0.7)
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=45,
                    ha='right', fontsize=7.5,
                    color='white')
ax1.axhline(0.5, color='white', linewidth=0.8,
            linestyle='--', alpha=0.5,
            label='Neutral (0.5)')

# Add tier boundary lines
n         = len(valid)
top_cutoff= n // 3
mid_cutoff= 2 * n // 3
ax1.axvline(top_cutoff - 0.5, color='#4ecdc4',
            linewidth=1.5, linestyle=':',
            alpha=0.7, label='TOP / MIDDLE')
ax1.axvline(mid_cutoff - 0.5, color='#ff6b6b',
            linewidth=1.5, linestyle=':',
            alpha=0.7, label='MIDDLE / BOTTOM')
ax1.set_title('Factor Score Ranking — All 50 Stocks '
              '(🟢TOP  🟡MIDDLE  🔴BOTTOM)',
              color='white', fontsize=12)
ax1.set_ylabel('Composite Factor Score', color='white')
ax1.legend(fontsize=8)
ax1.set_facecolor('#1a1a2e')
ax1.tick_params(colors='white')
ax1.set_xlim(-0.5, len(valid) - 0.5)

# Panel 2: Factor scatter — Value vs Momentum
ax2 = fig.add_subplot(gs[1, 0])
for tier, col in TIER_COLORS.items():
    subset = [r for r in valid
              if r['tier'] == tier
              and r['value_score']
              and r['momentum_score']]
    if subset:
        ax2.scatter(
            [r['value_score'] for r in subset],
            [r['momentum_score'] for r in subset],
            c=col, s=80, alpha=0.8,
            label=tier, edgecolors='none'
        )
        for r in subset[:5]:
            ax2.annotate(
                r['ticker'].replace('.NS', ''),
                (r['value_score'],
                 r['momentum_score']),
                fontsize=6.5, color='white',
                xytext=(3, 3),
                textcoords='offset points'
            )
ax2.axhline(0.5, color='white', linewidth=0.5,
            alpha=0.3, linestyle='--')
ax2.axvline(0.5, color='white', linewidth=0.5,
            alpha=0.3, linestyle='--')
ax2.set_title('Value vs Momentum Factor',
              color='white', fontsize=11)
ax2.set_xlabel('Value Score', color='white')
ax2.set_ylabel('Momentum Score', color='white')
ax2.legend(fontsize=8)
ax2.set_facecolor('#1a1a2e')
ax2.tick_params(colors='white')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Quadrant labels
for txt, x, y in [
    ('High value\nHigh mom', 0.75, 0.85),
    ('Low value\nHigh mom',  0.15, 0.85),
    ('High value\nLow mom',  0.75, 0.15),
    ('Low value\nLow mom',   0.15, 0.15),
]:
    ax2.text(x, y, txt, ha='center', va='center',
             fontsize=6, color='#555577',
             transform=ax2.transAxes)

# Panel 3: Value vs Quality scatter
ax3 = fig.add_subplot(gs[1, 1])
for tier, col in TIER_COLORS.items():
    subset = [r for r in valid
              if r['tier'] == tier
              and r['value_score']
              and r['quality_score']]
    if subset:
        ax3.scatter(
            [r['value_score'] for r in subset],
            [r['quality_score'] for r in subset],
            c=col, s=80, alpha=0.8,
            label=tier, edgecolors='none'
        )
        for r in subset[:5]:
            ax3.annotate(
                r['ticker'].replace('.NS', ''),
                (r['value_score'],
                 r['quality_score']),
                fontsize=6.5, color='white',
                xytext=(3, 3),
                textcoords='offset points'
            )
ax3.axhline(0.5, color='white', linewidth=0.5,
            alpha=0.3, linestyle='--')
ax3.axvline(0.5, color='white', linewidth=0.5,
            alpha=0.3, linestyle='--')
ax3.set_title('Value vs Quality Factor',
              color='white', fontsize=11)
ax3.set_xlabel('Value Score', color='white')
ax3.set_ylabel('Quality Score', color='white')
ax3.legend(fontsize=8)
ax3.set_facecolor('#1a1a2e')
ax3.tick_params(colors='white')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# Panel 4: Factor exposure by sector
ax4 = fig.add_subplot(gs[1, 2])
sectors      = {}
for r in valid:
    sec = r.get('sector', 'Unknown')
    if sec not in sectors:
        sectors[sec] = []
    sectors[sec].append(r['composite_score'])

sec_names  = list(sectors.keys())
sec_avg    = [np.mean(sectors[s]) for s in sec_names]
sec_colors = ['#4ecdc4' if v >= 0.5 else '#ff6b6b'
              for v in sec_avg]

sorted_pairs = sorted(zip(sec_avg, sec_names,
                           sec_colors), reverse=True)
sec_avg_s, sec_names_s, sec_col_s = zip(
    *sorted_pairs
) if sorted_pairs else ([], [], [])

bars4 = ax4.barh(list(sec_names_s),
                  list(sec_avg_s),
                  color=list(sec_col_s),
                  edgecolor='none')
ax4.axvline(0.5, color='white', linewidth=0.8,
            linestyle='--', alpha=0.5)
ax4.set_title('Avg Factor Score by Sector',
              color='white', fontsize=11)
ax4.set_facecolor('#1a1a2e')
ax4.tick_params(colors='white', labelsize=8)
ax4.set_xlabel('Avg Composite Score', color='white')
ax4.set_xlim(0, 1)

plt.suptitle(
    'QuantAI — Factor Investing Model '
    '(Value + Momentum + Quality)',
    color='white', fontsize=14, y=1.01
)
plt.savefig('models/factor_scores.png', dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.show()
print("  Chart saved → models/factor_scores.png")