"""
scan_patterns.py
Runs pattern scanner on all 50 stocks and shows results.
"""
import matplotlib.pyplot as plt# type: ignore
import matplotlib.gridspec as gridspec# type: ignore
import matplotlib.patches as mpatches# type: ignore
import numpy as np# type: ignore
from src.pattern_scanner import scan_patterns, load_ohlcv
from src.data_collector import STOCK_UNIVERSE

print("\n" + "="*65)
print("  QuantAI — Pattern Scanner")
print("="*65)

all_results  = []
bullish_hits = []
bearish_hits = []

total = len(STOCK_UNIVERSE)
for i, ticker in enumerate(STOCK_UNIVERSE, 1):
    print(f"  Scanning {i}/{total}: {ticker:<22}", end='\r')
    patterns, score = scan_patterns(ticker)
    name = STOCK_UNIVERSE[ticker][0]

    if patterns:
        all_results.append({
            'ticker'  : ticker,
            'name'    : name,
            'patterns': patterns,
            'score'   : score,
        })
        if score > 0:
            bullish_hits.append({
                'ticker': ticker,
                'name'  : name,
                'score' : score,
                'patterns': patterns
            })
        else:
            bearish_hits.append({
                'ticker': ticker,
                'name'  : name,
                'score' : score,
                'patterns': patterns
            })

# Sort
bullish_hits.sort(key=lambda x: x['score'],   reverse=True)
bearish_hits.sort(key=lambda x: x['score'])

print(f"\n  Scan complete — {total} stocks checked\n")

# ── Print bullish patterns ────────────────────────────────
print(f"{'='*65}")
print(f"  🟢 BULLISH PATTERNS DETECTED ({len(bullish_hits)} stocks)")
print(f"{'='*65}")
for r in bullish_hits:
    print(f"\n  {r['ticker'].replace('.NS',''):<15} "
          f"Score: {r['score']:+.2f}")
    for p in r['patterns']:
        if p['direction'] == 'BULLISH':
            print(f"    {p['emoji']} {p['pattern']:<28} "
                  f"conf: {p['confidence']:.2f}")
            print(f"       {p['description']}")

# ── Print bearish patterns ────────────────────────────────
print(f"\n{'='*65}")
print(f"  🔴 BEARISH PATTERNS DETECTED ({len(bearish_hits)} stocks)")
print(f"{'='*65}")
for r in bearish_hits:
    print(f"\n  {r['ticker'].replace('.NS',''):<15} "
          f"Score: {r['score']:+.2f}")
    for p in r['patterns']:
        if p['direction'] == 'BEARISH':
            print(f"    {p['emoji']} {p['pattern']:<28} "
                  f"conf: {p['confidence']:.2f}")
            print(f"       {p['description']}")

print(f"\n{'='*65}\n")

# ── Chart: Pattern heatmap ────────────────────────────────
if all_results:
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0d0d1a')
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # Panel 1: Score bar chart
    ax1 = fig.add_subplot(gs[0, :])
    sorted_results = sorted(all_results,
                            key=lambda x: x['score'],
                            reverse=True)
    labels  = [r['ticker'].replace('.NS', '')
               for r in sorted_results]
    scores  = [r['score'] for r in sorted_results]
    colors  = ['#4ecdc4' if s > 0 else '#ff6b6b'
               for s in scores]

    bars = ax1.bar(labels, scores,
                   color=colors, edgecolor='none', width=0.6)
    ax1.axhline(0, color='white', linewidth=0.8, alpha=0.5)
    ax1.axhline( 0.5, color='#4ecdc4', linewidth=0.5,
                linestyle='--', alpha=0.5)
    ax1.axhline(-0.5, color='#ff6b6b', linewidth=0.5,
                linestyle='--', alpha=0.5)

    ax1.set_title('Pattern Score — All Stocks '
                  '(Green=Bullish / Red=Bearish)',
                  color='white', fontsize=12)
    ax1.set_facecolor('#1a1a2e')
    ax1.tick_params(colors='white', rotation=45, labelsize=8)
    ax1.set_ylabel('Net Pattern Score', color='white')

    # Panel 2: Candlestick for top bullish stock
    if bullish_hits:
        ax2  = fig.add_subplot(gs[1, 0])
        best = bullish_hits[0]
        df   = load_ohlcv(best['ticker'], 60)

        for j, (idx, row) in enumerate(df.iterrows()):
            color = '#4ecdc4' if row['Close'] >= row['Open'] \
                    else '#ff6b6b'
            body_bot = min(row['Open'], row['Close'])
            body_top = max(row['Open'], row['Close'])
            ax2.add_patch(mpatches.Rectangle(
                (j - 0.3, body_bot),
                0.6, body_top - body_bot,
                facecolor=color, edgecolor='none'
            ))
            ax2.plot([j, j], [row['Low'], row['High']],
                     color=color, linewidth=0.8)

        for p in best['patterns']:
            if p['direction'] == 'BULLISH':
                ax2.annotate(
                    f"{p['emoji']} {p['pattern']}",
                    xy=(len(df)-1, float(df['High'].iloc[-1])),
                    xytext=(len(df)-15,
                            float(df['High'].max()) * 1.02),
                    color='#4ecdc4', fontsize=7,
                    arrowprops=dict(arrowstyle='->', color='#4ecdc4')
                )

        ax2.set_title(
            f"Candlestick — {best['ticker'].replace('.NS','')} "
            f"(Score: {best['score']:+.2f})",
            color='white', fontsize=11
        )
        ax2.set_facecolor('#1a1a2e')
        ax2.tick_params(colors='white', labelsize=7)
        ax2.set_ylabel('Price (₹)', color='white')
        ax2.set_xlim(-2, len(df) + 2)

    # Panel 3: Pattern frequency pie chart
    ax3 = fig.add_subplot(gs[1, 1])
    pattern_counts = {}
    for r in all_results:
        for p in r['patterns']:
            name = p['pattern']
            pattern_counts[name] = pattern_counts.get(name, 0) + 1

    if pattern_counts:
        pie_colors = [
            '#4ecdc4', '#ff6b6b', '#ffd700', '#c39bd3',
            '#ff9f43', '#54a0ff', '#5f27cd', '#00d2d3'
        ]
        wedges, texts, autotexts = ax3.pie(
            list(pattern_counts.values()),
            labels=list(pattern_counts.keys()),
            autopct='%1.0f%%',
            colors=pie_colors[:len(pattern_counts)],
            textprops={'color': 'white', 'fontsize': 8}
        )
        for at in autotexts:
            at.set_fontsize(7)
    ax3.set_title('Pattern Frequency Distribution',
                  color='white', fontsize=11)
    ax3.set_facecolor('#0d0d1a')

    plt.suptitle('QuantAI — Chart Pattern Scanner',
                 color='white', fontsize=14, y=1.01)
    plt.savefig('models/pattern_scan.png', dpi=150,
                bbox_inches='tight', facecolor='#0d0d1a')
    plt.show()
    print("  Chart saved → models/pattern_scan.png")