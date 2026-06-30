"""
correlation_matrix.py

Correlation matrix & diversification view across all 50 Nifty stocks.

Four panels:

  Panel 1 (main): Full 50×50 return-correlation heatmap, stocks grouped
    by sector. Red = highly correlated (move together), blue = negatively
    correlated (hedge), white = uncorrelated (diversifies well).

  Panel 2: Sector-level correlation block — the 16×16 average pairwise
    correlation between every sector pair. Shows which sectors move
    together as a group and which genuinely diversify each other.

  Panel 3: Diversification score per stock — average absolute correlation
    with all other 49 stocks. Low score = this stock moves independently
    (valuable diversifier). High score = it moves with the crowd
    (adds concentration risk).

  Panel 4: Rolling 60-day correlation of each sector vs Nifty 50 index
    proxy (equal-weight average of all 50 stocks). Shows how sector
    betas shift over time — useful for spotting regime changes.

Data:
  Same DB-first / yfinance-fallback pattern as drawdown_heatmap.py.
  Uses daily log-returns (not price levels) so the correlation captures
  co-movement of risk, not co-movement of price scale.

Run:
    python correlation_matrix.py
    python correlation_matrix.py --period 1y
    python correlation_matrix.py --save
"""
import argparse
import os
import sqlite3
import warnings
warnings.filterwarnings('ignore')

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as mcolors # type: ignore
import matplotlib.ticker as mtick # type: ignore
from matplotlib.patches import Patch # type: ignore

from src.data_collector import STOCK_UNIVERSE

SECTOR_COLORS = {
    'Banking'           : '#3b82f6',
    'Technology'        : '#8b5cf6',
    'Energy'            : '#f97316',
    'Power'             : '#eab308',
    'FMCG'             : '#22c55e',
    'Automobile'        : '#ec4899',
    'Pharma'            : '#14b8a6',
    'Metals'            : '#a16207',
    'Cement'            : '#94a3b8',
    'Consumer'          : '#f43f5e',
    'Healthcare'        : '#06b6d4',
    'Insurance'         : '#6366f1',
    'Financial Services': '#d946ef',
    'Infrastructure'    : '#0ea5e9',
    'Telecom'           : '#84cc16',
    'Conglomerate'      : '#fb923c',
}

PERIOD_DAYS = {'1y': 252, '2y': 504, '3y': 756, '5y': 1260}


# ── Data loading (same pattern as drawdown_heatmap.py) ───

def _load_from_db(ticker):
    db_path = os.path.join('data', 'quantai.db')
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT date, close FROM prices WHERE ticker=? ORDER BY date ASC",
            conn, params=(ticker,)
        )
        conn.close()
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.columns = ['Close']
        return df
    except Exception:
        return None


def _load_from_yf(ticker, period_days):
    try:
        import yfinance as yf # type: ignore
        years = max(1, round(period_days / 252))
        df = yf.download(ticker, period=f'{years}y', progress=False,
                         auto_adjust=True)
        if df.empty:
            return None
        if hasattr(df.columns, 'levels'):
            df.columns = [c[0] for c in df.columns]
        return df[['Close']]
    except Exception:
        return None


def load_close(ticker, period_days=504):
    df = _load_from_db(ticker)
    if df is None:
        df = _load_from_yf(ticker, period_days)
    if df is None or df.empty:
        return None
    cutoff = df.index[-1] - pd.Timedelta(days=period_days)
    df = df[df.index >= cutoff].copy()
    df = df[~df.index.duplicated()].sort_index()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(inplace=True)
    return df if len(df) >= 30 else None


# ── Build returns matrix ──────────────────────────────────

def build_returns_matrix(period_days=504):
    """
    Returns a DataFrame of daily log-returns: rows=dates, cols=tickers.
    Only tickers with enough data are included.
    Also returns sector_map and short_map for labelling.
    """
    all_tickers = list(STOCK_UNIVERSE.keys())
    print(f"\n⚙️  Loading price data for {len(all_tickers)} stocks...\n")

    closes = {}
    for i, ticker in enumerate(all_tickers):
        df = load_close(ticker, period_days)
        if df is not None:
            closes[ticker] = df['Close']
        pct = (i + 1) / len(all_tickers) * 100
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"  [{bar}] {pct:5.1f}%  {ticker:<20}", end='\r')

    print(f"\n\n✅  Loaded {len(closes)}/{len(all_tickers)} stocks\n")

    price_df = pd.DataFrame(closes).dropna(how='all')
    price_df = price_df.ffill().dropna()

    # Daily log-returns
    returns = np.log(price_df / price_df.shift(1)).dropna()

    sector_map = {t: STOCK_UNIVERSE[t][1] for t in returns.columns}
    short_map  = {t: t.replace('.NS', '') for t in returns.columns}

    return returns, sector_map, short_map


# ── Main plot ─────────────────────────────────────────────

def plot_correlation(period='2y', save=False):
    period_days = PERIOD_DAYS.get(period, 504)
    returns, sector_map, short_map = build_returns_matrix(period_days)

    if returns.empty or len(returns.columns) < 5:
        print("❌  Not enough data. Run `python pipeline.py` first or check "
              "internet connection.")
        return

    # Sort tickers by sector for the heatmap grouping
    tickers = sorted(returns.columns,
                     key=lambda t: (sector_map.get(t, 'ZZ'), t))
    returns = returns[tickers]

    corr = returns.corr()
    n = len(tickers)
    row_labels = [short_map[t] for t in tickers]

    # ── Colour maps ───────────────────────────────────────
    # Correlation: -1 (blue) → 0 (white) → +1 (red)
    corr_cmap = mcolors.LinearSegmentedColormap.from_list(
        'corr',
        [(0.0, '#1d4ed8'), (0.3, '#93c5fd'), (0.5, '#ffffff'),
         (0.7, '#fca5a5'), (1.0, '#991b1b')]
    )

    # ── Figure ────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 20), facecolor='#0d0d1a')
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=[2.4, 1],
        height_ratios=[1.6, 1],
        hspace=0.35, wspace=0.30,
        left=0.07, right=0.97, top=0.93, bottom=0.05
    )

    # ─────────────────────────────────────────────────────
    # PANEL 1 — Full 50×50 correlation heatmap
    # ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')

    matrix = corr.values
    im = ax1.imshow(matrix, cmap=corr_cmap, vmin=-1, vmax=1,
                     aspect='auto', interpolation='nearest')

    # Sector boundary lines + colour strip
    prev_sector = None
    boundaries  = []
    label_pos   = []
    for i, t in enumerate(tickers):
        s = sector_map.get(t, '')
        if s != prev_sector:
            if i > 0:
                boundaries.append(i - 0.5)
            label_pos.append((i, s))
            prev_sector = s

    for b in boundaries:
        ax1.axhline(b, color='#0d0d1a', linewidth=1.5, zorder=3)
        ax1.axvline(b, color='#0d0d1a', linewidth=1.5, zorder=3)

    for (idx, sector_name) in label_pos:
        color = SECTOR_COLORS.get(sector_name, '#888888')
        # Left strip
        ax1.add_patch(plt.Rectangle(
            (-0.5, idx - 0.5), 0.4, 1,
            color=color, clip_on=False, zorder=4, alpha=0.85
        ))
        # Top strip
        ax1.add_patch(plt.Rectangle(
            (idx - 0.5, -0.5), 1, 0.4,
            color=color, clip_on=False, zorder=4, alpha=0.85
        ))

    # Cell value text for high correlations only
    if n <= 55:
        for ri in range(n):
            for ci in range(n):
                val = matrix[ri, ci]
                if ri != ci and abs(val) >= 0.65:
                    txt_c = 'white' if abs(val) >= 0.80 else '#1a1a1a'
                    ax1.text(ci, ri, f'{val:.2f}', ha='center', va='center',
                              fontsize=4.5, color=txt_c, fontweight='bold')

    ax1.set_xticks(range(n))
    ax1.set_xticklabels(row_labels, rotation=90, fontsize=6.5,
                         color='#cbd5e1')
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(row_labels, fontsize=6.5, color='#e2e8f0')
    ax1.tick_params(length=0)
    ax1.set_title(
        f'Return Correlation Matrix — Nifty 50  ({period} daily log-returns)',
        color='white', fontsize=13, fontweight='bold', pad=14
    )

    cbar = fig.colorbar(im, ax=ax1, orientation='horizontal',
                         pad=0.02, fraction=0.022, shrink=0.55)
    cbar.set_label('Pearson correlation  (−1 = hedge  ·  0 = uncorrelated  ·  +1 = move together)',
                   color='#94a3b8', fontsize=8)
    cbar.ax.tick_params(colors='#94a3b8', labelsize=7)

    # ─────────────────────────────────────────────────────
    # PANEL 2 — Sector-level average correlation block
    # ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1a1a2e')

    sectors_ordered = list(dict.fromkeys(
        sector_map[t] for t in tickers
    ))
    sec_corr = pd.DataFrame(index=sectors_ordered, columns=sectors_ordered,
                              dtype=float)
    for s1 in sectors_ordered:
        t1 = [t for t in tickers if sector_map[t] == s1]
        for s2 in sectors_ordered:
            t2 = [t for t in tickers if sector_map[t] == s2]
            # Average pairwise correlation between all stocks in s1 vs s2
            pairs = corr.loc[t1, t2]
            if s1 == s2:
                # Within-sector: exclude the diagonal (self-correlation = 1)
                vals = pairs.values
                mask = ~np.eye(len(vals), dtype=bool) if len(vals) > 1 else np.ones((1,1), dtype=bool)
                sec_corr.loc[s1, s2] = vals[mask].mean() if mask.any() else 1.0
            else:
                sec_corr.loc[s1, s2] = pairs.values.mean()

    sec_matrix = sec_corr.values.astype(float)
    im2 = ax2.imshow(sec_matrix, cmap=corr_cmap, vmin=-0.2, vmax=1,
                      aspect='auto', interpolation='nearest')

    sec_n = len(sectors_ordered)
    for ri in range(sec_n):
        for ci in range(sec_n):
            val = sec_matrix[ri, ci]
            txt_c = 'white' if val > 0.7 or val < 0.1 else '#1a1a1a'
            ax2.text(ci, ri, f'{val:.2f}', ha='center', va='center',
                      fontsize=7, color=txt_c, fontweight='bold')

    sec_short = [s[:6] for s in sectors_ordered]
    ax2.set_xticks(range(sec_n))
    ax2.set_xticklabels(sec_short, rotation=45, ha='right',
                         fontsize=7.5, color='#cbd5e1')
    ax2.set_yticks(range(sec_n))
    ax2.set_yticklabels(sectors_ordered, fontsize=7.5, color='#e2e8f0')
    ax2.tick_params(length=0)
    ax2.set_title('Sector-Level Correlation\n(avg pairwise within / between sectors)',
                   color='white', fontsize=10, fontweight='bold', pad=8)

    fig.colorbar(im2, ax=ax2, orientation='horizontal',
                  pad=0.02, fraction=0.03, shrink=0.7).ax.tick_params(
        colors='#94a3b8', labelsize=7)

    # ─────────────────────────────────────────────────────
    # PANEL 3 — Diversification score per stock
    # ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#1a1a2e')

    # Average absolute correlation with all other stocks
    # (lower = moves more independently = better diversifier)
    abs_corr = corr.abs()
    abs_corr_arr = abs_corr.values.copy()          # copy() makes it writable
    np.fill_diagonal(abs_corr_arr, np.nan)
    div_score = pd.Series(
       np.nanmean(abs_corr_arr, axis=1),
       index=abs_corr.index
)
    bar_colors = [SECTOR_COLORS.get(sector_map.get(t, ''), '#64748b')
                  for t in div_score.index]
    y_pos = range(len(div_score))
    bars  = ax3.barh(y_pos, div_score.values, color=bar_colors,
                      alpha=0.82, height=0.75)

    for i, (t, val) in enumerate(div_score.items()):
        ax3.text(val + 0.005, i, f'{val:.2f}',
                  va='center', fontsize=6, color='white')

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(
        [short_map.get(t, t) for t in div_score.index],
        fontsize=6.5, color='#e2e8f0'
    )
    ax3.axvline(div_score.mean(), color='#fbbf24', linewidth=1.0,
                 linestyle='--', alpha=0.8,
                 label=f'Average ({div_score.mean():.2f})')
    ax3.axvline(0.5, color='#ef4444', linewidth=0.8, linestyle=':',
                 alpha=0.7, label='0.50 (high concentration risk)')
    ax3.set_xlabel('Avg |correlation| with all other stocks',
                   color='#94a3b8', fontsize=8)
    ax3.set_title(
        'Diversification Score  (lower = moves independently = adds more diversification)',
        color='white', fontsize=10, fontweight='bold', pad=8
    )
    ax3.legend(fontsize=7, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640', loc='lower right')
    ax3.tick_params(colors='#94a3b8', labelsize=7, length=0)
    ax3.invert_yaxis()   # highest corr (worst diversifier) at top

    # Sector legend
    legend_patches = [
        Patch(facecolor=SECTOR_COLORS.get(s, '#64748b'), label=s, alpha=0.85)
        for s in sorted(set(sector_map.values()))
        if s in SECTOR_COLORS
    ]
    ax3.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color='#fbbf24', linestyle='--', label=f'Avg ({div_score.mean():.2f})'),
        plt.Line2D([0], [0], color='#ef4444', linestyle=':', label='0.50 threshold'),
    ], fontsize=6, facecolor='#12121f', labelcolor='white',
    edgecolor='#262640', loc='lower right', ncol=2)

    # ─────────────────────────────────────────────────────
    # PANEL 4 — Rolling 60-day sector correlation vs index
    # ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#1a1a2e')

    # Index proxy = equal-weight average return of all 50 stocks
    index_ret = returns.mean(axis=1)

    # For each sector, compute rolling 60-day correlation of
    # sector-average-return vs the index proxy
    roll_window = 60
    for sector_name in sectors_ordered:
        sect_tickers = [t for t in tickers if sector_map[t] == sector_name]
        if not sect_tickers:
            continue
        sect_ret = returns[sect_tickers].mean(axis=1)
        rolling_corr = sect_ret.rolling(roll_window).corr(index_ret).dropna()
        if rolling_corr.empty:
            continue
        color = SECTOR_COLORS.get(sector_name, '#64748b')
        ax4.plot(rolling_corr.index, rolling_corr.values,
                  color=color, linewidth=1.1, alpha=0.8,
                  label=sector_name)

    ax4.axhline(1.0, color='white', linewidth=0.5, linestyle=':', alpha=0.3)
    ax4.axhline(0.5, color='#fbbf24', linewidth=0.7, linestyle='--',
                 alpha=0.5, label='0.50 reference')
    ax4.set_ylabel('Rolling 60-day correlation vs index', color='#94a3b8',
                   fontsize=8)
    ax4.set_title('Sector Beta Drift vs Nifty Index Proxy\n(rolling 60-day correlation)',
                   color='white', fontsize=10, fontweight='bold', pad=8)
    ax4.tick_params(colors='#94a3b8', labelsize=7, length=0)
    ax4.legend(fontsize=5.5, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640', loc='lower left', ncol=2)

    # ── Super-title ───────────────────────────────────────
    fig.suptitle(
        'QuantAI — Nifty 50 Correlation Matrix & Diversification View',
        color='white', fontsize=15, fontweight='bold', y=0.97
    )

    # ── Print key findings ────────────────────────────────
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    flat  = upper.stack()
    high_pairs  = flat[flat >= 0.80].sort_values(ascending=False)
    low_pairs   = flat[flat <= 0.10].sort_values()
    best_div    = div_score.index[-1]
    worst_div   = div_score.index[0]

    print(f"{'='*56}")
    print(f"  CORRELATION SNAPSHOT — Nifty 50 ({period})")
    print(f"{'='*56}")
    print(f"  {'Average pairwise correlation':<36} {flat.mean():+.3f}")
    print(f"  {'Median pairwise correlation':<36} {flat.median():+.3f}")
    print(f"  {'Pairs with corr ≥ 0.80 (move together)':<36} {len(high_pairs)}")
    print(f"  {'Pairs with corr ≤ 0.10 (independent)':<36} {len(low_pairs)}")
    print(f"{'─'*56}")
    print(f"  Best diversifier  (lowest avg |corr|)  "
          f"{short_map[best_div]:<12} {div_score[best_div]:.3f}")
    print(f"  Worst diversifier (highest avg |corr|) "
          f"{short_map[worst_div]:<12} {div_score[worst_div]:.3f}")
    if not high_pairs.empty:
        (t1, t2), v = high_pairs.index[0], high_pairs.iloc[0]
        print(f"  Highest corr pair: "
              f"{short_map[t1]} ↔ {short_map[t2]}  ({v:.3f})")
    print(f"{'='*56}\n")

    # ── Save / show ───────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    out_path = f'models/correlation_matrix_{period}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"📊 Chart saved → {out_path}")

    if not save:
        plt.show()

    return corr, div_score


# ── Entry point ───────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Correlation matrix & diversification view — Nifty 50'
    )
    parser.add_argument('--period', default='2y',
                        choices=['1y', '2y', '3y', '5y'],
                        help='Lookback window (default: 2y)')
    parser.add_argument('--save', action='store_true',
                        help='Save PNG without showing the window')
    args = parser.parse_args()

    plot_correlation(period=args.period, save=args.save)
