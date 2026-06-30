"""
drawdown_heatmap.py

Drawdown heatmap across all 50 Nifty stocks — four panels in one figure:

  Panel 1 (main): Monthly drawdown heatmap — stocks × months as a grid.
    Each cell is coloured by the worst intra-month drawdown that stock
    experienced that month (peak-to-trough within the month relative to
    the running all-time peak). Red = deep drawdown, white = flat/near
    peak, green = recovering/at new ATH.

  Panel 2: Current drawdown bar chart — where each stock sits RIGHT NOW
    vs its all-time high in the data window. Sorted worst → best.

  Panel 3: Sector drawdown summary — average current drawdown by sector,
    so you can see which entire sectors are under water together.

  Panel 4: Max-drawdown-ever ranking — which stocks have had the deepest
    single peak-to-trough drop in the full history window. Useful for
    understanding each stock's tail-risk profile.

Data source:
  Primary: SQLite DB (data/quantai.db) via src/database.py — used when
           the pipeline has already been run and prices are stored locally.
  Fallback: yfinance live download — used automatically if the DB is
            empty or a ticker is missing from it (e.g. fresh install,
            running before `python pipeline.py`).

Run:
    python drawdown_heatmap.py
    python drawdown_heatmap.py --period 1y    # 1y / 2y / 3y / 5y
    python drawdown_heatmap.py --save         # save PNG without showing
"""
import argparse
import os
import sqlite3
import warnings
warnings.filterwarnings('ignore')

import numpy as np # type: ignore # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as mcolors # type: ignore # type: ignore
import matplotlib.ticker as mtick # type: ignore
from matplotlib.patches import Patch # type: ignore

# ── Stock universe (mirrors src/data_collector.py) ───────
from src.data_collector import STOCK_UNIVERSE

# ── Sector colour palette (one colour per sector) ────────
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


# ── Data loading ──────────────────────────────────────────

def _load_from_db(ticker):
    """Try to load OHLCV from local SQLite. Returns None if missing."""
    db_path = os.path.join('data', 'quantai.db')
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date ASC",
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
    """Fallback: download from yfinance."""
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
    """Load closing prices, DB first then yfinance fallback."""
    df = _load_from_db(ticker)
    if df is None:
        df = _load_from_yf(ticker, period_days)
    if df is None or df.empty:
        return None
    # Trim to requested period
    cutoff = df.index[-1] - pd.Timedelta(days=period_days)
    df = df[df.index >= cutoff].copy()
    df = df[~df.index.duplicated()].sort_index()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(inplace=True)
    return df if len(df) >= 20 else None


# ── Drawdown calculation ──────────────────────────────────

def rolling_drawdown(close: pd.Series) -> pd.Series:
    """
    Daily drawdown series: how far (%) each day's close is from the
    running all-time peak up to that day. Always ≤ 0.
    """
    peak = close.cummax()
    return (close - peak) / peak * 100


def monthly_worst_drawdown(dd: pd.Series) -> pd.Series:
    """
    For each calendar month, return the worst (most negative) single-day
    drawdown value that occurred in that month.
    """
    return dd.resample('ME').min()


# ── Main ─────────────────────────────────────────────────

def build_heatmap_data(period_days=504):
    """
    Returns:
      monthly_dd  : DataFrame (tickers as rows, month-end dates as cols)
      current_dd  : Series (ticker → current drawdown from ATH)
      max_dd      : Series (ticker → worst ever drawdown in window)
      sector_map  : dict   (ticker → sector)
      name_map    : dict   (ticker → short display name)
    """
    monthly_dd = {}
    current_dd = {}
    max_dd_map = {}
    tickers_ok = []

    all_tickers = list(STOCK_UNIVERSE.keys())
    print(f"\n⚙️  Loading price data for {len(all_tickers)} stocks "
          f"({'DB + yfinance fallback'})...\n")

    for i, ticker in enumerate(all_tickers):
        df = load_close(ticker, period_days)
        if df is None:
            print(f"  ⚠️  {ticker:<20} no data — skipped")
            continue

        dd = rolling_drawdown(df['Close'])
        monthly_worst = monthly_worst_drawdown(dd)

        monthly_dd[ticker] = monthly_worst
        current_dd[ticker] = float(dd.iloc[-1])
        max_dd_map[ticker] = float(dd.min())
        tickers_ok.append(ticker)

        pct = (i + 1) / len(all_tickers) * 100
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"  [{bar}] {pct:5.1f}%  {ticker:<20} "
              f"current: {float(dd.iloc[-1]):+.1f}%   "
              f"max: {float(dd.min()):+.1f}%",
              end='\r')

    print(f"\n\n✅  Loaded {len(tickers_ok)}/{len(all_tickers)} stocks\n")

    # Align all monthly series onto a common date index
    dd_df = pd.DataFrame(monthly_dd).T  # tickers × months
    dd_df = dd_df.sort_index(axis=1)   # chronological columns

    sector_map = {t: STOCK_UNIVERSE[t][1] for t in tickers_ok}
    name_map   = {t: STOCK_UNIVERSE[t][0] for t in tickers_ok}

    # Short ticker labels (strip .NS)
    short_map  = {t: t.replace('.NS', '') for t in tickers_ok}

    current_s  = pd.Series(current_dd)
    max_s      = pd.Series(max_dd_map)

    return dd_df, current_s, max_s, sector_map, name_map, short_map


def plot_heatmap(period='2y', save=False):
    period_days = PERIOD_DAYS.get(period, 504)
    dd_df, current_s, max_s, sector_map, name_map, short_map = \
        build_heatmap_data(period_days)

    if dd_df.empty:
        print("❌  No data loaded — run `python pipeline.py` first or check "
              "your internet connection for the yfinance fallback.")
        return

    # ── Sort stocks: sector first, then by current drawdown within sector ──
    tickers = list(dd_df.index)
    tickers.sort(key=lambda t: (sector_map.get(t, 'ZZ'),
                                 current_s.get(t, 0)))
    dd_df    = dd_df.loc[tickers]
    current_s = current_s.reindex(tickers)
    max_s     = max_s.reindex(tickers)

    # ── Column labels: "Jan 24", "Feb 24" … ──────────────
    col_labels = [c.strftime('%b %y') for c in dd_df.columns]
    n_rows, n_cols = dd_df.shape
    row_labels = [short_map.get(t, t) for t in tickers]

    # ── Colour map: deep red → white → green ─────────────
    # Anchor: -40% = darkest red, 0% = white, +2% = light green
    # (stocks rarely go above their ATH during the month so green is rare
    #  and kept subtle — it just means "closed the month at ATH")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'dd_heat',
        [
            (0.00, '#7f0000'),   # -40%+ : very deep red
            (0.15, '#dc2626'),   # -34%
            (0.35, '#f97316'),   # -26%
            (0.55, '#fbbf24'),   # -18%
            (0.72, '#fef08a'),   # -11%
            (0.85, '#f0fdf4'),   # -6%
            (0.94, '#ffffff'),   # -2% : at/near peak (white)
            (1.00, '#86efac'),   # 0%  : at ATH (light green)
        ]
    )
    vmin, vmax = -40, 0

    # ── Figure layout ─────────────────────────────────────
    fig = plt.figure(figsize=(max(22, n_cols * 0.72 + 6), 22),
                     facecolor='#0d0d1a')

    # Grid: [heatmap | current_dd bar] top row
    #       [sector summary | max_dd bar] bottom row
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3.2, 1],
        height_ratios=[1, 1],
        hspace=0.38, wspace=0.28,
        left=0.09, right=0.97, top=0.93, bottom=0.04
    )

    # ─────────────────────────────────────────────────────
    # PANEL 1 — Monthly drawdown heatmap
    # ─────────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_heat.set_facecolor('#1a1a2e')

    matrix = dd_df.values.astype(float)
    im = ax_heat.imshow(matrix, aspect='auto', cmap=cmap,
                         vmin=vmin, vmax=vmax, interpolation='nearest')

    # Sector dividers — horizontal lines between sector groups
    prev_sector = None
    sector_boundaries = []
    sector_label_pos  = []
    for i, t in enumerate(tickers):
        s = sector_map.get(t, '')
        if s != prev_sector:
            if i > 0:
                sector_boundaries.append(i - 0.5)
            sector_label_pos.append((i, s))
            prev_sector = s

    for b in sector_boundaries:
        ax_heat.axhline(b, color='#0d0d1a', linewidth=1.8, zorder=3)

    # Sector colour strip on the left
    for (row_i, sector_name) in sector_label_pos:
        color = SECTOR_COLORS.get(sector_name, '#888888')
        ax_heat.add_patch(plt.Rectangle(
            (-0.5, row_i - 0.5), 0.45, 1,
            color=color, clip_on=False, zorder=4, alpha=0.85
        ))

    # Cell text — show value only if deep enough to matter (< -5%)
    if n_rows <= 55 and n_cols <= 36:
        for ri in range(n_rows):
            for ci in range(n_cols):
                val = matrix[ri, ci]
                if np.isnan(val):
                    continue
                if val < -5:
                    txt_color = 'white' if val < -18 else '#1a1a1a'
                    ax_heat.text(ci, ri, f'{val:.0f}',
                                  ha='center', va='center',
                                  fontsize=5.2, color=txt_color, fontweight='bold')

    ax_heat.set_xticks(range(n_cols))
    ax_heat.set_xticklabels(col_labels, rotation=45, ha='right',
                             fontsize=7, color='#cbd5e1')
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(row_labels, fontsize=7.5, color='#e2e8f0')
    ax_heat.tick_params(axis='both', length=0)

    ax_heat.set_title(
        f'Monthly Peak-to-Trough Drawdown — Nifty 50  ({period} window)',
        color='white', fontsize=13, fontweight='bold', pad=12
    )

    # Colour-bar
    cbar = fig.colorbar(im, ax=ax_heat, orientation='horizontal',
                         pad=0.02, fraction=0.025, shrink=0.6)
    cbar.set_label('Drawdown from running ATH (%)', color='#94a3b8', fontsize=8)
    cbar.ax.tick_params(colors='#94a3b8', labelsize=7)

    # ─────────────────────────────────────────────────────
    # PANEL 2 — Current drawdown bar chart (sorted worst→best)
    # ─────────────────────────────────────────────────────
    ax_cur = fig.add_subplot(gs[0, 1])
    ax_cur.set_facecolor('#1a1a2e')

    cur_sorted = current_s.sort_values()          # worst first (most negative)
    bar_colors = [
        SECTOR_COLORS.get(sector_map.get(t, ''), '#64748b')
        for t in cur_sorted.index
    ]
    bars = ax_cur.barh(range(len(cur_sorted)), cur_sorted.values,
                        color=bar_colors, alpha=0.85, height=0.72)

    # Value labels on bars
    for i, (val, bar) in enumerate(zip(cur_sorted.values, bars)):
        x_pos = val - 0.5 if val < -3 else 0.3
        ax_cur.text(x_pos, i, f'{val:.1f}%',
                     va='center', ha='right' if val < -3 else 'left',
                     fontsize=6, color='white')

    ax_cur.set_yticks(range(len(cur_sorted)))
    ax_cur.set_yticklabels(
        [short_map.get(t, t) for t in cur_sorted.index],
        fontsize=6.5, color='#e2e8f0'
    )
    ax_cur.axvline(0, color='white', linewidth=0.8, alpha=0.4)
    ax_cur.axvline(-20, color='#f97316', linewidth=0.7, linestyle='--',
                    alpha=0.6, label='-20% warning')
    ax_cur.set_xlabel('Current Drawdown from ATH (%)', color='#94a3b8', fontsize=8)
    ax_cur.set_title('Current Drawdown', color='white', fontsize=10,
                      fontweight='bold', pad=8)
    ax_cur.tick_params(colors='#94a3b8', labelsize=7, length=0)
    ax_cur.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d%%'))
    ax_cur.legend(fontsize=7, facecolor='#12121f', labelcolor='white',
                   edgecolor='#262640', loc='lower right')
    ax_cur.invert_yaxis()   # worst at top

    # ─────────────────────────────────────────────────────
    # PANEL 3 — Sector average current drawdown
    # ─────────────────────────────────────────────────────
    ax_sec = fig.add_subplot(gs[1, 0])
    ax_sec.set_facecolor('#1a1a2e')

    sector_dd = (
        current_s.rename('dd')
        .to_frame()
        .assign(sector=lambda df: [sector_map.get(t, 'Unknown') for t in df.index])
        .groupby('sector')['dd']
        .agg(['mean', 'min', 'max', 'count'])
        .sort_values('mean')
    )

    sec_colors = [SECTOR_COLORS.get(s, '#64748b') for s in sector_dd.index]
    y_pos = range(len(sector_dd))

    # Error bars = min/max spread within sector
    xerr_lo = (sector_dd['mean'] - sector_dd['min']).abs()
    xerr_hi = (sector_dd['max'] - sector_dd['mean']).abs()

    ax_sec.barh(y_pos, sector_dd['mean'], color=sec_colors,
                 alpha=0.82, height=0.6, zorder=3)
    ax_sec.errorbar(sector_dd['mean'], y_pos,
                     xerr=[xerr_lo.values, xerr_hi.values],
                     fmt='none', color='white', alpha=0.45,
                     capsize=3, linewidth=0.9, zorder=4)

    for i, (sec, row) in enumerate(sector_dd.iterrows()):
        ax_sec.text(row['mean'] - 0.4, i,
                     f"{row['mean']:.1f}%  (n={int(row['count'])})",
                     va='center', ha='right', fontsize=7, color='white')

    ax_sec.set_yticks(y_pos)
    ax_sec.set_yticklabels(sector_dd.index, fontsize=8, color='#e2e8f0')
    ax_sec.axvline(0, color='white', linewidth=0.8, alpha=0.4)
    ax_sec.axvline(-20, color='#f97316', linewidth=0.7, linestyle='--', alpha=0.6)
    ax_sec.set_xlabel('Average Current Drawdown (%)', color='#94a3b8', fontsize=8)
    ax_sec.set_title(
        'Sector Drawdown Summary  (bar = sector avg, whiskers = min/max)',
        color='white', fontsize=10, fontweight='bold', pad=8
    )
    ax_sec.tick_params(colors='#94a3b8', labelsize=7, length=0)
    ax_sec.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d%%'))
    ax_sec.invert_yaxis()

    # Sector legend patches
    legend_patches = [
        Patch(facecolor=SECTOR_COLORS.get(s, '#64748b'), label=s, alpha=0.85)
        for s in sorted(SECTOR_COLORS)
        if s in sector_dd.index
    ]
    ax_sec.legend(handles=legend_patches, fontsize=6.5,
                   facecolor='#12121f', labelcolor='white',
                   edgecolor='#262640', loc='lower right',
                   ncol=2, title='Sectors', title_fontsize=7)

    # ─────────────────────────────────────────────────────
    # PANEL 4 — Max drawdown ever (in window) ranking
    # ─────────────────────────────────────────────────────
    ax_max = fig.add_subplot(gs[1, 1])
    ax_max.set_facecolor('#1a1a2e')

    max_sorted = max_s.sort_values()   # worst first
    bar_colors_max = [
        SECTOR_COLORS.get(sector_map.get(t, ''), '#64748b')
        for t in max_sorted.index
    ]
    ax_max.barh(range(len(max_sorted)), max_sorted.values,
                 color=bar_colors_max, alpha=0.82, height=0.72)

    for i, val in enumerate(max_sorted.values):
        ax_max.text(val - 0.5, i, f'{val:.1f}%',
                     va='center', ha='right', fontsize=6, color='white')

    ax_max.set_yticks(range(len(max_sorted)))
    ax_max.set_yticklabels(
        [short_map.get(t, t) for t in max_sorted.index],
        fontsize=6.5, color='#e2e8f0'
    )
    ax_max.axvline(0,   color='white',   linewidth=0.8, alpha=0.4)
    ax_max.axvline(-30, color='#ef4444', linewidth=0.7, linestyle='--',
                    alpha=0.6, label='-30% danger')
    ax_max.set_xlabel('Max Drawdown in Window (%)', color='#94a3b8', fontsize=8)
    ax_max.set_title(f'Worst Ever Drawdown\n({period} window)',
                      color='white', fontsize=10, fontweight='bold', pad=8)
    ax_max.tick_params(colors='#94a3b8', labelsize=7, length=0)
    ax_max.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d%%'))
    ax_max.legend(fontsize=7, facecolor='#12121f', labelcolor='white',
                   edgecolor='#262640', loc='lower right')
    ax_max.invert_yaxis()

    # ── Super-title ───────────────────────────────────────
    fig.suptitle(
        'QuantAI — Nifty 50 Drawdown Heatmap',
        color='white', fontsize=16, fontweight='bold', y=0.97
    )

    # ── Print summary stats ───────────────────────────────
    stocks_at_ath   = (current_s >= -2).sum()
    stocks_mild     = ((current_s < -2)  & (current_s >= -10)).sum()
    stocks_moderate = ((current_s < -10) & (current_s >= -20)).sum()
    stocks_deep     = (current_s < -20).sum()

    print(f"{'='*56}")
    print(f"  DRAWDOWN SNAPSHOT — Nifty 50 ({period} window)")
    print(f"{'='*56}")
    print(f"  {'At / near ATH   (< 2% off peak)':<36} {stocks_at_ath:>3} stocks")
    print(f"  {'Mild pullback   (2–10%)':<36} {stocks_mild:>3} stocks")
    print(f"  {'Moderate drawdown (10–20%)':<36} {stocks_moderate:>3} stocks")
    print(f"  {'Deep drawdown   (> 20%)':<36} {stocks_deep:>3} stocks")
    print(f"{'─'*56}")
    print(f"  {'Worst current drawdown':<36} "
          f"{current_s.idxmin().replace('.NS',''):<12} {current_s.min():+.1f}%")
    print(f"  {'Best current drawdown':<36} "
          f"{current_s.idxmax().replace('.NS',''):<12} {current_s.max():+.1f}%")
    print(f"  {'Average current drawdown':<36} {current_s.mean():+.1f}%")
    print(f"  {'Worst ever (in window)':<36} "
          f"{max_s.idxmin().replace('.NS',''):<12} {max_s.min():+.1f}%")
    print(f"{'='*56}\n")

    # ── Save / show ───────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    out_path = f'models/drawdown_heatmap_{period}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"📊 Chart saved → {out_path}")

    if not save:
        plt.show()

    return dd_df, current_s, max_s


# ── Entry point ───────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Drawdown heatmap across all 50 Nifty stocks'
    )
    parser.add_argument('--period', default='2y',
                        choices=['1y', '2y', '3y', '5y'],
                        help='Lookback window (default: 2y)')
    parser.add_argument('--save', action='store_true',
                        help='Save PNG without showing the window')
    args = parser.parse_args()

    plot_heatmap(period=args.period, save=args.save)