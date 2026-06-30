"""
tail_risk_monitor.py

Runs the Tail Risk Monitor / Black Swan Detector and produces a
four-panel dashboard showing the current market stress level.

  Panel 1: Tail Risk Index gauge — the single number (0–1) with
    colour band (green → yellow → orange → red) and a 30-day
    rolling TRI history line so you can see how stress has evolved.

  Panel 2: Component score bar chart — shows exactly which of the
    six detectors (vol, kurtosis, skew, correlation, liquidity, VaR)
    is driving the TRI up. Coloured by contribution weight.

  Panel 3: Rolling 20-day realized volatility of the equal-weight
    Nifty index, with 1σ / 2σ bands from the 1-year baseline.
    The area above 2σ is where historical crashes lived.

  Panel 4: Rolling 20-day average pairwise correlation — one of the
    most reliable leading indicators: correlation spikes just before
    everything falls together.

Run:
    python tail_risk_monitor.py
    python tail_risk_monitor.py --period 1y
    python tail_risk_monitor.py --save
    python tail_risk_monitor.py --history 60    # show 60-day TRI history
"""
import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np# type: ignore
import pandas as pd# type: ignore
import matplotlib.pyplot as plt# type: ignore
import matplotlib.patches as mpatches# type: ignore
import matplotlib.colors as mcolors# type: ignore
from matplotlib.gridspec import GridSpec# type: ignore

from src.tail_risk import TailRiskMonitor, LEVEL_EMOJI, TRI_THRESHOLDS

PERIOD_DAYS = {'1y': 252, '2y': 504, '3y': 756}


def _load_prices_volumes(period_days=252):
    """Load all 50 stocks — DB first, yfinance fallback."""
    from src.data_collector import STOCK_UNIVERSE
    import sqlite3

    db_path = os.path.join('data', 'quantai.db')
    price_dict  = {}
    volume_dict = {}

    print(f"\n⚙️  Loading {len(STOCK_UNIVERSE)} stocks for tail risk scan...\n")

    for i, ticker in enumerate(STOCK_UNIVERSE):
        df = None

        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                df   = pd.read_sql_query(
                    "SELECT date, close, volume FROM prices WHERE ticker=? ORDER BY date ASC",
                    conn, params=(ticker,)
                )
                conn.close()
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df.columns = ['Close', 'Volume']
                    df[['Close', 'Volume']] = df[['Close', 'Volume']].apply(
                        pd.to_numeric, errors='coerce'
                    )
                    df.dropna(subset=['Close'], inplace=True)
                else:
                    df = None
            except Exception:
                df = None

        if df is None:
            try:
                import yfinance as yf # type: ignore
                years = max(1, round(period_days / 252))
                raw = yf.download(ticker, period=f'{years}y',
                                  progress=False, auto_adjust=True)
                if not raw.empty:
                    if hasattr(raw.columns, 'levels'):
                        raw.columns = [c[0] for c in raw.columns]
                    df = raw[['Close', 'Volume']].copy()
            except Exception:
                pass

        if df is None or df.empty:
            continue

        cutoff = df.index[-1] - pd.Timedelta(days=period_days)
        df     = df[df.index >= cutoff]
        if len(df) >= 30:
            price_dict[ticker]  = df['Close']
            volume_dict[ticker] = df['Volume']

        pct = (i + 1) / len(STOCK_UNIVERSE) * 100
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"  [{bar}] {pct:5.1f}%  {ticker:<22}", end='\r')

    print(f"\n\n✅  Loaded {len(price_dict)}/{len(STOCK_UNIVERSE)} stocks\n")
    return price_dict, volume_dict


def _build_rolling_tri(price_dict, volume_dict, history_days=60):
    """
    Computes TRI for each of the last `history_days` days so we can
    plot how tail risk has evolved over time.
    This is expensive (one scan per day) so we sub-sample to every 3 days.
    """
    monitor = TailRiskMonitor()

    price_df  = pd.DataFrame(price_dict).dropna(how='all').ffill().dropna()
    vol_df    = pd.DataFrame(volume_dict).reindex(price_df.index).ffill() \
                if volume_dict else None

    dates     = price_df.index[-history_days:]
    tri_series = {}

    for i, date in enumerate(dates[::3]):   # every 3 days
        snap_prices  = {col: price_df[col][:date] for col in price_df.columns}
        snap_volumes = ({col: vol_df[col][:date] for col in vol_df.columns}
                        if vol_df is not None else None)
        try:
            rep = monitor.scan(snap_prices, snap_volumes)
            tri_series[date] = rep.tri
        except Exception:
            pass

    return pd.Series(tri_series).sort_index()


def plot_tail_risk(period='1y', history_days=60, save=False):
    period_days = PERIOD_DAYS.get(period, 252)

    price_dict, volume_dict = _load_prices_volumes(period_days)
    if not price_dict:
        print("❌  No data. Run `python pipeline.py` first.")
        return

    monitor = TailRiskMonitor()
    report  = monitor.scan(price_dict, volume_dict)

    print(f"{'='*60}")
    print(f"  QuantAI Tail Risk Monitor  —  Black Swan Detector")
    print(f"{'='*60}")
    print(f"  {LEVEL_EMOJI[report.level]} Level: {report.level}")
    print(f"  Tail Risk Index (TRI): {report.tri:.3f} / 1.000")
    print(f"  Stocks scanned: {report.n_stocks}")
    print(f"{'─'*60}")
    print(f"  Component Scores:")
    print(f"    Volatility stress      {report.vol_score:>6.3f}  (weight 30%)")
    print(f"    Fat-tail kurtosis      {report.kurtosis_score:>6.3f}  (weight 20%)")
    print(f"    Negative skew          {report.skew_score:>6.3f}  (weight 15%)")
    print(f"    Correlation spike      {report.correlation_score:>6.3f}  (weight 20%)")
    print(f"    Liquidity stress       {report.liquidity_score:>6.3f}  (weight  7%)")
    print(f"    VaR breach count       {report.var_breach_score:>6.3f}  (weight  8%)")
    print(f"{'─'*60}")
    print(f"  Supporting Data:")
    print(f"    Realized vol (ann.)    {report.realized_vol_ann:>6.1f}%")
    print(f"    Avg excess kurtosis    {report.avg_kurtosis:>6.3f}")
    print(f"    Avg return skew        {report.avg_skew:>6.3f}")
    print(f"    Avg pairwise corr      {report.avg_pairwise_corr:>6.3f}")
    print(f"    VaR breaches today     {report.var_breach_count:>6}  / {report.n_stocks}")
    print(f"{'─'*60}")
    print(f"  Signals:")
    for r in report.reasons:
        print(f"    {r}")
    print(f"{'─'*60}")
    print(f"  Recommendation:")
    print(f"    {report.recommendation}")
    print(f"{'─'*60}")
    if report.halt_trading:
        print(f"  🚫 TRADING HALTED BY TAIL RISK MONITOR")
    print(f"{'='*60}\n")

    # ── Rolling TRI history ────────────────────────────────
    print(f"📈 Computing {history_days}-day TRI history (sub-sampled)...")
    rolling_tri = _build_rolling_tri(price_dict, volume_dict, history_days)

    # ── Rolling vol + correlation for charts ──────────────
    price_df = pd.DataFrame(price_dict).dropna(how='all').ffill().dropna()
    returns  = np.log(price_df / price_df.shift(1)).dropna()
    idx_ret  = returns.mean(axis=1)

    roll_vol  = idx_ret.rolling(20).std() * np.sqrt(252) * 100
    baseline  = idx_ret.rolling(252, min_periods=60).std() * np.sqrt(252) * 100
    vol_1sig  = baseline + baseline.rolling(60).std()
    vol_2sig  = baseline + 2 * baseline.rolling(60).std()

    def _rolling_avg_corr(ret_df, window=20):
        out = {}
        dates = ret_df.index[window:]
        for d in dates[::2]:
            snap = ret_df.loc[:d].iloc[-window:]
            c    = snap.corr()
            arr  = c.values.copy().astype(float)
            np.fill_diagonal(arr, np.nan)
            out[d] = float(np.nanmean(arr))
        return pd.Series(out).sort_index()

    roll_corr = _rolling_avg_corr(returns.iloc[-max(history_days * 2, 180):])

    # ── Build figure ───────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor='#0d0d1a')
    gs  = GridSpec(2, 2, hspace=0.38, wspace=0.30,
                   left=0.07, right=0.97, top=0.92, bottom=0.06)

    TRI_COLOR = {
        'NORMAL'  : '#22c55e',
        'ELEVATED': '#fbbf24',
        'HIGH'    : '#f97316',
        'CRITICAL': '#ef4444',
    }
    current_color = TRI_COLOR[report.level]

    # ─── Panel 1: TRI gauge + history ─────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')

    # Coloured background bands
    bands = [
        (0.00, 0.30, '#22c55e', 0.15, 'NORMAL'),
        (0.30, 0.50, '#fbbf24', 0.15, 'ELEVATED'),
        (0.50, 0.70, '#f97316', 0.15, 'HIGH'),
        (0.70, 1.00, '#ef4444', 0.15, 'CRITICAL'),
    ]
    for lo, hi, col, alpha, lbl in bands:
        ax1.axhspan(lo, hi, color=col, alpha=alpha, zorder=1)
        ax1.text(rolling_tri.index[0] if not rolling_tri.empty else 0,
                  (lo + hi) / 2, lbl,
                  color=col, fontsize=7.5, alpha=0.7, va='center')

    if not rolling_tri.empty:
        ax1.plot(rolling_tri.index, rolling_tri.values,
                  color='white', linewidth=1.4, alpha=0.7, zorder=3,
                  label='Rolling TRI (sub-sampled)')
        ax1.fill_between(rolling_tri.index, rolling_tri.values, 0,
                          color=current_color, alpha=0.18, zorder=2)

    # Current TRI as a bold horizontal line + annotation
    ax1.axhline(report.tri, color=current_color, linewidth=2.5,
                 linestyle='-', zorder=4, label=f'Current TRI = {report.tri:.3f}')
    ax1.annotate(
        f' {LEVEL_EMOJI[report.level]} {report.level}\n TRI = {report.tri:.3f}',
        xy=(rolling_tri.index[-1] if not rolling_tri.empty else 0.5, report.tri),
        color=current_color, fontsize=11, fontweight='bold', va='center'
    )

    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Tail Risk Index', color='white', fontsize=9)
    ax1.set_title(f'Tail Risk Index (TRI) — {history_days}-day History',
                   color='white', fontsize=11, fontweight='bold', pad=10)
    ax1.legend(fontsize=8, facecolor='#12121f', labelcolor='white', edgecolor='#262640')
    ax1.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ─── Panel 2: Component score bars ────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1a1a2e')

    components = [
        ('Volatility\nStress',    report.vol_score,          '#ef4444', 0.30),
        ('Fat-Tail\nKurtosis',    report.kurtosis_score,     '#f97316', 0.20),
        ('Negative\nSkew',        report.skew_score,         '#fbbf24', 0.15),
        ('Correlation\nSpike',    report.correlation_score,  '#a78bfa', 0.20),
        ('Liquidity\nStress',     report.liquidity_score,    '#38bdf8', 0.07),
        ('VaR Breach\nCount',     report.var_breach_score,   '#4ade80', 0.08),
    ]
    labels      = [c[0] for c in components]
    scores      = [c[1] for c in components]
    bar_colors  = [c[2] for c in components]
    weights     = [c[3] for c in components]

    x = np.arange(len(labels))
    bars = ax2.bar(x, scores, color=bar_colors, alpha=0.82, width=0.55)
    # Weighted contribution overlay (lighter, transparent)
    weighted = [s * w for s, w in zip(scores, weights)]
    ax2.bar(x, weighted, color=bar_colors, alpha=0.35, width=0.55,
             label='Weighted contribution to TRI')

    for bar, score, wt in zip(bars, scores, weights):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.02,
                  f'{score:.2f}\n({wt:.0%} wt)',
                  ha='center', va='bottom', fontsize=7, color='white')

    ax2.axhline(0.70, color='#ef4444', linestyle='--', linewidth=0.8,
                 alpha=0.7, label='0.70 critical threshold')
    ax2.axhline(0.40, color='#fbbf24', linestyle=':', linewidth=0.7,
                 alpha=0.6, label='0.40 watch threshold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, color='#e2e8f0')
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Component Score (0–1)', color='white', fontsize=9)
    ax2.set_title('Black Swan Detector — Component Breakdown',
                   color='white', fontsize=11, fontweight='bold', pad=10)
    ax2.legend(fontsize=7, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640', loc='upper right')
    ax2.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ─── Panel 3: Rolling realized vol ────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#1a1a2e')

    recent = roll_vol.iloc[-history_days * 2:]
    ax3.plot(recent.index, recent.values, color='#4ecdc4',
              linewidth=1.4, label='20-day Realized Vol (ann.)')
    ax3.plot(baseline.iloc[-history_days * 2:].index,
              baseline.iloc[-history_days * 2:].values,
              color='white', linewidth=0.8, linestyle='--',
              alpha=0.5, label='1-year baseline')

    if not vol_1sig.dropna().empty:
        r1 = vol_1sig.iloc[-history_days * 2:]
        r2 = vol_2sig.iloc[-history_days * 2:]
        ax3.fill_between(r1.index, r1, r2, color='#fbbf24', alpha=0.15,
                          label='1σ–2σ zone')
        ax3.fill_between(r2.index,
                          r2, r2 + r2.std(),
                          color='#ef4444', alpha=0.15, label='>2σ (crash zone)')

    ax3.axhline(report.realized_vol_ann, color='#ef4444', linewidth=1.2,
                 linestyle='-', label=f'Current: {report.realized_vol_ann:.1f}%')
    ax3.set_ylabel('Annualized Volatility (%)', color='white', fontsize=9)
    ax3.set_title('Rolling Realized Volatility — Nifty Index Proxy',
                   color='white', fontsize=11, fontweight='bold', pad=10)
    ax3.legend(fontsize=7.5, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640')
    ax3.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ─── Panel 4: Rolling avg pairwise correlation ────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#1a1a2e')

    if not roll_corr.empty:
        ax4.plot(roll_corr.index, roll_corr.values, color='#a78bfa',
                  linewidth=1.4, label='20-day Avg Pairwise Correlation')
        ax4.fill_between(roll_corr.index, roll_corr.values,
                          roll_corr.mean(),
                          where=(roll_corr.values > roll_corr.mean()),
                          color='#ef4444', alpha=0.18)
        ax4.axhline(roll_corr.mean(), color='white', linewidth=0.7,
                     linestyle='--', alpha=0.5,
                     label=f'Mean ({roll_corr.mean():.2f})')
        ax4.axhline(0.65, color='#ef4444', linewidth=1.0, linestyle='--',
                     alpha=0.7, label='0.65 — diversification failure zone')
        ax4.axhline(report.avg_pairwise_corr, color='#a78bfa',
                     linewidth=1.5, linestyle='-',
                     label=f'Current: {report.avg_pairwise_corr:.3f}')

    ax4.set_ylim(0, 1)
    ax4.set_ylabel('Avg Pairwise Return Correlation', color='white', fontsize=9)
    ax4.set_title('Cross-Stock Correlation — Diversification Health',
                   color='white', fontsize=11, fontweight='bold', pad=10)
    ax4.legend(fontsize=7.5, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640')
    ax4.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ── Super-title ────────────────────────────────────────
    status_str = f"{LEVEL_EMOJI[report.level]} {report.level}  |  TRI = {report.tri:.3f}"
    fig.suptitle(
        f'QuantAI Tail Risk Monitor — Black Swan Detector    [{status_str}]',
        color=current_color, fontsize=14, fontweight='bold', y=0.97
    )

    # ── Save / show ────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    out_path = f'models/tail_risk_monitor_{period}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"📊 Chart saved → {out_path}")

    if not save:
        plt.show()

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='QuantAI Tail Risk Monitor — Black Swan Detector'
    )
    parser.add_argument('--period', default='1y', choices=['1y', '2y', '3y'],
                        help='Lookback for data (default: 1y)')
    parser.add_argument('--history', type=int, default=60,
                        help='Days of TRI history to show on chart (default: 60)')
    parser.add_argument('--save', action='store_true',
                        help='Save PNG without showing the window')
    args = parser.parse_args()

    plot_tail_risk(period=args.period, history_days=args.history, save=args.save)