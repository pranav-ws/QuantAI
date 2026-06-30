"""
risk_parity_analyser.py

Standalone script that shows how risk parity would have allocated
across the current Nifty 50 BUY signals, and compares it against
the existing fixed-fraction sizing from RiskManager.

Two modes:

  LIVE mode (default):
    Scans all 50 stocks using the ensemble model to find today's BUY
    signals, then runs both RiskManager and RiskParityAllocator on
    them side-by-side, printing a comparison table and saving a chart.

  DEMO mode (--demo):
    Uses a synthetic set of 5–8 BUY signals with varied volatilities
    to illustrate the difference without needing trained models or
    live data. Useful for understanding the math before deployment.

Charts produced:

  Panel 1: Capital allocation (% of portfolio) — fixed-fraction vs
    risk parity side by side. Usually nearly identical by capital.

  Panel 2: Risk contribution (% of daily portfolio risk) — this is
    where the difference is stark. Fixed-fraction concentrates risk
    in high-vol names; risk parity equalises it.

  Panel 3: Volatility profile of selected stocks — shows WHY risk
    parity gives different sizes (the high-vol names get less capital).

  Panel 4: Cumulative "risk budget" usage — how efficiently each
    method deploys the daily risk budget across positions.

Run:
    python risk_parity_analyser.py
    python risk_parity_analyser.py --demo
    python risk_parity_analyser.py --capital 500000
    python risk_parity_analyser.py --save
"""
import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.gridspec import GridSpec # type: ignore

from src.risk_parity import RiskParityAllocator, RiskParityResult
from src.risk import RiskManager


# ── Demo signal generator (no models needed) ─────────────

def _demo_signals(capital: float) -> tuple[list, dict]:
    """
    Creates a realistic synthetic basket of 6 BUY signals spanning
    different sectors and volatility profiles, to demonstrate the
    difference between fixed-fraction and risk parity sizing.
    """
    import sqlite3
    specs = [
        # (ticker,        price,  approx_daily_vol)
        ('HDFCBANK.NS',   1650,   0.011),   # low vol banking
        ('TCS.NS',        3900,   0.012),   # low vol IT
        ('RELIANCE.NS',   2800,   0.014),   # medium vol energy/conglom
        ('TATAMOTORS.NS',  900,   0.022),   # high vol auto
        ('ADANIENT.NS',   2400,   0.030),   # high vol infra
        ('BAJFINANCE.NS',  700,   0.020),   # medium-high vol NBFC
    ]

    rm        = RiskManager(initial_capital=capital)
    rm.capital = capital
    signals   = []
    close_map = {}
    np.random.seed(42)

    for ticker, price, daily_vol in specs:
        conf = np.random.uniform(0.60, 0.76)
        shares, stop_loss, _ = rm.calculate_position_size(
            capital=capital, price=price, confidence=conf
        )
        if shares > 0:
            signals.append({
                'ticker'     : ticker,
                'price'      : price,
                'confidence' : round(conf, 4),
                'shares'     : shares,
                'stop_loss'  : round(stop_loss, 2),
                'trade_value': round(shares * price, 2),
            })
            # Simulate a close series with the desired vol
            n      = 60
            dates  = pd.bdate_range(end='2024-12-31', periods=n)
            ret    = np.random.normal(0.0, daily_vol, n)
            close  = pd.Series(price * np.exp(np.cumsum(ret)), index=dates)
            close_map[ticker] = close

    return signals, close_map


# ── Live signal generator (uses ensemble model) ───────────

def _live_signals(capital: float) -> tuple[list, dict]:
    """
    Runs the full ensemble scan on all 50 stocks and returns BUY signals
    plus a close_map pre-loaded from DB / yfinance.
    """
    from src.data_collector import STOCK_UNIVERSE
    from src.paper_trader   import fetch_latest_data
    from src.risk           import RiskManager
    from src.risk_parity    import RiskParityAllocator

    rm = RiskManager(initial_capital=capital)
    rm.capital = capital
    signals    = []
    close_map  = {}

    print(f"\n⚙️  Scanning {len(STOCK_UNIVERSE)} stocks for BUY signals...\n")

    for i, ticker in enumerate(STOCK_UNIVERSE):
        try:
            df = fetch_latest_data(ticker)
            if df is None or len(df) < 30:
                continue

            from src.ensemble_model import get_ensemble_confidence
            ens_conf, _, models_used = get_ensemble_confidence(ticker, df)
            if ens_conf is None or ens_conf < 0.58:
                continue

            price = float(df.iloc[-1]['Close'])
            shares, stop_loss, _ = rm.calculate_position_size(
                capital=capital, price=price, confidence=ens_conf
            )
            if shares > 0:
                signals.append({
                    'ticker'    : ticker,
                    'price'     : price,
                    'confidence': round(ens_conf, 4),
                    'shares'    : shares,
                    'stop_loss' : round(stop_loss, 2),
                    'trade_value': round(shares * price, 2),
                })
                close_map[ticker] = df['Close']

            pct = (i + 1) / len(STOCK_UNIVERSE) * 100
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:5.1f}%  {ticker:<22}", end='\r')

        except Exception:
            pass

    print(f"\n\n✅  Found {len(signals)} BUY signal(s)\n")
    return signals, close_map


# ── Comparison table ──────────────────────────────────────

def _print_comparison(signals_fixed: list, signals_rp: list,
                       capital: float, result_rp: RiskParityResult):
    rp_map = {s['ticker']: s for s in signals_rp}

    print(f"\n{'='*72}")
    print(f"  RISK PARITY vs FIXED-FRACTION SIZING COMPARISON")
    print(f"  Capital: ₹{capital:,.0f}")
    print(f"{'='*72}")
    print(f"  {'Ticker':<18} │ {'Fixed-Frac':^22} │ {'Risk Parity':^22}")
    print(f"  {'':─<18}─┼─{'':─^22}─┼─{'':─^22}")
    print(f"  {'':18} │ {'Shares':>7} {'Value':>10} Wt │ "
          f"{'Shares':>7} {'Value':>10} Wt")
    print(f"  {'':─<18}─┼─{'':─^22}─┼─{'':─^22}")

    for s in signals_fixed:
        t   = s['ticker'].replace('.NS', '')
        rp  = rp_map.get(s['ticker'], {})
        f_w = s['trade_value'] / capital * 100
        r_w = rp.get('rp_weight', 0) * 100
        vol = rp.get('rp_daily_vol', 0)
        diff= r_w - f_w
        arrow = '▲' if diff > 0.5 else ('▼' if diff < -0.5 else '=')
        print(f"  {t:<18} │ {s['shares']:>7} ₹{s['trade_value']:>9,.0f} {f_w:4.1f}% │ "
              f"{rp.get('shares', 0):>7} ₹{rp.get('trade_value', 0):>9,.0f} "
              f"{r_w:4.1f}% {arrow} ({vol:.2f}%σ)")

    tf_fixed = sum(s['trade_value'] for s in signals_fixed)
    tf_rp    = result_rp.total_deployed
    print(f"  {'':─<18}─┼─{'':─^22}─┼─{'':─^22}")
    print(f"  {'TOTAL':<18} │ {'':>7} ₹{tf_fixed:>9,.0f} {'':5} │ "
          f"{'':>7} ₹{tf_rp:>9,.0f}")
    print(f"\n  (σ = estimated daily volatility  │  ▲/▼/= = more/less/same vs fixed-fraction)")
    print(f"{'='*72}\n")

    # Risk concentration table
    print(f"  RISK CONTRIBUTION COMPARISON  (daily ₹ risk per position)")
    print(f"  {'':─<60}")
    print(f"  {'Ticker':<18} {'Fixed-Frac Risk':>16} {'Risk-Parity Risk':>16} {'Δ':>8}")
    print(f"  {'':─<60}")

    for s in signals_fixed:
        t     = s['ticker']
        vol   = result_rp.vol_estimates.get(t, 0) / 100
        f_dr  = s['trade_value'] * vol
        rp_dr = result_rp.actual_risk_contribs.get(t, 0)
        delta = rp_dr - f_dr
        print(f"  {t.replace('.NS',''):<18} ₹{f_dr:>13,.0f}   ₹{rp_dr:>13,.0f}  "
              f"{'▲' if delta > 100 else ('▼' if delta < -100 else '=')}{abs(delta):>6,.0f}")

    print(f"  {'':─<60}")
    total_f_risk = sum(
        s['trade_value'] * result_rp.vol_estimates.get(s['ticker'], 0) / 100
        for s in signals_fixed
    )
    total_rp_risk = sum(result_rp.actual_risk_contribs.values())
    print(f"  {'TOTAL':<18} ₹{total_f_risk:>13,.0f}   ₹{total_rp_risk:>13,.0f}")
    print(f"\n  Key insight: Risk parity reduces the gap between highest and lowest "
          f"risk contributors — each position truly 'punches equally'.\n")


# ── Chart ─────────────────────────────────────────────────

def _plot_comparison(signals_fixed: list, signals_rp: list,
                      capital: float, result_rp: RiskParityResult,
                      save: bool = False, suffix: str = 'live'):

    tickers_short = [s['ticker'].replace('.NS', '') for s in signals_fixed]
    rp_map        = {s['ticker']: s for s in signals_rp}
    x             = np.arange(len(tickers_short))
    w             = 0.38

    # Capital weights
    fixed_cap_w = [s['trade_value'] / capital * 100 for s in signals_fixed]
    rp_cap_w    = [rp_map.get(s['ticker'], {}).get('rp_weight', 0) * 100
                    for s in signals_fixed]

    # Daily vol estimates
    vols = [result_rp.vol_estimates.get(s['ticker'], 0) for s in signals_fixed]

    # Daily rupee risk contributions
    fixed_risk = [s['trade_value'] * result_rp.vol_estimates.get(s['ticker'], 0) / 100
                   for s in signals_fixed]
    rp_risk    = [result_rp.actual_risk_contribs.get(s['ticker'], 0)
                   for s in signals_fixed]

    # Risk budget usage (running total)
    target_per = result_rp.target_risk_per_pos
    cum_fixed  = np.cumsum(fixed_risk) / (target_per * len(signals_fixed)) * 100
    cum_rp     = np.cumsum(rp_risk)    / (target_per * len(signals_fixed)) * 100

    fig = plt.figure(figsize=(16, 12), facecolor='#0d0d1a')
    gs  = GridSpec(2, 2, hspace=0.38, wspace=0.32,
                   left=0.07, right=0.97, top=0.92, bottom=0.07)

    FIXED_COL = '#4ecdc4'
    RP_COL    = '#fbbf24'

    # ── Panel 1: Capital allocation ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')
    ax1.bar(x - w/2, fixed_cap_w, width=w, color=FIXED_COL, alpha=0.82,
             label='Fixed-fraction (Kelly)')
    ax1.bar(x + w/2, rp_cap_w,   width=w, color=RP_COL,    alpha=0.82,
             label='Risk Parity (ERC)')
    for i, (f, r) in enumerate(zip(fixed_cap_w, rp_cap_w)):
        ax1.text(i - w/2, f + 0.2, f'{f:.1f}%', ha='center', fontsize=7,
                  color='white')
        ax1.text(i + w/2, r + 0.2, f'{r:.1f}%', ha='center', fontsize=7,
                  color='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers_short, fontsize=8, color='#e2e8f0')
    ax1.set_ylabel('% of Portfolio Capital', color='white', fontsize=9)
    ax1.set_title('Capital Allocation per Position',
                   color='white', fontsize=11, fontweight='bold', pad=10)
    ax1.legend(fontsize=8, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640')
    ax1.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ── Panel 2: Risk contribution ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1a1a2e')

    total_f = sum(fixed_risk) or 1
    total_r = sum(rp_risk)    or 1
    fixed_risk_pct = [v / total_f * 100 for v in fixed_risk]
    rp_risk_pct    = [v / total_r * 100 for v in rp_risk]
    equal_line     = 100 / len(tickers_short)

    ax2.bar(x - w/2, fixed_risk_pct, width=w, color=FIXED_COL, alpha=0.82,
             label='Fixed-fraction')
    ax2.bar(x + w/2, rp_risk_pct,   width=w, color=RP_COL,    alpha=0.82,
             label='Risk Parity')
    ax2.axhline(equal_line, color='white', linewidth=1.2, linestyle='--',
                 alpha=0.6, label=f'Equal share ({equal_line:.1f}%)')

    for i, (f, r) in enumerate(zip(fixed_risk_pct, rp_risk_pct)):
        ax2.text(i - w/2, f + 0.3, f'{f:.1f}%', ha='center', fontsize=7,
                  color='white')
        ax2.text(i + w/2, r + 0.3, f'{r:.1f}%', ha='center', fontsize=7,
                  color='white')

    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers_short, fontsize=8, color='#e2e8f0')
    ax2.set_ylabel('% of Total Portfolio Risk', color='white', fontsize=9)
    ax2.set_title('Risk Contribution per Position\n'
                   '(Risk Parity ≈ equal bars = the goal)',
                   color='white', fontsize=11, fontweight='bold', pad=10)
    ax2.legend(fontsize=8, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640')
    ax2.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ── Panel 3: Volatility profile ───────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#1a1a2e')

    vol_colors = ['#ef4444' if v > 2.0 else
                   '#f97316' if v > 1.5 else
                   '#fbbf24' if v > 1.0 else '#22c55e'
                   for v in vols]
    bars = ax3.bar(x, vols, color=vol_colors, alpha=0.85, width=0.55)
    for bar, v in zip(bars, vols):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.03,
                  f'{v:.2f}%', ha='center', fontsize=8, color='white',
                  fontweight='bold')

    ax3.axhline(1.5, color='#fbbf24', linewidth=0.9, linestyle='--',
                 alpha=0.7, label='1.5% daily = ~24% annual')
    ax3.axhline(2.0, color='#ef4444', linewidth=0.9, linestyle='--',
                 alpha=0.7, label='2.0% daily = ~32% annual (high risk)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(tickers_short, fontsize=8, color='#e2e8f0')
    ax3.set_ylabel('Estimated Daily Volatility (%)', color='white', fontsize=9)
    ax3.set_title('Stock Volatility Profile\n'
                   '(why risk parity gives different sizes)',
                   color='white', fontsize=11, fontweight='bold', pad=10)
    ax3.legend(fontsize=7.5, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640')
    ax3.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ── Panel 4: Cumulative risk budget usage ─────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#1a1a2e')

    pos_labels = [f'Pos {i+1}\n({t})' for i, t in enumerate(tickers_short)]
    ax4.plot(range(len(pos_labels)), cum_fixed, color=FIXED_COL, linewidth=2.0,
              marker='o', markersize=6, label='Fixed-fraction')
    ax4.plot(range(len(pos_labels)), cum_rp, color=RP_COL, linewidth=2.0,
              marker='s', markersize=6, label='Risk Parity')
    ax4.fill_between(range(len(pos_labels)), cum_fixed, cum_rp,
                      alpha=0.12, color='white')

    ax4.axhline(100, color='#22c55e', linewidth=0.9, linestyle='--',
                 alpha=0.7, label='100% of risk budget used')
    ax4.set_xticks(range(len(pos_labels)))
    ax4.set_xticklabels(tickers_short, fontsize=7.5, color='#e2e8f0')
    ax4.set_ylabel('Cumulative Risk Budget Used (%)', color='white', fontsize=9)
    ax4.set_title('Cumulative Risk Budget Consumption\n'
                   '(risk parity should track the 100% line evenly)',
                   color='white', fontsize=11, fontweight='bold', pad=10)
    ax4.legend(fontsize=8, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640')
    ax4.tick_params(colors='#94a3b8', labelsize=7, length=0)

    fig.suptitle(
        'QuantAI Risk Parity — Equal Risk Contribution Allocation',
        color='white', fontsize=14, fontweight='bold', y=0.97
    )

    os.makedirs('models', exist_ok=True)
    out_path = f'models/risk_parity_{suffix}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"📊 Chart saved → {out_path}")
    if not save:
        plt.show()


# ── Entry point ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='QuantAI Risk Parity — Equal Risk Contribution Analyser'
    )
    parser.add_argument('--demo',    action='store_true',
                        help='Use synthetic signals (no models needed)')
    parser.add_argument('--capital', type=float, default=100_000,
                        help='Portfolio capital in ₹ (default: 100000)')
    parser.add_argument('--save',    action='store_true',
                        help='Save chart without showing window')
    args = parser.parse_args()

    capital = args.capital

    # ── Get signals ────────────────────────────────────────
    if args.demo:
        print("\n🔬 DEMO MODE — using synthetic signals\n")
        signals_fixed, close_map = _demo_signals(capital)
        suffix = 'demo'
    else:
        signals_fixed, close_map = _live_signals(capital)
        suffix = 'live'

    if not signals_fixed:
        print("❌  No BUY signals found. Try --demo mode or run pipeline.py first.")
        return

    # ── Fixed-fraction baseline (already computed by _demo/_live) ──
    print(f"  Fixed-fraction: {len(signals_fixed)} position(s) "
          f"totalling ₹{sum(s['trade_value'] for s in signals_fixed):,.0f}")

    # ── Risk parity allocation ─────────────────────────────
    allocator      = RiskParityAllocator()
    signals_rp, result_rp = allocator.allocate(signals_fixed, capital, close_map)

    # ── Print allocation table ─────────────────────────────
    allocator.print_allocation(result_rp, signals_rp)

    # ── Print comparison ───────────────────────────────────
    _print_comparison(signals_fixed, signals_rp, capital, result_rp)

    # ── Chart ──────────────────────────────────────────────
    _plot_comparison(signals_fixed, signals_rp, capital, result_rp,
                     save=args.save, suffix=suffix)


if __name__ == '__main__':
    main()