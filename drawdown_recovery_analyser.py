"""
drawdown_recovery_analyser.py

Simulates the DrawdownRecoveryManager against your historical
paper_trades.json and shows exactly what position sizes would have
been applied to each trade.

Two modes:

  LIVE mode (default):
    Reads data/paper_trades.json, replays every closed trade through
    the recovery manager in chronological order, and shows:
      - What multiplier was active on each trade entry
      - Whether the multiplier would have reduced losses in bad patches
      - How quickly recovery kicked back in after wins

  DEMO mode (--demo):
    Generates a synthetic 40-trade sequence (mix of wins/losses with
    two deliberate loss streaks and a drawdown episode) to illustrate
    the tier transitions without needing real trade history.

Charts (4 panels):

  Panel 1: Size multiplier over time — the main output.
    Shows how the multiplier ticked down during bad patches and
    recovered afterward. Green = full size, red = reduced.

  Panel 2: Drawdown tier vs streak tier over time.
    Shows which signal was doing the work at each point —
    sometimes it's the drawdown, sometimes it's the loss streak.

  Panel 3: Capital curve with and without recovery scaling.
    Hypothetical comparison: "What if recovery had been active from
    day 1?" vs "What if every trade was full-sized regardless?"

  Panel 4: Tier distribution pie — how many trading days spent
    in each recovery tier. Tells you how often the system would
    have been in a reduced-size state historically.

Run:
    python drawdown_recovery_analyser.py
    python drawdown_recovery_analyser.py --demo
    python drawdown_recovery_analyser.py --save
"""
import argparse
import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as mpatches # type: ignore
from matplotlib.gridspec import GridSpec # type: ignore

from src.drawdown_recovery import (
    DrawdownRecoveryManager, DRAWDOWN_TIERS, CONSEC_LOSS_TIERS,
    RECOVERY_STEP, MIN_WINS_TO_RECOVER
)


# ── Synthetic demo sequence ───────────────────────────────

def _demo_trades(initial_capital: float = 100_000.0) -> list:
    """
    40 synthetic trades: normal run → loss streak → recovery →
    drawdown episode → recovery. Covers all tier transitions.
    """
    np.random.seed(42)
    sequence = (
        ['WIN'] * 5 +
        ['LOSS', 'LOSS', 'LOSS'] +       # streak tier triggers
        ['WIN'] * 3 +                     # partial recovery
        ['LOSS', 'WIN', 'LOSS', 'LOSS', 'LOSS', 'LOSS'] +  # deeper streak
        ['WIN', 'WIN', 'WIN', 'WIN'] +    # recovery
        ['LOSS'] * 4 +                    # drawdown episode begins
        ['LOSS'] * 3 +                    # deeper
        ['WIN'] * 8 +                     # full recovery
        ['WIN', 'LOSS', 'WIN', 'WIN']     # stable end
    )

    trades = []
    capital = initial_capital
    peak    = initial_capital

    for i, result in enumerate(sequence):
        if result == 'WIN':
            pnl_pct = np.random.uniform(1.5, 6.0)
        else:
            pnl_pct = -np.random.uniform(1.0, 4.5)

        # Scale trade size by 5% of capital
        trade_value = capital * 0.05
        pnl_inr     = trade_value * pnl_pct / 100
        capital    += pnl_inr
        peak        = max(peak, capital)

        trades.append({
            'trade_num'  : i + 1,
            'result'     : result,
            'pnl_pct'    : round(pnl_pct, 2),
            'pnl'        : round(pnl_inr, 2),
            'capital'    : round(capital, 2),
            'peak'       : round(peak, 2),
            'status'     : 'CLOSED',
        })

    return trades


# ── Load real trade log ───────────────────────────────────

def _load_real_trades(initial_capital: float = 100_000.0) -> list:
    path = os.path.join('data', 'paper_trades.json')
    if not os.path.exists(path):
        print(f"  ⚠️  {path} not found — run paper_trade.py first or use --demo")
        return []

    with open(path) as f:
        raw = json.load(f)

    closed = [t for t in raw if t.get('status') == 'CLOSED' and 'pnl' in t]
    if not closed:
        print("  ℹ️  No closed trades in paper_trades.json yet.")
        print("  Use --demo mode to see how recovery would behave.")
        return []

    # Reconstruct capital curve from PnL
    capital = initial_capital
    peak    = initial_capital
    trades  = []
    for i, t in enumerate(sorted(closed, key=lambda x: x.get('date', ''))):
        pnl_inr = t.get('pnl', 0)
        capital += pnl_inr
        peak     = max(peak, capital)
        trade_value = t.get('trade_value', capital * 0.05) or capital * 0.05
        pnl_pct = pnl_inr / trade_value * 100 if trade_value else 0

        trades.append({
            'trade_num'  : i + 1,
            'ticker'     : t.get('ticker', '?'),
            'result'     : 'WIN' if pnl_inr > 0 else 'LOSS',
            'pnl_pct'    : round(pnl_pct, 2),
            'pnl'        : round(pnl_inr, 2),
            'capital'    : round(capital, 2),
            'peak'       : round(peak, 2),
            'status'     : 'CLOSED',
        })

    return trades


# ── Simulation ────────────────────────────────────────────

def simulate(trades: list, initial_capital: float = 100_000.0) -> pd.DataFrame:
    """
    Replay every trade through the recovery manager and record the
    multiplier that would have been active for each trade entry.
    Returns a DataFrame with one row per trade.
    """
    drm    = DrawdownRecoveryManager(initial_capital=initial_capital)
    rows   = []
    capital_with_recovery    = initial_capital
    capital_without_recovery = initial_capital

    for t in trades:
        # Multiplier at ENTRY (before this trade's outcome is known)
        mult_at_entry = drm.state.size_multiplier

        # ── "With recovery" equity ────────────────────────
        # Imagine trade_value was scaled by the multiplier
        # (We use 5% of capital as the base trade size)
        base_trade_val    = capital_with_recovery * 0.05
        scaled_trade_val  = base_trade_val * mult_at_entry
        pnl_with          = scaled_trade_val * t['pnl_pct'] / 100
        capital_with_recovery += pnl_with

        # ── "Without recovery" equity ─────────────────────
        base_trade_val2   = capital_without_recovery * 0.05
        pnl_without       = base_trade_val2 * t['pnl_pct'] / 100
        capital_without_recovery += pnl_without

        peak_w = max(drm.state.peak_capital, capital_with_recovery)

        # Update recovery state with actual outcome
        new_mult = drm.update(
            result         = t['result'],
            pnl_pct        = t['pnl_pct'],
            current_capital= capital_with_recovery,
            peak_capital   = peak_w
        )

        rows.append({
            'trade_num'           : t['trade_num'],
            'ticker'              : t.get('ticker', ''),
            'result'              : t['result'],
            'pnl_pct'             : t['pnl_pct'],
            'mult_at_entry'       : round(mult_at_entry, 3),
            'mult_after'          : round(new_mult, 3),
            'dd_tier'             : drm.state.dd_tier,
            'streak_tier'         : drm.state.streak_tier,
            'drawdown_pct'        : round(drm.state.current_drawdown_pct, 2),
            'consec_losses'       : drm.state.consecutive_losses,
            'consec_wins'         : drm.state.consecutive_wins,
            'capital_with'        : round(capital_with_recovery, 2),
            'capital_without'     : round(capital_without_recovery, 2),
            'pnl_with'            : round(pnl_with, 2),
            'pnl_without'         : round(pnl_without, 2),
        })

    return pd.DataFrame(rows)


# ── Print table ───────────────────────────────────────────

def _print_table(df: pd.DataFrame):
    print(f"\n{'='*80}")
    print(f"  DRAWDOWN RECOVERY SIMULATION — Trade-by-Trade")
    print(f"{'='*80}")
    print(f"  {'#':>3} {'Result':>6} {'PnL%':>6} {'Mult':>5} "
          f"{'DD Tier':>10} {'Str Tier':>10} {'DD%':>6} {'L':>3} {'W':>3}")
    print(f"  {'─'*72}")

    tier_emoji = {
        'NORMAL': '🟢', 'CAUTION': '🟡', 'REDUCED': '🟠',
        'DEFENSIVE': '🔴', 'CRITICAL': '🔴', 'HALTED': '🚫',
        'WATCH': '🟡', 'STREAK': '🟠', 'BAD_STREAK': '🔴', 'CRISIS': '🔴',
    }

    for _, r in df.iterrows():
        res_sym  = '✅' if r['result'] == 'WIN' else '❌'
        dd_sym   = tier_emoji.get(r['dd_tier'], '⚪')
        str_sym  = tier_emoji.get(r['streak_tier'], '⚪')
        mult_str = f"{r['mult_at_entry']*100:.0f}%"
        if r['mult_at_entry'] < 1.0:
            mult_str = f"*{mult_str}*"   # highlight reduced sizes
        print(f"  {r['trade_num']:>3} {res_sym}     {r['pnl_pct']:>+5.1f}% {mult_str:>5} "
              f"{dd_sym}{r['dd_tier']:>9} {str_sym}{r['streak_tier']:>9} "
              f"{r['drawdown_pct']:>+5.1f}% {r['consec_losses']:>3} {r['consec_wins']:>3}")

    # Summary
    reduced = (df['mult_at_entry'] < 1.0).sum()
    avg_mult = df['mult_at_entry'].mean()
    final_with    = df['capital_with'].iloc[-1]
    final_without = df['capital_without'].iloc[-1]
    initial       = 100_000.0

    print(f"  {'─'*72}")
    print(f"  Trades with reduced size : {reduced} / {len(df)}  "
          f"({reduced/len(df)*100:.1f}%)")
    print(f"  Average size multiplier  : {avg_mult*100:.1f}%")
    print(f"  Final capital (with RP)  : ₹{final_with:>12,.0f}  "
          f"({(final_with-initial)/initial*100:+.1f}%)")
    print(f"  Final capital (no RP)    : ₹{final_without:>12,.0f}  "
          f"({(final_without-initial)/initial*100:+.1f}%)")
    print(f"  Recovery benefit         : ₹{final_with - final_without:>+12,.0f}")
    print(f"{'='*80}\n")


# ── Chart ─────────────────────────────────────────────────

def _plot(df: pd.DataFrame, save: bool = False, suffix: str = 'live'):
    TIER_COLORS = {
        'NORMAL': '#22c55e', 'CAUTION': '#fbbf24', 'REDUCED': '#f97316',
        'DEFENSIVE': '#ef4444', 'CRITICAL': '#dc2626', 'HALTED': '#7f1d1d',
        'WATCH': '#fbbf24', 'STREAK': '#f97316', 'BAD_STREAK': '#ef4444',
        'CRISIS': '#dc2626',
    }

    x      = df['trade_num'].values
    mult   = df['mult_at_entry'].values
    dd_pct = df['drawdown_pct'].values

    fig = plt.figure(figsize=(16, 12), facecolor='#0d0d1a')
    gs  = GridSpec(2, 2, hspace=0.38, wspace=0.30,
                   left=0.07, right=0.97, top=0.92, bottom=0.06)

    # ── Panel 1: Size multiplier over time ────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')

    for i in range(len(x) - 1):
        color = TIER_COLORS.get(df['dd_tier'].iloc[i], '#64748b')
        ax1.fill_between([x[i], x[i+1]], [mult[i], mult[i+1]], 0,
                          color=color, alpha=0.35)
        ax1.plot([x[i], x[i+1]], [mult[i], mult[i+1]], color=color,
                  linewidth=1.8)

    ax1.axhline(1.0,  color='#22c55e', linewidth=0.9, linestyle='--',
                 alpha=0.6, label='Full size (100%)')
    ax1.axhline(0.60, color='#f97316', linewidth=0.7, linestyle=':',
                 alpha=0.5, label='60% (REDUCED tier)')
    ax1.axhline(0.20, color='#ef4444', linewidth=0.7, linestyle=':',
                 alpha=0.5, label='20% (CRITICAL tier)')

    # Mark losses with red triangles
    loss_mask = df['result'] == 'LOSS'
    ax1.scatter(df.loc[loss_mask, 'trade_num'],
                 df.loc[loss_mask, 'mult_at_entry'],
                 marker='v', color='#ef4444', s=55, zorder=5, label='Loss trade')
    win_mask = df['result'] == 'WIN'
    ax1.scatter(df.loc[win_mask, 'trade_num'],
                 df.loc[win_mask, 'mult_at_entry'],
                 marker='^', color='#22c55e', s=40, zorder=5,
                 alpha=0.6, label='Win trade')

    ax1.set_ylim(0, 1.15)
    ax1.set_xlabel('Trade Number', color='white', fontsize=9)
    ax1.set_ylabel('Size Multiplier', color='white', fontsize=9)
    ax1.set_title('Position Size Multiplier Over Trade History',
                   color='white', fontsize=11, fontweight='bold', pad=10)
    ax1.legend(fontsize=7.5, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640')
    ax1.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ── Panel 2: Drawdown tier vs streak tier ─────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1a1a2e')

    dd_mult    = df.apply(lambda r: {
        'NORMAL':1.0,'CAUTION':0.80,'REDUCED':0.60,
        'DEFENSIVE':0.40,'CRITICAL':0.20,'HALTED':0.0
    }.get(r['dd_tier'], 1.0), axis=1)
    str_mult   = df.apply(lambda r: {
        'NORMAL':1.0,'WATCH':0.85,'STREAK':0.65,
        'BAD_STREAK':0.45,'CRISIS':0.25
    }.get(r['streak_tier'], 1.0), axis=1)

    ax2.plot(x, dd_mult.values,  color='#4ecdc4', linewidth=1.5,
              label='Drawdown multiplier (Signal A)')
    ax2.plot(x, str_mult.values, color='#a78bfa', linewidth=1.5,
              linestyle='--', label='Streak multiplier (Signal B)')
    ax2.plot(x, mult, color='#fbbf24', linewidth=2.0,
              label='Effective (min of both)')
    ax2.fill_between(x, dd_mult.values, str_mult.values,
                      alpha=0.10, color='white')
    ax2.set_ylim(0, 1.15)
    ax2.set_xlabel('Trade Number', color='white', fontsize=9)
    ax2.set_ylabel('Multiplier', color='white', fontsize=9)
    ax2.set_title('Signal A (Drawdown) vs Signal B (Streak)\n'
                   'Effective = min(A, B)',
                   color='white', fontsize=11, fontweight='bold', pad=10)
    ax2.legend(fontsize=7.5, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640')
    ax2.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ── Panel 3: Capital curve with vs without recovery ───
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#1a1a2e')

    initial = 100_000.0
    cap_with    = df['capital_with'].values    / initial * 100
    cap_without = df['capital_without'].values / initial * 100

    ax3.plot(x, cap_without, color='#ef4444', linewidth=1.4,
              linestyle='--', label='Without recovery scaling', alpha=0.8)
    ax3.plot(x, cap_with,    color='#22c55e', linewidth=1.8,
              label='With recovery scaling')
    ax3.fill_between(x, cap_with, cap_without,
                      where=(cap_with >= cap_without),
                      color='#22c55e', alpha=0.12,
                      label='Recovery benefit')
    ax3.fill_between(x, cap_with, cap_without,
                      where=(cap_with < cap_without),
                      color='#ef4444', alpha=0.12)
    ax3.axhline(100, color='white', linewidth=0.7, linestyle=':',
                 alpha=0.4, label='Starting capital (100)')
    ax3.set_xlabel('Trade Number', color='white', fontsize=9)
    ax3.set_ylabel('Portfolio Value (base 100)', color='white', fontsize=9)
    ax3.set_title('Capital Curve: With vs Without Recovery Scaling',
                   color='white', fontsize=11, fontweight='bold', pad=10)
    ax3.legend(fontsize=7.5, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640')
    ax3.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ── Panel 4: Tier distribution ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#1a1a2e')

    tier_counts = df['dd_tier'].value_counts()
    tier_order  = ['NORMAL', 'CAUTION', 'REDUCED', 'DEFENSIVE', 'CRITICAL', 'HALTED']
    sizes       = [tier_counts.get(t, 0) for t in tier_order if tier_counts.get(t, 0) > 0]
    labels      = [t for t in tier_order if tier_counts.get(t, 0) > 0]
    colors      = [TIER_COLORS.get(t, '#64748b') for t in labels]

    wedges, texts, autotexts = ax4.pie(
        sizes, labels=labels, colors=colors,
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        startangle=90,
        textprops={'color': 'white', 'fontsize': 8},
        wedgeprops={'linewidth': 1.5, 'edgecolor': '#0d0d1a'},
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(8)
    ax4.set_title('Drawdown Tier Distribution\n(how often in each state)',
                   color='white', fontsize=11, fontweight='bold', pad=10)

    # Legend patches
    patches = [mpatches.Patch(color=TIER_COLORS.get(t, '#64748b'),
                               label=f'{t} ({tier_counts.get(t,0)} trades)')
               for t in tier_order if tier_counts.get(t, 0) > 0]
    ax4.legend(handles=patches, fontsize=7.5, facecolor='#12121f',
                labelcolor='white', edgecolor='#262640',
                loc='lower right', bbox_to_anchor=(1.3, -0.1))

    fig.suptitle('QuantAI — Drawdown Recovery (Auto-Reduce on Loss) Simulation',
                  color='white', fontsize=14, fontweight='bold', y=0.97)

    os.makedirs('models', exist_ok=True)
    out_path = f'models/drawdown_recovery_{suffix}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"📊 Chart saved → {out_path}")
    if not save:
        plt.show()


# ── Entry point ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='QuantAI Drawdown Recovery Analyser'
    )
    parser.add_argument('--demo', action='store_true',
                        help='Use synthetic trades (no real history needed)')
    parser.add_argument('--capital', type=float, default=100_000.0,
                        help='Initial capital ₹ (default: 100000)')
    parser.add_argument('--save', action='store_true',
                        help='Save chart without showing window')
    args = parser.parse_args()

    capital = args.capital

    if args.demo:
        print("\n🔬 DEMO MODE — synthetic 40-trade sequence\n")
        trades = _demo_trades(capital)
        suffix = 'demo'
    else:
        trades = _load_real_trades(capital)
        suffix = 'live'

    if not trades:
        return

    print(f"  Replaying {len(trades)} closed trade(s)...\n")
    df = simulate(trades, initial_capital=capital)

    _print_table(df)
    _plot(df, save=args.save, suffix=suffix)


if __name__ == '__main__':
    main()