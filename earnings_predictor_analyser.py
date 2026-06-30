"""
earnings_predictor_analyser.py  (root-level runner)

Runs the Earnings Beat/Miss Estimator and produces charts.

Modes:

  Single-ticker deep dive (default: RELIANCE.NS):
    python earnings_predictor.py
    python earnings_predictor.py --ticker TCS.NS

  Scan all 50 stocks for upcoming earnings (next 14 days):
    python earnings_predictor.py --scan
    python earnings_predictor.py --scan --days 21

  4-panel chart (single ticker):
    Panel 1: Historical EPS beat/miss timeline with surprise %
    Panel 2: Signal score radar showing each of the 6 signals
    Panel 3: Pre-earnings price drift pattern (average of all past events)
    Panel 4: Upcoming earnings calendar for all 50 stocks

  Save without showing:
    python earnings_predictor.py --save
"""
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as mpatches # type: ignore
from matplotlib.gridspec import GridSpec # type: ignore
from datetime import date, timedelta

from src.earnings_predictor import EarningsPredictor, EarningsPrediction


# ── 4-panel chart for single ticker ──────────────────────

def _plot_single(pred: EarningsPrediction,
                 predictor: EarningsPredictor,
                 save: bool = False):
    fig = plt.figure(figsize=(18, 12), facecolor='#0d0d1a')
    gs  = GridSpec(2, 2, hspace=0.42, wspace=0.32,
                   left=0.07, right=0.97, top=0.92, bottom=0.06)

    pred_emoji = {'BEAT': '🟢', 'MISS': '🔴', 'IN-LINE': '🟡',
                  'INSUFFICIENT_DATA': '⚪'}.get(pred.prediction, '⚪')
    fig.suptitle(
        f"QuantAI Earnings Predictor — {pred.ticker}  |  "
        f"{pred_emoji} {pred.prediction}  |  "
        f"Beat Prob: {pred.beat_probability*100:.1f}%  |  "
        f"Confidence: {pred.confidence*100:.1f}%",
        color='white', fontsize=12, fontweight='bold', y=0.97
    )

    # ── Panel 1: Historical EPS surprise timeline ─────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')

    if pred.raw_surprises:
        dates   = [s['date'][:7] for s in pred.raw_surprises]   # YYYY-MM
        surp    = [s['surprise_pct'] for s in pred.raw_surprises]
        colors  = ['#22c55e' if s > 0 else '#ef4444' for s in surp]
        x       = range(len(dates))
        bars    = ax1.bar(x, surp, color=colors, alpha=0.82, width=0.7)

        for bar, val in zip(bars, surp):
            ax1.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.4 if val >= 0 else bar.get_height() - 1.0,
                      f'{val:+.1f}%', ha='center', fontsize=7.5,
                      color='white', fontweight='bold')

        ax1.axhline(0, color='white', linewidth=0.8, alpha=0.5)
        avg_line = np.mean(surp)
        ax1.axhline(avg_line, color='#fbbf24', linewidth=1.2,
                     linestyle='--', alpha=0.8,
                     label=f'Avg: {avg_line:+.1f}%')
        ax1.set_xticks(x)
        ax1.set_xticklabels(dates, rotation=45, ha='right',
                              fontsize=7, color='#e2e8f0')
        ax1.set_ylabel('EPS Surprise vs Consensus (%)', color='white', fontsize=8)
        ax1.set_title(f'Historical EPS Surprise — {pred.ticker}\n'
                       f'Beat Rate: {pred.historical_beat_rate*100:.0f}%  |  '
                       f'Avg Surprise: {pred.avg_eps_surprise_pct:+.1f}%',
                       color='white', fontsize=10, fontweight='bold', pad=8)
        ax1.legend(fontsize=7.5, facecolor='#12121f', labelcolor='white',
                    edgecolor='#262640')
        ax1.tick_params(colors='#94a3b8', labelsize=7, length=0)

    else:
        ax1.text(0.5, 0.5, 'No earnings history\navailable on Yahoo Finance',
                  ha='center', va='center', color='#94a3b8', fontsize=11,
                  transform=ax1.transAxes)
        ax1.set_title('Historical EPS Surprise', color='white',
                       fontsize=10, fontweight='bold', pad=8)

    # ── Panel 2: Signal score radar / bar ─────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1a1a2e')

    signal_names = ['Beat Rate\nHistory', 'Surprise\nTrend',
                     'Pre-earnings\nDrift', 'Volume\nPattern',
                     'Revenue\nTrend', 'News\nSentiment']
    signal_scores = [pred.signal_beat_rate, pred.signal_surprise_trend,
                      pred.signal_price_drift, pred.signal_volume,
                      pred.signal_revenue, pred.signal_sentiment]
    weights       = [0.30, 0.15, 0.20, 0.15, 0.10, 0.10]

    bar_colors = ['#22c55e' if s > 0.60 else
                   '#ef4444' if s < 0.40 else
                   '#fbbf24' for s in signal_scores]

    x = range(len(signal_names))
    bars = ax2.bar(x, signal_scores, color=bar_colors, alpha=0.82, width=0.6)

    for bar, score, w in zip(bars, signal_scores, weights):
        ax2.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.02,
                  f'{score:.2f}\n({w:.0%})',
                  ha='center', va='bottom', fontsize=7.5, color='white')

    ax2.axhline(0.50, color='white', linewidth=0.8, linestyle='--',
                 alpha=0.5, label='Neutral (0.50)')
    ax2.axhline(0.60, color='#22c55e', linewidth=0.7, linestyle=':',
                 alpha=0.5, label='BEAT signal (0.60)')
    ax2.axhline(0.40, color='#ef4444', linewidth=0.7, linestyle=':',
                 alpha=0.5, label='MISS signal (0.40)')
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(signal_names, fontsize=8, color='#e2e8f0')
    ax2.set_ylabel('Signal Score (0–1)', color='white', fontsize=8)
    ax2.set_title(f'6-Signal Beat Estimator\n'
                   f'Composite Beat Probability: {pred.beat_probability*100:.1f}%  '
                   f'→  {pred.prediction}',
                   color='white', fontsize=10, fontweight='bold', pad=8)
    ax2.legend(fontsize=7, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640')
    ax2.tick_params(colors='#94a3b8', labelsize=7, length=0)

    # ── Panel 3: Pre-earnings price drift ─────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#1a1a2e')

    close, volume = predictor._load_price_volume(pred.ticker, days=180)
    if close is not None and pred.raw_surprises:
        past_dates = [s['date'] for s in pred.raw_surprises]
        all_windows = []
        for d_str in past_dates:
            try:
                ed   = pd.Timestamp(d_str)
                mask = close.index <= ed
                if not mask.any():
                    continue
                idx_end = close.index[mask][-1]
                loc_end = close.index.get_loc(idx_end)
                loc_start = max(0, loc_end - 20)
                window = close.iloc[loc_start:loc_end + 1]
                if len(window) >= 5:
                    normed = (window / window.iloc[0] - 1) * 100
                    normed.index = range(len(normed))
                    all_windows.append(normed)
            except Exception:
                pass

        if all_windows:
            # Pad all windows to same length
            max_len = max(len(w) for w in all_windows)
            padded  = [w.reindex(range(max_len)).interpolate() for w in all_windows]
            avg_drift  = pd.concat(padded, axis=1).mean(axis=1)
            std_drift  = pd.concat(padded, axis=1).std(axis=1)
            x_days     = range(-len(avg_drift) + 1, 1)

            ax3.fill_between(x_days,
                              avg_drift - std_drift,
                              avg_drift + std_drift,
                              color='#4ecdc4', alpha=0.15)
            ax3.plot(x_days, avg_drift, color='#4ecdc4', linewidth=1.8,
                      label='Avg pre-earnings drift')
            ax3.axhline(0, color='white', linewidth=0.6, alpha=0.4)
            ax3.axvline(0, color='#fbbf24', linewidth=1.2, linestyle='--',
                         alpha=0.7, label='Earnings day')
            ax3.set_xlabel('Days before earnings announcement', color='white', fontsize=8)
            ax3.set_ylabel('Cumulative Return (%)', color='white', fontsize=8)
            ax3.set_title(f'Historical Pre-Earnings Price Drift Pattern\n'
                           f'(average of {len(all_windows)} past events)',
                           color='white', fontsize=10, fontweight='bold', pad=8)
            ax3.legend(fontsize=7.5, facecolor='#12121f', labelcolor='white',
                        edgecolor='#262640')
            ax3.tick_params(colors='#94a3b8', labelsize=7, length=0)
        else:
            ax3.text(0.5, 0.5, 'Insufficient price/earnings\ndate overlap for drift analysis',
                      ha='center', va='center', color='#94a3b8', fontsize=10,
                      transform=ax3.transAxes)
            ax3.set_title('Pre-Earnings Price Drift', color='white',
                           fontsize=10, fontweight='bold', pad=8)
    else:
        ax3.text(0.5, 0.5, 'Price data\nnot available',
                  ha='center', va='center', color='#94a3b8', fontsize=11,
                  transform=ax3.transAxes)
        ax3.set_title('Pre-Earnings Price Drift', color='white',
                       fontsize=10, fontweight='bold', pad=8)

    # ── Panel 4: Beat probability gauge ───────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#1a1a2e')

    # Stacked bar showing probability breakdown
    prob_beat    = pred.beat_probability
    prob_miss    = max(0, 1.0 - pred.beat_probability - 0.15)
    prob_inline  = max(0, 1.0 - prob_beat - prob_miss)

    bars_data = [
        ('BEAT',    prob_beat,   '#22c55e'),
        ('IN-LINE', prob_inline, '#fbbf24'),
        ('MISS',    prob_miss,   '#ef4444'),
    ]
    bottom = 0
    for label, val, color in bars_data:
        ax4.bar(0.5, val, bottom=bottom, color=color, alpha=0.85,
                 width=0.5, label=f'{label}: {val*100:.1f}%')
        if val > 0.05:
            ax4.text(0.5, bottom + val/2, f'{label}\n{val*100:.1f}%',
                      ha='center', va='center', fontsize=10, color='white',
                      fontweight='bold')
        bottom += val

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_xticks([])
    ax4.set_yticks([0, 0.4, 0.6, 1.0])
    ax4.set_yticklabels(['0%', '40%\n(MISS)', '60%\n(BEAT)', '100%'],
                          color='#94a3b8', fontsize=8)
    ax4.axhline(0.60, color='#22c55e', linewidth=1.0, linestyle='--', alpha=0.7)
    ax4.axhline(0.40, color='#ef4444', linewidth=1.0, linestyle='--', alpha=0.7)
    ax4.set_title(f'Beat Probability Breakdown\n'
                   f'Data Quality: {pred.data_quality}  |  '
                   f'{pred.n_quarters} quarters of history',
                   color='white', fontsize=10, fontweight='bold', pad=8)
    ax4.legend(fontsize=8, facecolor='#12121f', labelcolor='white',
                edgecolor='#262640', loc='lower right')
    ax4.tick_params(colors='#94a3b8', labelsize=7, length=0)

    os.makedirs('models', exist_ok=True)
    suffix   = pred.ticker.replace('.NS', '')
    out_path = f'models/earnings_predictor_{suffix}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"📊 Chart saved → {out_path}")
    if not save:
        plt.show()


# ── Scan mode table ───────────────────────────────────────

def _print_scan_table(predictions: list[EarningsPrediction]):
    emoji = {'BEAT': '🟢', 'MISS': '🔴', 'IN-LINE': '🟡',
             'INSUFFICIENT_DATA': '⚪'}
    print(f"\n{'='*76}")
    print(f"  UPCOMING EARNINGS SCAN — next {max((p.days_to_earnings or 0) for p in predictions)} days")
    print(f"{'='*76}")
    print(f"  {'Ticker':<16} {'Company':<22} {'Date':>12} {'Days':>5} "
          f"{'Pred':>8} {'BeatProb':>9} {'BeatRate':>9}")
    print(f"  {'─'*72}")
    for p in predictions:
        em = emoji.get(p.prediction, '⚪')
        print(f"  {p.ticker:<16} {p.company_name[:21]:<22} "
              f"{(p.next_earnings_date or 'Unknown'):>12} "
              f"{(p.days_to_earnings or 0):>5}  "
              f"{em}{p.prediction:<8} "
              f"{p.beat_probability*100:>8.1f}%  "
              f"{p.historical_beat_rate*100:>8.1f}%")
    print(f"{'='*76}\n")


# ── Entry point ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='QuantAI Earnings Predictor — Beat/Miss Estimator'
    )
    parser.add_argument('--ticker', default='RELIANCE.NS',
                        help='Ticker for single-stock analysis (default: RELIANCE.NS)')
    parser.add_argument('--scan',   action='store_true',
                        help='Scan all 50 stocks for upcoming earnings')
    parser.add_argument('--days',   type=int, default=14,
                        help='Look-ahead window for scan mode (default: 14 days)')
    parser.add_argument('--save',   action='store_true',
                        help='Save chart without showing window')
    args = parser.parse_args()

    predictor = EarningsPredictor()

    if args.scan:
        results = predictor.scan_upcoming(days_ahead=args.days)
        if results:
            _print_scan_table(results)
            for r in results:
                predictor.print_prediction(r)
        else:
            print(f"\n  ℹ️  No earnings found in the next {args.days} days.\n")
    else:
        ticker = args.ticker
        print(f"\n⚙️  Running earnings prediction for {ticker}...\n")
        pred = predictor.predict(ticker)
        predictor.print_prediction(pred)
        _plot_single(pred, predictor, save=args.save)


if __name__ == '__main__':
    main()