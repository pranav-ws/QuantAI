"""
check_drift.py
Daily drift check — run this first thing each day.
Shows model health and triggers retraining if needed.
"""
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
import numpy as np # type: ignore
import json, os

from src.drift_detector import (
    update_outcomes, check_accuracy_drift,
    check_regime_drift, _load_json,
    PRED_LOG, RETRAIN_LOG, DRIFT_LOG
)
from src.adaptive_trainer import run_adaptive_retraining

# ── Update outcomes ───────────────────────────────────────
print("\n" + "="*58)
print("  QuantAI — Model Drift Monitor")
print("="*58)

print("\n  Updating prediction outcomes...")
updated = update_outcomes()
print(f"  Resolved: {updated} predictions\n")

# ── Check global accuracy ─────────────────────────────────
drift, accuracy, details = check_accuracy_drift()
print(f"  Rolling Accuracy (last 20 days):")
if accuracy:
    bar = '█' * int(accuracy * 30)
    spc = '░' * (30 - int(accuracy * 30))
    status = '🔴 DRIFT' if drift else '🟢 HEALTHY'
    print(f"  [{bar}{spc}] {accuracy:.1%}  {status}")
    print(f"  Predictions evaluated : "
          f"{details.get('n_predictions', 0)}")
    print(f"  Correct predictions   : "
          f"{details.get('n_correct', 0)}")
else:
    print(f"  Not enough predictions yet "
          f"({details.get('resolved', 0)} resolved)")

# ── Check regime ──────────────────────────────────────────
reg_drift, regime, reg_det = check_regime_drift()
print(f"\n  Market Regime: {regime}")
if reg_drift:
    print(f"  Regime changed: "
          f"{reg_det['previous_regime']} → {regime}")
    print(f"  Models trained in old regime may be stale")

# ── Retrain history ───────────────────────────────────────
history = _load_json(RETRAIN_LOG, [])
if history:
    recent = history[-10:]
    print(f"\n  Recent Retraining History:")
    print(f"  {'Ticker':<18} {'Old':>7} {'New':>7} "
          f"{'Delta':>8} {'When':<12}")
    print(f"  {'─'*55}")
    for r in reversed(recent):
        old   = f"{r['old_acc']:.1%}" if r.get('old_acc') else 'N/A'
        new   = f"{r['new_acc']:.1%}"
        delta = (f"{r['acc_delta']:+.1%}"
                 if r.get('acc_delta') else 'N/A')
        when  = r['retrained_at'][:10]
        color = '▲' if (r.get('acc_delta') or 0) > 0 else '▼'
        print(f"  {r['ticker'].replace('.NS',''):<18} "
              f"{old:>7} {new:>7} {color}{delta:>7} {when}")

# ── Run retraining if needed ──────────────────────────────
if drift or reg_drift:
    print(f"\n  Drift detected — starting retraining...")
    retrained = run_adaptive_retraining(
        max_retrain=5,
        reason='auto_drift_detection'
    )
else:
    print(f"\n  All models healthy — no retraining needed")

print(f"\n{'='*58}\n")

# ── Chart ─────────────────────────────────────────────────
pred_logs = _load_json(PRED_LOG, [])
resolved  = [p for p in pred_logs
             if p.get('correct') is not None]

if len(resolved) >= 10:
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#0d0d1a')
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)

    # Panel 1: Rolling accuracy over time
    ax1 = fig.add_subplot(gs[0, :2])
    window    = 20
    roll_accs = []
    dates_r   = []

    for i in range(window, len(resolved) + 1):
        window_preds = resolved[i-window:i]
        acc = np.mean([p['correct'] for p in window_preds])
        roll_accs.append(acc)
        dates_r.append(i)

    if roll_accs:
        colors_line = ['#4ecdc4' if a >= 0.48
                       else '#ff6b6b' for a in roll_accs]
        for i in range(len(roll_accs) - 1):
            ax1.plot([dates_r[i], dates_r[i+1]],
                     [roll_accs[i], roll_accs[i+1]],
                     color=colors_line[i], linewidth=2)

        ax1.axhline(0.48, color='#ff6b6b',
                    linewidth=1.5, linestyle='--',
                    label='Drift threshold (48%)')
        ax1.axhline(0.55, color='#4ecdc4',
                    linewidth=1.0, linestyle=':',
                    alpha=0.7, label='Target (55%)')
        ax1.fill_between(dates_r, roll_accs, 0.48,
                         where=[a >= 0.48 for a in roll_accs],
                         alpha=0.15, color='#4ecdc4')
        ax1.fill_between(dates_r, roll_accs, 0.48,
                         where=[a < 0.48 for a in roll_accs],
                         alpha=0.15, color='#ff6b6b')

        if history:
            for r in history[-5:]:
                idx = len(resolved)
                ax1.axvline(idx, color='#ffd700',
                            linewidth=1.5, linestyle=':',
                            alpha=0.7)
                ax1.text(idx, 0.52, 'Retrained',
                         rotation=90, fontsize=7,
                         color='#ffd700')

        ax1.set_ylim(0.35, 0.75)
        ax1.set_title('Rolling Prediction Accuracy '
                      '(20-prediction window)',
                      color='white', fontsize=11)
        ax1.set_xlabel('Prediction Number', color='white')
        ax1.set_ylabel('Accuracy', color='white')
        ax1.legend(fontsize=8)
        ax1.set_facecolor('#1a1a2e')
        ax1.tick_params(colors='white')
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.0%}')
        )

    # Panel 2: Accuracy improvement from retraining
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#1a1a2e')

    if history:
        tickers_h = [r['ticker'].replace('.NS', '')
                     for r in history[-8:]]
        deltas    = [(r.get('acc_delta') or 0) * 100
                     for r in history[-8:]]
        cols      = ['#4ecdc4' if d >= 0 else '#ff6b6b'
                     for d in deltas]

        bars = ax2.barh(tickers_h, deltas,
                        color=cols, edgecolor='none')
        ax2.axvline(0, color='white',
                    linewidth=0.8, alpha=0.5)
        for bar, delta in zip(bars, deltas):
            ax2.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height()/2,
                f'{delta:+.1f}%',
                va='center', color='white', fontsize=8
            )
        ax2.set_title('Accuracy Change After Retrain',
                      color='white', fontsize=11)
        ax2.set_facecolor('#1a1a2e')
        ax2.tick_params(colors='white', labelsize=8)
        ax2.set_xlabel('Accuracy Δ (%)', color='white')
    else:
        ax2.text(0.5, 0.5, 'No retraining\nhistory yet',
                 ha='center', va='center',
                 transform=ax2.transAxes,
                 color='#888888', fontsize=11)
        ax2.set_title('Accuracy Change After Retrain',
                      color='white', fontsize=11)

    plt.suptitle(
        'QuantAI — Adaptive Model Drift Monitor',
        color='white', fontsize=13, y=1.01
    )
    plt.savefig('models/drift_monitor.png', dpi=150,
                bbox_inches='tight', facecolor='#0d0d1a')
    plt.show()
    print("  Chart saved → models/drift_monitor.png")
else:
    print("  Not enough resolved predictions for chart yet.")
    print("  Run paper_trade.py daily and check back in 2 weeks.")