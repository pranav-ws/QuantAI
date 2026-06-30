"""
compare_all_models.py
3-way comparison: Random Forest vs XGBoost vs LSTM across all 50 stocks.
Generates a 3-panel figure and prints a summary table.
"""
import os, joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.features import get_feature_dataset
from src.model import prepare_data, time_series_split
from src.lstm_model import prepare_lstm_data
from src.data_collector import STOCK_UNIVERSE

rf_scores   = {}
xgb_scores  = {}
lstm_scores = {}

print(f"\n⚙️  Evaluating all 3 models for {len(STOCK_UNIVERSE)} stocks...\n")

for i, ticker in enumerate(STOCK_UNIVERSE, 1):
    print(f"  [{i:02d}/{len(STOCK_UNIVERSE)}] {ticker}", end=' ', flush=True)
    try:
        df = get_feature_dataset(ticker)
        X, y = prepare_data(df)
        _, X_test, _, y_test = time_series_split(X, y, test_size=0.2)

        # Random Forest
        rf_path = os.path.join('models', f'{ticker}_rf_model.pkl')
        if os.path.exists(rf_path):
            m = joblib.load(rf_path)
            rf_scores[ticker] = accuracy_score(y_test, m.predict(X_test))

        # XGBoost
        xgb_path = os.path.join('models', f'{ticker}_xgb_model.pkl')
        if os.path.exists(xgb_path):
            m = joblib.load(xgb_path)
            xgb_scores[ticker] = accuracy_score(y_test, m.predict(X_test))

        # SeqNN (sklearn — no TF required)
        seqnn_path  = os.path.join('models', f'{ticker}_seqnn_model.pkl')
        scaler_path = os.path.join('models', f'{ticker}_seqnn_scaler.pkl')
        if os.path.exists(seqnn_path) and os.path.exists(scaler_path):
            seqnn_m = joblib.load(seqnn_path)
            scaler  = joblib.load(scaler_path)
            _, X_seqnn, _, y_seqnn, _ = prepare_lstm_data(df)
            y_p = seqnn_m.predict_proba(X_seqnn)[:, 1]
            lstm_scores[ticker] = accuracy_score(y_seqnn, (y_p >= 0.58).astype(int))

        # TF LSTM fallback (only if SeqNN not available)
        if ticker not in lstm_scores:
            lstm_path   = os.path.join('models', f'{ticker}_lstm_model.h5')
            lstm_scaler = os.path.join('models', f'{ticker}_lstm_scaler.pkl')
            if os.path.exists(lstm_path) and os.path.exists(lstm_scaler):
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR')          # suppress absl warning
                lstm_m = tf.keras.models.load_model(lstm_path, compile=False)
                _, X_lstm, _, y_lstm, _ = prepare_lstm_data(df)
                y_p = lstm_m.predict(X_lstm, verbose=0).flatten()
                lstm_scores[ticker] = accuracy_score(y_lstm, (y_p >= 0.58).astype(int))

        print("✓")
    except Exception as e:
        print(f"✗ ({e})")

# ── Summary table ─────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  {'Ticker':<18} {'Sector':<16} {'RF':>8} {'XGBoost':>9} {'SeqNN':>8} {'Best':>10}")
print(f"  {'-'*66}")

model_wins = {'RF': 0, 'XGBoost': 0, 'SeqNN': 0}
for ticker in STOCK_UNIVERSE:
    rf  = rf_scores.get(ticker)
    xgb = xgb_scores.get(ticker)
    lst = lstm_scores.get(ticker)
    sector = STOCK_UNIVERSE[ticker][1]

    scores = {k: v for k, v in [('RF', rf), ('XGBoost', xgb), ('SeqNN', lst)] if v}
    if scores:
        best_name = max(scores, key=scores.get)
        best_val  = scores[best_name]
        if best_name in model_wins:
            model_wins[best_name] += 1
    else:
        best_name, best_val = 'N/A', 0

    print(f"  {ticker:<18} {sector:<16}"
          f" {rf_scores.get(ticker, 0):>7.2%}"
          f" {xgb_scores.get(ticker, 0):>9.2%}"
          f" {lstm_scores.get(ticker, 0):>8.2%}"
          f"  {best_name}")

print(f"{'='*72}")
print(f"  Wins — RF: {model_wins['RF']} | XGBoost: {model_wins['XGBoost']} | SeqNN: {model_wins['SeqNN']}")

avgs = {}
for name, d in [('RF', rf_scores), ('XGBoost', xgb_scores), ('SeqNN', lstm_scores)]:
    if d: avgs[name] = sum(d.values()) / len(d)
print("  Average accuracy — " + " | ".join(f"{k}: {v:.2%}" for k, v in avgs.items()))
print(f"{'='*72}\n")

# ── 3-panel chart ─────────────────────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor='#0d0d1a')
gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

# --- Panel 1: Average accuracy per model ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#1a1a2e')
model_names = list(avgs.keys())
avg_vals    = [avgs[m] for m in model_names]
colors_bar  = ['#ff6b6b', '#4ecdc4', '#a78bfa']
bars = ax1.bar(model_names, avg_vals, color=colors_bar[:len(model_names)], alpha=0.88, width=0.45)
for bar, val in zip(bars, avg_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.003,
             f'{val:.2%}', ha='center', va='bottom',
             color='white', fontfamily='monospace', fontsize=11, fontweight='bold')
ax1.axhline(0.50, color='white',   linewidth=0.8, linestyle='--', alpha=0.4, label='50% baseline')
ax1.axhline(0.55, color='#fbbf24', linewidth=0.8, linestyle=':', alpha=0.7, label='55% edge')
ax1.set_ylim(0.43, 0.75)
ax1.set_title('Average Accuracy by Model', color='white', fontsize=12, pad=12)
ax1.set_ylabel('Test Accuracy', color='white', fontsize=10)
ax1.tick_params(colors='white')
ax1.legend(fontsize=8, facecolor='#12121f', labelcolor='white', edgecolor='#262640')

# --- Panel 2: Win count pie chart ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#1a1a2e')
pie_labels = [k for k, v in model_wins.items() if v > 0]
pie_vals   = [model_wins[k] for k in pie_labels]
pie_colors = [{'RF': '#ff6b6b', 'XGBoost': '#4ecdc4', 'SeqNN': '#a78bfa'}[k] for k in pie_labels]
wedges, texts, autotexts = ax2.pie(
    pie_vals, labels=pie_labels, autopct='%1.0f%%',
    colors=pie_colors, startangle=90,
    textprops={'color': 'white', 'fontsize': 11})
for at in autotexts: at.set_fontsize(10)
ax2.set_title(f'Best Model Win Rate ({len(STOCK_UNIVERSE)} stocks)', color='white', fontsize=12, pad=12)

# --- Panel 3: Per-stock grouped bars sorted by SeqNN ---
ax3 = fig.add_subplot(gs[1, :])
ax3.set_facecolor('#1a1a2e')

sorted_t = sorted(
    [t for t in STOCK_UNIVERSE if t in lstm_scores],
    key=lambda t: lstm_scores.get(t, 0), reverse=True)
names = [t.replace('.NS','') for t in sorted_t]
n = len(names)
x = np.arange(n)
w = 0.26

ax3.bar(x - w, [rf_scores.get(t, 0)   for t in sorted_t], width=w, color='#ff6b6b', alpha=0.85, label='Random Forest')
ax3.bar(x,     [xgb_scores.get(t, 0)  for t in sorted_t], width=w, color='#4ecdc4', alpha=0.85, label='XGBoost')
ax3.bar(x + w, [lstm_scores.get(t, 0) for t in sorted_t], width=w, color='#a78bfa', alpha=0.85, label='SeqNN')
ax3.axhline(0.50, color='white',   linewidth=0.6, linestyle='--', alpha=0.3)
ax3.axhline(0.55, color='#fbbf24', linewidth=0.6, linestyle=':', alpha=0.5)
ax3.set_xticks(x)
ax3.set_xticklabels(names, rotation=45, ha='right',
                    color='white', fontfamily='monospace', fontsize=7.5)
ax3.set_ylim(0.43, 0.78)
ax3.set_ylabel('Test Accuracy', color='white', fontsize=10)
ax3.set_title('RF vs XGBoost vs SeqNN — All 50 Stocks (sorted by SeqNN accuracy)',
              color='white', fontsize=12, pad=12)
ax3.tick_params(colors='white')
ax3.legend(fontsize=9, facecolor='#12121f', labelcolor='white', edgecolor='#262640')

plt.suptitle('QuantAI — Full Model Comparison', color='white', fontsize=15, y=1.01)
plt.savefig('models/all_models_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.show()
print("📊 Chart saved → models/all_models_comparison.png")
