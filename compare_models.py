"""
compare_models.py
Loads saved RF and XGBoost models for all 50 stocks,
evaluates both on the same test set, and plots a
horizontal grouped bar chart sorted by XGBoost accuracy.
"""
import os, joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.features import get_feature_dataset
from src.model import prepare_data, time_series_split
from src.data_collector import STOCK_UNIVERSE

rf_scores  = {}
xgb_scores = {}

print(f"\n⚙️  Evaluating all {len(STOCK_UNIVERSE)} models on test sets...\n")

for ticker in STOCK_UNIVERSE:
    try:
        df = get_feature_dataset(ticker)
        X, y = prepare_data(df)
        _, X_test, _, y_test = time_series_split(X, y, test_size=0.2)

        rf_path  = os.path.join('models', f'{ticker}_rf_model.pkl')
        xgb_path = os.path.join('models', f'{ticker}_xgb_model.pkl')

        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)
            rf_scores[ticker] = accuracy_score(y_test, rf_model.predict(X_test))

        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            xgb_scores[ticker] = accuracy_score(y_test, xgb_model.predict(X_test))

    except Exception as e:
        print(f"  ⚠️  {ticker}: {e}")

# ── Print summary table ───────────────────────────────────
tickers     = list(STOCK_UNIVERSE.keys())
short_names = [t.replace('.NS','') for t in tickers]

print(f"\n{'='*62}")
print(f"  {'Ticker':<18} {'Sector':<18} {'RF':>8} {'XGBoost':>9} {'Winner':>8}")
print(f"  {'-'*56}")

xgb_wins = rf_wins = ties = 0
for t in tickers:
    rf  = rf_scores.get(t)
    xgb = xgb_scores.get(t)
    sector = STOCK_UNIVERSE[t][1]
    if rf and xgb:
        if xgb > rf:
            winner = '🤖 XGB'; xgb_wins += 1
        elif rf > xgb:
            winner = '🌲 RF '; rf_wins  += 1
        else:
            winner = '🤝 TIE'; ties     += 1
        print(f"  {t:<18} {sector:<18} {rf:>7.2%} {xgb:>9.2%} {winner}")
    elif rf:
        print(f"  {t:<18} {sector:<18} {rf:>7.2%} {'N/A':>9}")
    elif xgb:
        print(f"  {t:<18} {sector:<18} {'N/A':>7} {xgb:>9.2%}")

print(f"{'='*62}")
print(f"  XGBoost wins: {xgb_wins}  |  RF wins: {rf_wins}  |  Ties: {ties}")
avg_xgb = sum(xgb_scores.values()) / len(xgb_scores) if xgb_scores else 0
avg_rf  = sum(rf_scores.values())  / len(rf_scores)  if rf_scores  else 0
print(f"  Average accuracy — XGBoost: {avg_xgb:.2%}  |  RF: {avg_rf:.2%}")
print(f"{'='*62}\n")

# ── Plot: horizontal grouped bars sorted by XGB accuracy ─
# Sort by XGBoost score descending
sorted_tickers = sorted(
    [t for t in tickers if t in xgb_scores],
    key=lambda t: xgb_scores.get(t, 0),
    reverse=True
)
names   = [t.replace('.NS','') for t in sorted_tickers]
xgb_vals = [xgb_scores.get(t, 0) for t in sorted_tickers]
rf_vals  = [rf_scores.get(t, 0)  for t in sorted_tickers]

n   = len(names)
fig, ax = plt.subplots(figsize=(12, max(8, n * 0.38)))
fig.patch.set_facecolor('#0d0d1a')
ax.set_facecolor('#1a1a2e')

y   = range(n)
h   = 0.36

bars_xgb = ax.barh([i + h/2 for i in y], xgb_vals, height=h,
                    color='#4ecdc4', alpha=0.88, label='XGBoost')
bars_rf  = ax.barh([i - h/2 for i in y], rf_vals,  height=h,
                    color='#ff6b6b', alpha=0.88, label='Random Forest')

# Value labels
for i, (xv, rv) in enumerate(zip(xgb_vals, rf_vals)):
    ax.text(xv + 0.002, i + h/2, f'{xv:.1%}', va='center',
            fontsize=7.5, color='#4ecdc4', fontfamily='monospace')
    if rv:
        ax.text(rv + 0.002, i - h/2, f'{rv:.1%}', va='center',
                fontsize=7.5, color='#ff6b6b', fontfamily='monospace')

ax.axvline(0.50, color='white', linewidth=0.8, linestyle='--',
           alpha=0.4, label='50% baseline')
ax.axvline(0.55, color='#fbbf24', linewidth=0.8, linestyle=':',
           alpha=0.6, label='55% edge threshold')

ax.set_yticks(list(y))
ax.set_yticklabels(names, color='white', fontfamily='monospace', fontsize=9)
ax.set_xlabel('Test Accuracy', color='white', fontsize=10)
ax.set_xlim(0.44, 0.78)
ax.tick_params(colors='white')
ax.invert_yaxis()
ax.legend(fontsize=9, facecolor='#12121f', labelcolor='white',
          edgecolor='#262640', loc='lower right')
ax.set_title(f'XGBoost vs Random Forest — All {n} Nifty 50 Stocks',
             color='white', fontsize=13, pad=14)

plt.tight_layout()
plt.savefig('models/model_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.show()
print("📊 Chart saved → models/model_comparison.png")
