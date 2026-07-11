import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from src.features import get_feature_dataset
from src.model import (prepare_data, time_series_split,
                       train_random_forest, evaluate_model,
                       get_feature_importance, save_model)

TICKER = 'RELIANCE.NS'

print(f"\n🚀 Training ML model for {TICKER}\n")

# ── 1. Load feature dataset ──────────────────────────────
df = get_feature_dataset(TICKER)

# ── 2. Prepare X and y ───────────────────────────────────
X, y = prepare_data(df)
print(f"\n📊 Dataset: {len(X)} rows × {len(X.columns)} features")
print(f"   Target balance — UP: {y.sum()} days | DOWN: {(y==0).sum()} days\n")

# ── 3. Time-series split (80% train / 20% test) ──────────
print("📅 Chronological Train/Test Split:")
X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)

# ── 4. Train Random Forest ───────────────────────────────
print("\n⚙️  Training Random Forest (200 trees)...")
model = train_random_forest(X_train, y_train)
print("✅ Training complete!")

# ── 5. Evaluate ──────────────────────────────────────────
acc, y_pred, y_proba = evaluate_model(model, X_test, y_test, TICKER)

# ── 6. Save model ────────────────────────────────────────
save_model(model, TICKER)

# ── 7. Feature importance ────────────────────────────────
importance = get_feature_importance(model, top_n=10)
print("🏆 Top 10 Most Important Features:")
for feat, score in importance.items():
    bar = '█' * int(score * 200)
    print(f"  {feat:<20} {bar} {score:.4f}")

# ── 8. Plot results ──────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#0d0d1a')
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# Panel 1: Actual vs Predicted direction on price chart
ax1 = fig.add_subplot(gs[0, :])
test_dates  = X_test.index
test_prices = df.loc[test_dates, 'Close']

ax1.plot(test_dates, test_prices, color='white', linewidth=1.0, label='Close Price', zorder=2)

# Mark correct UP predictions in green, wrong in red
for i, (date, actual, pred) in enumerate(zip(test_dates, y_test, y_pred)):
    if pred == 1 and actual == 1:
        ax1.axvline(date, color='lime',  alpha=0.3, linewidth=0.5)
    elif pred == 1 and actual == 0:
        ax1.axvline(date, color='red',   alpha=0.3, linewidth=0.5)

ax1.set_title('Test Period — Green: Correct UP call | Red: Wrong UP call',
              color='white', fontsize=11)
ax1.set_facecolor('#1a1a2e')
ax1.tick_params(colors='white')
ax1.legend(fontsize=9)

# Panel 2: Feature importance bar chart
ax2 = fig.add_subplot(gs[1, 0])
colors = ['#4ecdc4' if i < 3 else '#ff6b6b' for i in range(len(importance))]
ax2.barh(importance.index[::-1], importance.values[::-1], color=colors[::-1])
ax2.set_title('Top 10 Feature Importances', color='white', fontsize=11)
ax2.set_facecolor('#1a1a2e')
ax2.tick_params(colors='white')
ax2.set_xlabel('Importance Score', color='white')

# Panel 3: Prediction confidence distribution
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(y_proba[y_test == 1], bins=20, alpha=0.6,
         color='lime', label='Actual UP days')
ax3.hist(y_proba[y_test == 0], bins=20, alpha=0.6,
         color='red',  label='Actual DOWN days')
ax3.axvline(0.5, color='white', linestyle='--', linewidth=1.5, label='Decision boundary')
ax3.set_title('Prediction Confidence Distribution', color='white', fontsize=11)
ax3.set_xlabel('P(UP)', color='white')
ax3.set_ylabel('Count', color='white')
ax3.legend(fontsize=8)
ax3.set_facecolor('#1a1a2e')
ax3.tick_params(colors='white')

plt.suptitle(f'QuantAI — Random Forest Model Results ({TICKER})',
             color='white', fontsize=13, y=1.01)
plt.savefig('models/model_results.png', dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.show()
print("\n📊 Chart saved → models/model_results.png")
