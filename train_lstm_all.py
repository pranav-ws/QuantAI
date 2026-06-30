"""
train_lstm_all.py
Trains a Sequence Neural Network (MLP on 30-day sequences) for every
stock in the Nifty 50 universe.

No TensorFlow required — uses scikit-learn which is already installed.
Expected time: 5–12 minutes on CPU.
"""
import os, time
from src.features import get_feature_dataset
from src.lstm_model import (prepare_lstm_data, train_lstm,
                             evaluate_model, save_model, FEATURE_COLS)
from src.data_collector import STOCK_UNIVERSE

print("\n" + "="*62)
print("  QuantAI — SeqNN Training | All Nifty 50 Stocks")
print("  (Sequence Neural Network — no TensorFlow needed)")
print("  ⏱  Estimated time: 5–12 minutes")
print("="*62)

results    = {}
total_t    = time.time()
n_features = len(FEATURE_COLS)

for i, ticker in enumerate(STOCK_UNIVERSE, 1):
    t0 = time.time()
    print(f"\n🧠 [{i:02d}/{len(STOCK_UNIVERSE)}] {ticker}")
    try:
        df = get_feature_dataset(ticker)
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df)
        model = train_lstm(X_train, y_train, n_features=n_features)
        acc, _, _ = evaluate_model(model, X_test, y_test, ticker)
        save_model(model, scaler, ticker)
        results[ticker] = acc
        elapsed = time.time() - t0
        remaining = (len(STOCK_UNIVERSE) - i) * elapsed
        print(f"  ⏱  Done in {elapsed:.0f}s  |  ~{remaining/60:.1f} min remaining")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        results[ticker] = None

total_elapsed = time.time() - total_t

# ── Final summary ─────────────────────────────────────────
print("\n" + "="*62)
print(f"  LSTM TRAINING COMPLETE  ({total_elapsed/60:.1f} min total)")
print("="*62)
print(f"  {'Ticker':<22} {'LSTM Accuracy':>14}")
print(f"  {'-'*38}")

good = 0
for ticker, acc in results.items():
    if acc:
        bar  = '█' * int(acc * 50)
        icon = '✅' if acc >= 0.52 else '⚠️ '
        print(f"  {icon} {ticker:<20} {acc:>13.2%}  {bar}")
        if acc >= 0.52: good += 1
    else:
        print(f"  ❌ {ticker:<20} {'FAILED':>14}")

print(f"\n  ✅ {good}/{len(results)} stocks with accuracy >= 52%")
print("  API and paper trader now automatically use SeqNN signals.")
print("  Run:  python compare_all_models.py  to see the full comparison.")
print("="*62 + "\n")
