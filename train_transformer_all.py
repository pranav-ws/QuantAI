"""
train_transformer_all.py
Trains the Attention Transformer for all 50 Nifty stocks.
Run: python train_transformer_all.py
Expected time: 8-15 minutes (faster than SeqNN because features are smaller).
"""
import os, time
from src.features import get_feature_dataset
from src.transformer_model import (prepare_transformer_data, train_transformer,
                                    evaluate_model, save_model)
from src.data_collector import STOCK_UNIVERSE

print("\n" + "="*64)
print("  QuantAI — Transformer Training | All Nifty 50 Stocks")
print("  Self-attention learns which past days matter most")
print("="*64)

results   = {}
total_t   = time.time()

for i, ticker in enumerate(STOCK_UNIVERSE, 1):
    t0 = time.time()
    print(f"\n🔮 [{i:02d}/{len(STOCK_UNIVERSE)}] {ticker}")
    try:
        df = get_feature_dataset(ticker)
        X_train, X_test, y_train, y_test, scaler, _ = prepare_transformer_data(df)
        model = train_transformer(X_train, y_train)
        acc, _, _ = evaluate_model(model, X_test, y_test, ticker)
        save_model(model, scaler, ticker)
        results[ticker] = acc
        remaining = (len(STOCK_UNIVERSE) - i) * (time.time() - t0)
        print(f"  ⏱  {time.time()-t0:.0f}s  |  ~{remaining/60:.1f} min left")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        results[ticker] = None

elapsed = time.time() - total_t
print("\n" + "="*64)
print(f"  TRANSFORMER TRAINING COMPLETE  ({elapsed/60:.1f} min)")
print("="*64)
print(f"  {'Ticker':<22} {'Transformer':>12}")
print(f"  {'-'*36}")

good = 0
for ticker, acc in results.items():
    if acc:
        bar  = '█' * int(acc * 50)
        icon = '✅' if acc >= 0.52 else '⚠️ '
        print(f"  {icon} {ticker:<20} {acc:>11.2%}  {bar}")
        if acc >= 0.52: good += 1
    else:
        print(f"  ❌ {ticker:<20} {'FAILED':>12}")

print(f"\n  ✅ {good}/{len(results)} stocks accuracy >= 52%")
print("  Transformer now auto-included in the ensemble.")
print("="*64 + "\n")
