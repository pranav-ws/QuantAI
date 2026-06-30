from src.features import get_feature_dataset
from src.model import (prepare_data, time_series_split,
                       train_random_forest, evaluate_model,
                       save_model)
from src.data_collector import STOCK_UNIVERSE

results = {}

print("\n" + "="*55)
print("  QuantAI — Training RF Models for All Nifty 50 Stocks")
print("="*55)

for ticker in STOCK_UNIVERSE:
    try:
        print(f"\n🚀 Training: {ticker}")

        # 1. Load features
        df = get_feature_dataset(ticker)

        # 2. Prepare data
        X, y = prepare_data(df)

        # 3. Time-series split
        X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)

        # 4. Train
        model = train_random_forest(X_train, y_train)

        # 5. Evaluate
        acc, y_pred, y_proba = evaluate_model(model, X_test, y_test, ticker)

        # 6. Save
        save_model(model, ticker)

        results[ticker] = acc

    except Exception as e:
        print(f"  ❌ {ticker} failed: {e}")
        results[ticker] = None

# ── Final summary ────────────────────────────────────────
print("\n" + "="*55)
print("  TRAINING COMPLETE — All Model Accuracies")
print("="*55)
print(f"  {'Ticker':<22} {'Accuracy':>10}")
print(f"  {'-'*34}")
for ticker, acc in results.items():
    if acc:
        bar    = '█' * int(acc * 50)
        status = '✅' if acc >= 0.52 else '⚠️ '
        print(f"  {status} {ticker:<20} {acc:>8.2%}  {bar}")
    else:
        print(f"  ❌ {ticker:<20} {'FAILED':>10}")
print("="*55 + "\n")
