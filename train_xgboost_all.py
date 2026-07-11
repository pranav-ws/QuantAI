from src.features import get_feature_dataset
from src.model import prepare_data, time_series_split
from src.xgboost_model import (train_xgboost, evaluate_model,
                                get_feature_importance, save_model)
from src.data_collector import STOCK_UNIVERSE
import joblib, os

print("\n" + "="*60)
print("  QuantAI — Training XGBoost Models for All Nifty 50 Stocks")
print("="*60)

xgb_results = {}

for ticker in STOCK_UNIVERSE:
    try:
        print(f"\n🚀 Training XGBoost: {ticker}")

        df = get_feature_dataset(ticker)
        X, y = prepare_data(df)
        X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)

        model = train_xgboost(X_train, y_train)
        acc, _, _ = evaluate_model(model, X_test, y_test, ticker)
        save_model(model, ticker)
        xgb_results[ticker] = acc

    except Exception as e:
        print(f"  ❌ {ticker} failed: {e}")
        xgb_results[ticker] = None

# ── Side-by-side comparison with Random Forest ───────────
print("\n" + "="*60)
print("  XGBoost vs Random Forest — Accuracy Comparison")
print("="*60)
print(f"  {'Ticker':<22} {'XGBoost':>10} {'Rand.Forest':>12} {'Delta':>8}")
print(f"  {'-'*54}")

for ticker, xgb_acc in xgb_results.items():
    rf_path = os.path.join('models', f'{ticker}_rf_model.pkl')
    rf_acc = None
    if os.path.exists(rf_path):
        try:
            from src.features import get_feature_dataset
            from src.model import prepare_data, time_series_split, FEATURE_COLS
            from sklearn.metrics import accuracy_score
            rf_model = joblib.load(rf_path)
            df = get_feature_dataset(ticker)
            X, y = prepare_data(df)
            _, X_test, _, y_test = time_series_split(X, y, test_size=0.2)
            rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
        except Exception:
            pass

    if xgb_acc and rf_acc:
        delta = xgb_acc - rf_acc
        symbol = '✅' if xgb_acc >= rf_acc else '⚠️ '
        bar = '█' * int(xgb_acc * 50)
        print(f"  {symbol} {ticker:<20} {xgb_acc:>9.2%} {rf_acc:>12.2%} {delta:>+8.2%}")
    elif xgb_acc:
        bar = '█' * int(xgb_acc * 50)
        print(f"  ✅ {ticker:<20} {xgb_acc:>9.2%} {'N/A':>12} {'N/A':>8}")
    else:
        print(f"  ❌ {ticker:<20} {'FAILED':>9}")

print("="*60)
print("\n✅ XGBoost training complete.")
print("   API and paper trader now automatically use XGBoost signals.")
print("   Run:  python paper_trade.py")
print("   or:   uvicorn src.api:app --reload --port 8000\n")
