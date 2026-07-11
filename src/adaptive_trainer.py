"""
src/adaptive_trainer.py

Adaptive retraining engine for QuantAI.
Retrains RF + XGBoost models when drift is detected.
"""

import os
import json
import joblib# type: ignore
import numpy as np# type: ignore
from datetime import datetime
from src.features import get_feature_dataset
from src.model import (prepare_data, time_series_split,
                       train_random_forest, evaluate_model,
                       save_model, FEATURE_COLS)
from src.drift_detector import (
    mark_retrained,
    _load_json,
    _save_json,
    DRIFT_LOG,
    RETRAIN_LOG
)

# RETRAIN_LOG = os.path.join('data', 'retrain_history.json')

def retrain_single_model(ticker, reason='scheduled'):
    """
    Retrains RF + XGBoost for one ticker.
    Returns accuracy delta vs old model.
    """
    print(f"\n  Retraining: {ticker} (reason: {reason})")

    try:
        df = get_feature_dataset(ticker)
        if len(df) < 100:
            print(f"  Skipped — not enough data")
            return None

        X, y = prepare_data(df)
        X_train, X_test, y_train, y_test = \
            time_series_split(X, y, test_size=0.2)

        # Get OLD accuracy before retraining
        old_acc = None
        rf_path = os.path.join('models',
                               f'{ticker}_rf_model.pkl')
        if os.path.exists(rf_path):
            try:
                old_rf   = joblib.load(rf_path)
                old_preds= old_rf.predict(X_test)
                from sklearn.metrics import accuracy_score # type: ignore
                old_acc  = accuracy_score(y_test, old_preds)
            except Exception:
                pass

        # Retrain RF
        print(f"  Training Random Forest...", end=' ')
        new_rf  = train_random_forest(X_train, y_train)
        new_acc, _, _ = evaluate_model(
            new_rf, X_test, y_test, ticker
        )
        save_model(new_rf, ticker)
        print(f"Accuracy: {new_acc:.2%}")

        # Retrain XGBoost
        xgb_path = os.path.join('models',
                                f'{ticker}_xgb_model.pkl')
        if os.path.exists(xgb_path):
            try:
                print(f"  Training XGBoost...", end=' ')
                from src.xgboost_model import (
                    train_xgboost, save_xgboost
                )
                new_xgb = train_xgboost(X_train, y_train)
                save_xgboost(new_xgb, ticker)
                print(f"Done")
            except Exception as e:
                print(f"XGB skip: {e}")

        # Log retraining
        acc_delta = ((new_acc - old_acc)
                     if old_acc else None)
        log_entry = {
            'ticker'     : ticker,
            'reason'     : reason,
            'old_acc'    : round(old_acc, 4) if old_acc else None,
            'new_acc'    : round(new_acc, 4),
            'acc_delta'  : round(acc_delta, 4) if acc_delta else None,
            'retrained_at': datetime.now().isoformat(),
            'n_train_rows': len(X_train),
            'n_test_rows' : len(X_test),
        }

        history = _load_json(RETRAIN_LOG, [])
        history.append(log_entry)
        history = history[-200:]
        _save_json(RETRAIN_LOG, history)

        mark_retrained(ticker)

        improved = acc_delta > 0 if acc_delta else None
        print(f"  Result: {old_acc:.1%} → {new_acc:.1%} "
              f"({'improved' if improved else 'declined'})"
              if old_acc else
              f"  Result: {new_acc:.1%}")

        return log_entry

    except Exception as e:
        print(f"  Retrain failed for {ticker}: {e}")
        return None

def run_adaptive_retraining(
    max_retrain=5,
    force_all=False,
    reason='drift_detected'
):
    """
    Main adaptive retraining function.

    Args:
      max_retrain: max stocks to retrain per run
      force_all  : ignore drift check, retrain everything
      reason     : why retraining is happening
    """
    from src.drift_detector import (
        check_all_drift, get_retrain_queue,
        add_to_retrain_queue, update_outcomes
    )
    from src.data_collector import STOCK_UNIVERSE

    print(f"\n{'='*58}")
    print(f"  QuantAI Adaptive Retraining")
    print(f"  {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    print(f"{'='*58}")

    # Step 1: Update prediction outcomes
    print(f"\n  Updating prediction outcomes...")
    updated = update_outcomes()
    print(f"  Resolved {updated} predictions")

    # Step 2: Check for drift
    if force_all:
        to_retrain = [{'ticker': t, 'reasons': ['forced']}
                      for t in STOCK_UNIVERSE]
    else:
        to_retrain, results, summary = check_all_drift()

        # Add to queue
        if to_retrain:
            tickers = [r['ticker'] for r in to_retrain]
            add_to_retrain_queue(tickers, reason)

        # Also grab pending queue items
        pending = get_retrain_queue()
        existing_tickers = {r['ticker'] for r in to_retrain}
        for q in pending:
            if q['ticker'] not in existing_tickers:
                to_retrain.append({
                    'ticker' : q['ticker'],
                    'reasons': [q['reason']],
                })

    # Step 3: Retrain (limit per run)
    if not to_retrain:
        print(f"\n  No drift detected — all models healthy")
        print(f"{'='*58}\n")
        return []

    print(f"\n  Retraining {min(len(to_retrain), max_retrain)}"
          f" model(s)...")

    retrained = []
    for item in to_retrain[:max_retrain]:
        ticker  = item['ticker']
        reasons = item.get('reasons', ['drift'])
        result  = retrain_single_model(
            ticker,
            reason=', '.join(reasons)
        )
        if result:
            retrained.append(result)

    # Step 4: Update regime record
    try:
        from src.regime_detector import detect_regime
        regime_data = detect_regime()
        drift_log   = _load_json(DRIFT_LOG, {})
        drift_log['last_train_regime'] = \
            regime_data.get('regime', 'UNKNOWN')
        drift_log['last_retrain_at']   = \
            datetime.now().isoformat()
        _save_json(DRIFT_LOG, drift_log)
    except Exception:
        pass

    # Step 5: Summary
    print(f"\n{'='*58}")
    print(f"  Retraining Complete")
    print(f"{'='*58}")
    print(f"  Models retrained : {len(retrained)}")
    for r in retrained:
        delta = r.get('acc_delta')
        delta_str = (f" ({delta:+.1%})" if delta else "")
        print(f"  {r['ticker'].replace('.NS',''):<18} "
              f"{r['new_acc']:.1%}{delta_str}")
    print(f"{'='*58}\n")

    return retrained