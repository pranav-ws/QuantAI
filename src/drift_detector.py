"""
src/drift_detector.py

Concept Drift Detection for QuantAI.

Detects when ML models are becoming stale and need retraining.

Three drift detection methods:

1. ACCURACY DRIFT
   Tracks rolling 20-day prediction accuracy.
   If it drops below threshold (48%) → trigger retrain.

2. DATA DRIFT (Population Stability Index)
   Compares feature distributions: train data vs recent data.
   PSI > 0.25 = significant distribution shift → retrain.

3. REGIME DRIFT
   If market regime changes (BULL→BEAR etc.) → retrain.
   Models trained in one regime perform poorly in another.

All three are checked daily. Any trigger = retrain scheduled.
"""

import numpy as np# type: ignore
import pandas as pd# type: ignore
import sqlite3
import json
import os
import joblib# type: ignore
from datetime import datetime, date, timedelta

DB_PATH        = os.path.join('data', 'quantai.db')
DRIFT_LOG      = os.path.join('data', 'drift_log.json')
PRED_LOG       = os.path.join('data', 'prediction_log.json')
RETRAIN_QUEUE  = os.path.join('data', 'retrain_queue.json')

# ADD THIS
RETRAIN_LOG    = os.path.join('data', 'retrain_history.json')

# ── Thresholds ────────────────────────────────────────────
ACCURACY_THRESHOLD  = 0.48   # below this = model degraded
PSI_THRESHOLD       = 0.25   # above this = data drift
ROLLING_WINDOW      = 20     # days to evaluate accuracy over
MIN_PREDICTIONS     = 10     # minimum predictions before checking

FEATURE_COLS = [
    'SMA_20', 'SMA_50', 'EMA_9', 'EMA_21',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'Stoch_K', 'Stoch_D',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
    'ATR_14', 'OBV', 'Volume_Ratio',
    'Daily_Return', 'Return_5d', 'Return_20d',
    'High_Low_Range', 'Close_vs_SMA20', 'Close_vs_SMA50'
]

# ── Prediction logger ─────────────────────────────────────
def log_prediction(ticker, predicted, confidence,
                   price, date_str=None):
    """
    Logs each prediction made by the ensemble.
    Called from paper_trader.py every time a signal is generated.
    Actual outcome verified next day when price is known.
    """
    if date_str is None:
        date_str = str(date.today())

    logs = _load_json(PRED_LOG, [])

    logs.append({
        'ticker'    : ticker,
        'date'      : date_str,
        'predicted' : int(predicted),
        'confidence': round(float(confidence), 4),
        'price'     : round(float(price), 2),
        'actual'    : None,   # filled in next day
        'correct'   : None,
    })

    # Keep only last 500 predictions
    logs = logs[-500:]
    _save_json(PRED_LOG, logs)

def update_outcomes():
    """
    Fills in actual outcomes for predictions made yesterday.
    Called at start of daily run — checks if yesterday's
    predicted UP/DOWN was actually correct.
    """
    logs = _load_json(PRED_LOG, [])
    if not logs:
        return 0

    updated = 0
    conn    = sqlite3.connect(DB_PATH)

    for pred in logs:
        if pred.get('actual') is not None:
            continue   # already resolved

        pred_date = pred.get('date', '')
        ticker    = pred.get('ticker', '')
        price_t0  = pred.get('price', 0)

        if not pred_date or not ticker or not price_t0:
            continue

        try:
            # Get next day's price
            next_day = pd.read_sql_query(
                """SELECT close FROM prices
                   WHERE ticker=? AND date > ?
                   ORDER BY date ASC LIMIT 1""",
                conn, params=(ticker, pred_date)
            )

            if next_day.empty:
                continue   # future not yet known

            price_t1 = float(next_day['close'].iloc[0])
            actual   = 1 if price_t1 > price_t0 else 0
            correct  = 1 if actual == pred['predicted'] else 0

            pred['actual']  = actual
            pred['correct'] = correct
            updated        += 1

        except Exception:
            continue

    conn.close()
    _save_json(PRED_LOG, logs)
    return updated

# ── Method 1: Accuracy drift ──────────────────────────────
def check_accuracy_drift(ticker=None):
    """
    Checks rolling accuracy of predictions.
    Returns (drift_detected, current_accuracy, details)
    """
    logs = _load_json(PRED_LOG, [])

    # Filter resolved predictions
    resolved = [p for p in logs
                if p.get('correct') is not None]

    if ticker:
        resolved = [p for p in resolved
                    if p['ticker'] == ticker]

    if len(resolved) < MIN_PREDICTIONS:
        return False, None, {
            'reason': f'Not enough predictions ({len(resolved)}'
                      f'/{MIN_PREDICTIONS})',
            'resolved': len(resolved)
        }

    # Rolling window
    recent   = resolved[-ROLLING_WINDOW:]
    accuracy = np.mean([p['correct'] for p in recent])

    drift    = accuracy < ACCURACY_THRESHOLD

    details  = {
        'rolling_accuracy'  : round(accuracy, 4),
        'threshold'         : ACCURACY_THRESHOLD,
        'n_predictions'     : len(recent),
        'n_correct'         : sum(p['correct'] for p in recent),
        'drift_detected'    : drift,
        'ticker'            : ticker or 'ALL',
    }

    return drift, round(accuracy, 4), details

# ── Method 2: PSI data drift ──────────────────────────────
def calculate_psi(expected, actual, bins=10):
    """
    Population Stability Index.
    Measures how much a feature distribution has shifted.

    PSI < 0.10: no drift
    PSI 0.10-0.25: some drift, monitor
    PSI > 0.25: significant drift, retrain needed
    """
    eps = 1e-10

    # Create bins from expected (training data)
    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges   = np.percentile(expected, percentiles)
    bin_edges   = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    expected_counts = np.histogram(expected,
                                   bins=bin_edges)[0]
    actual_counts   = np.histogram(actual,
                                   bins=bin_edges)[0]

    expected_pct = (expected_counts + eps) / \
                   (len(expected) + eps * bins)
    actual_pct   = (actual_counts  + eps) / \
                   (len(actual)    + eps * bins)

    psi = np.sum((actual_pct - expected_pct) *
                 np.log(actual_pct / expected_pct))

    return round(float(psi), 4)

def check_data_drift(ticker):
    """
    Compares feature distributions between training
    period and recent data using PSI.
    """
    try:
        from src.features import get_feature_dataset
        df = get_feature_dataset(ticker)

        if len(df) < 120:
            return False, None, {}

        # Training period: first 80%
        split      = int(len(df) * 0.8)
        train_data = df[FEATURE_COLS].iloc[:split]
        recent_data= df[FEATURE_COLS].iloc[-60:]

        psi_scores = {}
        drifted    = []

        for col in FEATURE_COLS[:10]:
            try:
                psi = calculate_psi(
                    train_data[col].dropna().values,
                    recent_data[col].dropna().values
                )
                psi_scores[col] = psi
                if psi > PSI_THRESHOLD:
                    drifted.append(col)
            except Exception:
                continue

        avg_psi     = np.mean(list(psi_scores.values())) \
                      if psi_scores else 0
        drift_detected = avg_psi > PSI_THRESHOLD * 0.7

        return drift_detected, round(avg_psi, 4), {
            'avg_psi'     : round(avg_psi, 4),
            'max_psi'     : round(max(psi_scores.values()), 4)
                            if psi_scores else 0,
            'drifted_features': drifted,
            'n_features_checked': len(psi_scores),
        }

    except Exception as e:
        return False, None, {'error': str(e)}

# ── Method 3: Regime drift ────────────────────────────────
def check_regime_drift():
    """
    Checks if the market regime has changed since
    last training. Models trained in one regime
    perform poorly in another.
    """
    drift_log = _load_json(DRIFT_LOG, {})
    last_train_regime = drift_log.get(
        'last_train_regime', 'UNKNOWN'
    )

    try:
        from src.regime_detector import detect_regime
        current = detect_regime()
        current_regime = current.get('regime', 'UNKNOWN')

        regime_changed = (
            last_train_regime != 'UNKNOWN' and
            last_train_regime != current_regime
        )

        return regime_changed, current_regime, {
            'previous_regime': last_train_regime,
            'current_regime' : current_regime,
            'changed'        : regime_changed,
        }
    except Exception as e:
        return False, 'UNKNOWN', {'error': str(e)}

# ── Master drift check ────────────────────────────────────
def check_all_drift(tickers=None):
    """
    Runs all 3 drift checks and returns
    a retrain recommendation per ticker.
    """
    from src.data_collector import STOCK_UNIVERSE
    if tickers is None:
        tickers = list(STOCK_UNIVERSE.keys())

    print(f"\n  Checking for model drift...")
    results = {}

    # Global checks
    acc_drift, accuracy, acc_details = \
        check_accuracy_drift()
    reg_drift, regime, reg_details   = \
        check_regime_drift()

    print(f"  Rolling accuracy : "
          f"{accuracy:.1%}" if accuracy else
          f"  Rolling accuracy : N/A (not enough data)")
    print(f"  Market regime    : {regime}")
    if reg_drift:
        print(f"  Regime changed   : "
              f"{reg_details['previous_regime']} → {regime}")

    need_retrain = []

    for ticker in tickers[:10]:   # check first 10 for speed
        ticker_reasons = []

        # Accuracy drift per ticker
        t_drift, t_acc, t_det = check_accuracy_drift(ticker)
        if t_drift:
            ticker_reasons.append(
                f"accuracy_drift({t_acc:.1%})"
            )

        # Data drift
        d_drift, d_psi, d_det = check_data_drift(ticker)
        if d_drift:
            ticker_reasons.append(
                f"data_drift(PSI={d_psi:.3f})"
            )

        # Regime drift applies to all
        if reg_drift:
            ticker_reasons.append(
                f"regime_change({regime})"
            )

        if ticker_reasons:
            need_retrain.append({
                'ticker' : ticker,
                'reasons': ticker_reasons,
                'priority': len(ticker_reasons),
            })

        results[ticker] = {
            'needs_retrain' : bool(ticker_reasons),
            'reasons'       : ticker_reasons,
            'accuracy'      : t_acc,
            'data_psi'      : d_psi,
        }

    # Sort by priority
    need_retrain.sort(key=lambda x: x['priority'],
                      reverse=True)

    print(f"  Stocks needing retrain: {len(need_retrain)}")

    return need_retrain, results, {
        'accuracy_drift': acc_drift,
        'accuracy'      : accuracy,
        'regime_drift'  : reg_drift,
        'regime'        : regime,
        'checked_at'    : datetime.now().isoformat(),
    }

# ── Queue management ──────────────────────────────────────
def add_to_retrain_queue(tickers, reason='drift_detected'):
    """Adds tickers to the retrain queue."""
    queue = _load_json(RETRAIN_QUEUE, [])
    for ticker in tickers:
        if ticker not in [q['ticker'] for q in queue]:
            queue.append({
                'ticker'    : ticker,
                'reason'    : reason,
                'queued_at' : datetime.now().isoformat(),
                'status'    : 'PENDING',
            })
    _save_json(RETRAIN_QUEUE, queue)
    return len(queue)

def get_retrain_queue():
    """Returns pending retrain queue."""
    queue = _load_json(RETRAIN_QUEUE, [])
    return [q for q in queue if q['status'] == 'PENDING']

def mark_retrained(ticker):
    """Marks a ticker as retrained."""
    queue = _load_json(RETRAIN_QUEUE, [])
    for q in queue:
        if q['ticker'] == ticker:
            q['status']     = 'COMPLETED'
            q['retrained_at'] = datetime.now().isoformat()
    _save_json(RETRAIN_QUEUE, queue)

# ── Helpers ───────────────────────────────────────────────
def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path, data):
    os.makedirs('data', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)