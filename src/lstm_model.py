"""
src/lstm_model.py

Sequence Neural Network for stock direction prediction.
Uses scikit-learn's MLPClassifier (already installed, works on all Python
versions including 3.14) instead of TensorFlow.

How it works:
  Instead of a recurrent LSTM cell, we flatten the last SEQUENCE_LEN days
  of technical indicators into one wide input vector:
      30 days × 23 features = 690 inputs → MLP → P(UP tomorrow)

  The MLP still learns temporal patterns ("RSI was falling 5 days ago AND
  MACD crossed bullish 2 days ago → BUY") — it just does so through
  wide feedforward connections rather than recurrence.
  In practice on financial data, accuracy is comparable to LSTM.

Output: model_type shown as "SeqNN" in dashboard and paper trader.
"""
import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)
import joblib

MODELS_DIR   = 'models'
SEQUENCE_LEN = 30    # look back 30 trading days (~6 weeks)
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'SMA_20', 'SMA_50', 'EMA_9', 'EMA_21',
    'RSI_14',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'Stoch_K', 'Stoch_D',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
    'ATR_14',
    'OBV', 'Volume_Ratio',
    'Daily_Return', 'Return_5d', 'Return_20d',
    'High_Low_Range', 'Close_vs_SMA20', 'Close_vs_SMA50'
]

# ── Sequence builder ──────────────────────────────────────
def build_sequences(X, y, seq_len=SEQUENCE_LEN):
    """
    For each day t, flatten X[t-seq_len:t] into one row.
    Shape: (n_samples, seq_len * n_features)
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i].flatten())
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# ── Data preparation ──────────────────────────────────────
def prepare_lstm_data(df, test_size=0.2):
    """
    Scale features (fit on train only — no look-ahead bias),
    build sequences, split chronologically.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    X_raw = df[FEATURE_COLS].values.astype(float)
    y_raw = df['Target'].values

    split_idx = int(len(X_raw) * (1 - test_size))

    # Fit scaler on TRAIN data only
    scaler   = StandardScaler()
    X_scaled = X_raw.copy()
    X_scaled[:split_idx] = scaler.fit_transform(X_raw[:split_idx])
    X_scaled[split_idx:] = scaler.transform(X_raw[split_idx:])

    X_seq, y_seq = build_sequences(X_scaled, y_raw)

    new_split   = max(0, split_idx - SEQUENCE_LEN)
    X_train, X_test = X_seq[:new_split], X_seq[new_split:]
    y_train, y_test = y_seq[:new_split], y_seq[new_split:]

    print(f"  Sequences — Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler

# ── Model ─────────────────────────────────────────────────
def train_lstm(X_train, y_train, n_features=None, seq_len=SEQUENCE_LEN):
    """
    Trains a 3-layer MLP on flattened 30-day sequences.
    hidden_layer_sizes=(256, 128, 64): gradually narrows down,
    learning increasingly abstract temporal patterns.
    early_stopping=True: stops when val loss plateaus (no overfitting).
    """
    model = MLPClassifier(
        hidden_layer_sizes  = (256, 128, 64),
        activation          = 'relu',
        solver              = 'adam',
        alpha               = 0.001,      # L2 regularisation
        batch_size          = 32,
        learning_rate       = 'adaptive',
        learning_rate_init  = 0.001,
        max_iter            = 300,
        early_stopping      = True,
        validation_fraction = 0.15,
        n_iter_no_change    = 15,         # patience
        random_state        = 42,
        verbose             = False,
    )
    model.fit(X_train, y_train)
    iters = model.n_iter_
    best  = model.best_validation_score_
    print(f"  Stopped at iteration {iters}  |  Best val_score: {best:.2%}")
    return model

# ── Evaluation ────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, ticker):
    """Full evaluation metrics — identical format to RF / XGBoost."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.58).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  SeqNN Results — {ticker}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.2%}")
    print(f"  Precision : {prec:.2%}")
    print(f"  Recall    : {rec:.2%}")
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  {'':10} Pred DOWN  Pred UP")
    print(f"  Actual DOWN  {cm[0][0]:>6}     {cm[0][1]:>6}")
    print(f"  Actual UP    {cm[1][0]:>6}     {cm[1][1]:>6}")
    print(f"{'='*50}\n")

    return acc, y_pred, y_proba

# ── Save / Load ───────────────────────────────────────────
def save_model(model, scaler, ticker):
    """Saves MLP model + fitted scaler using joblib."""
    model_path  = os.path.join(MODELS_DIR, f'{ticker}_seqnn_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'{ticker}_seqnn_scaler.pkl')
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  💾 SeqNN saved → {model_path}")

def load_model(ticker):
    """Loads saved SeqNN model + scaler."""
    model_path  = os.path.join(MODELS_DIR, f'{ticker}_seqnn_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'{ticker}_seqnn_scaler.pkl')
    return joblib.load(model_path), joblib.load(scaler_path)

# ── Live prediction ───────────────────────────────────────
def predict_lstm(model, scaler, df_recent):
    """
    Live prediction from the last SEQUENCE_LEN rows.
    Works identically to the TensorFlow version — same call signature.
    Returns confidence (float) = P(price goes UP tomorrow).
    """
    if len(df_recent) < SEQUENCE_LEN:
        raise ValueError(f"Need at least {SEQUENCE_LEN} rows, got {len(df_recent)}")
    X_raw    = df_recent[FEATURE_COLS].values[-SEQUENCE_LEN:].astype(float)
    X_scaled = scaler.transform(X_raw)
    X_flat   = X_scaled.flatten().reshape(1, -1)
    return float(model.predict_proba(X_flat)[0][1])
