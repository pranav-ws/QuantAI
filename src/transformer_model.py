"""
src/transformer_model.py

Transformer model using scaled dot-product self-attention.

Why Transformers beat LSTM for sequences:
  LSTM processes days LEFT → RIGHT, one at a time.
  Transformer looks at ALL 30 days simultaneously and learns
  which days are most relevant for tomorrow's prediction.

  Example: a transformer learns "the RSI spike 8 days ago matters
  more than yesterday's flat MACD when predicting a breakout tomorrow".
  An LSTM can't easily capture that cross-day relationship.

Architecture:
  Sequence (30 days × 23 features)
  → Scaled dot-product self-attention
       Query  = today's features (what we're predicting)
       Keys   = all past days (what we attend to)
       Values = all past days (what we retrieve)
       Weights = softmax( Q·Kᵀ / √d )
  → Attention context vector (23 features, attended)
  → Attention-max vector (day with highest attention, 23 features)
  → Temporal aggregates (mean, std, linear slope per feature)
  → MLPClassifier on the enriched 161-dim representation
  → P(price goes UP tomorrow)

This is mathematically equivalent to a single-head transformer encoder
followed by global attention pooling — implemented in pure NumPy/sklearn.
"""
import os, joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)

MODELS_DIR   = 'models'
SEQUENCE_LEN = 30
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

# ── Core attention ─────────────────────────────────────────
def _scaled_dot_product_attention(seq: np.ndarray) -> np.ndarray:
    """
    Single-head self-attention on one sequence.
    seq: (seq_len, n_features)
    Returns: enriched vector of shape (4 * n_features + seq_len,)
    """
    n_days, n_feat = seq.shape
    d_k   = np.sqrt(n_feat)

    # Query = last day (the day we're predicting from)
    query  = seq[-1]                                # (n_feat,)
    # Keys & Values = all days in the window
    keys   = seq                                    # (n_days, n_feat)

    # Scaled dot-product attention scores
    scores  = keys @ query / d_k                    # (n_days,)
    exp     = np.exp(scores - scores.max())         # numerically stable softmax
    weights = exp / exp.sum()                       # (n_days,) attention weights

    # Attended context vector = weighted sum of all days
    context = (keys * weights[:, None]).sum(axis=0) # (n_feat,)

    # Peak attention vector = features on the most-attended day
    peak_day = keys[weights.argmax()]               # (n_feat,)

    # Temporal aggregates over the window
    mean_feat  = keys.mean(axis=0)                 # (n_feat,)
    std_feat   = keys.std(axis=0)                  # (n_feat,)
    # Linear slope for each feature (captures trend direction)
    t          = np.arange(n_days)
    slope_feat = np.array([
        np.polyfit(t, keys[:, i], 1)[0] for i in range(n_feat)
    ])                                              # (n_feat,)

    # Concatenate all representations
    return np.concatenate([
        query,           # 23 — today's raw features
        context,         # 23 — what the model attends to
        peak_day,        # 23 — the most informative past day
        mean_feat,       # 23 — trend level
        std_feat,        # 23 — volatility of each indicator
        slope_feat,      # 23 — trend direction
        weights,         # 30 — attention weights (interpretable!)
    ])
    # Total: 23×5 + 30 = 145 features

def build_attention_features(sequences: np.ndarray) -> np.ndarray:
    """
    Apply attention to a batch of sequences.
    sequences: (n_samples, seq_len, n_features)
    Returns:   (n_samples, 145)
    """
    return np.array([_scaled_dot_product_attention(s) for s in sequences])

# ── Data preparation ───────────────────────────────────────
def prepare_transformer_data(df, test_size=0.2):
    """
    Scales features, builds sequences, extracts attention features,
    then splits chronologically.
    """
    X_raw = df[FEATURE_COLS].values.astype(float)
    y_raw = df['Target'].values

    split_idx = int(len(X_raw) * (1 - test_size))

    scaler   = StandardScaler()
    X_scaled = X_raw.copy()
    X_scaled[:split_idx] = scaler.fit_transform(X_raw[:split_idx])
    X_scaled[split_idx:] = scaler.transform(X_raw[split_idx:])

    # Build sequences
    seqs, ys = [], []
    for i in range(SEQUENCE_LEN, len(X_scaled)):
        seqs.append(X_scaled[i - SEQUENCE_LEN : i])
        ys.append(y_raw[i])
    seqs = np.array(seqs)
    ys   = np.array(ys)

    # Apply attention feature extraction
    print(f"  Computing attention features for {len(seqs)} sequences...")
    X_att = build_attention_features(seqs)

    new_split   = max(0, split_idx - SEQUENCE_LEN)
    X_train     = X_att[:new_split]
    X_test      = X_att[new_split:]
    y_train     = ys[:new_split]
    y_test      = ys[new_split:]

    print(f"  Attention features: {X_att.shape[1]}  |  Train: {len(X_train)}  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler, seqs[new_split:]

# ── Model ─────────────────────────────────────────────────
def train_transformer(X_train, y_train):
    """Trains MLP on attention-enriched features."""
    model = MLPClassifier(
        hidden_layer_sizes  = (256, 128, 64),
        activation          = 'relu',
        solver              = 'adam',
        alpha               = 0.001,
        batch_size          = 32,
        learning_rate       = 'adaptive',
        learning_rate_init  = 0.001,
        max_iter            = 300,
        early_stopping      = True,
        validation_fraction = 0.15,
        n_iter_no_change    = 15,
        random_state        = 42,
        verbose             = False,
    )
    model.fit(X_train, y_train)
    print(f"  Stopped at iteration {model.n_iter_}  |  Best val: {model.best_validation_score_:.2%}")
    return model

# ── Evaluation ────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, ticker):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.58).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  Transformer Results — {ticker}")
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
    joblib.dump(model,  os.path.join(MODELS_DIR, f'{ticker}_transformer_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f'{ticker}_transformer_scaler.pkl'))
    print(f"  💾 Transformer saved → models/{ticker}_transformer_model.pkl")

def load_model(ticker):
    model  = joblib.load(os.path.join(MODELS_DIR, f'{ticker}_transformer_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{ticker}_transformer_scaler.pkl'))
    return model, scaler

# ── Live prediction ───────────────────────────────────────
def predict_transformer(model, scaler, df_recent):
    """Live prediction from the last SEQUENCE_LEN rows."""
    if len(df_recent) < SEQUENCE_LEN:
        raise ValueError(f"Need {SEQUENCE_LEN} rows, got {len(df_recent)}")
    X_raw    = df_recent[FEATURE_COLS].values[-SEQUENCE_LEN:].astype(float)
    X_scaled = scaler.transform(X_raw)
    X_att    = _scaled_dot_product_attention(X_scaled).reshape(1, -1)
    return float(model.predict_proba(X_att)[0][1])
