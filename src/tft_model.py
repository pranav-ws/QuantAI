"""
src/tft_model.py — Temporal Fusion Transformer for QuantAI
===========================================================

ARCHITECTURE: Full TFT using tf.keras.Model subclassing API.
  All components are proper tf.keras.layers.Layer subclasses.
  This avoids all KerasTensor errors that occur when plain Python
  classes with __call__ methods are used inside the Keras functional API.

Components:
  GRNLayer      — Gated Residual Network (universal building block)
  VSNLayer      — Variable Selection Network (feature importance)
  IMHALayer     — Interpretable Multi-Head Attention (shared Value)
  TFTModel      — Full model combining all components

Signal outputs:
  P10 / P50 / P90 — quantile 5-day return forecasts
  BUY / SELL / HOLD + confidence — for ensemble integration
"""

import io
import json
import math
import os
import warnings

import joblib
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ── Hyperparameters ───────────────────────────────────────
D_MODEL    = 64
N_HEADS    = 4
LSTM_UNITS = 64
DROPOUT    = 0.1
LOOKBACK   = 30
HORIZON    = 5
QUANTILES  = [0.1, 0.5, 0.9]
BATCH_SIZE = 64
EPOCHS     = 80
LR         = 1e-3

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'SMA_20', 'SMA_50', 'EMA_9', 'EMA_21',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'Stoch_K', 'Stoch_D',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
    'ATR_14', 'OBV', 'Volume_Ratio',
    'Daily_Return', 'Return_5d', 'Return_20d',
    'High_Low_Range', 'Close_vs_SMA20', 'Close_vs_SMA50',
]
N_FEATURES = len(FEATURE_COLS)   # 23


def _tf():
    """Lazy TF import — keeps the module importable without TF."""
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    return tf


# ══════════════════════════════════════════════════════════
#  SECTION 1 — Building blocks (proper Layer subclasses)
# ══════════════════════════════════════════════════════════

def _make_grn_layer(tf, units, dropout_rate=DROPOUT, name='grn'):
    """
    Gated Residual Network as a tf.keras.layers.Layer.

    Forward pass:
        h = ELU(Dense(x))
        h = Dense(h); h = Dropout(h)
        h = Dense(h) * sigmoid(Dense(h))   ← GLU gate
        out = LayerNorm(Dense_skip(x) + h) ← residual
    """
    class GRNLayer(tf.keras.layers.Layer):
        def __init__(self, units, dropout_rate, **kwargs):
            super().__init__(**kwargs)
            self.d1      = tf.keras.layers.Dense(units, activation='elu')
            self.d2      = tf.keras.layers.Dense(units)
            self.drop    = tf.keras.layers.Dropout(dropout_rate)
            self.gate_l  = tf.keras.layers.Dense(units)
            self.gate_s  = tf.keras.layers.Dense(units, activation='sigmoid')
            self.skip    = tf.keras.layers.Dense(units, use_bias=False)
            self.norm    = tf.keras.layers.LayerNormalization()

        def call(self, x, training=False):
            h = self.d1(x)
            h = self.d2(h)
            h = self.drop(h, training=training)
            h = self.gate_l(h) * self.gate_s(h)
            return self.norm(self.skip(x) + h)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({'units': units, 'dropout_rate': dropout_rate})
            return cfg

    return GRNLayer(units, dropout_rate, name=name)


def _make_vsn_layer(tf, num_vars, units, dropout=DROPOUT, name='vsn'):
    """
    Variable Selection Network.
    Learns per-feature importance weights at every time step.
    """
    class VSNLayer(tf.keras.layers.Layer):
        def __init__(self, num_vars, units, dropout, **kwargs):
            super().__init__(**kwargs)
            self.num_vars = num_vars
            # One projection Dense + one GRN per variable
            self.projs = [
                tf.keras.layers.Dense(units, name=f'proj_{i}')
                for i in range(num_vars)
            ]
            self.grns = [
                _make_grn_layer(tf, units, dropout, name=f'grn_{i}')
                for i in range(num_vars)
            ]
            # Softmax selection: input features → importance weights
            self.selector = tf.keras.layers.Dense(
                num_vars, activation='softmax', name='selector'
            )

        def call(self, x, training=False):
            # x: (batch, T, num_vars) — each feature is a scalar
            var_outs = []
            for i in range(self.num_vars):
                xi = x[..., i:i+1]                      # (batch, T, 1)
                xi = self.projs[i](xi)                   # (batch, T, units)
                xi = self.grns[i](xi, training=training) # (batch, T, units)
                var_outs.append(xi)

            stacked  = tf.stack(var_outs, axis=2)        # (batch, T, num_vars, units)
            weights  = self.selector(x)                  # (batch, T, num_vars)
            w_exp    = tf.expand_dims(weights, axis=-1)  # (batch, T, num_vars, 1)
            out      = tf.reduce_sum(stacked * w_exp, axis=2)   # (batch, T, units)
            return out, weights

        def get_config(self):
            cfg = super().get_config()
            cfg.update({'num_vars': num_vars, 'units': units, 'dropout': dropout})
            return cfg

    return VSNLayer(num_vars, units, dropout, name=name)


def _make_imha_layer(tf, d_model, n_heads, dropout=DROPOUT, name='imha'):
    """
    Interpretable Multi-Head Attention.
    Key difference from standard MHA: Value network is SHARED across all heads.
    This means head-averaged attention weights are directly interpretable.
    """
    class IMHALayer(tf.keras.layers.Layer):
        def __init__(self, d_model, n_heads, dropout, **kwargs):
            super().__init__(**kwargs)
            assert d_model % n_heads == 0
            self.n_heads = n_heads
            self.d_head  = d_model // n_heads
            self.scale   = math.sqrt(self.d_head)

            self.Wq = [
                tf.keras.layers.Dense(self.d_head, use_bias=False, name=f'q{i}')
                for i in range(n_heads)
            ]
            self.Wk = [
                tf.keras.layers.Dense(self.d_head, use_bias=False, name=f'k{i}')
                for i in range(n_heads)
            ]
            self.Wv_shared = tf.keras.layers.Dense(
                self.d_head, use_bias=False, name='v_shared'
            )
            self.out_proj = tf.keras.layers.Dense(d_model, name='out_proj')
            self.attn_drop = tf.keras.layers.Dropout(dropout)

        def call(self, query, key, value, training=False):
            V = self.Wv_shared(value)   # single shared Value

            head_outs = []
            for h in range(self.n_heads):
                Q       = self.Wq[h](query)
                K       = self.Wk[h](key)
                scores  = tf.matmul(Q, K, transpose_b=True) / self.scale
                w       = tf.nn.softmax(scores, axis=-1)
                w       = self.attn_drop(w, training=training)
                head_outs.append(tf.matmul(w, V))

            concat = tf.concat(head_outs, axis=-1)  # (batch, T, d_model)
            return self.out_proj(concat)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({'d_model': d_model, 'n_heads': n_heads, 'dropout': dropout})
            return cfg

    return IMHALayer(d_model, n_heads, dropout, name=name)


# ══════════════════════════════════════════════════════════
#  SECTION 2 — Full TFT Model (tf.keras.Model subclass)
# ══════════════════════════════════════════════════════════

def _build_tft_model(tf, n_features=N_FEATURES, d_model=D_MODEL,
                     n_heads=N_HEADS, lstm_units=LSTM_UNITS, dropout=DROPOUT):
    """
    Build TFT as a tf.keras.Model subclass.

    Using the subclassing API instead of the Functional API avoids the
    'KerasTensor cannot be used as input to a TensorFlow function' error
    that occurs when plain Python classes are used inside Model(inputs, outputs).

    All forward-pass logic lives in call() which TF traces correctly.
    """
    class TFTModel(tf.keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # ── Variable Selection Network ──────────────
            self.vsn = _make_vsn_layer(tf, n_features, d_model, dropout, 'vsn')

            # ── LSTM Encoder (2 layers with residuals) ──
            self.lstm1  = tf.keras.layers.LSTM(
                lstm_units, return_sequences=True,
                dropout=dropout, name='lstm1'
            )
            self.lstm2  = tf.keras.layers.LSTM(
                lstm_units, return_sequences=True,
                dropout=dropout, name='lstm2'
            )
            self.ln_l1  = tf.keras.layers.LayerNormalization(name='ln_lstm1')
            self.ln_l2  = tf.keras.layers.LayerNormalization(name='ln_lstm2')

            # Projection to match lstm_units if d_model != lstm_units
            self.vsn_proj = tf.keras.layers.Dense(lstm_units, use_bias=False,
                                                   name='vsn_proj')

            # ── Interpretable Multi-Head Attention ──────
            self.imha   = _make_imha_layer(tf, lstm_units, n_heads, dropout, 'imha')
            self.ln_attn= tf.keras.layers.LayerNormalization(name='ln_attn')

            # ── Position-wise GRN ────────────────────────
            self.pw_d1  = tf.keras.layers.Dense(lstm_units, activation='elu', name='pw_d1')
            self.pw_d2  = tf.keras.layers.Dense(lstm_units, name='pw_d2')
            self.pw_drop= tf.keras.layers.Dropout(dropout, name='pw_drop')
            self.pw_gl  = tf.keras.layers.Dense(lstm_units, name='pw_gl')
            self.pw_gs  = tf.keras.layers.Dense(lstm_units, activation='sigmoid', name='pw_gs')
            self.ln_grn = tf.keras.layers.LayerNormalization(name='ln_grn')

            # ── Temporal pooling + pre-head ──────────────
            self.pool       = tf.keras.layers.GlobalAveragePooling1D(name='pool')
            self.pre_head   = tf.keras.layers.Dense(lstm_units // 2, activation='elu',
                                                    name='pre_head')
            self.head_drop  = tf.keras.layers.Dropout(dropout, name='head_drop')

            # ── Quantile output heads ────────────────────
            # Enforce monotonicity: P10 free; P50=P10+softplus(Δ50); P90=P50+softplus(Δ90)
            self.h_p10   = tf.keras.layers.Dense(1, name='h_p10')
            self.h_d50   = tf.keras.layers.Dense(1, name='h_d50')
            self.h_d90   = tf.keras.layers.Dense(1, name='h_d90')

        def call(self, x, training=False):
            # x: (batch, LOOKBACK, n_features)

            # 1. Variable Selection
            vsn_out, _vsn_w = self.vsn(x, training=training)

            # 2. LSTM encoder with residuals
            vsn_proj = self.vsn_proj(vsn_out)
            h1 = self.lstm1(vsn_out, training=training)
            h1 = self.ln_l1(vsn_proj + h1)
            h2 = self.lstm2(h1, training=training)
            h2 = self.ln_l2(h1 + h2)

            # 3. Interpretable Multi-Head Attention
            attn_out = self.imha(h2, h2, h2, training=training)
            h3 = self.ln_attn(h2 + attn_out)

            # 4. Position-wise GRN
            g  = self.pw_d1(h3)
            g  = self.pw_d2(g)
            g  = self.pw_drop(g, training=training)
            g  = self.pw_gl(g) * self.pw_gs(g)
            h4 = self.ln_grn(h3 + g)

            # 5. Temporal pooling
            pooled = self.pool(h4)
            dense  = self.pre_head(pooled)
            dense  = self.head_drop(dense, training=training)

            # 6. Quantile heads (monotonic by construction)
            p10    = self.h_p10(dense)
            delta50= self.h_d50(dense)
            delta90= self.h_d90(dense)

            p50 = p10 + tf.math.softplus(delta50)
            p90 = p10 + tf.math.softplus(delta50) + tf.math.softplus(delta90)

            return tf.concat([p10, p50, p90], axis=-1)   # (batch, 3)

    model = TFTModel(name='TFT')
    return model


# ══════════════════════════════════════════════════════════
#  SECTION 3 — Pinball (quantile regression) loss
# ══════════════════════════════════════════════════════════

def _make_pinball_loss(tf, quantiles=QUANTILES):
    """
    Pinball loss as a proper tf.keras.losses.Loss subclass.

    For quantile τ:
      L = τ·max(y-ŷ, 0) + (1-τ)·max(ŷ-y, 0)

    Minimising across τ ∈ {0.1, 0.5, 0.9} gives calibrated prediction
    intervals — the model learns to be uncertain when uncertainty is warranted.
    """
    class PinballLoss(tf.keras.losses.Loss):
        def __init__(self, quantiles, **kwargs):
            super().__init__(name='pinball_loss', **kwargs)
            self.quantiles = quantiles

        def call(self, y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            total  = tf.constant(0.0)
            for i, q in enumerate(self.quantiles):
                e      = y_true - y_pred[:, i:i+1]
                total  = total + tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))
            return total / tf.cast(len(self.quantiles), tf.float32)

        def get_config(self):
            return {'quantiles': self.quantiles}

    return PinballLoss(quantiles)


# ══════════════════════════════════════════════════════════
#  SECTION 4 — Data preparation
# ══════════════════════════════════════════════════════════

def make_dataset(df: pd.DataFrame, lookback=LOOKBACK, horizon=HORIZON):
    """Build (X, y) windows. y = forward horizon-day return."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    feat      = df[available].values.astype(np.float32)
    close     = df['Close'].values.astype(np.float32)
    fwd_ret   = np.zeros(len(close), dtype=np.float32)
    for i in range(len(close) - horizon):
        fwd_ret[i] = (close[i + horizon] - close[i]) / (close[i] + 1e-8)

    Xs, ys = [], []
    for i in range(lookback, len(feat) - horizon):
        Xs.append(feat[i - lookback: i])
        ys.append(fwd_ret[i])

    if not Xs:
        return None, None
    return np.array(Xs, np.float32), np.array(ys, np.float32).reshape(-1, 1)


def prepare_tft_data(df: pd.DataFrame, test_frac=0.15):
    """Scale features (train only), build windows, split chronologically."""
    from sklearn.preprocessing import StandardScaler

    available = [c for c in FEATURE_COLS if c in df.columns]
    n         = len(df)
    split     = int(n * (1 - test_frac))

    scaler    = StandardScaler()
    feat_arr  = df[available].values.astype(np.float32)
    feat_arr[:split] = scaler.fit_transform(feat_arr[:split])
    feat_arr[split:] = scaler.transform(feat_arr[split:])

    df_s = df.copy()
    df_s[available] = feat_arr

    X_all, y_all = make_dataset(df_s)
    if X_all is None:
        return None, None, None, None, None

    seq_split = max(0, split - LOOKBACK - HORIZON)
    return (X_all[:seq_split], X_all[seq_split:],
            y_all[:seq_split], y_all[seq_split:],
            scaler)


# ══════════════════════════════════════════════════════════
#  SECTION 5 — Training
# ══════════════════════════════════════════════════════════

def train_tft(df: pd.DataFrame, ticker: str,
              epochs=EPOCHS, verbose=0) -> dict:
    """Train TFT for one ticker and save to models/."""
    tf = _tf()

    result = prepare_tft_data(df)
    if result[0] is None:
        return {'error': 'Insufficient data'}
    X_train, X_test, y_train, y_test, scaler = result

    if len(X_train) < 50:
        return {'error': f'Only {len(X_train)} training samples — need 50+'}

    n_feat = X_train.shape[2]
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}  "
          f"Features: {n_feat}  Lookback: {X_train.shape[1]}")

    model = _build_tft_model(tf, n_features=n_feat)

    # Build weights by running a forward pass before compile
    dummy = X_train[:1]
    _ = model(dummy, training=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=_make_pinball_loss(tf),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=12,
            min_delta=1e-4, restore_best_weights=True, verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0,
        ),
    ]

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=verbose,
    )

    # Evaluation
    y_pred = model.predict(X_test, verbose=0)
    p50    = y_pred[:, 1]
    y_true = y_test[:, 0]
    mae      = float(np.mean(np.abs(p50 - y_true)))
    hit_rate = float(np.mean(np.sign(p50) == np.sign(y_true)))
    coverage = float(np.mean((y_true >= y_pred[:, 0]) & (y_true <= y_pred[:, 2])))

    print(f"  ✅ TFT {ticker}: MAE={mae:.4f}  "
          f"Direction={hit_rate:.1%}  Coverage={coverage:.1%}  "
          f"Epochs={len(hist.history['loss'])}")

    _save_tft(model, scaler, ticker, n_features=n_feat)
    return {
        'ticker'    : ticker,
        'mae'       : round(mae, 6),
        'hit_rate'  : round(hit_rate, 4),
        'coverage'  : round(coverage, 4),
        'n_features': n_feat,
        'epochs'    : len(hist.history['loss']),
        'train_loss': round(hist.history['loss'][-1], 6),
        'val_loss'  : round(hist.history['val_loss'][-1], 6),
    }


# ══════════════════════════════════════════════════════════
#  SECTION 6 — Save / Load
# ══════════════════════════════════════════════════════════

def _model_dir(ticker):  return os.path.join(MODELS_DIR, f'{ticker}_tft_model')
def _scaler_path(ticker):return os.path.join(MODELS_DIR, f'{ticker}_tft_scaler.pkl')
def _config_path(ticker):return os.path.join(MODELS_DIR, f'{ticker}_tft_config.json')
def _weights_path(ticker):return os.path.join(MODELS_DIR, f'{ticker}_tft_weights.weights.h5')


def _save_tft(model, scaler, ticker, n_features=N_FEATURES):
    """Save TFT weights + scaler + config (subclassed models save weights only)."""
    # Save weights (more reliable than full model save for subclassed models)
    model.save_weights(_weights_path(ticker))
    joblib.dump(scaler, _scaler_path(ticker))
    with open(_config_path(ticker), 'w') as f:
        json.dump({'n_features': n_features}, f)
    print(f"  💾 TFT saved → models/{ticker}_tft_weights*")


def tft_available(ticker):
    """Check if a trained TFT exists for this ticker."""
    return (os.path.exists(_config_path(ticker)) and
            os.path.exists(_scaler_path(ticker)) and
            os.path.exists(_weights_path(ticker)))


def load_tft(ticker):
    """Rebuild TFT model and load saved weights. Returns (model, scaler) or (None, None)."""
    if not tft_available(ticker):
        return None, None

    tf = _tf()
    try:
        with open(_config_path(ticker)) as f:
            cfg = json.load(f)
        n_features = cfg.get('n_features', N_FEATURES)

        model = _build_tft_model(tf, n_features=n_features)

        # Build weights by running a dummy forward pass first
        dummy = np.zeros((1, LOOKBACK, n_features), dtype=np.float32)
        _ = model(dummy, training=False)

        model.load_weights(_weights_path(ticker))
        scaler = joblib.load(_scaler_path(ticker))
        return model, scaler
    except Exception as exc:
        print(f"  ⚠️  Could not load TFT for {ticker}: {exc}")
        return None, None


# ══════════════════════════════════════════════════════════
#  SECTION 7 — Inference + signal generation
# ══════════════════════════════════════════════════════════

def predict_tft(model, scaler, df_recent: pd.DataFrame) -> dict | None:
    """
    Run TFT inference on the most recent LOOKBACK rows.

    Returns dict with P10/P50/P90 forecasts + BUY/SELL/HOLD signal.
    """
    available = [c for c in FEATURE_COLS if c in df_recent.columns]
    if len(df_recent) < LOOKBACK or not available:
        return None

    feat   = df_recent[available].values[-LOOKBACK:].astype(np.float32)
    feat_s = scaler.transform(feat)
    X      = feat_s[np.newaxis, ...]   # (1, LOOKBACK, n_features)

    pred   = model.predict(X, verbose=0)[0]   # (3,)
    p10, p50, p90 = float(pred[0]), float(pred[1]), float(pred[2])
    spread = p90 - p10

    # Signal rules
    if p10 > 0.005:                          # P10 > +0.5% → strong BUY
        signal   = 'BUY'
        raw_conf = 0.80 + min(p10 / 0.05, 0.15)
    elif p50 > 0.010:                        # P50 > +1% → moderate BUY
        signal   = 'BUY'
        raw_conf = 0.63 + min(p50 / 0.08, 0.15)
    elif p90 < -0.005:                       # P90 < -0.5% → strong SELL
        signal   = 'SELL'
        raw_conf = 0.80 + min(abs(p90) / 0.05, 0.15)
    elif p50 < -0.010:                       # P50 < -1% → moderate SELL
        signal   = 'SELL'
        raw_conf = 0.63 + min(abs(p50) / 0.08, 0.15)
    else:
        signal   = 'HOLD'
        raw_conf = 0.50

    # Penalise wide (high-uncertainty) intervals
    uncertainty_penalty = min(spread / 0.10, 0.15)
    confidence = round(max(0.50, raw_conf - uncertainty_penalty), 4)

    binary_prob = (confidence if signal == 'BUY'
                   else (1 - confidence if signal == 'SELL'
                         else 0.5))

    return {
        'p10'        : round(p10, 6),
        'p50'        : round(p50, 6),
        'p90'        : round(p90, 6),
        'spread'     : round(spread, 6),
        'signal'     : signal,
        'confidence' : confidence,
        'binary_prob': round(binary_prob, 4),
    }


def get_tft_signal(ticker: str, df: pd.DataFrame) -> float | None:
    """Ensemble-compatible: returns P(UP) float or None."""
    model, scaler = load_tft(ticker)
    if model is None:
        return None
    result = predict_tft(model, scaler, df)
    return result['binary_prob'] if result else None