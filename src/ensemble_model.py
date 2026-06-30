"""
src/ensemble_model.py

Weighted ensemble of Random Forest + XGBoost + SeqNN/LSTM, now joined
by three rule-based (non-ML) strategy votes: Mean Reversion, Momentum
Breakout, and MACD+RSI Confluence.

Why ensembles outperform individual models:
  - Each model has different strengths and blind spots
  - When ALL models agree → very high confidence → better signal
  - When models disagree → lower confidence → system skips (avoids bad trades)
  - Errors that one model makes are often cancelled out by others
  - In live quant trading, ensembles typically reduce false signals by 20-35%

Weighting strategy:
  - RF:                  weight 1.0  (baseline)
  - XGBoost:              weight 1.2  (gradient boosting usually more accurate)
  - SeqNN / LSTM:         weight 1.2  (captures temporal patterns other models miss)
  - MeanReversion:        weight 0.6  (rule-based, fires rarely, lower than trained models)
  - MomentumBreakout:     weight 0.6  (rule-based, fires rarely, lower than trained models)
  - MACD_RSI_Confluence:  weight 0.7  (rule-based, but triple-confirmed — slightly higher
                                       than the other two rule-based strategies)
  When a model is missing, remaining models split its share proportionally.

How the rule-based strategies are folded in:
  Unlike the ML models, src/mean_reversion.py, src/momentum_breakout.py,
  and src/macd_rsi_confluence.py output HOLD on most days — there's
  nothing statistically interesting happening. Including a neutral 0.5
  in the weighted average every single day would just dilute genuine
  ML signal with noise. So a rule-based strategy only enters the vote
  on days it actually considers itself "tradeable" (signal fired AND
  cleared its own confidence floor) — the rest of the time it sits out
  the vote entirely, exactly like a model that isn't available.
"""
import os, joblib# type: ignore
import pandas as pd# type: ignore
import numpy as np# type: ignore
from src.model import FEATURE_COLS
from src.lstm_model import predict_lstm, SEQUENCE_LEN
from src.mean_reversion import MeanReversionStrategy
from src.momentum_breakout import MomentumBreakoutStrategy
from src.macd_rsi_confluence import MACDRSIConfluenceStrategy

MODELS_DIR = 'models'

# Base weights per model type
MODEL_WEIGHTS = {
    'RF'                  : 1.0,
    'XGBoost'             : 1.2,
    'SeqNN'               : 1.2,
    'LSTM'                : 1.2,
    'MeanReversion'       : 0.6,
    'MomentumBreakout'    : 0.6,
    'MACD_RSI_Confluence' : 0.7,
    'TFT'     : 1.8,
}

# Rule-based strategies are stateless and ticker-agnostic, so a single
# shared instance of each is enough — no per-ticker loading needed.
_mean_reversion_strategy   = MeanReversionStrategy()
_momentum_breakout_strategy = MomentumBreakoutStrategy()
_macd_rsi_confluence_strategy = MACDRSIConfluenceStrategy()

# ── Module-level cache (models loaded once per session) ───
_cache: dict = {}

def _load_models(ticker):
    """
    Load all available models for a ticker, with in-memory caching.
    Returns a dict: {'RF': model, 'XGBoost': model, 'SeqNN': (model, scaler)}
    """
    if ticker in _cache:
        return _cache[ticker]

    models = {}

    # Random Forest
    rf_path = os.path.join(MODELS_DIR, f'{ticker}_rf_model.pkl')
    if os.path.exists(rf_path):
        try:
            models['RF'] = joblib.load(rf_path)
        except Exception:
            pass

    # XGBoost
    xgb_path = os.path.join(MODELS_DIR, f'{ticker}_xgb_model.pkl')
    if os.path.exists(xgb_path):
        try:
            models['XGBoost'] = joblib.load(xgb_path)
        except Exception:
            pass

    # SeqNN (sklearn — no TF required)
    seqnn_path  = os.path.join(MODELS_DIR, f'{ticker}_seqnn_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'{ticker}_seqnn_scaler.pkl')
    if os.path.exists(seqnn_path) and os.path.exists(scaler_path):
        try:
            models['SeqNN'] = (joblib.load(seqnn_path), joblib.load(scaler_path))
        except Exception:
            pass

    # TF LSTM (fallback if SeqNN not available)
    if 'SeqNN' not in models:
        lstm_path   = os.path.join(MODELS_DIR, f'{ticker}_lstm_model.h5')
        lstm_scaler = os.path.join(MODELS_DIR, f'{ticker}_lstm_scaler.pkl')
        if os.path.exists(lstm_path) and os.path.exists(lstm_scaler):
            try:
                import os as _os
                _os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                import importlib as _importlib
                tf = _importlib.import_module('tensorflow')
                tf.get_logger().setLevel('ERROR')
                model  = tf.keras.models.load_model(lstm_path, compile=False)
                scaler = joblib.load(lstm_scaler)
                models['LSTM'] = (model, scaler)
            except Exception:
                pass

    # ── Temporal Fusion Transformer ──────────────────────
    from src.tft_model import tft_available, load_tft
    if tft_available(ticker):
        try:
            tft_model, tft_scaler = load_tft(ticker)
            if tft_model is not None:
               models['TFT'] = (tft_model, tft_scaler)
        except Exception:
            pass

    _cache[ticker] = models
    return models


def _get_rule_based_confidences(df):
    """
    Runs the three rule-based strategies and returns a dict of
    {name: confidence} for ONLY the strategies that fired a tradeable
    signal today. A strategy that's sitting on HOLD (the vast majority
    of days) contributes nothing — same as a missing ML model.

    Also returns a `details` dict with the full signal/reason payload
    per strategy that fired, so callers (e.g. the API) can surface
    *why* a rule-based vote happened, not just the number.
    """
    confidences = {}
    details     = {}

    strategies = {
        'MeanReversion'       : _mean_reversion_strategy,
        'MomentumBreakout'    : _momentum_breakout_strategy,
        'MACD_RSI_Confluence' : _macd_rsi_confluence_strategy,
    }

    for name, strategy in strategies.items():
        try:
            result = strategy.get_latest_signal(df)
            if result['tradeable']:
                confidences[name] = result['confidence']
                details[name]     = result
        except Exception:
            # Not enough history yet (e.g. momentum breakout needs ~100+
            # rows for its 52-week window) — skip this strategy quietly,
            # same as a model file that doesn't exist.
            pass

    return confidences, details


def get_ensemble_confidence(ticker, df):
    """
    Gets confidence from every available ML model AND every rule-based
    strategy that fired today, then returns a weighted-average
    ensemble confidence score.

    Returns:
        ensemble_conf  (float)  — weighted average P(UP)
        individual     (dict)   — {'RF': 0.61, 'XGBoost': 0.59, 'MeanReversion': 0.71, ...}
        model_names    (list)   — which models/strategies contributed
    Returns (None, None, None) if nothing — ML or rule-based — has a signal.
    """
    models = _load_models(ticker)

    confidences = {}
    latest = df.iloc[-1]

    for name, model_obj in models.items():
        try:
            if name in ('SeqNN', 'LSTM'):
                # Sequence model — needs last SEQUENCE_LEN rows
                if len(df) >= SEQUENCE_LEN:
                    model, scaler = model_obj
                    confidences[name] = predict_lstm(model, scaler, df)
            else:
                # RF or XGBoost — single-row prediction
                X = pd.DataFrame([latest[FEATURE_COLS]], columns=FEATURE_COLS)
                confidences[name] = float(model_obj.predict_proba(X)[0][1])
        except Exception:
            pass

    # ── Fold in the rule-based strategy votes (only ones that fired) ──
    rule_based_confidences, _ = _get_rule_based_confidences(df)
    confidences.update(rule_based_confidences)

    if not confidences:
        return None, None, None

    total_w    = sum(MODEL_WEIGHTS.get(k, 1.0) for k in confidences)
    ens_conf   = sum(confidences[k] * MODEL_WEIGHTS.get(k, 1.0)
                     for k in confidences) / total_w

    return float(ens_conf), confidences, list(confidences.keys())


def get_model_agreement(individual_confs):
    """
    Returns a string describing how much the models agree.
    Separates ML models from rule-based strategies in the spread
    calculation — it's expected and healthy for a rule-based
    strategy to disagree with ML on direction (that's its value),
    so overall spread is reported but ML-internal agreement is
    the primary conviction signal.
    """
    if not individual_confs or len(individual_confs) < 2:
        return 'single'

    ML_NAMES = {'RF', 'XGBoost', 'SeqNN', 'LSTM'}
    RB_NAMES = {'MeanReversion', 'MomentumBreakout', 'MACD_RSI_Confluence'}

    ml_vals = [v for k, v in individual_confs.items() if k in ML_NAMES]
    rb_vals = [v for k, v in individual_confs.items() if k in RB_NAMES]
    all_vals = list(individual_confs.values())

    # ML-internal spread is the primary quality signal
    ml_spread  = (max(ml_vals) - min(ml_vals)) if len(ml_vals) >= 2 else 0.0
    all_spread = (max(all_vals) - min(all_vals)) if len(all_vals) >= 2 else 0.0

    if ml_spread < 0.05 and all_spread < 0.12:
        return 'strong'    # ML models tightly agree, rule-based broadly agrees
    elif ml_spread < 0.12 or all_spread < 0.18:
        return 'moderate'
    else:
        return 'weak'      # Either ML models split, or rule-based strongly contradicts


def get_ensemble_signal_full(ticker, df):
    """
    Extended version of get_ensemble_confidence that also returns
    the full rule-based strategy details (signal text, reason, stop
    levels, etc.) for whichever strategies fired today.

    This is the function the API should call so it can surface
    *why* a rule-based vote happened, not just the confidence number.

    Returns:
        ensemble_conf   (float)        — weighted average P(UP)
        individual      (dict)         — {name: confidence} for every voter
        model_names     (list[str])    — ordered list of who voted
        rule_based_details (dict)      — {strategy_name: full get_latest_signal() dict}
        ml_count        (int)          — how many ML models voted
        rb_count        (int)          — how many rule-based strategies voted
    Returns (None, None, None, {}, 0, 0) if nothing has a signal.
    """
    ML_NAMES = {'RF', 'XGBoost', 'SeqNN', 'LSTM'}

    models = _load_models(ticker)
    confidences = {}
    latest = df.iloc[-1]

    for name, model_obj in models.items():
        try:
            if name == 'TFT':
             # Temporal Fusion Transformer — returns P(UP) float
                from src.tft_model import predict_tft, LOOKBACK
                if len(df) >= LOOKBACK:
                    tft_model, tft_scaler = model_obj
                    result = predict_tft(tft_model, tft_scaler, df)
                    if result is not None:
                        confidences['TFT'] = result['binary_prob']
            elif name in ('SeqNN', 'LSTM'):
             # Sequence model — needs last SEQUENCE_LEN rows
                if len(df) >= SEQUENCE_LEN:
                    model, scaler = model_obj
                    confidences[name] = predict_lstm(model, scaler, df)
            else:
              # RF or XGBoost — single-row prediction
                X = pd.DataFrame([latest[FEATURE_COLS]], columns=FEATURE_COLS)
                confidences[name] = float(model_obj.predict_proba(X)[0][1])
        except Exception:
            pass

    rule_based_confidences, rule_based_details = _get_rule_based_confidences(df)
    confidences.update(rule_based_confidences)

    if not confidences:
        return None, None, None, {}, 0, 0

    total_w   = sum(MODEL_WEIGHTS.get(k, 1.0) for k in confidences)
    ens_conf  = sum(confidences[k] * MODEL_WEIGHTS.get(k, 1.0)
                    for k in confidences) / total_w

    ml_count = sum(1 for k in confidences if k in ML_NAMES)
    rb_count = sum(1 for k in confidences if k not in ML_NAMES)

    return (float(ens_conf), confidences, list(confidences.keys()),
            rule_based_details, ml_count, rb_count)