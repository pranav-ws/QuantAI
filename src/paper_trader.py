import pandas as pd # type: ignore # type: ignore
import numpy as np# type: ignore
import sqlite3
import joblib# type: ignore
import os
import json
from datetime import datetime, date
import yfinance as yf# type: ignore
from src.features import add_features
from src.risk import RiskManager
from src.model import FEATURE_COLS
from src.pattern_scanner import scan_patterns, get_pattern_confidence_boost
from src.regime_detector import detect_regime, get_regime_threshold, get_regime_summary
from src.sector_rotation import (is_rotation_buy,
    is_rotation_avoid, get_sector_for_ticker)
from src.factor_model import get_factor_confidence_boost
from src.drift_detector import log_prediction


DB_PATH      = os.path.join('data', 'quantai.db')
TRADES_PATH  = os.path.join('data', 'paper_trades.json')
CAPITAL_PATH = os.path.join('data', 'paper_capital.json')

# ── Persistence helpers ──────────────────────────────────
def load_state():
    """Loads paper trading state from disk."""
    if os.path.exists(CAPITAL_PATH):
        with open(CAPITAL_PATH) as f:
            cap = json.load(f)
    else:
        cap = {'capital': 100000.0, 'peak': 100000.0, 'start': str(date.today())}

    if os.path.exists(TRADES_PATH):
        with open(TRADES_PATH) as f:
            trades = json.load(f)
    else:
        trades = []

    return cap, trades

def save_state(cap, trades):
    """Saves paper trading state to disk."""
    os.makedirs('data', exist_ok=True)
    with open(CAPITAL_PATH, 'w') as f:
        json.dump(cap, f, indent=2)
    with open(TRADES_PATH, 'w') as f:
        json.dump(trades, f, indent=2, default=str)

# ── Live data fetcher ─────────────────────────────────────
def fetch_latest_data(ticker, lookback_days=120):
    """Downloads recent data and builds features for prediction."""
    df = yf.download(ticker, period=f'{lookback_days}d',
                     progress=False, auto_adjust=True)
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = add_features(df)
    return df

# ── Signal generator ──────────────────────────────────────
def get_signal(ticker):
    """
    Gets ensemble signal combining RF + XGBoost + SeqNN/LSTM.
    Returns (prediction, confidence, current_price, model_type).
    """
    df = fetch_latest_data(ticker)
    if df is None or len(df) < 30:
        return None, None, None, None

    try:
        from src.ensemble_model import get_ensemble_confidence
        ens_conf, individual, models_used = get_ensemble_confidence(ticker, df)

        if ens_conf is None:
            return None, None, None, None

        current_price = float(df.iloc[-1]['Close'])
        n             = len(models_used) if models_used else 1
        model_type    = f"Ensemble({n})"

        # ── Pattern Scanner boost ─────────────────────────
        try:
            patterns, net_score = scan_patterns(ticker)
            if patterns:
                boost      = get_pattern_confidence_boost(
                    patterns, net_score
                )
                ens_conf   = round(ens_conf + boost, 4)
                model_type = f"Ensemble({n})+Patterns"

                # Hard block: strongly bearish pattern → skip
                if net_score < -1.0:
                    return 0, ens_conf, current_price, model_type
        except Exception:
            pass


        # ── Factor Model boost ────────────────────────────
        try:
            factor_boost = get_factor_confidence_boost(ticker)
            if factor_boost != 0:
                ens_conf   = round(
                    min(max(ens_conf + factor_boost, 0), 1), 4
                )
                model_type += f'+F({factor_boost:+.2f})'
        except Exception:
            pass

        # ── Sector Rotation Filter ────────────────────────
        try:
            if is_rotation_avoid(ticker):
                # Stock in bottom sector — skip regardless
                return 0, ens_conf, current_price, model_type
            if is_rotation_buy(ticker):
                # Stock in top sector — small confidence boost
                ens_conf   = round(
                    min(ens_conf + 0.03, 1.0), 4
                )
                model_type += '+Rotation'
        except Exception:
            pass

        prediction = 1 if ens_conf >= 0.58 else 0

        # Log prediction for drift monitoring
        try:
            log_prediction(ticker, prediction,
                           ens_conf, current_price)
        except Exception:
            pass

        return prediction, ens_conf, current_price, model_type
    except Exception as e:
        print(f"  ⚠️  {ticker} ensemble error: {e}")
        return None, None, None, None
