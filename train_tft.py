"""
train_tft.py — Train Temporal Fusion Transformer for all Nifty 50 stocks
=========================================================================
Trains one TFT per stock, saves to models/{ticker}_tft_model/.
After training, TFT is automatically picked up by the ensemble.

Usage:
    python3 train_tft.py                          # train all 50 stocks
    python3 train_tft.py --ticker RELIANCE        # single stock
    python3 train_tft.py --tickers RELIANCE INFY TCS
    python3 train_tft.py --epochs 50              # fewer epochs (faster)
    python3 train_tft.py --skip-trained           # skip already-trained tickers
    python3 train_tft.py --eval                   # evaluate without retraining

Training time (approximate):
    Per stock  : 3–8 min on CPU (Mac M-series / Intel)
    All 50     : 3–7 hours on CPU  (run overnight)
    With GPU   : ~20 min total

After training, re-run pipeline.py to get updated ensemble signals.
"""

import argparse
import io
import json
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np# type: ignore
import yfinance as yf# type: ignore

from src.features import add_features
from src.tft_model import (
    train_tft, load_tft, predict_tft,
    tft_available, FEATURE_COLS, LOOKBACK,
)

# ── Nifty 50 tickers ─────────────────────────────────────
NIFTY50 = [
    'RELIANCE.NS',  'TCS.NS',       'HDFCBANK.NS',  'ICICIBANK.NS',
    'INFY.NS',      'HINDUNILVR.NS','ITC.NS',        'SBIN.NS',
    'BAJFINANCE.NS','BHARTIARTL.NS','KOTAKBANK.NS',  'LT.NS',
    'AXISBANK.NS',  'ASIANPAINT.NS','MARUTI.NS',     'SUNPHARMA.NS',
    'TITAN.NS',     'ULTRACEMCO.NS','WIPRO.NS',      'NTPC.NS',
    'POWERGRID.NS', 'HCLTECH.NS',   'TECHM.NS',      'NESTLEIND.NS',
    'BRITANNIA.NS', 'DRREDDY.NS',   'CIPLA.NS',      'DIVISLAB.NS',
    'BPCL.NS',      'COALINDIA.NS', 'ONGC.NS',       'TATAMOTORS.NS',
    'TATASTEEL.NS', 'JSWSTEEL.NS',  'HINDALCO.NS',   'GRASIM.NS',
    'ADANIPORTS.NS','BAJAJ-AUTO.NS','EICHERMOT.NS',  'HEROMOTOCO.NS',
    'M&M.NS',       'TRENT.NS',     'ADANIENT.NS',   'HDFCLIFE.NS',
    'SBILIFE.NS',   'APOLLOHOSP.NS','BAJAJFINSV.NS', 'INDUSINDBK.NS',
    'SHRIRAMFIN.NS','BEL.NS',
]


def fetch_data(ticker: str, years: int = 3):
    """
    Download OHLCV + compute features for one ticker.

    FIX: yfinance only supports these period strings:
         1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
         '3y' is INVALID — use start/end dates instead.
    """
    end_date   = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    try:
        _buf = io.StringIO()
        with redirect_stdout(_buf), redirect_stderr(_buf):
            df = yf.download(
                ticker,
                start      = start_date,
                end        = end_date,
                progress   = False,
                auto_adjust= True,
                multi_level_index=False,
            )
    except Exception as exc:
        print(f'download error ({exc.__class__.__name__})')
        return None

    if df is None or df.empty:
        print('empty download')
        return None

    # Flatten MultiIndex columns if present
    if hasattr(df.columns, 'levels'):
        df.columns = [c[0] for c in df.columns]

    # Verify required OHLCV columns exist
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f'missing columns: {missing}')
        return None

    if len(df) < LOOKBACK + 60:
        print(f'only {len(df)} rows (need {LOOKBACK+60}+)')
        return None

    _buf2 = io.StringIO()
    with redirect_stdout(_buf2), redirect_stderr(_buf2):
        try:
            df = add_features(df)
        except Exception as exc:
            print(f'feature error ({exc.__class__.__name__})')
            return None

    # Keep only the columns we actually need (some may be missing)
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < len(FEATURE_COLS):
        print(f'only {len(available)}/{len(FEATURE_COLS)} features available')
        if len(available) < 10:      # too few features to be useful
            return None

    needed = available + (['Close'] if 'Close' in df.columns else [])
    df = df[needed].dropna()

    if len(df) < LOOKBACK + 60:
        print(f'after dropna: only {len(df)} rows')
        return None

    return df


def train_one(ticker: str, epochs: int, verbose: int = 0) -> dict:
    """Download data, train TFT, return result dict."""
    df = fetch_data(ticker)
    if df is None:
        return {'ticker': ticker, 'status': 'skipped', 'reason': 'insufficient data'}

    print(f"  [{ticker.replace('.NS','')}]  {len(df)} rows  ", end='', flush=True)
    try:
        result = train_tft(df, ticker, epochs=epochs, verbose=verbose)
        if 'error' in result:
            return {'ticker': ticker, 'status': 'failed', 'reason': result['error']}
        result['status'] = 'trained'
        return result
    except Exception as exc:
        return {'ticker': ticker, 'status': 'failed', 'reason': str(exc)[:80]}


def evaluate_one(ticker: str) -> dict:
    """Run inference on latest data and show forecast."""
    df = fetch_data(ticker, years=1)
    if df is None:
        return {'ticker': ticker, 'status': 'no data'}

    model, scaler = load_tft(ticker)
    if model is None:
        return {'ticker': ticker, 'status': 'not trained'}

    result = predict_tft(model, scaler, df)
    if result is None:
        return {'ticker': ticker, 'status': 'prediction failed'}

    spot = float(df['Close'].iloc[-1])
    return {
        'ticker'    : ticker,
        'spot'      : round(spot, 2),
        'p10_pct'   : round(result['p10'] * 100, 2),
        'p50_pct'   : round(result['p50'] * 100, 2),
        'p90_pct'   : round(result['p90'] * 100, 2),
        'signal'    : result['signal'],
        'confidence': result['confidence'],
        'status'    : 'ok',
    }


def print_header():
    print(f"\n{'═'*62}")
    print(f"  QuantAI — Temporal Fusion Transformer Training")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print(f"{'═'*62}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train TFT models for Nifty 50 stocks'
    )
    parser.add_argument('--ticker',       type=str,  default=None)
    parser.add_argument('--tickers',      nargs='+', default=None)
    parser.add_argument('--epochs',       type=int,  default=80)
    parser.add_argument('--skip-trained', action='store_true',
                        help='Skip tickers that already have a saved model')
    parser.add_argument('--eval',         action='store_true',
                        help='Evaluate saved models without retraining')
    parser.add_argument('--save-results', action='store_true',
                        help='Save training results to data/tft_training.json')
    parser.add_argument('--verbose',      type=int,  default=0,
                        help='Keras verbosity (0=silent, 1=progress bar)')
    args = parser.parse_args()

    print_header()

    # ── Determine ticker list ─────────────────────────────
    if args.ticker:
        t = args.ticker.upper()
        tickers = [t if t.endswith('.NS') else t + '.NS']
    elif args.tickers:
        tickers = [
            t.upper() if t.upper().endswith('.NS') else t.upper() + '.NS'
            for t in args.tickers
        ]
    else:
        tickers = NIFTY50

    # ── Evaluation mode ───────────────────────────────────
    if args.eval:
        print("  Evaluating saved TFT models…\n")
        print(f"  {'TICKER':<16} {'SIGNAL':<6} {'CONF':>5}  "
              f"{'P10%':>7}  {'P50%':>7}  {'P90%':>7}")
        print(f"  {'─'*55}")
        for ticker in tickers:
            r = evaluate_one(ticker)
            if r['status'] == 'ok':
                icon = '🟢' if r['signal'] == 'BUY' else ('🔴' if r['signal'] == 'SELL' else '⚪')
                print(
                    f"  {ticker.replace('.NS',''):<16} "
                    f"{icon} {r['signal']:<4} {r['confidence']:.0%}  "
                    f"{r['p10_pct']:>+7.2f}%  {r['p50_pct']:>+7.2f}%  {r['p90_pct']:>+7.2f}%"
                )
            else:
                print(f"  {ticker.replace('.NS',''):<16} {r['status']}")
        print()
        return

    # ── Training mode ─────────────────────────────────────
    if args.skip_trained:
        tickers_to_train = [t for t in tickers if not tft_available(t)]
        skipped = len(tickers) - len(tickers_to_train)
        if skipped:
            print(f"  Skipping {skipped} already-trained ticker(s).")
        tickers = tickers_to_train

    if not tickers:
        print("  All tickers already trained. Use --eval to see forecasts.")
        return

    print(f"  Training {len(tickers)} ticker(s)  |  epochs={args.epochs}\n")

    results   = []
    trained   = 0
    failed    = 0

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:>2}/{len(tickers)}] ", end='', flush=True)
        r = train_one(ticker, args.epochs, args.verbose)
        results.append(r)

        if r['status'] == 'trained':
            trained += 1
            print(f"  MAE={r.get('mae','?'):.4f}  "
                  f"Dir={r.get('hit_rate','?'):.1%}  "
                  f"Cov={r.get('coverage','?'):.1%}")
        else:
            failed += 1
            print(f"  ❌ {r.get('reason','unknown error')}")

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  Training complete:  {trained} trained  |  {failed} failed")
    print(f"\n  Metric glossary:")
    print(f"    MAE        — mean absolute error on 5-day return forecast")
    print(f"    Direction  — % of times P50 sign matches actual return sign")
    print(f"    Coverage   — % of actual returns inside [P10, P90] band")
    print(f"                 target ≈ 80% (well-calibrated uncertainty)")

    if trained:
        dirs = [r['hit_rate'] for r in results if r.get('hit_rate')]
        covs = [r['coverage'] for r in results if r.get('coverage')]
        if dirs:
            print(f"\n  Avg direction accuracy : {sum(dirs)/len(dirs):.1%}")
        if covs:
            print(f"  Avg P10-P90 coverage   : {sum(covs)/len(covs):.1%}")

    print(f"\n  Next steps:")
    print(f"    python3 train_tft.py --eval          # see current forecasts")
    print(f"    python3 pipeline.py                  # retrain full ensemble")
    print(f"    python3 options_scanner.py           # options with TFT signals")

    if args.save_results:
        os.makedirs('data', exist_ok=True)
        path = 'data/tft_training.json'
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  💾 Results saved → {path}")

    print(f"\n{'═'*62}\n")


if __name__ == '__main__':
    main()