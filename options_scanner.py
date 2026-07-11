"""
options_scanner.py — QuantAI Options Scanner CLI
=================================================
Scans for options opportunities using either the trained ensemble model
or a built-in technical fallback (RSI + momentum) if models aren't ready.

Usage:
    python3 options_scanner.py                        # auto-scan (ensemble or technical fallback)
    python3 options_scanner.py --ticker RELIANCE      # single stock deep-dive
    python3 options_scanner.py --ticker RELIANCE --signal BUY --confidence 0.70
    python3 options_scanner.py --expiry 1             # 2nd nearest expiry
    python3 options_scanner.py --min-confidence 0.60  # stricter filter
    python3 options_scanner.py --show-chain           # print full chain table
    python3 options_scanner.py --save                 # save JSON to data/
    python3 options_scanner.py --price RELIANCE 2850 35 0.24  # BSM quick pricer
"""

import argparse
import io
from contextlib import redirect_stdout, redirect_stderr
import json
import os
import sys
import warnings
from datetime import date, datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np # type: ignore
import yfinance as yf # type: ignore

from src.options_pricing import (
    bsm_price, bsm_greeks,
    get_lot_size, RISK_FREE_RATE,
)
from src.options_chain import (
    fetch_full_chain, print_chain_summary,
)
from src.options_strategies import (
    suggest_strategy, select_strikes_from_chain,
    long_call, long_put, bull_call_spread, bear_put_spread,
    long_straddle, iron_condor,
)

# ── Nifty 50 tickers — all using .NS (NSE) suffix ────────
# .BO (Bombay SE) symbols are no longer supported by yfinance.
NIFTY50_NS = [
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


# ══════════════════════════════════════════════════════════
#  SECTION 1 — Technical fallback signal (no models needed)
# ══════════════════════════════════════════════════════════

def _technical_signal(ticker: str) -> dict | None:
    """
    Lightweight signal using RSI-14 + 20/50 SMA crossover + momentum.
    Used when ensemble models haven't been trained yet.

    Rules:
      BUY  : RSI < 60 AND close > SMA20 > SMA50 AND 5d-return > 0
      SELL : RSI > 40 AND close < SMA20 < SMA50 AND 5d-return < 0
      else : HOLD (skip)

    Confidence = 0.58 + (0.12 × number of confirming signals / 3)
    """
    try:
        df = yf.download(ticker, period='120d', progress=False,
                         auto_adjust=True, multi_level_index=False)
        if df.empty or len(df) < 55:
            return None
        if hasattr(df.columns, 'levels'):
            df.columns = [c[0] for c in df.columns]

        close = df['Close'].squeeze()

        # Indicators
        sma20  = close.rolling(20).mean()
        sma50  = close.rolling(50).mean()
        ret5   = (close.iloc[-1] / close.iloc[-6] - 1)

        # RSI-14
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / (loss + 1e-10)
        rsi    = 100 - 100 / (1 + rs)

        last_close = float(close.iloc[-1])
        last_sma20 = float(sma20.iloc[-1])
        last_sma50 = float(sma50.iloc[-1])
        last_rsi   = float(rsi.iloc[-1])
        last_ret5  = float(ret5)

        # Score conditions
        buy_conditions = [
            last_rsi   <  60,
            last_close >  last_sma20,
            last_sma20 >  last_sma50,
            last_ret5  >  0,
        ]
        sell_conditions = [
            last_rsi   >  40,
            last_close <  last_sma20,
            last_sma20 <  last_sma50,
            last_ret5  <  0,
        ]

        buy_score  = sum(buy_conditions)
        sell_score = sum(sell_conditions)

        if buy_score >= 3:
            conf   = 0.56 + 0.04 * buy_score   # 0.68 at 3/4, 0.72 at 4/4
            signal = 'BUY'
        elif sell_score >= 3:
            conf   = 0.56 + 0.04 * sell_score
            signal = 'SELL'
        else:
            return None

        return {
            'ticker'    : ticker,
            'signal'    : signal,
            'confidence': round(conf, 4),
            'source'    : 'technical_fallback',
            'rsi'       : round(last_rsi, 1),
            'sma20'     : round(last_sma20, 2),
            'sma50'     : round(last_sma50, 2),
            'ret_5d_pct': round(last_ret5 * 100, 2),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════
#  SECTION 2 — Signal fetcher (ensemble → technical fallback)
# ══════════════════════════════════════════════════════════

def get_signals(min_confidence: float = 0.58, tickers: list = None) -> list[dict]:
    """
    Try the trained ensemble model first.
    If models aren't trained yet, fall back to the technical signal.
    Always uses .NS tickers — .BO symbols are silently replaced.
    """
    scan_list = tickers or NIFTY50_NS

    # ── Try ensemble models ───────────────────────────────
    ensemble_available = False
    try:
        from src.features import add_features
        from src.ensemble_model import get_ensemble_confidence
        ensemble_available = True
    except ImportError:
        pass

    results   = []
    mode      = 'ensemble' if ensemble_available else 'technical'
    n         = len(scan_list)

    print(f"\n  Scanning {n} stocks  [mode: {mode}]…")
    if not ensemble_available:
        print("  ℹ️  Ensemble models not found → using RSI+SMA technical signal.")
        print("     Run python3 pipeline.py to train models for stronger signals.\n")

    skipped = 0
    for i, ticker in enumerate(scan_list, 1):
        # Force .NS suffix — replace .BO if present
        t = ticker.upper().replace('.BO', '.NS')
        if not t.endswith('.NS') and t not in ('NIFTY', 'BANKNIFTY'):
            t += '.NS'

        display = t.replace('.NS', '')
        print(f"  [{i:>2}/{n}] {display:<16}", end=' ', flush=True)

        try:
            if ensemble_available:
                _buf2 = io.StringIO()
                with redirect_stdout(_buf2), redirect_stderr(_buf2):
                    df = yf.download(t, period='120d', progress=False,
                                     auto_adjust=True, multi_level_index=False)
                if df.empty or len(df) < 30:
                    print('no data')
                    skipped += 1
                    continue
                if hasattr(df.columns, 'levels'):
                    df.columns = [c[0] for c in df.columns]
                # Redirect stdout+stderr to suppress internal debug prints
                # from add_features() and get_ensemble_confidence()
                _buf = io.StringIO()
                with redirect_stdout(_buf), redirect_stderr(_buf):
                    df = add_features(df)
                    conf, signal, _ = get_ensemble_confidence(t, df)
                if conf is None or conf < min_confidence or signal == 'HOLD':
                    print(f'HOLD ({conf:.0%})' if conf else 'skip')
                    continue
                rec = {'ticker': t, 'signal': signal,
                       'confidence': conf, 'source': 'ensemble'}
            else:
                rec = _technical_signal(t)
                if rec is None or rec['confidence'] < min_confidence:
                    print(f"skip")
                    continue

            icon = '🟢' if rec['signal'] == 'BUY' else '🔴'
            print(f"{icon} {rec['signal']}  conf={rec['confidence']:.0%}")
            results.append(rec)

        except Exception as exc:
            print(f'error: {str(exc)[:40]}')
            skipped += 1

    if skipped:
        print(f"\n  ⚠️  {skipped} ticker(s) skipped (delisted / no data)")

    results.sort(key=lambda x: -x['confidence'])
    return results


# ══════════════════════════════════════════════════════════
#  SECTION 3 — Per-stock options analysis
# ══════════════════════════════════════════════════════════

def analyse_ticker(
    ticker     : str,
    signal     : str   = 'BUY',
    confidence : float = 0.65,
    expiry_idx : int   = 0,
    show_chain : bool  = False,
    r          : float = RISK_FREE_RATE,
) -> dict | None:
    """Full options analysis for one ticker."""

    # Normalise ticker
    t = ticker.upper().replace('.BO', '.NS')
    if not t.endswith('.NS') and t not in ('NIFTY', 'BANKNIFTY', 'FINNIFTY'):
        t += '.NS'

    print(f"\n  {'─'*62}")
    print(f"  {t.replace('.NS','')}  |  Signal: {signal}  ({confidence:.0%})")

    chain_data = fetch_full_chain(t, expiry_idx=expiry_idx, r=r)
    if chain_data is None:
        print(f"  ⚠️  Could not fetch options chain — market may be closed "
              f"or {t} has no listed options.")
        return None

    spot     = chain_data['spot']
    T        = chain_data['T']
    dte      = chain_data['dte']
    atm_iv   = chain_data['atm_iv'] / 100
    ivr      = chain_data['iv_rank']
    hv30     = chain_data['hv_30']
    lot_size = get_lot_size(t)
    sigma    = atm_iv if atm_iv > 0.05 else 0.25

    print(f"  Spot ₹{spot:,.1f}  |  Expiry {chain_data['expiry']} ({dte} DTE)")
    print(f"  ATM IV: {atm_iv*100:.1f}%  |  HV30: {hv30:.1f}%  |  "
          f"IV Rank: {ivr:.0f}/100  |  PCR: {chain_data['pcr_oi']:.2f}  |  "
          f"Max Pain: ₹{chain_data['max_pain']:,.0f}")

    if show_chain:
        print_chain_summary(chain_data, n_strikes=8)

    calls = chain_data['calls']
    puts  = chain_data['puts']

    result, rationale = suggest_strategy(
        signal=signal, confidence=confidence, iv_rank=ivr,
        spot=spot, T=T, sigma=sigma, lot_size=lot_size, r=r,
    )

    if result is None:
        print(f"  💤  {rationale}")
        return None

    _print_recommendation(result, calls, puts, spot, T, sigma, lot_size, r)

    return {
        'ticker'      : t,
        'signal'      : signal,
        'confidence'  : round(confidence, 4),
        'spot'        : spot,
        'expiry'      : chain_data['expiry'],
        'dte'         : dte,
        'atm_iv_pct'  : round(atm_iv * 100, 2),
        'hv30_pct'    : hv30,
        'iv_rank'     : ivr,
        'pcr_oi'      : chain_data['pcr_oi'],
        'max_pain'    : chain_data['max_pain'],
        'lot_size'    : lot_size,
        'strategy'    : result.name,
        'rationale'   : rationale,
        'net_premium' : round(result.net_premium * lot_size, 2),
        'max_profit'  : round(result.max_profit * lot_size, 2) if result.max_profit else None,
        'max_loss'    : round(result.max_loss   * lot_size, 2) if result.max_loss   else None,
        'breakevens'  : result.breakevens,
        'net_greeks'  : result.net_greeks,
        'legs'        : [{k: v for k, v in leg.items() if k != 'sigma'}
                         for leg in result.legs],
    }


def _print_recommendation(result, calls, puts, spot, T, sigma, lot_size, r):
    """Print strategy details with live strike data."""
    import pandas as pd # type: ignore

    ls = lot_size
    print(f"\n  🎯  Recommended: {result.name}")
    print(f"  {result.rationale}\n")

    call_by_k = calls.set_index('strike') if not calls.empty else pd.DataFrame()
    put_by_k  = puts.set_index('strike')  if not puts.empty  else pd.DataFrame()

    print(f"  {'ACTION':<6} {'TYPE':<5} {'STRIKE':>9}  {'PREMIUM':>10}  "
          f"{'DELTA':>7}  {'THETA/d':>8}  {'IV%':>6}")
    print(f"  {'─'*62}")

    for leg in result.legs:
        K     = leg['K']
        otype = leg['option_type'].upper()
        act   = leg['action'].upper()
        src   = call_by_k if otype == 'CALL' else put_by_k
        live  = src.loc[K] if K in src.index else None

        prem  = float(live['mid'])   if live is not None else leg['premium']
        delta = float(live['delta']) if live is not None else bsm_greeks(spot, K, T, r, sigma, otype.lower())['delta']
        theta = float(live['theta']) if live is not None else bsm_greeks(spot, K, T, r, sigma, otype.lower())['theta']
        iv    = float(live['iv_pct'])if live is not None else sigma * 100

        print(f"  {act:<6} {otype:<5} ₹{K:>8,.0f}  ₹{prem:>9.2f}  "
              f"{delta:>7.3f}  ₹{theta:>7.2f}  {iv:>6.1f}%")

    print(f"\n  {'─'*62}")
    mp   = f"₹{result.max_profit*ls:>10,.0f}" if result.max_profit else "  Unlimited  "
    ml   = f"₹{abs(result.max_loss)*ls:>10,.0f}" if result.max_loss  else "  Unlimited  "
    sign = "CREDIT" if result.net_premium >= 0 else "DEBIT"
    be   = ' / '.join(f"₹{b:,.1f}" for b in result.breakevens) or "Path-dependent"

    print(f"  Net Premium : ₹{abs(result.net_premium*ls):,.0f} {sign} per contract ({ls} shares)")
    print(f"  Max Profit  : {mp}")
    print(f"  Max Loss    : {ml}")
    print(f"  Breakeven(s): {be}")

    g = result.net_greeks
    print(f"\n  Position Greeks (per contract = {ls} shares):")
    print(f"    Δ Delta  : {g.get('delta',0):>+.4f}  "
          f"(₹{g.get('delta',0)*ls:+.0f} per ₹1 spot move)")
    print(f"    Γ Gamma  : {g.get('gamma',0):>+.6f}")
    print(f"    Θ Theta  : ₹{g.get('theta',0)*ls:>+.2f}/day  "
          f"({'earns' if g.get('theta',0)>0 else 'costs'} ₹{abs(g.get('theta',0)*ls):.1f}/day)")
    print(f"    ν Vega   : ₹{g.get('vega',0)*ls:>+.2f} per 1% IV move")


# ══════════════════════════════════════════════════════════
#  SECTION 4 — Manual BSM pricer
# ══════════════════════════════════════════════════════════

def manual_price_options(ticker: str, spot: float, dte: int, sigma: float):
    """Quick BSM table — no live data needed."""
    T        = dte / 365
    lot_size = get_lot_size(ticker)
    step     = 100 if spot > 5000 else 50
    strikes  = [round(spot / step) * step + i * step for i in range(-3, 4)]

    print(f"\n  {'═'*70}")
    print(f"  BSM Pricer — {ticker.replace('.NS','')}  |  "
          f"Spot ₹{spot:,.0f}  |  {dte}d  |  σ {sigma*100:.0f}%")
    print(f"  {'═'*70}")
    print(f"  {'STRIKE':>9}  {'CALL':>8}  {'C-Δ':>6}  {'C-Θ':>7}  "
          f"│  {'PUT':>8}  {'P-Δ':>6}  {'P-Θ':>7}")
    print(f"  {'─'*70}")

    for K in strikes:
        cp = bsm_price(spot, K, T, RISK_FREE_RATE, sigma, 'call')
        pp = bsm_price(spot, K, T, RISK_FREE_RATE, sigma, 'put')
        cg = bsm_greeks(spot, K, T, RISK_FREE_RATE, sigma, 'call')
        pg = bsm_greeks(spot, K, T, RISK_FREE_RATE, sigma, 'put')
        atm = " ◄ ATM" if K == round(spot / step) * step else ""
        print(
            f"  {K:>9,.0f}  ₹{cp:>7.2f}  {cg['delta']:>6.3f}  ₹{cg['theta']:>6.2f}  "
            f"│  ₹{pp:>7.2f}  {pg['delta']:>6.3f}  ₹{pg['theta']:>6.2f}{atm}"
        )

    atm_k    = round(spot / step) * step
    c_cost   = bsm_price(spot, atm_k, T, RISK_FREE_RATE, sigma, 'call') * lot_size
    p_cost   = bsm_price(spot, atm_k, T, RISK_FREE_RATE, sigma, 'put')  * lot_size
    print(f"\n  Lot size: {lot_size}  |  "
          f"ATM Call/lot: ₹{c_cost:,.0f}  |  "
          f"ATM Put/lot: ₹{p_cost:,.0f}  |  "
          f"Straddle/lot: ₹{c_cost+p_cost:,.0f}")
    print()


# ══════════════════════════════════════════════════════════
#  SECTION 5 — Main
# ══════════════════════════════════════════════════════════

def print_header():
    print(f"\n{'═'*64}")
    print(f"  QuantAI Options Scanner — {date.today()}")
    print(f"  Black-Scholes · All Greeks · AI Strategy Recommender")
    print(f"{'═'*64}")


def main():
    parser = argparse.ArgumentParser(
        description='QuantAI Options Scanner — Greeks + AI strategy recommendations'
    )
    parser.add_argument('--ticker',         type=str,   default=None,
                        help='Single ticker, e.g. RELIANCE or NIFTY')
    parser.add_argument('--signal',         type=str,   default='BUY',
                        choices=['BUY', 'SELL', 'HOLD'],
                        help='Direction for --ticker mode (default BUY)')
    parser.add_argument('--confidence',     type=float, default=0.68,
                        help='Confidence for --ticker mode (default 0.68)')
    parser.add_argument('--expiry',         type=int,   default=0,
                        help='Expiry index: 0=nearest (default), 1=next, etc.')
    parser.add_argument('--min-confidence', type=float, default=0.60,
                        help='Min signal confidence for auto scan (default 0.60)')
    parser.add_argument('--show-chain',     action='store_true',
                        help='Print full options chain table for each ticker')
    parser.add_argument('--save',           action='store_true',
                        help='Save results to data/options_scan_YYYYMMDD.json')
    parser.add_argument('--price',          nargs=4,
                        metavar=('TICKER', 'SPOT', 'DTE', 'IV'),
                        help='BSM pricer: --price RELIANCE 2850 35 0.24')
    args = parser.parse_args()

    print_header()

    # ── Manual BSM pricer ─────────────────────────────────
    if args.price:
        ticker, spot, dte, iv = args.price
        t = ticker.upper()
        if not t.endswith('.NS') and t not in ('NIFTY','BANKNIFTY'):
            t += '.NS'
        manual_price_options(t, float(spot), int(dte), float(iv))
        return

    # ── Single ticker mode ────────────────────────────────
    if args.ticker:
        result = analyse_ticker(
            args.ticker,
            signal     = args.signal,
            confidence = args.confidence,
            expiry_idx = args.expiry,
            show_chain = args.show_chain,
        )
        if args.save and result:
            _save([result])
        print(f"\n{'═'*64}\n")
        return

    # ── Auto scan (ensemble or technical fallback) ────────
    signals = get_signals(args.min_confidence)

    if not signals:
        print("\n  No BUY/SELL signals found above the confidence threshold.")
        print("  Try:  python3 options_scanner.py --ticker RELIANCE --signal BUY")
        print("  Or:   python3 pipeline.py   (to train ensemble models)")
        print(f"\n{'═'*64}\n")
        return

    print(f"\n  {len(signals)} candidate(s) found — analysing options chains…")

    all_results = []
    for s in signals:
        res = analyse_ticker(
            s['ticker'],
            signal     = s['signal'],
            confidence = s['confidence'],
            expiry_idx = args.expiry,
            show_chain = args.show_chain,
        )
        if res:
            all_results.append(res)

    # ── Summary table ─────────────────────────────────────
    if all_results:
        print(f"\n  {'═'*66}")
        print(f"  SUMMARY — {len(all_results)} recommendation(s)")
        print(f"  {'═'*66}")
        print(f"  {'TICKER':<14} {'SIG':<5} {'CONF':>5}  "
              f"{'STRATEGY':<22} {'NET PREMIUM':>12} {'MAX PROFIT':>12}")
        print(f"  {'─'*66}")
        for r in all_results:
            mp    = f"₹{r['max_profit']:>10,.0f}" if r['max_profit'] else "  Unlimited"
            label = "CR" if r['net_premium'] >= 0 else "DR"
            print(
                f"  {r['ticker'].replace('.NS',''):<14} {r['signal']:<5} "
                f"{r['confidence']:.0%}  "
                f"{r['strategy']:<22} "
                f"₹{abs(r['net_premium']):>9,.0f} {label}  {mp}"
            )

    if args.save and all_results:
        _save(all_results)

    print(f"\n{'═'*64}\n")


def _save(results: list):
    os.makedirs('data', exist_ok=True)
    fname = f"data/options_scan_{date.today().strftime('%Y%m%d')}.json"
    with open(fname, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  💾 Saved → {fname}")


if __name__ == '__main__':
    main()