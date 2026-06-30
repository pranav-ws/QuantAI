"""
src/options_chain.py — QuantAI NSE Options Chain
=================================================

Fetches live options chains from yfinance, enriches every row with
BSM Greeks (Delta, Gamma, Theta, Vega), classifies moneyness, flags
unusual OI/volume, and builds an IV surface across all expiries.

Key output columns added on top of yfinance defaults:
  iv_pct     : IV as percentage (e.g. 24.5 for 24.5%)
  delta      : BSM delta
  gamma      : BSM gamma
  theta      : daily theta in ₹ per share
  vega       : vega per 1% IV move
  moneyness  : 'ATM' | 'ITM' | 'OTM'
  mid        : (bid + ask) / 2 — better than lastPrice for illiquid strikes
  oi_pcr     : put-call ratio by open interest at this strike
  flag       : 'unusual_vol' | 'high_oi' | '' — alert for unusual activity
  lot_size   : NSE lot size for this ticker
  lot_premium: mid × lot_size = cost to buy one contract (₹)
"""

import warnings
import numpy as np# type: ignore
import pandas as pd# type: ignore
from datetime import datetime, date
import yfinance as yf# type: ignore

from src.options_pricing import (
    RISK_FREE_RATE,
    bsm_price,
    bsm_greeks,
    implied_volatility,
    moneyness,
    days_to_expiry,
    expiry_to_T,
    historical_volatility,
    iv_rank,
    iv_percentile,
    get_lot_size,
)

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════
#  SECTION 1 — Fetch raw chain from yfinance
# ══════════════════════════════════════════════════════════

def fetch_expiries(ticker: str) -> list[str]:
    """
    Return all available expiry date strings for a ticker.
    Format: ['2024-01-25', '2024-02-29', ...]
    Equity → monthly expiries (last Thursday).
    NIFTY / BANKNIFTY → weekly + monthly expiries.
    """
    try:
        t = yf.Ticker(ticker)
        return list(t.options)
    except Exception as exc:
        print(f"  ⚠️  Could not fetch expiries for {ticker}: {exc}")
        return []


def fetch_spot(ticker: str) -> float | None:
    """Return latest close price for the underlying."""
    try:
        t    = yf.Ticker(ticker)
        info = t.fast_info
        return float(info.get('lastPrice') or info.get('regularMarketPrice', 0))
    except Exception:
        return None


def _clean_chain_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise yfinance chain columns — fill NaN, clip bad IV values."""
    df = df.copy()

    # yfinance sometimes ships impliedVolatility as 0 or NaN
    df['impliedVolatility'] = pd.to_numeric(df['impliedVolatility'], errors='coerce')
    df['impliedVolatility'] = df['impliedVolatility'].clip(lower=0.01, upper=5.0)
    df['impliedVolatility'] = df['impliedVolatility'].fillna(0.30)

    for col in ('bid', 'ask', 'lastPrice', 'volume', 'openInterest'):
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

    # Mid-price (more stable than lastPrice for low-volume strikes)
    df['mid'] = ((df['bid'] + df['ask']) / 2).where(
        df['bid'] > 0, df['lastPrice']
    )
    df['mid'] = df['mid'].clip(lower=0)

    return df


# ══════════════════════════════════════════════════════════
#  SECTION 2 — Enrich with BSM Greeks
# ══════════════════════════════════════════════════════════

def enrich_with_greeks(
    df: pd.DataFrame,
    spot: float,
    T: float,
    option_type: str,
    r: float = RISK_FREE_RATE,
    lot_size: int = 1,
) -> pd.DataFrame:
    """
    Add BSM Greeks, moneyness classification, and contract-level metrics
    to a raw yfinance options DataFrame.

    For each row:
      1. Use yfinance IV as the vol input (market-calibrated)
      2. Recompute BSM price → compare to mid (shows model/market gap)
      3. Add all 5 primary Greeks
      4. Classify ATM / ITM / OTM
      5. Compute lot-level metrics (lot_premium, lot_theta)
    """
    records = []
    for _, row in df.iterrows():
        K   = float(row['strike'])
        iv  = float(row['impliedVolatility'])
        mid = float(row['mid'])

        try:
            greeks = bsm_greeks(spot, K, T, r, iv, option_type)
            bsm_px = bsm_price(spot, K, T, r, iv, option_type)
            m_class = moneyness(spot, K, option_type)
        except Exception:
            greeks  = {k: 0.0 for k in ('delta','gamma','theta','vega','rho','vanna','charm','vomma')}
            bsm_px  = 0.0
            m_class = 'OTM'

        # ── Unusual activity flags ────────────────────────
        vol = float(row.get('volume', 0))
        oi  = float(row.get('openInterest', 0))
        flag = ''
        if oi > 0 and vol > 5 * oi:
            flag = 'unusual_vol'
        elif oi > 100_000:
            flag = 'high_oi'

        records.append({
            'strike'        : K,
            'moneyness'     : m_class,
            'bid'           : round(float(row['bid']), 2),
            'ask'           : round(float(row['ask']), 2),
            'mid'           : round(mid, 2),
            'last'          : round(float(row['lastPrice']), 2),
            'volume'        : int(vol),
            'open_interest' : int(oi),
            'iv_pct'        : round(iv * 100, 2),
            'bsm_price'     : round(bsm_px, 2),
            'model_gap'     : round(mid - bsm_px, 2),
            'lot_size'      : lot_size,
            'lot_premium'   : round(mid * lot_size, 0),
            'lot_theta'     : round(greeks['theta'] * lot_size, 2),
            'flag'          : flag,
            **greeks,
        })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values('strike').reset_index(drop=True)
    return result


# ══════════════════════════════════════════════════════════
#  SECTION 3 — Full chain with PCR and max-pain
# ══════════════════════════════════════════════════════════

def fetch_full_chain(
    ticker: str,
    expiry_idx: int = 0,
    r: float = RISK_FREE_RATE,
) -> dict | None:
    """
    Fetch and enrich the complete calls + puts chain for one expiry.

    Returns dict:
        {
            'ticker'   : str,
            'spot'     : float,
            'expiry'   : str,
            'dte'      : int,          # days to expiry
            'T'        : float,        # years to expiry
            'calls'    : DataFrame,    # enriched calls
            'puts'     : DataFrame,    # enriched puts
            'pcr_oi'   : float,        # put-call ratio by OI
            'pcr_vol'  : float,        # put-call ratio by volume
            'max_pain' : float,        # max pain strike
            'atm_iv'   : float,        # ATM IV (%)
            'hv_30'    : float,        # 30-day historical vol (%)
            'iv_rank'  : float,        # IVR 0-100
        }
    """
    # ── 1. Fetch expiries + spot ──────────────────────────
    expiries = fetch_expiries(ticker)
    if not expiries:
        return None

    expiry   = expiries[min(expiry_idx, len(expiries) - 1)]
    dte      = days_to_expiry(expiry)
    T        = expiry_to_T(expiry)
    lot_size = get_lot_size(ticker)

    if T <= 0:
        # Try next expiry if current one expired
        if expiry_idx + 1 < len(expiries):
            return fetch_full_chain(ticker, expiry_idx + 1, r)
        return None

    spot = fetch_spot(ticker)
    if not spot or spot <= 0:
        return None

    # ── 2. Fetch raw chain ────────────────────────────────
    try:
        t     = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        calls_raw = _clean_chain_df(chain.calls)
        puts_raw  = _clean_chain_df(chain.puts)
    except Exception as exc:
        print(f"  ❌  Could not fetch chain for {ticker}: {exc}")
        return None

    if calls_raw.empty and puts_raw.empty:
        return None

    # ── 3. Enrich with Greeks ─────────────────────────────
    calls = enrich_with_greeks(calls_raw, spot, T, 'call', r, lot_size)
    puts  = enrich_with_greeks(puts_raw,  spot, T, 'put',  r, lot_size)

    # ── 4. Put-Call Ratio ─────────────────────────────────
    total_call_oi  = calls['open_interest'].sum()
    total_put_oi   = puts['open_interest'].sum()
    total_call_vol = calls['volume'].sum()
    total_put_vol  = puts['volume'].sum()

    pcr_oi  = round(total_put_oi  / (total_call_oi  + 1e-6), 2)
    pcr_vol = round(total_put_vol / (total_call_vol + 1e-6), 2)

    # ── 5. Max Pain ───────────────────────────────────────
    max_pain = _compute_max_pain(calls, puts)

    # ── 6. ATM IV (average of ATM call and put) ───────────
    atm_call = calls[calls['moneyness'] == 'ATM']
    atm_put  = puts[puts['moneyness'] == 'ATM']

    # Fallback: nearest-to-spot if no exact ATM
    if atm_call.empty:
        idx = (calls['strike'] - spot).abs().idxmin()
        atm_call = calls.loc[[idx]]
    if atm_put.empty:
        idx = (puts['strike'] - spot).abs().idxmin()
        atm_put = puts.loc[[idx]]

    atm_iv_call = float(atm_call['iv_pct'].iloc[0]) if not atm_call.empty else 0
    atm_iv_put  = float(atm_put['iv_pct'].iloc[0])  if not atm_put.empty  else 0
    atm_iv      = round((atm_iv_call + atm_iv_put) / 2, 2) if atm_iv_call and atm_iv_put else max(atm_iv_call, atm_iv_put)

    # ── 7. Historical vol + IV Rank ───────────────────────
    hv_30  = _get_historical_vol(ticker)
    ivr    = _get_iv_rank(ticker, atm_iv / 100)

    return {
        'ticker'   : ticker,
        'spot'     : round(spot, 2),
        'expiry'   : expiry,
        'dte'      : dte,
        'T'        : round(T, 6),
        'calls'    : calls,
        'puts'     : puts,
        'pcr_oi'   : pcr_oi,
        'pcr_vol'  : pcr_vol,
        'max_pain' : max_pain,
        'atm_iv'   : atm_iv,
        'hv_30'    : hv_30,
        'iv_rank'  : ivr,
    }


def _compute_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    """
    Max Pain Theory: the expiry price where option buyers suffer maximum loss
    (i.e. writers collect the most premium).

    For each candidate strike K*, compute:
        Pain = Σ OI_call_k × max(K* - k, 0) + Σ OI_put_k × max(k - K*, 0)
    The K* that minimises Pain is the max-pain strike.
    """
    if calls.empty or puts.empty:
        return 0.0

    all_strikes = sorted(set(calls['strike']) | set(puts['strike']))
    call_oi = dict(zip(calls['strike'], calls['open_interest']))
    put_oi  = dict(zip(puts['strike'],  puts['open_interest']))

    min_pain  = float('inf')
    max_pain_strike = all_strikes[0]

    for K_star in all_strikes:
        pain  = sum(call_oi.get(k, 0) * max(K_star - k, 0) for k in all_strikes)
        pain += sum(put_oi.get(k, 0)  * max(k - K_star, 0) for k in all_strikes)
        if pain < min_pain:
            min_pain        = pain
            max_pain_strike = K_star

    return float(max_pain_strike)


def _get_historical_vol(ticker: str, window: int = 30) -> float:
    """Fetch 6 months of prices and compute 30-day HV."""
    try:
        df = yf.download(ticker, period='6mo', progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        hv = historical_volatility(df['Close'], window=window)
        return round(hv * 100, 2)
    except Exception:
        return 0.0


def _get_iv_rank(ticker: str, current_iv: float) -> float:
    """Compute IV Rank using 1-year ATM IV history (proxy via HV)."""
    try:
        df = yf.download(ticker, period='1y', progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        # Proxy IV history with 30-day rolling HV
        returns  = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        hv_series = returns.rolling(30).std() * np.sqrt(252)
        hv_series = hv_series.dropna()
        return iv_rank(current_iv, hv_series)
    except Exception:
        return 50.0


# ══════════════════════════════════════════════════════════
#  SECTION 4 — IV Surface
# ══════════════════════════════════════════════════════════

def build_iv_surface(ticker: str, max_expiries: int = 4) -> pd.DataFrame:
    """
    Build an IV surface across multiple expiries.

    Returns a DataFrame with columns:
        expiry, dte, moneyness_pct (strike/spot × 100), iv_pct

    The moneyness axis uses strike/spot so surfaces from different
    price levels are comparable (100 = ATM, <100 = OTM call/ITM put, etc.)
    """
    expiries = fetch_expiries(ticker)[:max_expiries]
    spot     = fetch_spot(ticker)
    if not spot:
        return pd.DataFrame()

    rows = []
    for expiry in expiries:
        T   = expiry_to_T(expiry)
        dte = days_to_expiry(expiry)
        if T <= 0:
            continue
        try:
            t     = yf.Ticker(ticker)
            chain = t.option_chain(expiry)
            calls = _clean_chain_df(chain.calls)

            for _, row in calls.iterrows():
                K  = float(row['strike'])
                iv = float(row['impliedVolatility']) * 100
                rows.append({
                    'expiry'        : expiry,
                    'dte'           : dte,
                    'strike'        : K,
                    'moneyness_pct' : round(K / spot * 100, 1),
                    'iv_pct'        : round(iv, 2),
                })
        except Exception:
            continue

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
#  SECTION 5 — Print helpers
# ══════════════════════════════════════════════════════════

def print_chain_summary(chain_data: dict, n_strikes: int = 10):
    """Pretty-print a compact options chain around ATM."""
    spot   = chain_data['spot']
    calls  = chain_data['calls']
    puts   = chain_data['puts']
    expiry = chain_data['expiry']
    dte    = chain_data['dte']

    print(f"\n  {'═'*74}")
    print(f"  {chain_data['ticker']}  |  Spot ₹{spot:,.1f}  |  Expiry {expiry}  ({dte} DTE)")
    print(f"  ATM IV: {chain_data['atm_iv']:.1f}%  |  HV30: {chain_data['hv_30']:.1f}%  |  "
          f"IV Rank: {chain_data['iv_rank']:.0f}  |  "
          f"PCR (OI): {chain_data['pcr_oi']:.2f}  |  Max Pain: ₹{chain_data['max_pain']:,.0f}")
    print(f"  {'═'*74}")

    # Choose n_strikes closest to ATM from each side
    call_atm = calls.iloc[(calls['strike'] - spot).abs().argsort()[:n_strikes]]
    put_atm  = puts.iloc[(puts['strike'] - spot).abs().argsort()[:n_strikes]]
    all_strikes = sorted(set(call_atm['strike']) | set(put_atm['strike']))

    call_by_k = call_atm.set_index('strike')
    put_by_k  = put_atm.set_index('strike')

    hdr = (f"  {'CALL':>8} {'δ':>6} {'θ/d':>6} {'IV%':>6} {'OI':>8}  "
           f"{'STRIKE':^8}  "
           f"{'OI':>8} {'IV%':>6} {'θ/d':>6} {'δ':>6} {'PUT':>8}")
    print(hdr)
    print(f"  {'─'*74}")

    for K in all_strikes:
        c = call_by_k.loc[K] if K in call_by_k.index else None
        p = put_by_k.loc[K]  if K in put_by_k.index  else None

        c_mid   = f"₹{c['mid']:>6.1f}"   if c is not None else ' ' * 8
        c_delta = f"{c['delta']:>6.3f}"  if c is not None else ' ' * 6
        c_theta = f"{c['theta']:>6.2f}"  if c is not None else ' ' * 6
        c_iv    = f"{c['iv_pct']:>6.1f}" if c is not None else ' ' * 6
        c_oi    = f"{int(c['open_interest']):>8,}" if c is not None else ' ' * 8

        p_mid   = f"₹{p['mid']:>6.1f}"   if p is not None else ' ' * 8
        p_delta = f"{p['delta']:>6.3f}"  if p is not None else ' ' * 6
        p_theta = f"{p['theta']:>6.2f}"  if p is not None else ' ' * 6
        p_iv    = f"{p['iv_pct']:>6.1f}" if p is not None else ' ' * 6
        p_oi    = f"{int(p['open_interest']):>8,}" if p is not None else ' ' * 8

        atm_mark = " ◄" if abs(K - spot) / spot < 0.02 else ""
        print(f"  {c_mid} {c_delta} {c_theta} {c_iv} {c_oi}  "
              f"{K:^8,.0f}  "
              f"{p_oi} {p_iv} {p_theta} {p_delta} {p_mid}{atm_mark}")

    print(f"  {'─'*74}\n")