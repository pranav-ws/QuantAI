"""
src/options_pricing.py — QuantAI Options Pricing Engine
========================================================

Black-Scholes-Merton model with full Greek suite + implied volatility solver.

Supported Greeks:
  First-order  : Delta, Theta, Vega, Rho
  Second-order : Gamma, Vanna, Charm, Vomma (Volga)

All outputs are in rupee / per-1%-vol / per-day units so numbers
are directly usable in trade sizing — no unit conversion needed.

NSE context:
  • Index options (NIFTY, BANKNIFTY) are European-style → BSM exact.
  • Equity options are American-style → BSM slightly underprices early-
    exercise premium; for equity options treat BSM as a floor estimate.
  • T (time) is always in calendar years (not trading days).
  • Risk-free rate = India 10-yr G-sec yield (~6.5% p.a.).
"""

import math
import numpy as np# type: ignore
from scipy import stats, optimize# type: ignore
import pandas as pd# type: ignore

# ── Constants ─────────────────────────────────────────────
RISK_FREE_RATE  = 0.065   # India 10-yr G-sec yield
TRADING_DAYS    = 252
CALENDAR_DAYS   = 365

# NSE lot sizes — each contract covers this many shares
NSE_LOT_SIZES = {
    'NIFTY'      : 50,   'BANKNIFTY'  : 15,   'FINNIFTY'   : 40,
    'RELIANCE'   : 250,  'INFY'       : 400,   'TCS'        : 150,
    'HDFCBANK'   : 550,  'ICICIBANK'  : 700,   'SBIN'       : 1500,
    'AXISBANK'   : 1200, 'KOTAKBANK'  : 400,   'HDFC'       : 300,
    'LT'         : 300,  'ITC'        : 3200,  'HINDUNILVR' : 300,
    'BAJFINANCE' : 125,  'WIPRO'      : 3000,  'HCLTECH'    : 700,
    'ONGC'       : 1950, 'COALINDIA'  : 4200,  'NTPC'       : 3750,
    'POWERGRID'  : 4700, 'TATAMOTORS' : 1425,  'MARUTI'     : 100,
    'SUNPHARMA'  : 700,  'DRREDDY'    : 125,   'CIPLA'      : 650,
    'DIVISLAB'   : 200,  'BRITANNIA'  : 200,   'ASIANPAINT' : 200,
    'NESTLEIND'  : 50,   'ULTRACEMCO' : 100,   'GRASIM'     : 375,
    'JSWSTEEL'   : 600,  'TATASTEEL'  : 5500,  'HINDALCO'   : 2150,
    'ADANIPORTS' : 1250, 'ADANIENT'   : 250,   'BAJAJ-AUTO' : 250,
    'EICHERMOT'  : 150,  'HEROMOTOCO' : 300,   'M&M'        : 700,
    'TRENT'      : 375,  'TITAN'      : 375,   'BPCL'       : 4500,
}


def get_lot_size(ticker: str) -> int:
    """Return NSE lot size for a ticker (strips .NS suffix)."""
    key = ticker.replace('.NS', '').upper()
    return NSE_LOT_SIZES.get(key, 1)


# ══════════════════════════════════════════════════════════
#  SECTION 1 — Core BSM Pricing
# ══════════════════════════════════════════════════════════

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """
    Compute BSM d1 and d2.
    d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
    d2 = d1 - σ·√T
    """
    if T <= 0 or sigma <= 0:
        raise ValueError(f"T and sigma must be positive (got T={T}, sigma={sigma})")
    if S <= 0 or K <= 0:
        # math.log(S/K) is a literal division — K=0 (or S=0) crashes with
        # "float division by zero" rather than a useful message. This
        # happens in practice when an auto-computed ATM strike rounds
        # down to 0 for a very small spot price (e.g. spot=0.3 with a
        # ₹50 strike step rounds to 0) — a garbage/typo'd spot value
        # producing a garbage strike, not a real market scenario.
        raise ValueError(
            f"Spot and strike must both be positive (got spot={S}, strike={K}). "
            f"Check the spot price you entered — it looks too small to be a real stock price."
        )
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call',
) -> float:
    """
    Black-Scholes-Merton price for a European option.

    Args:
        S           : current spot price (₹)
        K           : strike price (₹)
        T           : time to expiry in years (e.g. 30/365 for 30 days)
        r           : risk-free rate (annualised, e.g. 0.065)
        sigma       : implied/historical volatility (annualised, e.g. 0.25 = 25%)
        option_type : 'call' or 'put'

    Returns:
        Option premium per share in ₹
    """
    if T <= 0:
        # At expiry: return intrinsic value only
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        return intrinsic

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    discount = math.exp(-r * T)

    if option_type == 'call':
        return S * stats.norm.cdf(d1) - K * discount * stats.norm.cdf(d2)
    else:
        return K * discount * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def option_intrinsic(S: float, K: float, option_type: str = 'call') -> float:
    """Intrinsic value = payoff if exercised right now."""
    if option_type == 'call':
        return max(S - K, 0.0)
    return max(K - S, 0.0)


def time_value(S, K, T, r, sigma, option_type='call') -> float:
    """Time value = total premium - intrinsic value."""
    premium   = bsm_price(S, K, T, r, sigma, option_type)
    intrinsic = option_intrinsic(S, K, option_type)
    return max(premium - intrinsic, 0.0)


# ══════════════════════════════════════════════════════════
#  SECTION 2 — Full Greek Suite
# ══════════════════════════════════════════════════════════

def bsm_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call',
) -> dict:
    """
    Compute the full Greek suite for a European BSM option.

    Units chosen for direct usability in NSE trading:
      Delta  → ₹ change in premium per ₹1 move in spot
      Gamma  → change in Delta per ₹1 move in spot
      Theta  → ₹ premium decay per calendar day  (negative = decay)
      Vega   → ₹ change in premium per 1% rise in IV
      Rho    → ₹ change in premium per 1% rise in risk-free rate
      Vanna  → change in Delta per 1% rise in IV
      Charm  → change in Delta per calendar day (Delta decay)
      Vomma  → change in Vega per 1% rise in IV (convexity of vol)

    Returns dict with all 8 Greeks.
    """
    if T <= 0:
        # At expiry all greeks collapse
        delta = 1.0 if (option_type == 'call' and S > K) else (
                -1.0 if (option_type == 'put' and S < K) else 0.0)
        return {k: 0.0 for k in ('delta','gamma','theta','vega','rho','vanna','charm','vomma')} | {'delta': delta}

    d1, d2   = _d1_d2(S, K, T, r, sigma)
    sqrt_T   = math.sqrt(T)
    pdf_d1   = stats.norm.pdf(d1)          # N'(d1) — standard normal PDF
    discount = math.exp(-r * T)

    # ── First-order Greeks ────────────────────────────────
    if option_type == 'call':
        delta = stats.norm.cdf(d1)
        theta = (
            (-S * pdf_d1 * sigma / (2 * sqrt_T))
            - r * K * discount * stats.norm.cdf(d2)
        ) / CALENDAR_DAYS                   # per calendar day
        rho   = K * T * discount * stats.norm.cdf(d2)  / 100   # per 1%
    else:
        delta = stats.norm.cdf(d1) - 1.0
        theta = (
            (-S * pdf_d1 * sigma / (2 * sqrt_T))
            + r * K * discount * stats.norm.cdf(-d2)
        ) / CALENDAR_DAYS
        rho   = -K * T * discount * stats.norm.cdf(-d2) / 100

    # ── Second-order (same sign for call and put) ─────────
    gamma  = pdf_d1 / (S * sigma * sqrt_T)
    vega   = S * pdf_d1 * sqrt_T / 100        # per 1% change in sigma

    # ── Cross-derivatives ─────────────────────────────────
    # Vanna: ∂Delta/∂sigma  (how delta changes when IV moves 1%)
    vanna  = -pdf_d1 * d2 / sigma / 100

    # Charm: ∂Delta/∂t  (how delta drifts as time passes — daily)
    charm  = -pdf_d1 * (
        2 * r * T - d2 * sigma * sqrt_T
    ) / (2 * T * sigma * sqrt_T) / CALENDAR_DAYS
    if option_type == 'put':
        charm = -charm    # put delta drifts toward 0 from -1

    # Vomma / Volga: ∂Vega/∂sigma  (convexity of vega — per 1%)
    vomma  = vega * d1 * d2 / sigma / 100

    return {
        'delta' : round(delta, 4),
        'gamma' : round(gamma, 6),
        'theta' : round(theta, 4),    # per calendar day
        'vega'  : round(vega,  4),    # per 1% sigma
        'rho'   : round(rho,   4),    # per 1% rate
        'vanna' : round(vanna, 6),
        'charm' : round(charm, 6),
        'vomma' : round(vomma, 4),
    }


def moneyness(S: float, K: float, option_type: str = 'call') -> str:
    """Classify strike as ATM / ITM / OTM (1% band around spot)."""
    ratio = S / K
    if option_type == 'call':
        if abs(ratio - 1.0) < 0.01:
            return 'ATM'
        return 'ITM' if ratio > 1.0 else 'OTM'
    else:
        if abs(ratio - 1.0) < 0.01:
            return 'ATM'
        return 'ITM' if ratio < 1.0 else 'OTM'


# ══════════════════════════════════════════════════════════
#  SECTION 3 — Implied Volatility Solver
# ══════════════════════════════════════════════════════════

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = RISK_FREE_RATE,
    option_type: str = 'call',
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """
    Back-solve Black-Scholes for implied volatility given a market price.

    Algorithm:
      1. Reject if price < intrinsic (arbitrage) → return None
      2. Corrado-Miller closed-form initial guess (fast convergence)
      3. Newton-Raphson refinement  (converges in 3-8 iterations usually)
      4. Brent's method fallback    (robust, guaranteed for well-defined cases)

    Returns:
        Annualised implied volatility (e.g. 0.25 = 25%), or None if unsolvable.
    """
    if T <= 0:
        return None

    intrinsic = option_intrinsic(S, K, option_type)
    if market_price < intrinsic - 0.01:
        return None   # price below intrinsic → arbitrage → no solution

    # ── Initial guess: Corrado-Miller (1996) ──────────────
    # Works for any moneyness; much better starting point than σ₀ = 0.3
    F   = S * math.exp(r * T)          # forward price
    mid = (market_price - (F - K) / 2)
    try:
        disc  = math.sqrt(
            mid ** 2 - (F - K) ** 2 / math.pi
        )
        sigma = (math.sqrt(2 * math.pi / T) * (mid + disc)) / (F + K)
        sigma = max(0.01, min(sigma, 3.0))
    except (ValueError, ZeroDivisionError):
        sigma = 0.30   # safe fallback

    # ── Newton-Raphson ────────────────────────────────────
    for _ in range(max_iter):
        try:
            price = bsm_price(S, K, T, r, sigma, option_type)
        except Exception:
            break

        diff = market_price - price
        if abs(diff) < tol:
            return round(sigma, 6)

        # Vega in natural units (not per 1%) for the update step
        d1, _  = _d1_d2(S, K, T, r, sigma)
        vega_raw = S * stats.norm.pdf(d1) * math.sqrt(T)
        if abs(vega_raw) < 1e-10:
            break

        sigma  += diff / vega_raw
        sigma   = max(0.001, min(sigma, 5.0))

    # ── Brent's method fallback ───────────────────────────
    try:
        iv = optimize.brentq(
            lambda s: bsm_price(S, K, T, r, s, option_type) - market_price,
            0.001, 5.0,
            xtol=tol, maxiter=500,
        )
        return round(iv, 6)
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════
#  SECTION 4 — Historical & Implied Volatility Utilities
# ══════════════════════════════════════════════════════════

def historical_volatility(
    prices: 'pd.Series',
    window: int = 30,
    annualise: bool = True,
) -> float:
    """
    Compute close-to-close historical volatility (annualised).

    Uses log returns — more stationary than simple returns.
    window: rolling window in trading days (30 = 1 month, 252 = 1 year).
    """
    import pandas as pd # type: ignore
    returns = np.log(prices / prices.shift(1)).dropna()
    hv = returns.rolling(window).std().iloc[-1]
    return float(hv * math.sqrt(TRADING_DAYS)) if annualise else float(hv)


def iv_rank(
    current_iv: float,
    iv_history: 'pd.Series',
) -> float:
    """
    IV Rank (IVR): where current IV sits within its 52-week range.

    IVR = (current_IV - 52wk_low) / (52wk_high - 52wk_low) × 100

    Returns 0-100:
      0-30   : Low IV  → options cheap → prefer buying
      30-70  : Medium  → balanced approach
      70-100 : High IV → options expensive → prefer selling
    """
    iv_min = iv_history.min()
    iv_max = iv_history.max()
    if iv_max <= iv_min:
        return 50.0
    return round((current_iv - iv_min) / (iv_max - iv_min) * 100, 1)


def iv_percentile(
    current_iv: float,
    iv_history: 'pd.Series',
) -> float:
    """
    IV Percentile: % of days in the past year where IV was below current IV.

    Preferred by many traders over IV Rank because it is not distorted
    by single extreme spikes that inflate iv_max.

    Returns 0-100.
    """
    pct = (iv_history < current_iv).mean() * 100
    return round(pct, 1)


def days_to_expiry(expiry_str: str) -> int:
    """Return calendar days from today to expiry (YYYY-MM-DD string)."""
    from datetime import datetime
    expiry_dt = datetime.strptime(expiry_str, '%Y-%m-%d')
    delta = expiry_dt - datetime.now()
    return max(0, delta.days)


def expiry_to_T(expiry_str: str) -> float:
    """Convert expiry string to T in years."""
    dte = days_to_expiry(expiry_str)
    return dte / CALENDAR_DAYS


# ══════════════════════════════════════════════════════════
#  SECTION 5 — P&L Simulation
# ══════════════════════════════════════════════════════════

def pnl_at_expiry(
    S_range: np.ndarray,
    legs: list,
) -> np.ndarray:
    """
    Compute combined P&L at expiry across a spot price range.

    legs: list of dicts, each:
        {
          'K'          : strike (₹),
          'option_type': 'call' or 'put',
          'action'     : 'buy' or 'sell',
          'premium'    : option premium paid/received (₹ per share),
          'quantity'   : number of contracts (positive int),
          'lot_size'   : shares per contract,
        }

    Returns P&L per lot in ₹ for each spot price in S_range.
    """
    pnl = np.zeros(len(S_range))
    for leg in legs:
        K        = leg['K']
        opt_type = leg['option_type']
        sign     = 1 if leg['action'] == 'buy' else -1
        premium  = leg['premium']
        qty      = leg.get('quantity', 1)
        lot      = leg.get('lot_size', 1)

        if opt_type == 'call':
            payoff = np.maximum(S_range - K, 0) - premium
        else:
            payoff = np.maximum(K - S_range, 0) - premium

        pnl += sign * payoff * qty * lot

    return pnl


def breakeven_points(legs: list, spot: float, precision: float = 0.5) -> list:
    """
    Find breakeven spot prices by scanning a fine grid around current spot.
    Returns list of ₹ breakeven prices.
    """
    S_range = np.arange(spot * 0.5, spot * 1.5, precision)
    pnl     = pnl_at_expiry(S_range, legs)

    # Zero-crossings
    crossings = []
    for i in range(len(pnl) - 1):
        if pnl[i] * pnl[i + 1] < 0:
            # Linear interpolation
            be = S_range[i] - pnl[i] * (S_range[i + 1] - S_range[i]) / (pnl[i + 1] - pnl[i])
            crossings.append(round(be, 1))
    return crossings