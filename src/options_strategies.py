"""
src/options_strategies.py — QuantAI Options Strategy Engine
============================================================

Nine production-ready strategies with:
  • P&L mechanics (max profit, max loss, breakevens)
  • Combined Greeks for the full position
  • Payoff curve data for charting
  • AI strategy suggester tied to ensemble signal + IV Rank

Strategy decision matrix (integrated with QuantAI ensemble):

  Signal   | Confidence | IV Rank | Recommended
  ─────────────────────────────────────────────────
  BUY      |  ≥ 70%    |  ≤ 30   | Long Call
  BUY      |  ≥ 70%    |  > 30   | Bull Call Spread
  BUY      |  ≥ 58%    |  ≤ 40   | Bull Call Spread
  BUY      |  ≥ 58%    |  > 40   | Covered Call (own stock)
  SELL     |  ≥ 70%    |  ≤ 30   | Long Put
  SELL     |  ≥ 70%    |  > 30   | Bear Put Spread
  SELL     |  ≥ 58%    |  any    | Bear Put Spread
  HOLD/any |  any      |  ≥ 70   | Iron Condor
  HOLD/any |  any      |  ≥ 80   | Short Straddle (aggressive)
  any      |  < 58%    |  any    | No trade — wait

For NSE: multiply all ₹/share metrics by lot_size to get per-contract ₹.
"""

import math
import numpy as np# type: ignore
import pandas as pd# type: ignore
from dataclasses import dataclass, field
from typing import Optional

from src.options_pricing import (
    RISK_FREE_RATE,
    bsm_price,
    bsm_greeks,
    pnl_at_expiry,
    breakeven_points,
    get_lot_size,
)


# ── Strategy result container ─────────────────────────────
@dataclass
class StrategyResult:
    name          : str
    legs          : list                  # list of leg dicts
    net_premium   : float                 # + = credit, - = debit
    max_profit    : Optional[float]       # None = "Unlimited"
    max_loss      : Optional[float]       # None = "Unlimited"
    breakevens    : list[float]
    net_greeks    : dict
    payoff_spots  : list[float]
    payoff_values : list[float]
    rationale     : str = ''
    lot_size      : int = 1

    def per_lot(self) -> dict:
        """Scale key metrics by lot size for one NSE contract."""
        ls = self.lot_size
        return {
            'net_premium_lot' : round(self.net_premium * ls, 0),
            'max_profit_lot'  : round(self.max_profit  * ls, 0) if self.max_profit  is not None else None,
            'max_loss_lot'    : round(self.max_loss    * ls, 0) if self.max_loss    is not None else None,
        }

    def summary(self) -> str:
        mp = f"₹{self.max_profit * self.lot_size:,.0f}" if self.max_profit is not None else "Unlimited"
        ml = f"₹{abs(self.max_loss) * self.lot_size:,.0f}" if self.max_loss is not None else "Unlimited"
        be = ' / '.join(f"₹{b:,.1f}" for b in self.breakevens)
        sign = 'DEBIT' if self.net_premium < 0 else 'CREDIT'
        return (
            f"  Strategy  : {self.name}\n"
            f"  Premium   : ₹{abs(self.net_premium) * self.lot_size:,.0f} {sign} per contract\n"
            f"  Max Profit: {mp} per contract\n"
            f"  Max Loss  : {ml} per contract\n"
            f"  Breakeven : {be}\n"
            f"  Delta     : {self.net_greeks.get('delta', '?')}\n"
            f"  Theta/day : ₹{self.net_greeks.get('theta', 0) * self.lot_size:+.2f}\n"
            f"  Vega/1%   : ₹{self.net_greeks.get('vega', 0) * self.lot_size:+.2f}\n"
        )


def _build_payoff_curve(legs: list, spot: float) -> tuple[list, list]:
    """Generate 200-point payoff curve around spot."""
    S_range = np.linspace(spot * 0.60, spot * 1.40, 200)
    pnl     = pnl_at_expiry(S_range, legs)
    return S_range.round(2).tolist(), pnl.round(2).tolist()


def _combine_greeks(legs: list, spot: float, T: float, r: float, lot: int = 1) -> dict:
    """Aggregate Greeks across all legs of a strategy."""
    combined = {k: 0.0 for k in ('delta','gamma','theta','vega','rho','vanna','charm','vomma')}
    for leg in legs:
        sign  = 1 if leg['action'] == 'buy' else -1
        qty   = leg.get('quantity', 1)
        sigma = leg.get('sigma', 0.25)
        K     = leg['K']
        otype = leg['option_type']
        g = bsm_greeks(spot, K, T, r, sigma, otype)
        for key in combined:
            combined[key] += sign * qty * g[key]
    return {k: round(v, 4) for k, v in combined.items()}


# ══════════════════════════════════════════════════════════
#  SECTION 1 — Individual Strategies
# ══════════════════════════════════════════════════════════

def long_call(
    spot: float, K: float, T: float, sigma: float,
    r: float = RISK_FREE_RATE, lot_size: int = 1, market_price: float = None,
) -> StrategyResult:
    """
    Buy 1 call.

    View    : Strongly bullish
    Risk    : Limited (premium paid)
    Reward  : Unlimited
    IV bias : Low IV preferred (options cheaper to buy)
    """
    premium = market_price if market_price else bsm_price(spot, K, T, r, sigma, 'call')
    legs = [{'K': K, 'option_type': 'call', 'action': 'buy',
              'premium': premium, 'quantity': 1, 'lot_size': lot_size, 'sigma': sigma}]

    spots, payoffs = _build_payoff_curve(legs, spot)
    return StrategyResult(
        name         = 'Long Call',
        legs         = legs,
        net_premium  = -premium,
        max_profit   = None,
        max_loss     = -premium,
        breakevens   = [round(K + premium, 2)],
        net_greeks   = _combine_greeks(legs, spot, T, r, lot_size),
        payoff_spots = spots,
        payoff_values= payoffs,
        rationale    = 'Strongly bullish outlook; unlimited upside with defined risk',
        lot_size     = lot_size,
    )


def long_put(
    spot: float, K: float, T: float, sigma: float,
    r: float = RISK_FREE_RATE, lot_size: int = 1, market_price: float = None,
) -> StrategyResult:
    """
    Buy 1 put.

    View    : Strongly bearish
    Risk    : Limited (premium paid)
    Reward  : Large (spot can fall toward 0)
    IV bias : Low IV preferred
    """
    premium = market_price if market_price else bsm_price(spot, K, T, r, sigma, 'put')
    legs = [{'K': K, 'option_type': 'put', 'action': 'buy',
              'premium': premium, 'quantity': 1, 'lot_size': lot_size, 'sigma': sigma}]

    spots, payoffs = _build_payoff_curve(legs, spot)
    return StrategyResult(
        name         = 'Long Put',
        legs         = legs,
        net_premium  = -premium,
        max_profit   = K - premium,         # spot can fall to 0
        max_loss     = -premium,
        breakevens   = [round(K - premium, 2)],
        net_greeks   = _combine_greeks(legs, spot, T, r, lot_size),
        payoff_spots = spots,
        payoff_values= payoffs,
        rationale    = 'Strongly bearish outlook; large downside capture with defined risk',
        lot_size     = lot_size,
    )


def bull_call_spread(
    spot: float, K1: float, K2: float, T: float, sigma: float,
    r: float = RISK_FREE_RATE, lot_size: int = 1,
) -> StrategyResult:
    """
    Buy K1 call, sell K2 call  (K1 < K2, both same expiry).

    View    : Moderately bullish
    Risk    : Net debit paid
    Reward  : K2 - K1 - net debit
    IV bias : Works in any IV; reduces IV risk vs naked long call

    Ideal when: bullish signal + IV rank > 30 (long call is expensive)
    """
    p_long  = bsm_price(spot, K1, T, r, sigma, 'call')
    p_short = bsm_price(spot, K2, T, r, sigma, 'call')
    net_debit = p_long - p_short

    legs = [
        {'K': K1, 'option_type': 'call', 'action': 'buy',  'premium': p_long,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
        {'K': K2, 'option_type': 'call', 'action': 'sell', 'premium': p_short,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
    ]
    spots, payoffs = _build_payoff_curve(legs, spot)
    return StrategyResult(
        name         = 'Bull Call Spread',
        legs         = legs,
        net_premium  = -net_debit,
        max_profit   = K2 - K1 - net_debit,
        max_loss     = -net_debit,
        breakevens   = [round(K1 + net_debit, 2)],
        net_greeks   = _combine_greeks(legs, spot, T, r, lot_size),
        payoff_spots = spots,
        payoff_values= payoffs,
        rationale    = 'Moderately bullish; lower cost than naked call, defined max profit',
        lot_size     = lot_size,
    )


def bear_put_spread(
    spot: float, K1: float, K2: float, T: float, sigma: float,
    r: float = RISK_FREE_RATE, lot_size: int = 1,
) -> StrategyResult:
    """
    Buy K2 put (higher), sell K1 put (lower)  (K1 < K2).

    View    : Moderately bearish
    Risk    : Net debit paid
    Reward  : K2 - K1 - net debit
    IV bias : Works in any IV; reduces IV risk vs naked long put
    """
    p_long  = bsm_price(spot, K2, T, r, sigma, 'put')   # buy the higher strike
    p_short = bsm_price(spot, K1, T, r, sigma, 'put')   # sell the lower strike
    net_debit = p_long - p_short

    legs = [
        {'K': K2, 'option_type': 'put', 'action': 'buy',  'premium': p_long,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
        {'K': K1, 'option_type': 'put', 'action': 'sell', 'premium': p_short,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
    ]
    spots, payoffs = _build_payoff_curve(legs, spot)
    return StrategyResult(
        name         = 'Bear Put Spread',
        legs         = legs,
        net_premium  = -net_debit,
        max_profit   = K2 - K1 - net_debit,
        max_loss     = -net_debit,
        breakevens   = [round(K2 - net_debit, 2)],
        net_greeks   = _combine_greeks(legs, spot, T, r, lot_size),
        payoff_spots = spots,
        payoff_values= payoffs,
        rationale    = 'Moderately bearish; cheaper than naked put, defined max profit',
        lot_size     = lot_size,
    )


def covered_call(
    spot: float, K: float, T: float, sigma: float,
    r: float = RISK_FREE_RATE, lot_size: int = 1,
) -> StrategyResult:
    """
    Long stock (already owned) + sell 1 OTM call.

    View    : Neutral to mildly bullish; happy to sell at K
    Risk    : Stock falls (reduced by premium collected)
    Reward  : Premium + (K - spot) if called away; premium if expires worthless
    IV bias : High IV preferred — sell when options are expensive

    NOTE: stock_delta is +1 per share; net delta = 1 - call_delta
    """
    premium = bsm_price(spot, K, T, r, sigma, 'call')

    # Stock leg (modelled as long the stock at current price)
    call_legs = [
        {'K': K, 'option_type': 'call', 'action': 'sell', 'premium': premium,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
    ]
    spots, call_payoffs = _build_payoff_curve(call_legs, spot)
    # Add stock payoff
    S_range    = np.array(spots)
    stock_pnl  = (S_range - spot) * lot_size   # long stock P&L
    payoffs    = (call_payoffs + stock_pnl).tolist()

    g       = bsm_greeks(spot, K, T, r, sigma, 'call')
    net_g   = {k: round(-g[k] + (1.0 if k == 'delta' else 0.0), 4) for k in g}

    max_p  = round((K - spot) + premium, 2)    # capped by K
    max_l  = round(-(spot - premium), 2)        # stock goes to 0

    return StrategyResult(
        name         = 'Covered Call',
        legs         = call_legs,
        net_premium  = premium,                 # credit received
        max_profit   = max_p,
        max_loss     = max_l,
        breakevens   = [round(spot - premium, 2)],
        net_greeks   = net_g,
        payoff_spots = spots,
        payoff_values= [round(p, 2) for p in payoffs],
        rationale    = 'Income on existing equity holding; sell expensive call in high IV',
        lot_size     = lot_size,
    )


def long_straddle(
    spot: float, T: float, sigma: float,
    r: float = RISK_FREE_RATE, lot_size: int = 1,
) -> StrategyResult:
    """
    Buy ATM call + Buy ATM put (same strike, same expiry).

    View    : Neutral on direction; expects large move either way
    Risk    : Both premiums paid (expensive — avoid in high IV)
    Reward  : Unlimited on both sides
    IV bias : LOW IV preferred — buy volatility cheap before catalyst
    Best for: earnings announcements, FOMC, Budget day

    NOTE: Strike = spot rounded to nearest 50 (NSE convention).
    """
    K = round(spot / 50) * 50   # round to nearest NSE strike (50-point grid)

    p_call = bsm_price(spot, K, T, r, sigma, 'call')
    p_put  = bsm_price(spot, K, T, r, sigma, 'put')
    total_premium = p_call + p_put

    legs = [
        {'K': K, 'option_type': 'call', 'action': 'buy', 'premium': p_call,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
        {'K': K, 'option_type': 'put',  'action': 'buy', 'premium': p_put,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
    ]
    spots, payoffs = _build_payoff_curve(legs, spot)
    return StrategyResult(
        name         = 'Long Straddle',
        legs         = legs,
        net_premium  = -total_premium,
        max_profit   = None,                    # unlimited both ways
        max_loss     = -total_premium,
        breakevens   = [round(K - total_premium, 2), round(K + total_premium, 2)],
        net_greeks   = _combine_greeks(legs, spot, T, r, lot_size),
        payoff_spots = spots,
        payoff_values= payoffs,
        rationale    = 'Big move expected; direction unknown. Best before earnings/catalysts.',
        lot_size     = lot_size,
    )


def long_strangle(
    spot: float, K_put: float, K_call: float, T: float, sigma: float,
    r: float = RISK_FREE_RATE, lot_size: int = 1,
) -> StrategyResult:
    """
    Buy OTM put (K_put < spot) + Buy OTM call (K_call > spot).

    View    : Expects large move; cheaper than straddle
    Risk    : Both premiums (less than straddle)
    Reward  : Large on big move; needs bigger move to profit vs straddle
    IV bias : Low IV preferred
    """
    p_call = bsm_price(spot, K_call, T, r, sigma, 'call')
    p_put  = bsm_price(spot, K_put,  T, r, sigma, 'put')
    total_premium = p_call + p_put

    legs = [
        {'K': K_put,  'option_type': 'put',  'action': 'buy', 'premium': p_put,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
        {'K': K_call, 'option_type': 'call', 'action': 'buy', 'premium': p_call,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
    ]
    spots, payoffs = _build_payoff_curve(legs, spot)
    return StrategyResult(
        name         = 'Long Strangle',
        legs         = legs,
        net_premium  = -total_premium,
        max_profit   = None,
        max_loss     = -total_premium,
        breakevens   = [round(K_put - total_premium, 2), round(K_call + total_premium, 2)],
        net_greeks   = _combine_greeks(legs, spot, T, r, lot_size),
        payoff_spots = spots,
        payoff_values= payoffs,
        rationale    = 'Big move expected; OTM wings make it cheaper than straddle',
        lot_size     = lot_size,
    )


def iron_condor(
    spot: float,
    K_put_buy: float, K_put_sell: float,
    K_call_sell: float, K_call_buy: float,
    T: float, sigma: float,
    r: float = RISK_FREE_RATE, lot_size: int = 1,
) -> StrategyResult:
    """
    Sell OTM put spread + Sell OTM call spread (4 legs).

    Strikes (low → high): K_put_buy < K_put_sell < K_call_sell < K_call_buy

    View    : Neutral; expects spot to stay within a range
    Risk    : Width of either spread minus net credit
    Reward  : Net credit collected (max when all 4 expire worthless)
    IV bias : HIGH IV preferred — collect expensive premium, profit from IV crush

    Best for: High IV Rank (≥ 70), stable stocks, post-earnings IV crush
    """
    p_pb  = bsm_price(spot, K_put_buy,   T, r, sigma, 'put')
    p_ps  = bsm_price(spot, K_put_sell,  T, r, sigma, 'put')
    p_cs  = bsm_price(spot, K_call_sell, T, r, sigma, 'call')
    p_cb  = bsm_price(spot, K_call_buy,  T, r, sigma, 'call')

    net_credit = (p_ps - p_pb) + (p_cs - p_cb)   # positive = credit received
    put_width  = K_put_sell  - K_put_buy
    call_width = K_call_buy  - K_call_sell
    max_risk   = max(put_width, call_width) - net_credit

    legs = [
        {'K': K_put_buy,   'option_type': 'put',  'action': 'buy',  'premium': p_pb,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
        {'K': K_put_sell,  'option_type': 'put',  'action': 'sell', 'premium': p_ps,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
        {'K': K_call_sell, 'option_type': 'call', 'action': 'sell', 'premium': p_cs,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
        {'K': K_call_buy,  'option_type': 'call', 'action': 'buy',  'premium': p_cb,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma},
    ]
    spots, payoffs = _build_payoff_curve(legs, spot)
    return StrategyResult(
        name         = 'Iron Condor',
        legs         = legs,
        net_premium  = net_credit,
        max_profit   = net_credit,
        max_loss     = -max_risk,
        breakevens   = [
            round(K_put_sell  - net_credit, 2),
            round(K_call_sell + net_credit, 2),
        ],
        net_greeks   = _combine_greeks(legs, spot, T, r, lot_size),
        payoff_spots = spots,
        payoff_values= payoffs,
        rationale    = 'High IV + neutral view; collect premium, profit from IV crush & range-bound move',
        lot_size     = lot_size,
    )


def calendar_spread(
    spot: float, K: float,
    T_near: float, T_far: float,
    sigma_near: float, sigma_far: float,
    r: float = RISK_FREE_RATE, lot_size: int = 1,
) -> StrategyResult:
    """
    Sell near-term ATM call + Buy far-term ATM call (same strike).

    View    : Neutral; profits from time decay differential
    Risk    : Net debit if near-term call expires worthless and far-term moves far OTM
    Reward  : Near-term decay > far-term decay
    IV bias : Low near-term IV, high far-term IV ideal (upward-sloping term structure)

    NOTE: Calendar spread Greeks shown at initiation (near expiry not yet expired).
    Payoff curve is approximate (single-expiry payoff at near-leg expiry).
    """
    p_near = bsm_price(spot, K, T_near, r, sigma_near, 'call')
    p_far  = bsm_price(spot, K, T_far,  r, sigma_far,  'call')
    net_debit = p_far - p_near

    legs = [
        {'K': K, 'option_type': 'call', 'action': 'sell', 'premium': p_near,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma_near},
        {'K': K, 'option_type': 'call', 'action': 'buy',  'premium': p_far,
         'quantity': 1, 'lot_size': lot_size, 'sigma': sigma_far},
    ]
    spots, payoffs = _build_payoff_curve(legs, spot)
    return StrategyResult(
        name         = 'Calendar Spread (Call)',
        legs         = legs,
        net_premium  = -net_debit,
        max_profit   = None,               # depends on IV at near expiry
        max_loss     = -net_debit,
        breakevens   = [],                 # path-dependent; omit
        net_greeks   = _combine_greeks(legs, spot, T_far, r, lot_size),
        payoff_spots = spots,
        payoff_values= payoffs,
        rationale    = 'Neutral view; sell fast-decaying near-term premium, hold longer exposure',
        lot_size     = lot_size,
    )


# ══════════════════════════════════════════════════════════
#  SECTION 2 — AI Strategy Suggester
# ══════════════════════════════════════════════════════════

def suggest_strategy(
    signal     : str,        # 'BUY' | 'SELL' | 'HOLD'
    confidence : float,      # 0–1 from ensemble model
    iv_rank    : float,      # 0–100 (IV Rank)
    spot       : float,
    T          : float,
    sigma      : float,
    lot_size   : int = 1,
    r          : float = RISK_FREE_RATE,
) -> tuple[Optional[StrategyResult], str]:
    """
    Recommend and construct an options strategy from QuantAI ensemble output.

    Integrates with the existing signals by reading:
        signal       = ensemble prediction ('BUY'/'SELL'/'HOLD')
        confidence   = ensemble confidence score (from get_ensemble_confidence)
        iv_rank      = IV Rank 0-100 (from options_chain.fetch_full_chain)

    The decision matrix balances direction (from ML signal) with
    volatility regime (from IV Rank) to select the optimal strategy.
    High IV → prefer selling strategies (collect expensive premium).
    Low  IV → prefer buying strategies (cheap options).

    Returns (StrategyResult, rationale_string) or (None, reason).
    """
    MIN_CONFIDENCE = 0.58

    if confidence < MIN_CONFIDENCE:
        return None, (
            f"Ensemble confidence {confidence:.1%} below threshold {MIN_CONFIDENCE:.0%}. "
            "Wait for stronger signal before entering options."
        )

    sig = signal.upper().strip()

    # ── Round to nearest NSE strike interval ─────────────
    # NSE uses ₹50 intervals for most stocks (₹100 for higher-priced)
    step = 100 if spot > 5000 else 50

    def round_to_step(x, s=step):
        return round(x / s) * s

    ATM   = round_to_step(spot)
    OTM1  = round_to_step(spot * 1.03)   # ~3% OTM
    OTM2  = round_to_step(spot * 1.06)   # ~6% OTM
    ITM1  = round_to_step(spot * 0.97)
    OTM_P = round_to_step(spot * 0.97)   # 3% OTM put
    OTM_P2= round_to_step(spot * 0.94)   # 6% OTM put

    # ── BULLISH SCENARIOS ─────────────────────────────────
    if sig == 'BUY':
        if confidence >= 0.70 and iv_rank <= 30:
            s = long_call(spot, OTM1, T, sigma, r, lot_size)
            s.rationale = (
                f"Strong bullish signal ({confidence:.0%}) + Low IV ({iv_rank:.0f}) → "
                "buy OTM call outright. Options cheap; unlimited upside."
            )
            return s, s.rationale

        elif confidence >= 0.58 or iv_rank > 30:
            s = bull_call_spread(spot, ATM, OTM2, T, sigma, r, lot_size)
            s.rationale = (
                f"Bullish signal ({confidence:.0%}) + IV Rank {iv_rank:.0f} → "
                "bull call spread. Defined risk; selling OTM call offsets high IV cost."
            )
            return s, s.rationale

    # ── BEARISH SCENARIOS ─────────────────────────────────
    elif sig == 'SELL':
        if confidence >= 0.70 and iv_rank <= 30:
            s = long_put(spot, ITM1, T, sigma, r, lot_size)
            s.rationale = (
                f"Strong bearish signal ({confidence:.0%}) + Low IV ({iv_rank:.0f}) → "
                "buy put. Options cheap; large downside capture."
            )
            return s, s.rationale

        else:
            s = bear_put_spread(spot, OTM_P2, OTM_P, T, sigma, r, lot_size)
            s.rationale = (
                f"Bearish signal ({confidence:.0%}) + IV Rank {iv_rank:.0f} → "
                "bear put spread. Lower cost; selling OTM put offsets debit."
            )
            return s, s.rationale

    # ── NEUTRAL / HIGH-IV SCENARIOS ───────────────────────
    if iv_rank >= 80:
        # Very high IV → aggressive premium selling
        s = long_straddle(spot, T, sigma, r, lot_size)
        # Actually in high IV, sell the straddle (short straddle is risky, use condor)
        s = iron_condor(spot, OTM_P2, OTM_P, OTM1, OTM2, T, sigma, r, lot_size)
        s.rationale = (
            f"Extreme IV ({iv_rank:.0f}) + neutral signal → "
            "iron condor. Collect rich premium; profit from IV crush."
        )
        return s, s.rationale

    elif iv_rank >= 65:
        s = iron_condor(spot, OTM_P2, OTM_P, OTM1, OTM2, T, sigma, r, lot_size)
        s.rationale = (
            f"High IV ({iv_rank:.0f}) + neutral/no-direction signal → "
            "iron condor. Range-bound bet; IV contraction helps."
        )
        return s, s.rationale

    elif iv_rank <= 25:
        s = long_straddle(spot, T, sigma, r, lot_size)
        s.rationale = (
            f"Low IV ({iv_rank:.0f}) + no clear direction → "
            "long straddle. Buy cheap volatility before potential catalyst."
        )
        return s, s.rationale

    return None, (
        f"Signal={sig}, Confidence={confidence:.0%}, IVR={iv_rank:.0f}: "
        "No high-conviction setup. Sit in cash."
    )


# ══════════════════════════════════════════════════════════
#  SECTION 3 — Strike selector (ATM / OTM picker)
# ══════════════════════════════════════════════════════════

def select_strikes_from_chain(
    calls_df: 'pd.DataFrame',
    puts_df:  'pd.DataFrame',
    spot: float,
    strategy_name: str,
) -> dict:
    """
    Pick optimal strikes from a live chain for a given strategy.

    Selects based on delta targets:
      ATM call : delta ~0.50
      OTM call : delta ~0.35 (for buying), ~0.20 (for selling in condor)
      ATM put  : delta ~-0.50
      OTM put  : delta ~-0.35 (buying), ~-0.20 (selling in condor)

    Returns {'K_atm_call', 'K_otm_call', 'K_atm_put', 'K_otm_put', ...}
    """
    import pandas as pd # type: ignore

    def nearest_delta(df, target_delta):
        if df.empty:
            return None
        idx = (df['delta'] - target_delta).abs().idxmin()
        return float(df.loc[idx, 'strike'])

    result = {
        'K_atm_call'  : nearest_delta(calls_df, 0.50),
        'K_otm_call'  : nearest_delta(calls_df, 0.35),
        'K_far_call'  : nearest_delta(calls_df, 0.20),
        'K_atm_put'   : nearest_delta(puts_df,  -0.50),
        'K_otm_put'   : nearest_delta(puts_df,  -0.35),
        'K_far_put'   : nearest_delta(puts_df,  -0.20),
    }
    return result