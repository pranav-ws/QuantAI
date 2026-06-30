"""
src/slippage.py — QuantAI Realistic Cost Simulator

Models the gap between a backtest's "paper" fill price and what you'd
actually get in the real market. Every real trade pays for:

  1. Brokerage / commission         — flat % charged by the broker
  2. Statutory charges (India, .NS)  — STT, exchange + SEBI fees, stamp
                                        duty, GST on brokerage (bundled
                                        into one realistic bps figure,
                                        asymmetric buy vs sell, matching
                                        NSE equity-delivery norms)
  3. Volatility slippage             — wider ATR ⇒ worse fill, because
                                        fast-moving prices are harder to
                                        execute at the exact signal price
  4. Liquidity slippage              — thin volume (low Volume_Ratio) ⇒
                                        worse fill, because your order
                                        moves the price more in a quiet
                                        stock

A BUY fills at a price >= the signal price (you pay up).
A SELL fills at a price <= the signal price (you receive less).
This asymmetry is what actually erodes paper returns in live trading.

Usage
-----
    from src.slippage import SlippageModel

    slippage = SlippageModel()  # or SlippageModel(profile='intraday')
    fill_price, cost_breakdown = slippage.get_fill_price(
        signal_price=2500.0,
        side='BUY',
        atr=row['ATR_14'],
        volume_ratio=row['Volume_Ratio']
    )
"""
import numpy as np # type: ignore


class SlippageModel:
    """
    Realistic transaction-cost simulator for Indian equity (.NS) trades.

    Parameters
    ----------
    profile : str
        'delivery' (default) — normal swing/positional trades, held
                    overnight, charged STT on both buy and sell.
        'intraday' — same-day square-off, lower STT (charged only on
                    sell side) but otherwise similar structure.
    commission_pct : float
        Broker commission as a fraction of trade value, each side.
        Default 0.0003 (0.03%) — typical discount-broker flat fee
        expressed as a %, conservative for larger trade sizes.
    base_slippage_bps : float
        Minimum slippage (in basis points, 1bps = 0.01%) applied to
        every trade even in calm, liquid conditions — covers bid-ask
        spread that a backtest assuming mid/close price ignores.
    atr_slippage_factor : float
        How strongly volatility (ATR as % of price) translates into
        extra slippage. Higher = more conservative (worse fills on
        volatile days).
    liquidity_slippage_factor : float
        How strongly thin volume (Volume_Ratio < 1) translates into
        extra slippage. Higher = more conservative (worse fills when
        volume is below its 20-day average).
    random_seed : int or None
        If set, adds a small reproducible random jitter to each fill
        to simulate the fact that real slippage isn't perfectly
        deterministic. Leave as None for default (no jitter, fully
        deterministic — better for comparing strategy variants).
    """

    def __init__(self, profile='delivery', commission_pct=0.0003,
                 base_slippage_bps=5, atr_slippage_factor=0.25,
                 liquidity_slippage_factor=0.15, random_seed=None):
        if profile not in ('delivery', 'intraday'):
            raise ValueError("profile must be 'delivery' or 'intraday'")

        self.profile = profile
        self.commission_pct = commission_pct
        self.base_slippage_bps = base_slippage_bps
        self.atr_slippage_factor = atr_slippage_factor
        self.liquidity_slippage_factor = liquidity_slippage_factor
        self.rng = np.random.default_rng(random_seed) if random_seed is not None else None

        # ── Statutory cost table (approximate NSE equity norms, as bps of trade value) ──
        if profile == 'delivery':
            self.stt_buy_bps   = 10.0   # STT on delivery: ~0.1% on BOTH buy & sell
            self.stt_sell_bps  = 10.0
        else:  # intraday
            self.stt_buy_bps   = 0.0    # intraday STT charged only on sell side
            self.stt_sell_bps  = 2.5    # ~0.025% on sell

        self.exchange_txn_bps = 0.345   # NSE transaction charge, both sides
        self.sebi_fee_bps     = 0.01    # SEBI turnover fee, both sides
        self.stamp_duty_bps   = 1.5     # stamp duty, buy side only
        self.gst_pct_of_comm  = 0.18    # 18% GST on (brokerage + exchange charges)

    # ──────────────────────────────────────────────────────
    def _slippage_bps(self, price, atr=None, volume_ratio=None):
        """
        Computes total price-impact slippage in basis points, driven by
        volatility (ATR as a fraction of price) and liquidity (Volume_Ratio).
        """
        bps = self.base_slippage_bps

        if atr is not None and price > 0 and not np.isnan(atr):
            atr_pct = (atr / price) * 100          # ATR as % of price
            bps += atr_pct * 100 * self.atr_slippage_factor  # scale into bps

        if volume_ratio is not None and not np.isnan(volume_ratio) and volume_ratio > 0:
            if volume_ratio < 1.0:
                # Below-average volume ⇒ thinner book ⇒ worse fill.
                # e.g. volume_ratio=0.5 (half normal volume) adds meaningfully more slippage.
                thinness = (1.0 / volume_ratio) - 1.0
                bps += thinness * 100 * self.liquidity_slippage_factor
            # volume_ratio >= 1.0 (above-average volume): no extra penalty,
            # deep liquidity doesn't reduce slippage below the base floor.

        if self.rng is not None:
            # Small reproducible random jitter (+/- up to 20% of the computed bps)
            bps *= (1.0 + self.rng.uniform(-0.2, 0.2))

        return max(bps, 0.0)

    # ──────────────────────────────────────────────────────
    def get_fill_price(self, signal_price, side, atr=None, volume_ratio=None):
        """
        Returns the realistic fill price for a trade, and a breakdown of
        every cost component (all expressed in ₹ per share AND as bps).

        Parameters
        ----------
        signal_price : float
            The "paper" price the backtest would have used (e.g. Close).
        side : str
            'BUY' or 'SELL'.
        atr : float or None
            ATR_14 value for that day (from src.features), if available.
        volume_ratio : float or None
            Volume_Ratio value for that day (from src.features), if available.

        Returns
        -------
        fill_price : float
            The realistic price you'd actually pay (BUY) or receive (SELL).
        breakdown : dict
            Per-component cost in bps and in ₹/share, plus the total.
        """
        side = side.upper()
        if side not in ('BUY', 'SELL'):
            raise ValueError("side must be 'BUY' or 'SELL'")

        slip_bps = self._slippage_bps(signal_price, atr, volume_ratio)

        commission_bps = self.commission_pct * 100 * 100   # pct → bps
        exchange_bps   = self.exchange_txn_bps + self.sebi_fee_bps
        stt_bps        = self.stt_buy_bps if side == 'BUY' else self.stt_sell_bps
        stamp_bps      = self.stamp_duty_bps if side == 'BUY' else 0.0
        gst_bps        = (commission_bps + exchange_bps) * self.gst_pct_of_comm

        total_cost_bps = slip_bps + commission_bps + exchange_bps + stt_bps + stamp_bps + gst_bps
        cost_fraction   = total_cost_bps / 10000.0           # bps → fraction

        if side == 'BUY':
            fill_price = signal_price * (1 + cost_fraction)   # pay MORE than signal price
        else:
            fill_price = signal_price * (1 - cost_fraction)   # receive LESS than signal price

        breakdown = {
            'side'              : side,
            'signal_price'      : signal_price,
            'fill_price'        : fill_price,
            'slippage_bps'      : round(slip_bps, 3),
            'commission_bps'    : round(commission_bps, 3),
            'exchange_fees_bps' : round(exchange_bps, 3),
            'stt_bps'           : round(stt_bps, 3),
            'stamp_duty_bps'    : round(stamp_bps, 3),
            'gst_bps'           : round(gst_bps, 3),
            'total_cost_bps'    : round(total_cost_bps, 3),
            'cost_per_share'    : round(abs(fill_price - signal_price), 4),
            'cost_pct_of_price' : round(cost_fraction * 100, 4),
        }

        return fill_price, breakdown

    # ──────────────────────────────────────────────────────
    def estimate_round_trip_cost_pct(self, price=1000, atr=None, volume_ratio=None):
        """
        Convenience helper: total % cost of a BUY followed by a SELL at the
        same reference price — useful for quickly sanity-checking how much
        edge a strategy needs per trade just to break even on costs.
        """
        _, buy_breakdown  = self.get_fill_price(price, 'BUY', atr, volume_ratio)
        _, sell_breakdown = self.get_fill_price(price, 'SELL', atr, volume_ratio)
        return buy_breakdown['cost_pct_of_price'] + sell_breakdown['cost_pct_of_price']


def print_cost_report(model: SlippageModel, sample_price=1000, sample_atr=15,
                       sample_volume_ratio=0.8):
    """Pretty-prints an example cost breakdown for both BUY and SELL,
    useful for sanity-checking a configured SlippageModel."""
    buy_fill, buy_bd   = model.get_fill_price(sample_price, 'BUY', sample_atr, sample_volume_ratio)
    sell_fill, sell_bd = model.get_fill_price(sample_price, 'SELL', sample_atr, sample_volume_ratio)
    round_trip = model.estimate_round_trip_cost_pct(sample_price, sample_atr, sample_volume_ratio)

    print(f"\n{'='*55}")
    print(f"  QuantAI Realistic Cost Simulator — {model.profile.upper()} profile")
    print(f"{'='*55}")
    print(f"  Sample price       : ₹{sample_price:,.2f}")
    print(f"  Sample ATR_14      : {sample_atr}")
    print(f"  Sample Volume_Ratio: {sample_volume_ratio}")
    print(f"\n  🟢 BUY")
    print(f"    Signal price     : ₹{buy_bd['signal_price']:>10,.2f}")
    print(f"    Realistic fill   : ₹{buy_bd['fill_price']:>10,.2f}  (you pay MORE)")
    print(f"    Slippage         : {buy_bd['slippage_bps']:>8.2f} bps")
    print(f"    Commission       : {buy_bd['commission_bps']:>8.2f} bps")
    print(f"    Exchange+SEBI    : {buy_bd['exchange_fees_bps']:>8.2f} bps")
    print(f"    STT              : {buy_bd['stt_bps']:>8.2f} bps")
    print(f"    Stamp duty       : {buy_bd['stamp_duty_bps']:>8.2f} bps")
    print(f"    GST              : {buy_bd['gst_bps']:>8.2f} bps")
    print(f"    TOTAL COST       : {buy_bd['total_cost_bps']:>8.2f} bps ({buy_bd['cost_pct_of_price']:.3f}%)")
    print(f"\n  🔴 SELL")
    print(f"    Signal price     : ₹{sell_bd['signal_price']:>10,.2f}")
    print(f"    Realistic fill   : ₹{sell_bd['fill_price']:>10,.2f}  (you receive LESS)")
    print(f"    Slippage         : {sell_bd['slippage_bps']:>8.2f} bps")
    print(f"    Commission       : {sell_bd['commission_bps']:>8.2f} bps")
    print(f"    Exchange+SEBI    : {sell_bd['exchange_fees_bps']:>8.2f} bps")
    print(f"    STT              : {sell_bd['stt_bps']:>8.2f} bps")
    print(f"    GST              : {sell_bd['gst_bps']:>8.2f} bps")
    print(f"    TOTAL COST       : {sell_bd['total_cost_bps']:>8.2f} bps ({sell_bd['cost_pct_of_price']:.3f}%)")
    print(f"\n  ⚠️  Round-trip cost (buy + sell): {round_trip:.3f}% of trade value")
    print(f"     Your strategy needs a bigger average edge than this per trade to be profitable.")
    print(f"{'='*55}\n")