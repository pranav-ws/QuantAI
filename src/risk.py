import pandas as pd# type: ignore
import numpy as np# type: ignore

class RiskManager:
    """
    Portfolio-level risk management system.
    Every trade must pass through this before execution.
    """

    def __init__(self, initial_capital=100000):
        self.initial_capital  = initial_capital
        self.capital          = initial_capital
        self.peak_capital     = initial_capital
        self.is_trading_halted = False

        self.tail_risk_override = False   # set True by TailRiskMonitor when TRI >= 0.70


        # ── Risk Parameters ──────────────────────────────
        self.max_portfolio_risk  = 0.02   # never risk more than 2% of capital per trade
        self.max_drawdown_limit  = 0.15   # halt all trading if portfolio drops 15%
        self.max_position_size   = 0.25   # never put more than 25% in one stock
        self.min_confidence      = 0.58   # minimum model confidence to enter trade
        self.max_positions       = 5      # hold max 5 stocks simultaneously
        self.stop_loss_pct       = 0.03   # 3% stop loss on every trade

    def calculate_position_size(self, capital, price,
                             confidence, regime='SIDEWAYS',
                             ticker=None, atr=None):
        """
        Kelly Criterion position sizing.
        Replaces old fixed 2% risk model.
        """
        if self.is_trading_halted or self.tail_risk_override:
            reason = ("Trading halted — max drawdown breached"
                      if self.is_trading_halted
                      else "🚫 Trading halted — BLACK SWAN ALERT (TRI ≥ 0.70)")
            return 0, 0.0, reason

        if confidence < self.min_confidence:
            return 0, 0.0, f"Confidence {confidence:.2%} below minimum"

        try:
            from src.kelly_sizer import kelly_position_size
            shares, kelly_f, position_val, details = \
                kelly_position_size(
                    capital    = capital,
                    price      = price,
                    model_confidence = confidence,
                    ticker     = ticker,
                    regime     = regime,
                    verbose    = False
                )
            stop_loss = details.get('stop_loss', price * 0.97)
            reason    = (
                f"{shares} shares @ ₹{price:.1f} = "
                f"₹{position_val:,.0f} "
                f"({kelly_f*100:.1f}% Kelly) | "
                f"SL: ₹{stop_loss:.1f}"
            )
            return shares, stop_loss, reason

        except Exception as e:
            # Fallback to old fixed sizing
            risk_amount   = capital * self.max_portfolio_risk
            if atr is not None and atr > 0:
                risk_per_share = atr          # ATR = more accurate vol estimate
                stop_loss_p    = price - atr
            else:
                stop_loss_p    = price * (1 - self.stop_loss_pct)
                risk_per_share = price - stop_loss_p
            shares        = int(risk_amount / risk_per_share)
            max_shares    = int(capital * self.max_position_size / price)
            shares        = min(shares, max_shares)
            reason        = f"Fixed fallback: {shares} shares"
            return shares, stop_loss_p, reason

    def check_portfolio_risk(self, current_value):
        """
        Monitors overall portfolio. Halts trading if drawdown too deep.
        Returns True if safe to trade.
        """
        if current_value > self.peak_capital:
            self.peak_capital = current_value

        drawdown = (current_value - self.peak_capital) / self.peak_capital

        if drawdown <= -self.max_drawdown_limit:
            self.is_trading_halted = True
            return False, drawdown

        self.is_trading_halted = False
        return True, drawdown

    def get_risk_report(self, positions, current_prices):
        """Prints a snapshot of current portfolio risk."""
        print(f"\n{'='*55}")
        print(f"  QuantAI Risk Dashboard")
        print(f"{'='*55}")
        print(f"  Capital          : ₹{self.capital:>12,.0f}")
        print(f"  Peak Capital     : ₹{self.peak_capital:>12,.0f}")

        portfolio_value = self.capital
        for ticker, (shares, entry, stop_loss) in positions.items():
            price = current_prices.get(ticker, entry)
            value = shares * price
            pnl   = (price - entry) * shares
            portfolio_value += value
            sl_distance = (price - stop_loss) / price * 100
            print(f"\n  {ticker}")
            print(f"    Shares     : {shares}")
            print(f"    Entry      : ₹{entry:.1f}  →  Now: ₹{price:.1f}")
            print(f"    P&L        : ₹{pnl:+,.0f}")
            print(f"    Stop Loss  : ₹{stop_loss:.1f} ({sl_distance:.1f}% away)")

        _, drawdown = self.check_portfolio_risk(portfolio_value)
        if self.is_trading_halted:
           status = "🚫 HALTED (drawdown)"
        elif self.tail_risk_override:
           status = "🔴 HALTED (black swan)"
        else:
           status = "✅ ACTIVE"
           
        print(f"\n  Total Value      : ₹{portfolio_value:>12,.0f}")
        print(f"  Drawdown         : {drawdown:>+11.2%}")
        print(f"  Trading Status   : {status}")
        print(f"{'='*55}\n")


    def apply_tail_risk_sizing(self, tri_level: str):
        """
        Called by paper_trade.py after the TRI check.
        Temporarily reduces max risk per trade when tail risk is elevated.
        Does NOT halt trading — that's handled by tail_risk_override.
        """
        BASE_RISK = 0.02   # the default set in __init__

        if tri_level == 'ELEVATED':
            self.max_portfolio_risk = BASE_RISK * 0.75   # 1.5% per trade
            print(f"  ⚠️  Risk per trade reduced to "
                  f"{self.max_portfolio_risk:.1%} (ELEVATED tail risk)")
        elif tri_level == 'HIGH':
            self.max_portfolio_risk = BASE_RISK * 0.50   # 1.0% per trade
            print(f"  🟠 Risk per trade reduced to "
                  f"{self.max_portfolio_risk:.1%} (HIGH tail risk)")
        else:
            self.max_portfolio_risk = BASE_RISK           # restore normal