"""
src/risk_parity.py

Risk Parity — Equal Risk Contribution (ERC) position sizing.

The problem with standard position sizing (src/risk.py):
  Kelly / fixed-fraction sizing allocates equal CAPITAL to each trade.
  But stocks have wildly different volatilities — a ₹1L position in a
  low-vol Banking stock carries a fraction of the risk of a ₹1L position
  in a high-vol Pharma or Small-cap stock. The result is that your
  portfolio risk is dominated by your most volatile holdings even if
  they have small capital weights.

What Risk Parity fixes:
  Instead of equal capital per trade, allocate so that each position
  contributes EQUAL RISK to the portfolio — measured in rupee volatility
  (₹ std-dev of daily P&L). A low-vol stock gets a bigger capital
  allocation; a high-vol stock gets a smaller one. After allocation,
  every position "punches equally" in terms of how much it can move
  your portfolio on a bad day.

The maths (simple one-asset-at-a-time version):
  1. Compute each stock's daily volatility σᵢ from recent returns
     (20-day rolling std of daily log-returns, annualized for display
     but used as a daily figure for sizing).
  2. Target daily rupee-risk per stock = total_capital × target_risk_pct
     (default: 0.5% of capital per position per day).
  3. Shares = target_daily_rupee_risk / (price × σᵢ)
     → a stock with 2× the vol gets half the shares.
  4. Apply guards: min position floor (don't buy 1 share), max position
     cap (never more than max_position_pct of capital in one name),
     and a correlation penalty (if two signals are highly correlated,
     shrink both to avoid doubling up on the same risk factor).

Why this is better than the existing RiskManager for a basket of signals:
  The existing RiskManager already does a great job of single-trade
  risk control (2% risk/trade, stop losses, drawdown halt). Risk parity
  is complementary — it runs AFTER you have a basket of BUY signals
  and reallocates the sizes so the portfolio-level risk is balanced
  across all of them together, not just capped per trade.

  Think of it this way:
    - RiskManager: "Should I take this trade, and how big?"
    - RiskParity:  "Given I'm taking all THESE trades, how do I
                    size each one so my risk is spread evenly?"

Integration with existing code:
  paper_trade.py builds signals_today → each entry has 'shares',
  'price', 'stop_loss' from RiskManager.calculate_position_size().
  After that list is built, call:

      from src.risk_parity import RiskParityAllocator
      allocator = RiskParityAllocator()
      signals_today = allocator.allocate(signals_today, capital, price_history_dict)

  allocate() returns the same list with 'shares' and 'trade_value'
  replaced by risk-parity-adjusted figures, and adds two new keys:
      'rp_weight'    : float  — this stock's fraction of total capital (0–1)
      'rp_daily_vol' : float  — stock's estimated daily volatility (%)
"""

import numpy as np # type: ignore
import pandas as pd# type: ignore
import sqlite3
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskParityResult:
    """Summary of a risk parity allocation run."""
    n_stocks           : int
    total_capital      : float
    total_deployed     : float
    cash_remaining     : float
    target_risk_per_pos: float          # ₹ daily risk per position
    actual_risk_contribs: dict          # {ticker: ₹ daily risk}
    weights            : dict           # {ticker: capital weight 0–1}
    vol_estimates      : dict           # {ticker: daily vol %}
    correlation_used   : bool
    rebalance_reason   : str


class RiskParityAllocator:
    """
    Computes equal-risk-contribution position sizes for a basket of
    BUY signals, using recent price history to estimate volatility.

    Parameters
    ----------
    target_risk_pct : float
        Fraction of total capital to risk per position per day.
        Default 0.005 = 0.5% daily capital at risk per stock.
        With 5 stocks this targets ~2.5% total portfolio daily risk,
        similar to what a 1% VaR model would cap at.

    vol_window : int
        Days of daily log-returns used to estimate each stock's
        volatility. 20 days = roughly one trading month.

    min_vol_floor : float
        Minimum daily vol assumed even if measured vol is lower.
        Prevents over-sizing into artificially quiet stocks.
        Default 0.005 = 0.5% per day.

    max_position_pct : float
        Hard cap: no single stock can exceed this fraction of capital.
        Default 0.25 = 25%, same as RiskManager.max_position_size.

    min_position_pct : float
        Hard floor: don't bother with positions smaller than this
        (too small to be meaningful after brokerage).
        Default 0.02 = 2% of capital.

    correlation_penalty : bool
        If True, pairs of signals with correlation > corr_threshold
        are shrunk proportionally — avoids doubling up on the same
        risk factor.

    corr_threshold : float
        Pairwise correlation above which the penalty is applied.
        Default 0.70.
    """

    def __init__(self,
                 target_risk_pct    = 0.005,
                 vol_window         = 20,
                 min_vol_floor      = 0.005,
                 max_position_pct   = 0.25,
                 min_position_pct   = 0.02,
                 correlation_penalty= True,
                 corr_threshold     = 0.70):

        self.target_risk_pct     = target_risk_pct
        self.vol_window          = vol_window
        self.min_vol_floor       = min_vol_floor
        self.max_position_pct    = max_position_pct
        self.min_position_pct    = min_position_pct
        self.correlation_penalty = correlation_penalty
        self.corr_threshold      = corr_threshold

    # ── Volatility estimation ─────────────────────────────

    def _estimate_vol(self, close: pd.Series) -> float:
        """
        Daily log-return standard deviation over the last vol_window days.
        Returns a daily fraction (e.g. 0.015 = 1.5% per day).
        """
        if len(close) < self.vol_window + 2:
            return self.min_vol_floor
        log_ret = np.log(close / close.shift(1)).dropna()
        vol = float(log_ret.iloc[-self.vol_window:].std())
        return max(vol, self.min_vol_floor)

    def _load_close(self, ticker: str, days: int = 60) -> Optional[pd.Series]:
        """
        Loads recent close prices for a ticker.
        DB first (quantai.db) → yfinance fallback.
        Same pattern used by tail_risk.py and drawdown_heatmap.py.
        """
        db_path = os.path.join('data', 'quantai.db')

        # Try DB first
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                df   = pd.read_sql_query(
                    "SELECT date, close FROM prices "
                    "WHERE ticker=? ORDER BY date DESC LIMIT ?",
                    conn, params=(ticker, days + 5)
                )
                conn.close()
                if not df.empty:
                    df['date']  = pd.to_datetime(df['date'])
                    df          = df.sort_values('date').set_index('date')
                    df['close'] = pd.to_numeric(df['close'], errors='coerce')
                    df.dropna(inplace=True)
                    if len(df) >= self.vol_window + 2:
                        return df['close']
            except Exception:
                pass

        # Fallback to yfinance
        try:
            import yfinance as yf # type: ignore
            raw = yf.download(ticker, period='3mo',
                              progress=False, auto_adjust=True)
            if raw.empty:
                return None
            if hasattr(raw.columns, 'levels'):
                raw.columns = [c[0] for c in raw.columns]
            close = raw['Close'].dropna()
            return close if len(close) >= self.vol_window + 2 else None
        except Exception:
            return None

    # ── Correlation matrix for penalty ────────────────────

    def _correlation_matrix(self,
                             tickers: list,
                             close_map: dict) -> pd.DataFrame:
        """
        Computes pairwise return correlations from the available close
        price series. Returns a DataFrame indexed/columned by ticker.
        """
        returns = {}
        for t in tickers:
            if t in close_map and close_map[t] is not None:
                log_ret = np.log(close_map[t] / close_map[t].shift(1)).dropna()
                returns[t] = log_ret.iloc[-self.vol_window:]

        if len(returns) < 2:
            return pd.DataFrame(np.eye(len(tickers)),
                                 index=tickers, columns=tickers)

        ret_df = pd.DataFrame(returns).dropna()
        if ret_df.empty or ret_df.shape[0] < 5:
            return pd.DataFrame(np.eye(len(tickers)),
                                 index=tickers, columns=tickers)

        return ret_df.corr()

    # ── Correlation penalty ───────────────────────────────

    def _apply_correlation_penalty(self,
                                    weights: dict,
                                    corr: pd.DataFrame) -> dict:
        """
        For each pair of tickers with correlation > corr_threshold,
        scale both weights down proportionally.

        Intuition: if HDFCBANK and ICICIBANK have 0.85 correlation,
        holding both at full weight is almost like holding one name at
        double weight. The penalty reduces each by √(1 - excess_corr)
        so combined risk exposure stays roughly flat.
        """
        adj_weights = dict(weights)
        tickers     = list(weights.keys())

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                ti, tj = tickers[i], tickers[j]
                if ti not in corr.index or tj not in corr.index:
                    continue
                c = float(corr.loc[ti, tj])
                if c > self.corr_threshold:
                    # penalty factor: 1.0 at threshold, 0.71 at corr=1.0
                    excess  = (c - self.corr_threshold) / (1.0 - self.corr_threshold)
                    penalty = np.sqrt(1.0 - excess * 0.50)
                    adj_weights[ti] = adj_weights[ti] * penalty
                    adj_weights[tj] = adj_weights[tj] * penalty

        return adj_weights

    # ── Main allocation ───────────────────────────────────

    def allocate(self,
                 signals: list,
                 capital: float,
                 close_map: Optional[dict] = None
                 ) -> tuple[list, RiskParityResult]:
        """
        Re-sizes a list of BUY signal dicts using risk parity.

        Parameters
        ----------
        signals   : list of dicts from paper_trade.py (must have 'ticker', 'price')
        capital   : available capital in ₹
        close_map : optional {ticker: pd.Series of Close} — if None, loads automatically

        Returns
        -------
        (adjusted_signals, RiskParityResult)
        adjusted_signals is the same list with 'shares', 'trade_value',
        'rp_weight', 'rp_daily_vol' updated/added per signal.
        """
        if not signals:
            return signals, RiskParityResult(
                n_stocks=0, total_capital=capital,
                total_deployed=0, cash_remaining=capital,
                target_risk_per_pos=0, actual_risk_contribs={},
                weights={}, vol_estimates={}, correlation_used=False,
                rebalance_reason='No signals to allocate'
            )

        tickers = [s['ticker'] for s in signals]

        # ── Step 1: load price history if not provided ──
        if close_map is None:
            close_map = {}
            for t in tickers:
                close_map[t] = self._load_close(t)

        # ── Step 2: estimate each stock's daily volatility ──
        vol_map = {}
        for t in tickers:
            if t in close_map and close_map[t] is not None:
                vol_map[t] = self._estimate_vol(close_map[t])
            else:
                # No history → use a conservative default
                vol_map[t] = 0.02   # 2% daily = roughly Nifty small-cap level

        # ── Step 3: compute raw risk-parity capital weights ──
        # Target rupee risk per stock = capital × target_risk_pct
        target_rupee_risk = capital * self.target_risk_pct

        raw_weights = {}
        for sig in signals:
            t     = sig['ticker']
            price = sig['price']
            vol   = vol_map[t]
            # Shares s.t. price × vol × shares = target_rupee_risk
            target_value = target_rupee_risk / vol
            raw_weights[t] = target_value / capital   # as fraction of capital

        # ── Step 4: apply correlation penalty ───────────────
        corr_used = False
        if self.correlation_penalty and len(tickers) >= 2:
            try:
                corr        = self._correlation_matrix(tickers, close_map)
                raw_weights = self._apply_correlation_penalty(raw_weights, corr)
                corr_used   = True
            except Exception:
                pass

        # ── Step 5: apply min/max position caps ─────────────
        capped_weights = {}
        for t, w in raw_weights.items():
            w = max(w, self.min_position_pct)
            w = min(w, self.max_position_pct)
            capped_weights[t] = w

        # ── Step 6: normalise so total ≤ 95% of capital ─────
        total_w = sum(capped_weights.values())
        if total_w > 0.95:
            scale = 0.95 / total_w
            capped_weights = {t: w * scale for t, w in capped_weights.items()}

        # ── Step 7: build adjusted signal list ──────────────
        adjusted    = []
        risk_contribs = {}

        for sig in signals:
            t     = sig['ticker']
            price = sig['price']
            w     = capped_weights.get(t, self.min_position_pct)
            vol   = vol_map.get(t, 0.02)

            alloc_capital = capital * w
            shares        = int(alloc_capital / price)

            if shares < 1:
                shares = 1

            trade_value      = shares * price
            daily_rupee_risk = trade_value * vol
            risk_contribs[t] = round(daily_rupee_risk, 2)

            new_sig = dict(sig)
            new_sig['shares']      = shares
            new_sig['trade_value'] = round(trade_value, 2)
            new_sig['rp_weight']   = round(w, 4)
            new_sig['rp_daily_vol']= round(vol * 100, 3)   # as %
            adjusted.append(new_sig)

        total_deployed = sum(s['trade_value'] for s in adjusted)
        cash_remaining = capital - total_deployed

        result = RiskParityResult(
            n_stocks            = len(adjusted),
            total_capital       = capital,
            total_deployed      = round(total_deployed, 2),
            cash_remaining      = round(cash_remaining, 2),
            target_risk_per_pos = round(target_rupee_risk, 2),
            actual_risk_contribs= risk_contribs,
            weights             = {s['ticker']: s['rp_weight'] for s in adjusted},
            vol_estimates       = {t: round(v * 100, 3) for t, v in vol_map.items()},
            correlation_used    = corr_used,
            rebalance_reason    = (
                f'Risk parity allocated {len(adjusted)} positions across '
                f'₹{total_deployed:,.0f} capital — '
                f'{"correlation penalty applied" if corr_used else "no correlation data"}'
            )
        )

        return adjusted, result

    def print_allocation(self, result: RiskParityResult, signals: list):
        """Pretty-prints the risk parity allocation table."""
        print(f"\n  {'─'*58}")
        print(f"  ⚖️  RISK PARITY ALLOCATION  (Equal Risk Contribution)")
        print(f"  {'─'*58}")
        print(f"  {'Ticker':<20} {'DailyVol':>8} {'Weight':>8} "
              f"{'₹ Value':>12} {'Daily Risk':>12}")
        print(f"  {'─'*58}")

        for sig in signals:
            t         = sig['ticker'].replace('.NS', '')
            vol_pct   = sig.get('rp_daily_vol', 0)
            weight    = sig.get('rp_weight', 0)
            value     = sig.get('trade_value', 0)
            drisk     = result.actual_risk_contribs.get(sig['ticker'], 0)
            print(f"  {t:<20} {vol_pct:>7.2f}% {weight:>7.1%} "
                  f"  ₹{value:>10,.0f}   ₹{drisk:>8,.0f}")

        total_drisk = sum(result.actual_risk_contribs.values())
        print(f"  {'─'*58}")
        print(f"  {'TOTAL':<20} {'':>8} {'':>8} "
              f"  ₹{result.total_deployed:>10,.0f}   ₹{total_drisk:>8,.0f}")
        print(f"  Cash remaining : ₹{result.cash_remaining:,.0f} "
              f"({result.cash_remaining / result.total_capital * 100:.1f}%)")
        print(f"  Target daily risk / position : ₹{result.target_risk_per_pos:,.0f} "
              f"({result.target_risk_per_pos / result.total_capital * 100:.2f}% of capital)")
        print(f"  Correlation penalty applied  : {result.correlation_used}")
        print(f"  {'─'*58}\n")