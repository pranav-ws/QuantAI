"""
src/tail_risk.py

Tail Risk Monitor — Black Swan Detector for QuantAI.

What it does:
  Normal risk managers (like src/risk.py) handle day-to-day drawdowns —
  "we're down 3%, tighten stops". This module handles a completely
  different class of problem: detecting the early statistical signatures
  of a regime-change or crash BEFORE the price fully collapses.

  Black swans don't announce themselves. But they do leave forensic
  traces in market microstructure before they arrive:

    1. Volatility clustering + spike      → VIX-equivalent (realized vol)
                                            jumps multiple standard devs
    2. Fat-tail return distribution       → Excess kurtosis suddenly rises,
                                            meaning daily moves are becoming
                                            abnormally large vs recent history
    3. Skew collapse                      → Negative skewness deepens — more
                                            extreme down days than up days
    4. Correlation spike                  → All stocks suddenly move together
                                            (diversification fails exactly when
                                            you need it most — a classic crash signal)
    5. Liquidity stress                   → Volume dries up on up-days and
                                            surges on down-days (OBV divergence)
    6. Value-at-Risk breach count         → How many stocks hit their 5th-
                                            percentile worst-day return today
                                            (many simultaneous VaR breaches =
                                            systemic, not idiosyncratic risk)

  Each signal is scored 0–1 and combined into a single "Tail Risk Index"
  (TRI) from 0 (no stress) to 1 (extreme stress). The TRI is then mapped
  to an alert level:

      TRI < 0.30  → 🟢 NORMAL     — business as usual
      TRI < 0.50  → 🟡 ELEVATED   — watch carefully, reduce new entries
      TRI < 0.70  → 🟠 HIGH       — stop new entries, tighten stops
      TRI >= 0.70 → 🔴 CRITICAL   — close positions, go to cash

  This plugs into RiskManager (src/risk.py) via an override flag —
  when TRI hits CRITICAL, RiskManager.is_trading_halted is forced True
  regardless of the portfolio drawdown level.
"""

import numpy as np# type: ignore
import pandas as pd# type: ignore
from dataclasses import dataclass, field
from typing import Optional
import sqlite3
import os


# ── Alert levels ─────────────────────────────────────────

LEVEL_NORMAL   = 'NORMAL'
LEVEL_ELEVATED = 'ELEVATED'
LEVEL_HIGH     = 'HIGH'
LEVEL_CRITICAL = 'CRITICAL'

TRI_THRESHOLDS = {
    LEVEL_NORMAL   : 0.30,
    LEVEL_ELEVATED : 0.50,
    LEVEL_HIGH     : 0.70,
    LEVEL_CRITICAL : 1.01,   # >= 0.70 triggers CRITICAL
}

LEVEL_EMOJI = {
    LEVEL_NORMAL   : '🟢',
    LEVEL_ELEVATED : '🟡',
    LEVEL_HIGH     : '🟠',
    LEVEL_CRITICAL : '🔴',
}


@dataclass
class TailRiskReport:
    """Full tail risk snapshot — one per run."""
    timestamp       : str
    tri             : float          # Tail Risk Index 0–1
    level           : str            # NORMAL / ELEVATED / HIGH / CRITICAL
    halt_trading    : bool           # True if TRI >= 0.70

    # Component scores (each 0–1)
    vol_score       : float          # realized volatility stress
    kurtosis_score  : float          # fat-tail score
    skew_score      : float          # negative skewness score
    correlation_score: float         # correlation spike score
    liquidity_score : float          # volume/OBV stress score
    var_breach_score: float          # simultaneous VaR breach score

    # Supporting detail
    realized_vol_ann: float          # annualized 20-day realized vol (%)
    avg_kurtosis    : float          # cross-stock average excess kurtosis
    avg_skew        : float          # cross-stock average skewness
    avg_pairwise_corr: float         # average pairwise return correlation
    var_breach_count: int            # how many stocks breached 5th-pct VaR today
    n_stocks        : int            # how many stocks were in the scan

    # Human-readable reasons behind the score
    reasons         : list = field(default_factory=list)
    recommendation  : str  = ''


# ── Core detector ─────────────────────────────────────────

class TailRiskMonitor:
    """
    Scans a basket of stocks and computes the Tail Risk Index.

    Usage:
        monitor = TailRiskMonitor()
        report  = monitor.scan(price_dict)   # price_dict: {ticker: pd.Series of Close}

    price_dict values must have a DatetimeIndex — exactly what
    load_close() in drawdown_heatmap.py / correlation_matrix.py returns.
    """

    def __init__(self,
                 lookback_short  = 20,    # short window for vol/kurtosis (trading days)
                 lookback_long   = 252,   # long window for baseline vol
                 vol_z_trigger   = 2.0,   # z-score above which vol is "stressed"
                 kurt_trigger    = 3.0,   # excess kurtosis above which tails are "fat"
                 skew_trigger    = -0.5,  # skewness below which we flag negative skew
                 corr_trigger    = 0.65,  # avg pairwise corr above which diversification fails
                 var_pct         = 0.05,  # VaR percentile (5th = 95% VaR)
                 var_breach_pct  = 0.20,  # fraction of stocks breaching VaR = systemic
                 # Component weights in the TRI (must sum to 1.0)
                 w_vol           = 0.30,
                 w_kurt          = 0.20,
                 w_skew          = 0.15,
                 w_corr          = 0.20,
                 w_liquidity     = 0.07,
                 w_var           = 0.08):

        self.lookback_short  = lookback_short
        self.lookback_long   = lookback_long
        self.vol_z_trigger   = vol_z_trigger
        self.kurt_trigger    = kurt_trigger
        self.skew_trigger    = skew_trigger
        self.corr_trigger    = corr_trigger
        self.var_pct         = var_pct
        self.var_breach_pct  = var_breach_pct
        self.w_vol           = w_vol
        self.w_kurt          = w_kurt
        self.w_skew          = w_skew
        self.w_corr          = w_corr
        self.w_liquidity     = w_liquidity
        self.w_var           = w_var

    # ── Internal helpers ──────────────────────────────────

    def _log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        return np.log(prices / prices.shift(1)).dropna()

    def _vol_score(self, returns: pd.DataFrame) -> tuple[float, float, list]:
        """
        Realized volatility stress score.
        Compares the rolling short-window vol to the long-window baseline.
        A z-score of 2.0 above the mean = score 1.0.
        """
        reasons = []
        # Equal-weight index return
        idx_ret = returns.mean(axis=1)

        short_vol = idx_ret.rolling(self.lookback_short).std().iloc[-1] * np.sqrt(252) * 100
        long_vol  = idx_ret.rolling(self.lookback_long,  min_periods=60).std()
        long_mean = long_vol.mean() * np.sqrt(252) * 100
        long_std  = long_vol.std()  * np.sqrt(252) * 100

        if long_std > 0:
            z = (short_vol - long_mean) / long_std
        else:
            z = 0.0

        score = float(np.clip(z / self.vol_z_trigger, 0, 1))

        if score >= 0.7:
            reasons.append(f"⚡ Realized vol {short_vol:.1f}% ann. is "
                           f"{z:.1f}σ above {long_mean:.1f}% baseline — volatility regime shift")
        elif score >= 0.4:
            reasons.append(f"⚠️  Realized vol {short_vol:.1f}% ann. is elevated "
                           f"({z:.1f}σ above baseline)")

        return score, short_vol, reasons

    def _kurtosis_score(self, returns: pd.DataFrame) -> tuple[float, float, list]:
        """
        Excess kurtosis score.
        High kurtosis = fat tails = crash risk.
        Normal distribution has kurtosis = 3, excess kurtosis = 0.
        """
        reasons = []
        recent  = returns.iloc[-self.lookback_short:]
        kurt    = recent.kurtosis()           # excess kurtosis per stock
        avg_kurt = float(kurt.mean())

        score = float(np.clip(avg_kurt / self.kurt_trigger, 0, 1))

        if score >= 0.7:
            reasons.append(f"💀 Excess kurtosis {avg_kurt:.2f} — fat tails, "
                           f"crash-sized moves becoming more frequent")
        elif score >= 0.4:
            reasons.append(f"⚠️  Excess kurtosis {avg_kurt:.2f} — tails thickening")

        return score, avg_kurt, reasons

    def _skew_score(self, returns: pd.DataFrame) -> tuple[float, float, list]:
        """
        Negative skewness score.
        More and larger down-days than up-days = classic pre-crash signature.
        """
        reasons = []
        recent   = returns.iloc[-self.lookback_short:]
        skew     = recent.skew()
        avg_skew = float(skew.mean())

        # Score increases as skew goes more negative (below skew_trigger)
        score = float(np.clip((self.skew_trigger - avg_skew) / abs(self.skew_trigger), 0, 1))

        if score >= 0.7:
            reasons.append(f"📉 Strong negative skew ({avg_skew:.2f}) — down-moves "
                           f"dominating, asymmetric crash risk building")
        elif score >= 0.4:
            reasons.append(f"⚠️  Negative return skew ({avg_skew:.2f}) — more bad days than good")

        return score, avg_skew, reasons

    def _correlation_score(self, returns: pd.DataFrame) -> tuple[float, float, list]:
        """
        Cross-stock correlation spike score.
        When everything falls together, diversification has failed.
        Average pairwise correlation rising sharply above historical mean
        is one of the most reliable early warning signals.
        """
        reasons = []

        if returns.shape[1] < 3:
            return 0.0, 0.0, []

        recent_corr = returns.iloc[-self.lookback_short:].corr()
        upper       = recent_corr.where(
            np.triu(np.ones(recent_corr.shape), k=1).astype(bool)
        )
        avg_corr = float(upper.stack().mean())

        score = float(np.clip(
            (avg_corr - 0.3) / (self.corr_trigger - 0.3), 0, 1
        ))

        if score >= 0.7:
            reasons.append(f"🔗 Correlation spike — avg pairwise {avg_corr:.2f} "
                           f"(normal ~0.30) — diversification has FAILED, "
                           f"systemic move in progress")
        elif score >= 0.4:
            reasons.append(f"⚠️  Rising cross-stock correlation ({avg_corr:.2f}) — "
                           f"stocks increasingly moving together")

        return score, avg_corr, reasons

    def _liquidity_score(self, prices: pd.DataFrame,
                          volumes: Optional[pd.DataFrame]) -> tuple[float, list]:
        """
        Liquidity/volume stress score.
        Pattern: volume surges on down-days, evaporates on up-days.
        This is the market-microstructure signature of distribution
        (smart money selling into retail buyers).
        """
        reasons = []

        if volumes is None or volumes.empty:
            return 0.0, []

        try:
            ret     = prices.pct_change().dropna()
            vol_chg = volumes.pct_change().dropna().reindex(ret.index).fillna(0)

            # Align
            common = ret.index.intersection(vol_chg.index)
            if len(common) < self.lookback_short:
                return 0.0, []

            ret     = ret.loc[common]
            vol_chg = vol_chg.loc[common]

            recent_ret = ret.iloc[-self.lookback_short:]
            recent_vol = vol_chg.iloc[-self.lookback_short:]

            # Average volume on up-days vs down-days (cross all stocks)
            scores = []
            for col in ret.columns:
                if col not in vol_chg.columns:
                    continue
                r = recent_ret[col]
                v = recent_vol[col]
                up_vol   = v[r > 0].mean()
                down_vol = v[r < 0].mean()
                if not (np.isnan(up_vol) or np.isnan(down_vol)):
                    # Positive score = more volume on down-days (distribution)
                    scores.append(down_vol - up_vol)

            if not scores:
                return 0.0, []

            stress = float(np.mean(scores))
            # Normalize: +0.5 daily vol change difference = score 1.0
            score = float(np.clip(stress / 0.5, 0, 1))

            if score >= 0.7:
                reasons.append(f"📦 Liquidity stress — heavy volume on down-days, "
                               f"thin volume on up-days (distribution pattern)")
            elif score >= 0.4:
                reasons.append(f"⚠️  Volume skewing toward down-days — watch for distribution")

            return score, reasons

        except Exception:
            return 0.0, []

    def _var_breach_score(self, returns: pd.DataFrame) -> tuple[float, int, list]:
        """
        Simultaneous VaR breach count.
        Checks how many stocks had a return today worse than their own
        historical 5th-percentile (i.e., today was a 1-in-20 bad day for them).
        Many stocks breaching simultaneously = systemic shock, not noise.
        """
        reasons = []

        if len(returns) < 60:
            return 0.0, 0, []

        historical = returns.iloc[:-1]          # all history except today
        today      = returns.iloc[-1]

        breach_count = 0
        for col in returns.columns:
            threshold = float(np.percentile(historical[col].dropna(),
                                             self.var_pct * 100))
            if float(today[col]) <= threshold:
                breach_count += 1

        breach_pct = breach_count / len(returns.columns)
        score      = float(np.clip(breach_pct / self.var_breach_pct, 0, 1))

        if score >= 0.7:
            reasons.append(f"🚨 {breach_count}/{len(returns.columns)} stocks breached "
                           f"their 5th-percentile VaR today — systemic shock, not noise")
        elif score >= 0.4:
            reasons.append(f"⚠️  {breach_count}/{len(returns.columns)} stocks hit "
                           f"VaR threshold — elevated systemic stress")

        return score, breach_count, reasons

    # ── Main scan ─────────────────────────────────────────

    def scan(self,
             price_dict  : dict,          # {ticker: pd.Series of Close}
             volume_dict : Optional[dict] = None  # {ticker: pd.Series of Volume}
             ) -> TailRiskReport:
        """
        Run the full black swan detector across all provided tickers.

        price_dict  : {ticker: pd.Series(Close, DatetimeIndex)}
        volume_dict : {ticker: pd.Series(Volume, DatetimeIndex)} — optional
        Returns a TailRiskReport dataclass.
        """
        from datetime import datetime as _dt

        # Align all price series onto common dates, forward-fill gaps
        price_df = pd.DataFrame(price_dict).dropna(how='all').ffill().dropna()
        if price_df.empty or price_df.shape[1] < 3:
            return TailRiskReport(
                timestamp=_dt.now().isoformat(),
                tri=0.0, level=LEVEL_NORMAL, halt_trading=False,
                vol_score=0.0, kurtosis_score=0.0, skew_score=0.0,
                correlation_score=0.0, liquidity_score=0.0, var_breach_score=0.0,
                realized_vol_ann=0.0, avg_kurtosis=0.0, avg_skew=0.0,
                avg_pairwise_corr=0.0, var_breach_count=0, n_stocks=0,
                reasons=['Insufficient data for tail risk scan'],
                recommendation='Collect more price history before relying on TRI.'
            )

        returns = self._log_returns(price_df)
        if len(returns) < self.lookback_short + 5:
            return TailRiskReport(
                timestamp=_dt.now().isoformat(),
                tri=0.0, level=LEVEL_NORMAL, halt_trading=False,
                vol_score=0.0, kurtosis_score=0.0, skew_score=0.0,
                correlation_score=0.0, liquidity_score=0.0, var_breach_score=0.0,
                realized_vol_ann=0.0, avg_kurtosis=0.0, avg_skew=0.0,
                avg_pairwise_corr=0.0, var_breach_count=0, n_stocks=len(price_df.columns),
                reasons=['Not enough return history for a meaningful scan (need ≥25 days)'],
                recommendation='Continue monitoring — data window is too short.'
            )

        vol_df  = None
        if volume_dict:
            vol_df = pd.DataFrame(volume_dict).reindex(price_df.index).ffill()

        # ── Run all six detectors ─────────────────────────
        v_score, real_vol, v_reasons    = self._vol_score(returns)
        k_score, avg_kurt, k_reasons    = self._kurtosis_score(returns)
        s_score, avg_skew, s_reasons    = self._skew_score(returns)
        c_score, avg_corr, c_reasons    = self._correlation_score(returns)
        l_score, l_reasons              = self._liquidity_score(price_df, vol_df)
        var_score, var_ct, var_reasons  = self._var_breach_score(returns)

        # ── Weighted Tail Risk Index ──────────────────────
        tri = (
            v_score   * self.w_vol       +
            k_score   * self.w_kurt      +
            s_score   * self.w_skew      +
            c_score   * self.w_corr      +
            l_score   * self.w_liquidity +
            var_score * self.w_var
        )
        tri = float(np.clip(tri, 0.0, 1.0))

        # ── Determine alert level ─────────────────────────
        if tri >= 0.70:
            level = LEVEL_CRITICAL
        elif tri >= 0.50:
            level = LEVEL_HIGH
        elif tri >= 0.30:
            level = LEVEL_ELEVATED
        else:
            level = LEVEL_NORMAL

        halt_trading = level == LEVEL_CRITICAL

        # ── Collect all reasons (only from elevated detectors) ──
        all_reasons = v_reasons + k_reasons + s_reasons + \
                      c_reasons + l_reasons + var_reasons
        if not all_reasons:
            all_reasons = ['✅ No individual stress signal exceeded threshold']

        # ── Recommendation text ───────────────────────────
        reco = {
            LEVEL_NORMAL  : ('Markets within normal statistical bounds. '
                              'Standard risk management applies.'),
            LEVEL_ELEVATED: ('Tail risk is building. Reduce size on new entries. '
                              'Tighten stops on existing positions. Watch closely.'),
            LEVEL_HIGH    : ('Significant tail risk detected. Stop all new entries. '
                              'Tighten stops to ATR×1.0. Consider partial exits.'),
            LEVEL_CRITICAL: ('🚨 BLACK SWAN ALERT 🚨 — Systemic stress at extreme levels. '
                              'HALT all new trades. Close or hedge all open positions. '
                              'Go to cash. Do not average down.'),
        }[level]

        return TailRiskReport(
            timestamp        = _dt.now().isoformat(),
            tri              = round(tri, 4),
            level            = level,
            halt_trading     = halt_trading,
            vol_score        = round(v_score, 4),
            kurtosis_score   = round(k_score, 4),
            skew_score       = round(s_score, 4),
            correlation_score= round(c_score, 4),
            liquidity_score  = round(l_score, 4),
            var_breach_score = round(var_score, 4),
            realized_vol_ann = round(real_vol, 2),
            avg_kurtosis     = round(avg_kurt, 3),
            avg_skew         = round(avg_skew, 3),
            avg_pairwise_corr= round(avg_corr, 4),
            var_breach_count = var_ct,
            n_stocks         = len(price_df.columns),
            reasons          = all_reasons,
            recommendation   = reco,
        )

    def scan_from_db(self, period_days: int = 252) -> TailRiskReport:
        """
        Convenience wrapper — loads all 50 Nifty stocks from the local
        SQLite DB (or yfinance fallback) and runs scan() automatically.
        Same data path used by drawdown_heatmap.py and correlation_matrix.py.
        """
        from src.data_collector import STOCK_UNIVERSE
        import sqlite3

        db_path = os.path.join('data', 'quantai.db')
        price_dict  = {}
        volume_dict = {}

        for ticker in STOCK_UNIVERSE:
            df = None

            # Try DB first
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    df   = pd.read_sql_query(
                        "SELECT date, close, volume FROM prices "
                        "WHERE ticker=? ORDER BY date ASC",
                        conn, params=(ticker,)
                    )
                    conn.close()
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        df.columns = ['Close', 'Volume']
                        df['Close']  = pd.to_numeric(df['Close'],  errors='coerce')
                        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
                        df.dropna(subset=['Close'], inplace=True)
                    else:
                        df = None
                except Exception:
                    df = None

            # Fallback to yfinance
            if df is None:
                try:
                    import yfinance as yf # type: ignore
                    years = max(1, round(period_days / 252))
                    raw = yf.download(ticker, period=f'{years}y',
                                      progress=False, auto_adjust=True)
                    if not raw.empty:
                        if hasattr(raw.columns, 'levels'):
                            raw.columns = [c[0] for c in raw.columns]
                        df = raw[['Close', 'Volume']].copy()
                except Exception:
                    pass

            if df is None or df.empty:
                continue

            cutoff = df.index[-1] - pd.Timedelta(days=period_days)
            df     = df[df.index >= cutoff]

            if len(df) >= 30:
                price_dict[ticker]  = df['Close']
                volume_dict[ticker] = df['Volume']

        return self.scan(price_dict, volume_dict)