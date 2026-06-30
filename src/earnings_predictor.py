"""
src/earnings_predictor.py

Earnings Predictor — Beat / Miss / In-Line Estimator.

What this does:
  Estimates the probability that a company will BEAT, MISS, or come
  IN-LINE with analyst EPS consensus for its next earnings release.

  Indian companies listed on NSE report quarterly results. The pattern
  of whether they beat or miss is not random — it's driven by recurring
  signals that are observable BEFORE the announcement:

    Signal 1 — Historical beat rate
      Has this company consistently beaten consensus? Companies with
      strong IR teams and conservative guidance tend to beat repeatedly.
      Weight: 30%

    Signal 2 — EPS surprise magnitude trend
      Is the size of beats/misses growing or shrinking? A company that
      used to beat by 15% now barely beating by 2% is a warning sign.
      Weight: 15%

    Signal 3 — Pre-earnings price drift
      How has the stock moved in the 20 days BEFORE past earnings dates?
      A stock that already ran up 12% into earnings has high expectations
      baked in — a beat is already priced, a miss is not. Conversely, a
      stock that drifted down into earnings often beats ("low bar effect").
      Weight: 20%

    Signal 4 — Volume accumulation pattern
      Is smart money positioning? Volume on up-days vs down-days in the
      20 days before earnings is a leading indicator of institutional
      positioning.
      Weight: 15%

    Signal 5 — Revenue growth direction
      Revenue is harder to manipulate than EPS. Accelerating revenue
      growth in the trailing two quarters is a leading signal for an
      EPS beat. Decelerating revenue = caution.
      Weight: 10%

    Signal 6 — News sentiment into earnings
      What is the tone of recent news about this company? Unusually
      positive sentiment heading into earnings reflects information
      asymmetry that often precedes beats.
      Weight: 10%

Output:
  An EarningsPrediction dataclass with:
    - beat_probability  : float 0–1
    - prediction        : 'BEAT' / 'MISS' / 'IN-LINE' / 'INSUFFICIENT_DATA'
    - confidence        : float 0–1 (data quality, not same as beat_prob)
    - next_earnings_date: str or None
    - days_to_earnings  : int or None
    - signal_scores     : dict of each signal's raw score
    - key_factors       : list of human-readable reasons for the prediction
    - risk_factors      : list of risks that could invalidate it
    - data_quality      : 'HIGH' / 'MEDIUM' / 'LOW' (n of quarters available)

Limitations:
  - Data depends on yfinance scraping Yahoo Finance. Indian stocks
    sometimes have incomplete or missing earnings history on Yahoo.
  - This is a PROBABILISTIC estimator, not a crystal ball. Even a
    BEAT prediction at 75% confidence means a 25% chance of miss.
  - Use alongside the ensemble signal, not as a replacement for it.
  - Never trade DURING the earnings announcement itself — only the
    PRE-earnings setup is modelled here.
"""

import os
import time
import sqlite3
import warnings
warnings.filterwarnings('ignore')

import numpy as np# type: ignore
import pandas as pd# type: ignore
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import Optional


# ── Prediction output ─────────────────────────────────────

@dataclass
class EarningsPrediction:
    ticker              : str
    company_name        : str   = ''
    sector              : str   = ''

    # Core output
    beat_probability    : float = 0.50
    prediction          : str   = 'INSUFFICIENT_DATA'   # BEAT / MISS / IN-LINE
    confidence          : float = 0.0    # 0–1: how much data we had

    # Earnings timing
    next_earnings_date  : Optional[str] = None
    days_to_earnings    : Optional[int] = None
    last_reported_date  : Optional[str] = None

    # Historical record
    historical_beat_rate: float = 0.50
    avg_eps_surprise_pct: float = 0.0    # average % surprise vs consensus
    n_quarters          : int   = 0      # how many quarters of history

    # Individual signal scores (each 0–1, higher = more bullish)
    signal_beat_rate    : float = 0.50
    signal_surprise_trend: float = 0.50
    signal_price_drift  : float = 0.50
    signal_volume       : float = 0.50
    signal_revenue      : float = 0.50
    signal_sentiment    : float = 0.50

    # Context
    data_quality        : str   = 'LOW'   # HIGH / MEDIUM / LOW
    key_factors         : list  = field(default_factory=list)
    risk_factors        : list  = field(default_factory=list)
    raw_surprises       : list  = field(default_factory=list)


# ── Weights ───────────────────────────────────────────────

SIGNAL_WEIGHTS = {
    'beat_rate'      : 0.30,
    'surprise_trend' : 0.15,
    'price_drift'    : 0.20,
    'volume'         : 0.15,
    'revenue'        : 0.10,
    'sentiment'      : 0.10,
}

BEAT_THRESHOLD   = 0.60
MISS_THRESHOLD   = 0.40
MIN_QUARTERS     = 3     # need at least this many quarters for meaningful prediction

# Cache — earnings data changes infrequently (avoids hammering Yahoo Finance)
_cache: dict = {}
_CACHE_TTL = 6 * 3600    # 6 hours


# ── Core predictor ────────────────────────────────────────

class EarningsPredictor:

    def __init__(self):
        pass

    # ── Data fetching ─────────────────────────────────────

    def _get_ticker_obj(self, ticker: str):
        """Returns a yfinance Ticker object."""
        import yfinance as yf # type: ignore
        return yf.Ticker(ticker)

    def _fetch_earnings_history(self, tk) -> pd.DataFrame:
        """
        Fetches historical quarterly EPS actual vs estimate.
        Returns DataFrame with columns: date, eps_actual, eps_estimate, surprise_pct
        Tries multiple yfinance attribute names for robustness across versions.
        """
        df = None

        # Try newer yfinance attribute first
        for attr in ['earnings_history', 'quarterly_earnings', 'earnings']:
            try:
                raw = getattr(tk, attr, None)
                if raw is None:
                    continue
                if callable(raw):
                    raw = raw()
                if isinstance(raw, pd.DataFrame) and not raw.empty:
                    df = raw.copy()
                    break
            except Exception:
                continue

        if df is None or df.empty:
            return pd.DataFrame()

        # Normalise column names across yfinance versions
        col_map = {}
        cols_lower = {c.lower().replace(' ', '_'): c for c in df.columns}

        for std, aliases in [
            ('eps_actual',   ['epsactual', 'reported_eps', 'eps', 'actual']),
            ('eps_estimate', ['epsestimate', 'eps_estimate', 'estimate']),
        ]:
            for alias in aliases:
                if alias in cols_lower:
                    col_map[cols_lower[alias]] = std
                    break

        df = df.rename(columns=col_map)

        # Must have both columns
        if 'eps_actual' not in df.columns or 'eps_estimate' not in df.columns:
            return pd.DataFrame()

        df['eps_actual']   = pd.to_numeric(df['eps_actual'],   errors='coerce')
        df['eps_estimate'] = pd.to_numeric(df['eps_estimate'], errors='coerce')
        df.dropna(subset=['eps_actual', 'eps_estimate'], inplace=True)

        # Compute surprise
        df['surprise'] = df['eps_actual'] - df['eps_estimate']
        df['surprise_pct'] = np.where(
            df['eps_estimate'].abs() > 0.001,
            df['surprise'] / df['eps_estimate'].abs() * 100,
            0.0
        )
        df['beat'] = df['surprise'] > 0

        # Ensure chronological order
        if df.index.dtype == 'datetime64[ns]' or hasattr(df.index, 'to_pydatetime'):
            df = df.sort_index()
        else:
            try:
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
            except Exception:
                pass

        return df.tail(12)   # last 12 quarters max

    def _fetch_next_earnings_date(self, tk) -> Optional[str]:
        """Returns the next earnings date as YYYY-MM-DD string, or None."""
        try:
            dates_df = tk.earnings_dates
            if dates_df is None or (hasattr(dates_df, 'empty') and dates_df.empty):
                return None
            # earnings_dates has future dates too — find first future date
            today = pd.Timestamp(date.today())
            if hasattr(dates_df.index, 'tz_localize'):
                try:
                    dates_df.index = dates_df.index.tz_localize(None)
                except Exception:
                    try:
                        dates_df.index = dates_df.index.tz_convert(None)
                    except Exception:
                        pass
            future = dates_df[dates_df.index >= today]
            if not future.empty:
                return str(future.index[-1].date())
        except Exception:
            pass
        return None

    def _fetch_quarterly_revenue(self, tk) -> Optional[pd.Series]:
        """Returns quarterly revenue series, most recent last."""
        for attr in ['quarterly_financials', 'quarterly_income_stmt']:
            try:
                fin = getattr(tk, attr, None)
                if fin is None or (hasattr(fin, 'empty') and fin.empty):
                    continue
                # Revenue row
                for row_name in ['Total Revenue', 'Revenue', 'Net Revenue']:
                    if row_name in fin.index:
                        rev = fin.loc[row_name].sort_index()
                        return pd.to_numeric(rev, errors='coerce').dropna()
            except Exception:
                continue
        return None

    # ── Individual signal computations ────────────────────

    def _score_beat_rate(self, history: pd.DataFrame) -> tuple[float, float, list]:
        """
        Historical beat rate → 0–1 signal.
        0.5 = 50% beat rate (neutral), 1.0 = 100% beat rate.
        Returns (signal_score, beat_rate, key_reasons)
        """
        if history.empty or len(history) < MIN_QUARTERS:
            return 0.50, 0.50, []

        beat_rate = float(history['beat'].mean())
        # Scale: 0% beat rate → signal 0.0, 100% → 1.0
        score = beat_rate
        reasons = []
        if beat_rate >= 0.75:
            reasons.append(f"✅ Strong historical beat rate: {beat_rate*100:.0f}% of quarters")
        elif beat_rate <= 0.35:
            reasons.append(f"❌ Weak historical beat rate: {beat_rate*100:.0f}% of quarters")

        return round(score, 3), round(beat_rate, 3), reasons

    def _score_surprise_trend(self, history: pd.DataFrame) -> tuple[float, float, list]:
        """
        Is the EPS surprise magnitude improving or shrinking?
        Recent quarters compared to older ones.
        Returns (signal_score, avg_surprise_pct, key_reasons)
        """
        if history.empty or len(history) < MIN_QUARTERS:
            return 0.50, 0.0, []

        surp = history['surprise_pct'].fillna(0)
        avg_all    = float(surp.mean())
        avg_recent = float(surp.iloc[-2:].mean()) if len(surp) >= 2 else avg_all
        avg_old    = float(surp.iloc[:-2].mean()) if len(surp) >= 4 else avg_all

        # Score: positive surprise and improving trend → high score
        # Clamp surprise_pct to ±50% before normalising
        norm_avg = float(np.clip(avg_all / 50, -1, 1))
        score = (norm_avg + 1) / 2   # map [-1,1] → [0,1]

        reasons = []
        if avg_all > 5:
            reasons.append(f"✅ Average EPS surprise: +{avg_all:.1f}% vs consensus")
        elif avg_all < -5:
            reasons.append(f"❌ Average EPS miss: {avg_all:.1f}% vs consensus")

        if len(surp) >= 4:
            if avg_recent > avg_old + 3:
                reasons.append("✅ Beat magnitude is improving in recent quarters")
            elif avg_recent < avg_old - 3:
                reasons.append("⚠️  Beat magnitude shrinking — lowering bar")

        return round(score, 3), round(avg_all, 2), reasons

    def _score_price_drift(self, close: pd.Series,
                           earnings_dates: list,
                           window: int = 20) -> tuple[float, list]:
        """
        How did the stock move in the 20 days BEFORE each past earnings?
        Negative drift into earnings → underpriced expectations → beat signal.
        Positive drift → expectations already high → beat already priced in.
        Returns (signal_score, key_reasons)
        """
        if not earnings_dates or len(close) < window + 5:
            return 0.50, []

        drifts = []
        for d in earnings_dates:
            try:
                ed = pd.Timestamp(d)
                # Find closest trading day at or before ed
                mask = close.index <= ed
                if not mask.any():
                    continue
                idx_end = close.index[mask][-1]
                loc_end = close.index.get_loc(idx_end)
                loc_start = max(0, loc_end - window)
                window_close = close.iloc[loc_start:loc_end + 1]
                if len(window_close) < 3:
                    continue
                drift = (window_close.iloc[-1] - window_close.iloc[0]) \
                        / window_close.iloc[0] * 100
                drifts.append(drift)
            except Exception:
                continue

        if not drifts:
            return 0.50, []

        avg_drift = float(np.mean(drifts))
        # Negative pre-earnings drift → historically led to beats (low bar)
        # Score: drift of -10% → high beat signal, drift of +10% → low beat signal
        score = float(np.clip(0.5 - avg_drift / 20, 0, 1))

        reasons = []
        if avg_drift < -5:
            reasons.append(f"✅ Stock typically drifts {avg_drift:.1f}% into earnings — low expectations = beat setup")
        elif avg_drift > 8:
            reasons.append(f"⚠️  Stock already ran {avg_drift:.1f}% into past earnings — high expectations already priced")

        return round(score, 3), reasons

    def _score_volume(self, close: pd.Series,
                      volume: pd.Series,
                      window: int = 20) -> tuple[float, list]:
        """
        Computes OBV-based accumulation signal in the last `window` days.
        More volume on up-days than down-days → accumulation → beat signal.
        """
        if len(close) < window + 5 or volume is None or len(volume) < window + 5:
            return 0.50, []

        ret    = close.pct_change().iloc[-window:]
        vol    = volume.iloc[-window:]

        up_vol   = vol[ret > 0].mean()
        down_vol = vol[ret < 0].mean()

        if pd.isna(up_vol) or pd.isna(down_vol) or down_vol == 0:
            return 0.50, []

        vol_ratio = float(up_vol / down_vol)
        # vol_ratio > 1.5 → strong accumulation
        score = float(np.clip((vol_ratio - 0.5) / 2, 0, 1))

        reasons = []
        if vol_ratio > 1.5:
            reasons.append(f"✅ Volume accumulation: {vol_ratio:.1f}x more buying than selling volume")
        elif vol_ratio < 0.7:
            reasons.append(f"⚠️  Distribution: more selling volume ({vol_ratio:.1f}x) than buying")

        return round(score, 3), reasons

    def _score_revenue(self, revenue: Optional[pd.Series]) -> tuple[float, list]:
        """
        Is quarterly revenue accelerating or decelerating?
        YoY revenue growth acceleration → bullish for EPS beat.
        """
        if revenue is None or len(revenue) < 4:
            return 0.50, []

        # YoY growth for each available quarter
        yoy = []
        for i in range(min(4, len(revenue))):
            if i + 4 < len(revenue):
                prior = revenue.iloc[-(i + 4 + 1)]
                curr  = revenue.iloc[-(i + 1)]
                if prior and prior != 0:
                    yoy.append((curr - prior) / abs(prior) * 100)

        if not yoy:
            return 0.50, []

        avg_yoy = float(np.mean(yoy))
        score   = float(np.clip(0.5 + avg_yoy / 40, 0, 1))

        reasons = []
        if avg_yoy > 10:
            reasons.append(f"✅ Revenue growing {avg_yoy:.1f}% YoY — strong top-line supports EPS beat")
        elif avg_yoy < -5:
            reasons.append(f"❌ Revenue declining {avg_yoy:.1f}% YoY — headwind for EPS")

        # Check acceleration (recent quarter vs older)
        if len(yoy) >= 2 and yoy[0] > yoy[-1] + 5:
            reasons.append("✅ Revenue growth accelerating")

        return round(score, 3), reasons

    def _score_sentiment(self, ticker: str) -> tuple[float, list]:
        """
        Pulls from src/news_sentiment.py — reuses the existing VADER pipeline.
        Recent bullish sentiment ahead of earnings is a leading signal.
        """
        try:
            from src.news_sentiment import get_sentiment
            score_raw, _ = get_sentiment(ticker, max_articles=10)
            # score_raw: -1 to +1 → signal: 0 to 1
            score = float(np.clip((score_raw + 1) / 2, 0, 1))
            reasons = []
            if score_raw > 0.15:
                reasons.append(f"✅ Bullish news sentiment ({score_raw:+.2f}) heading into earnings")
            elif score_raw < -0.15:
                reasons.append(f"⚠️  Bearish news sentiment ({score_raw:+.2f}) ahead of earnings")
            return round(score, 3), reasons
        except Exception:
            return 0.50, []

    # ── Main predict ──────────────────────────────────────

    def predict(self, ticker: str, verbose: bool = True) -> EarningsPrediction:
        """
        Full earnings beat/miss prediction for a single ticker.
        Returns EarningsPrediction dataclass.
        """
        from src.data_collector import STOCK_UNIVERSE

        # Check cache
        cache_key = ticker
        if cache_key in _cache:
            cached_at, cached_result = _cache[cache_key]
            if time.time() - cached_at < _CACHE_TTL:
                return cached_result

        name   = STOCK_UNIVERSE.get(ticker, (ticker, 'Unknown'))[0]
        sector = STOCK_UNIVERSE.get(ticker, (ticker, 'Unknown'))[1]

        pred = EarningsPrediction(ticker=ticker, company_name=name, sector=sector)

        try:
            tk = self._get_ticker_obj(ticker)

            # ── Fetch raw data ───────────────────────────
            history  = self._fetch_earnings_history(tk)
            next_ed  = self._fetch_next_earnings_date(tk)
            revenue  = self._fetch_quarterly_revenue(tk)

            # Price + volume from DB / yfinance
            close, volume = self._load_price_volume(ticker, days=120)

            # ── Fill in basic fields ─────────────────────
            pred.n_quarters = len(history)
            if next_ed:
                pred.next_earnings_date = next_ed
                try:
                    days = (datetime.strptime(next_ed, '%Y-%m-%d').date()
                            - date.today()).days
                    pred.days_to_earnings = days
                except Exception:
                    pass

            if not history.empty:
                pred.last_reported_date = str(history.index[-1].date()) \
                    if hasattr(history.index[-1], 'date') else str(history.index[-1])[:10]
                pred.raw_surprises = [
                    {'date': str(idx)[:10],
                     'eps_actual': round(float(row['eps_actual']), 2),
                     'eps_estimate': round(float(row['eps_estimate']), 2),
                     'surprise_pct': round(float(row['surprise_pct']), 1),
                     'beat': bool(row['beat'])}
                    for idx, row in history.iterrows()
                ]

            # ── Data quality ─────────────────────────────
            n = pred.n_quarters
            pred.data_quality = 'HIGH' if n >= 8 else ('MEDIUM' if n >= MIN_QUARTERS else 'LOW')

            if n < MIN_QUARTERS:
                pred.prediction = 'INSUFFICIENT_DATA'
                pred.confidence = 0.0
                pred.key_factors = [f'Only {n} quarters of earnings history available on Yahoo Finance']
                _cache[cache_key] = (time.time(), pred)
                return pred

            # ── Run all 6 signals ────────────────────────
            s_beat, beat_rate, r_beat   = self._score_beat_rate(history)
            s_surp, avg_surp, r_surp    = self._score_surprise_trend(history)

            past_dates = [str(idx)[:10] for idx in history.index]
            s_drift, r_drift = self._score_price_drift(close, past_dates) \
                if close is not None and len(close) > 25 else (0.50, [])
            s_vol,   r_vol   = self._score_volume(close, volume) \
                if close is not None and volume is not None else (0.50, [])
            s_rev,   r_rev   = self._score_revenue(revenue)
            s_sent,  r_sent  = self._score_sentiment(ticker)

            pred.signal_beat_rate      = s_beat
            pred.signal_surprise_trend = s_surp
            pred.signal_price_drift    = s_drift
            pred.signal_volume         = s_vol
            pred.signal_revenue        = s_rev
            pred.signal_sentiment      = s_sent
            pred.historical_beat_rate  = beat_rate
            pred.avg_eps_surprise_pct  = avg_surp

            # ── Weighted composite ───────────────────────
            beat_prob = (
                s_beat  * SIGNAL_WEIGHTS['beat_rate']       +
                s_surp  * SIGNAL_WEIGHTS['surprise_trend']  +
                s_drift * SIGNAL_WEIGHTS['price_drift']      +
                s_vol   * SIGNAL_WEIGHTS['volume']           +
                s_rev   * SIGNAL_WEIGHTS['revenue']          +
                s_sent  * SIGNAL_WEIGHTS['sentiment']
            )
            pred.beat_probability = round(float(np.clip(beat_prob, 0, 1)), 4)

            # ── Data quality → confidence ────────────────
            quality_mult = {'HIGH': 1.0, 'MEDIUM': 0.75, 'LOW': 0.50}
            pred.confidence = round(quality_mult[pred.data_quality]
                                    * (0.5 + abs(beat_prob - 0.5)), 4)

            # ── Prediction label ─────────────────────────
            if beat_prob >= BEAT_THRESHOLD:
                pred.prediction = 'BEAT'
            elif beat_prob <= MISS_THRESHOLD:
                pred.prediction = 'MISS'
            else:
                pred.prediction = 'IN-LINE'

            # ── Key factors and risk factors ─────────────
            pred.key_factors = [r for r in r_beat + r_surp + r_drift
                                  + r_vol + r_rev + r_sent if r]

            # Construct risk factors
            risks = []
            if pred.days_to_earnings and pred.days_to_earnings < 7:
                risks.append("⚠️  Earnings in less than 7 days — price may already move on rumours")
            if beat_rate > 0.7 and s_drift > 0.7:
                risks.append("⚠️  High historical beat rate + strong pre-earnings run = beat likely priced in")
            if beat_prob > 0.65 and s_vol < 0.35:
                risks.append("⚠️  Bullish fundamental signals but volume shows distribution — caution")
            if n < 5:
                risks.append(f"⚠️  Limited history ({n} quarters) — prediction less reliable")
            if pred.data_quality == 'MEDIUM':
                risks.append("⚠️  Medium data quality — treat with caution")
            if not risks:
                risks.append("No major risk flags detected — maintain standard position sizing")
            pred.risk_factors = risks

        except Exception as e:
            pred.prediction  = 'INSUFFICIENT_DATA'
            pred.key_factors = [f'Error fetching data: {str(e)[:80]}']

        _cache[cache_key] = (time.time(), pred)
        return pred

    # ── Scan upcoming earnings across all stocks ──────────

    def scan_upcoming(self, days_ahead: int = 14) -> list[EarningsPrediction]:
        """
        Scans all 50 Nifty stocks and returns predictions for those
        with earnings in the next `days_ahead` days, sorted by
        days_to_earnings ascending.
        """
        from src.data_collector import STOCK_UNIVERSE
        upcoming = []
        print(f"\n⚙️  Scanning {len(STOCK_UNIVERSE)} stocks for earnings "
              f"in next {days_ahead} days...\n")

        for i, ticker in enumerate(STOCK_UNIVERSE):
            try:
                pred = self.predict(ticker, verbose=False)
                if (pred.days_to_earnings is not None
                        and 0 <= pred.days_to_earnings <= days_ahead):
                    upcoming.append(pred)
            except Exception:
                pass

            pct = (i + 1) / len(STOCK_UNIVERSE) * 100
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:5.1f}%  {ticker:<22}", end='\r')

        print(f"\n\n✅  Found {len(upcoming)} upcoming earnings\n")
        return sorted(upcoming, key=lambda p: (p.days_to_earnings or 999))

    # ── Helpers ───────────────────────────────────────────

    def _load_price_volume(self, ticker: str,
                            days: int = 120
                            ) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
        """DB first, yfinance fallback. Returns (close, volume) Series."""
        db_path = os.path.join('data', 'quantai.db')
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                df   = pd.read_sql_query(
                    "SELECT date, close, volume FROM prices "
                    "WHERE ticker=? ORDER BY date DESC LIMIT ?",
                    conn, params=(ticker, days + 5)
                )
                conn.close()
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').set_index('date')
                    df['close']  = pd.to_numeric(df['close'],  errors='coerce')
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                    df.dropna(subset=['close'], inplace=True)
                    if len(df) >= 20:
                        return df['close'], df['volume']
            except Exception:
                pass

        try:
            import yfinance as yf # type: ignore
            raw = yf.download(ticker, period='6mo', progress=False,
                              auto_adjust=True)
            if not raw.empty:
                if hasattr(raw.columns, 'levels'):
                    raw.columns = [c[0] for c in raw.columns]
                return raw['Close'].dropna(), raw['Volume'].dropna()
        except Exception:
            pass
        return None, None

    # ── Pretty print ──────────────────────────────────────

    def print_prediction(self, pred: EarningsPrediction):
        emoji = {'BEAT': '🟢', 'MISS': '🔴', 'IN-LINE': '🟡',
                 'INSUFFICIENT_DATA': '⚪'}.get(pred.prediction, '⚪')
        conf_bar = '█' * int(pred.confidence * 20)

        print(f"\n{'='*60}")
        print(f"  QuantAI Earnings Predictor — {pred.ticker}")
        print(f"  {pred.company_name}  |  {pred.sector}")
        print(f"{'='*60}")
        print(f"  Prediction       : {emoji} {pred.prediction}")
        print(f"  Beat Probability : {pred.beat_probability*100:.1f}%")
        print(f"  Confidence       : {pred.confidence*100:.1f}%  {conf_bar}")
        print(f"  Data Quality     : {pred.data_quality}  ({pred.n_quarters} quarters)")
        if pred.next_earnings_date:
            print(f"  Next Earnings    : {pred.next_earnings_date}  "
                  f"({pred.days_to_earnings} days away)")
        print(f"\n  Historical Record:")
        print(f"    Beat Rate      : {pred.historical_beat_rate*100:.1f}%")
        print(f"    Avg Surprise   : {pred.avg_eps_surprise_pct:+.1f}% vs consensus")
        print(f"\n  Signal Scores (0.5 = neutral):")
        for name, score in [
            ('Beat Rate History',  pred.signal_beat_rate),
            ('Surprise Trend',     pred.signal_surprise_trend),
            ('Pre-earnings Drift', pred.signal_price_drift),
            ('Volume Pattern',     pred.signal_volume),
            ('Revenue Trend',      pred.signal_revenue),
            ('News Sentiment',     pred.signal_sentiment),
        ]:
            bar = '█' * int(score * 20)
            side = '↑' if score > 0.55 else ('↓' if score < 0.45 else '→')
            print(f"    {name:<22} {score:.2f}  {side}  {bar}")
        if pred.key_factors:
            print(f"\n  Key Factors:")
            for f in pred.key_factors:
                print(f"    {f}")
        if pred.risk_factors:
            print(f"\n  Risk Factors:")
            for r in pred.risk_factors:
                print(f"    {r}")
        print(f"{'='*60}\n")