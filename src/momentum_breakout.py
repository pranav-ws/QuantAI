"""
src/momentum_breakout.py

Momentum Breakout Strategy — a rule-based (non-ML) trading strategy
that bets price CONTINUES once it breaks out to a new 52-week high
on unusually heavy volume.

Core idea (the mirror opposite of src/mean_reversion.py):
  Mean reversion bets price snaps BACK to its average after an
  extreme move. Momentum breakout bets the opposite — that a stock
  punching through a year's worth of resistance, on volume that
  proves big players are involved, tends to keep going. This is the
  classic Darvas-box / Turtle-trading style of trend entry.

  We look for FOUR things lining up together before calling it a
  breakout:

    1. New 52-week high          → Close > highest High of the prior year
    2. Volume confirms it        → Volume_Ratio well above the 20-day average
    3. Trend already pointed up  → Close > SMA_50 and MACD_Hist > 0
    4. Momentum already building → positive 20-day return going in

  Same idea also runs in reverse for breakdowns (new 52-week LOW on
  heavy volume) — useful as a bearish/exit signal, mirroring how
  mean_reversion.py treats overbought conditions as a SELL signal.

  Confidence is reported on the SAME 0–1 P(up) scale the ML ensemble
  uses (confidence > 0.5 = bullish, < 0.5 = bearish, = 0.5 = neutral),
  so this can later be wired into ensemble_model.py as another vote.

  This is intentionally a TREND-FOLLOWING strategy — it trades WITH
  the recent move, which is why it pairs well alongside the mean
  reversion strategy rather than overlapping it: mean reversion fades
  exhaustion, momentum breakout rides confirmed strength.
"""
import pandas as pd# type: ignore
import numpy as np# type: ignore


class MomentumBreakoutStrategy:
    """
    Rule-based 52-week breakout + volume surge signal generator.
    Works directly on the feature DataFrame produced by src/features.py
    (needs: Close, High, Low, SMA_50, MACD_Hist, Volume_Ratio,
    Return_20d, ATR_14).
    """

    def __init__(self,
                 breakout_window       = 252,   # ~52 weeks of trading days
                 breakout_buffer       = 0.0,   # require close this far ABOVE prior high (0.02 = 2%) for a stricter, less whipsaw-prone signal
                 volume_surge_mult     = 1.5,   # Volume_Ratio must be at least this multiple of the 20-day average
                 require_trend_confirm = True,  # require Close > SMA_50 and MACD_Hist > 0
                 min_return_20d        = 0.0,   # require at least this much 20-day momentum already underway
                 stop_loss_atr_mult    = 2.0,   # initial hard stop = entry - 2x ATR
                 trail_atr_mult        = 3.0,   # chandelier trailing stop = highest close since entry - 3x ATR
                 max_holding_days      = 60,    # safety-net time stop if trend stalls sideways
                 min_confidence        = 0.62): # confidence floor to actually trade

        self.breakout_window       = breakout_window
        self.breakout_buffer       = breakout_buffer
        self.volume_surge_mult     = volume_surge_mult
        self.require_trend_confirm = require_trend_confirm
        self.min_return_20d        = min_return_20d
        self.stop_loss_atr_mult    = stop_loss_atr_mult
        self.trail_atr_mult        = trail_atr_mult
        self.max_holding_days      = max_holding_days
        self.min_confidence        = min_confidence

    # ── Indicator construction ───────────────────────────────

    def compute_indicators(self, df):
        """
        Adds breakout-specific columns to a copy of the feature
        DataFrame. The rolling high/low are SHIFTED by one day so the
        breakout level is always "the prior year's extreme, not
        including today" — avoids any lookahead bias.
        """
        df = df.copy()

        df['Rolling_High_252'] = (
            df['High'].rolling(self.breakout_window, min_periods=100).max().shift(1)
        )
        df['Rolling_Low_252'] = (
            df['Low'].rolling(self.breakout_window, min_periods=100).min().shift(1)
        )

        df['Breakout_Pct']  = (df['Close'] - df['Rolling_High_252']) / df['Rolling_High_252'] * 100
        df['Breakdown_Pct'] = (df['Rolling_Low_252'] - df['Close']) / df['Rolling_Low_252'] * 100

        df['Trend_Up_OK']   = (df['Close'] > df['SMA_50']) & (df['MACD_Hist'] > 0)
        df['Trend_Down_OK'] = (df['Close'] < df['SMA_50']) & (df['MACD_Hist'] < 0)

        df['Volume_Surge'] = df['Volume_Ratio'] >= self.volume_surge_mult

        return df

    # ── Signal generation ────────────────────────────────────

    def generate_signals(self, df):
        """
        Vectorized signal pass over the whole DataFrame.
        Adds: BO_Signal ('BUY' / 'SELL' / 'HOLD'), BO_Confidence (0-1).
        BUY  = confirmed 52-week-high breakout with volume + trend behind it.
        SELL = confirmed 52-week-low breakdown with volume + trend behind it
               (informational / exit signal — mirrors mean_reversion's SELL).
        """
        df = self.compute_indicators(df)

        roll_high = df['Rolling_High_252']
        roll_low  = df['Rolling_Low_252']
        vol_ratio = df['Volume_Ratio']
        ret20     = df['Return_20d']

        # ── Component scores (0→1, "1" = maximum strength of signal) ──
        breakout_strength  = (df['Breakout_Pct']  / 5.0).clip(0, 1)   # full score at 5%+ above prior high
        breakdown_strength = (df['Breakdown_Pct'] / 5.0).clip(0, 1)

        volume_score = ((vol_ratio - 1.0) / 2.0).clip(0, 1)            # 1x ratio = 0, 3x ratio = 1

        momentum_score_up   = (ret20 / 0.15).clip(0, 1)                 # full score at +15% in 20 days
        momentum_score_down = (-ret20 / 0.15).clip(0, 1)

        trend_bonus_up   = df['Trend_Up_OK'].astype(float)   * 0.15
        trend_bonus_down = df['Trend_Down_OK'].astype(float) * 0.15

        composite_buy  = (breakout_strength  * 0.35 + volume_score * 0.30 +
                          momentum_score_up   * 0.20 + trend_bonus_up)
        composite_sell = (breakdown_strength * 0.35 + volume_score * 0.30 +
                          momentum_score_down * 0.20 + trend_bonus_down)
        composite_buy  = composite_buy.clip(0, 1)
        composite_sell = composite_sell.clip(0, 1)

        # ── Trigger conditions — breakout/breakdown + volume + (optional) trend + momentum ──
        breakout_level  = roll_high * (1 + self.breakout_buffer)
        breakdown_level = roll_low  * (1 - self.breakout_buffer)

        buy_trigger  = (df['Close'] > breakout_level)  & (vol_ratio >= self.volume_surge_mult) & \
                       (ret20 >= self.min_return_20d)
        sell_trigger = (df['Close'] < breakdown_level) & (vol_ratio >= self.volume_surge_mult) & \
                       (ret20 <= -self.min_return_20d)

        if self.require_trend_confirm:
            buy_trigger  = buy_trigger  & df['Trend_Up_OK']
            sell_trigger = sell_trigger & df['Trend_Down_OK']

        confidence = pd.Series(0.5, index=df.index)
        confidence = confidence.where(~buy_trigger,  0.5 + composite_buy  * 0.45)
        confidence = confidence.where(~sell_trigger, 0.5 - composite_sell * 0.45)

        signal = pd.Series('HOLD', index=df.index)
        signal = signal.where(~buy_trigger,  'BUY')
        signal = signal.where(~sell_trigger, 'SELL')

        df['BO_Signal']     = signal
        df['BO_Confidence'] = confidence

        return df

    # ── Latest snapshot for live signals / API use ───────────

    def get_latest_signal(self, df):
        """
        Returns a dict describing the most recent bar's signal —
        same shape philosophy as mean_reversion.get_latest_signal()
        and ensemble_model.get_ensemble_confidence() so it can slot
        into the same dashboards/alerts.
        """
        scored = self.generate_signals(df)
        latest = scored.iloc[-1]

        signal     = latest['BO_Signal']
        confidence = float(latest['BO_Confidence'])
        price      = float(latest['Close'])
        atr        = float(latest['ATR_14'])

        if signal == 'BUY':
            stop_loss = price - self.stop_loss_atr_mult * atr
            reason = (f"🟢 BUY — 52-week high breakout: {latest['Breakout_Pct']:+.1f}% above "
                      f"prior 52w high, volume {latest['Volume_Ratio']:.1f}x average"
                      f"{', trend confirmed (above SMA50, MACD bullish)' if latest['Trend_Up_OK'] else ''}")
        elif signal == 'SELL':
            stop_loss = price + self.stop_loss_atr_mult * atr
            reason = (f"🔴 SELL — 52-week low breakdown: {latest['Breakdown_Pct']:+.1f}% below "
                      f"prior 52w low, volume {latest['Volume_Ratio']:.1f}x average"
                      f"{', downtrend confirmed' if latest['Trend_Down_OK'] else ''}")
        else:
            stop_loss = None
            dist_to_high = latest['Rolling_High_252'] - price if pd.notna(latest['Rolling_High_252']) else None
            reason = (f"⚪ HOLD — no confirmed breakout: "
                      f"{f'₹{dist_to_high:.1f} below 52w high, ' if dist_to_high is not None else ''}"
                      f"volume {latest['Volume_Ratio']:.1f}x average (need {self.volume_surge_mult}x+)")

        tradeable = (signal == 'BUY' and confidence >= self.min_confidence) or \
                    (signal == 'SELL' and (1 - confidence) >= self.min_confidence)

        return {
            'strategy'        : 'MomentumBreakout',
            'signal'          : signal,
            'confidence'      : round(confidence, 4),
            'tradeable'       : bool(tradeable),
            'price'           : round(price, 2),
            'stop_loss'       : round(stop_loss, 2) if stop_loss else None,
            'breakout_pct'    : round(float(latest['Breakout_Pct']), 2) if pd.notna(latest['Breakout_Pct']) else None,
            'breakdown_pct'   : round(float(latest['Breakdown_Pct']), 2) if pd.notna(latest['Breakdown_Pct']) else None,
            'volume_ratio'    : round(float(latest['Volume_Ratio']), 2),
            'volume_surge'    : bool(latest['Volume_Surge']),
            'trend_confirmed' : bool(latest['Trend_Up_OK']) if signal == 'BUY' else bool(latest['Trend_Down_OK']),
            'return_20d'      : round(float(latest['Return_20d']) * 100, 2),
            'reason'          : reason,
        }