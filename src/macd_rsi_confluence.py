"""
src/macd_rsi_confluence.py

MACD + RSI Confluence Strategy with multi-timeframe (daily + weekly)
confirmation — a rule-based (non-ML) strategy that only trusts a daily
momentum-shift signal if the higher (weekly) timeframe trend agrees.

Core idea:
  Daily MACD/RSI crossovers are noisy — a stock can flash a bullish
  MACD cross on the daily chart while still being in a weekly
  downtrend, and that signal usually fails. Multi-timeframe analysis
  fixes this by requiring confluence ACROSS timeframes before acting:

    1. Daily MACD crossover    → MACD histogram flips from - to + (bull)
                                   or + to - (bear) — momentum just shifted
    2. Daily RSI confirms      → RSI_14 sitting in a "confirming" zone,
                                   not already overbought/oversold
    3. Weekly trend agrees     → the SAME direction trend exists one
                                   timeframe up (weekly close vs weekly
                                   SMA, weekly MACD histogram sign)

  Only when all three line up is this called a genuine signal — a
  daily-chart trigger fired with the weekly-chart "wind" behind it,
  rather than against it.

  Weekly indicators are computed by resampling the daily OHLCV with
  pandas (no dependency on src/features.py for this — entirely
  self-contained) and joined back onto the daily index with
  merge_asof(direction='backward'), so every daily row only ever sees
  the most recently COMPLETED weekly bar — no lookahead bias.

  Confidence is reported on the same 0–1 P(up) scale the ML ensemble
  and the other two strategies use (>0.5 bullish, <0.5 bearish, 0.5
  neutral), so this can later be wired into ensemble_model.py too.
"""
import pandas as pd
import numpy as np


class MACDRSIConfluenceStrategy:
    """
    Daily MACD-cross + RSI-zone signal, gated by weekly trend agreement.
    Works directly on the feature DataFrame produced by src/features.py
    (needs: Close, High, Low, MACD_Hist, RSI_14, ATR_14).
    """

    def __init__(self,
                 rsi_bull_min          = 45,    # daily RSI zone that confirms a BUY (momentum up, not yet overbought)
                 rsi_bull_max          = 70,
                 rsi_bear_min          = 30,    # daily RSI zone that confirms a SELL (momentum down, not yet oversold)
                 rsi_bear_max          = 55,
                 require_weekly_confirm = True,
                 weekly_sma_window     = 10,    # 10-week ≈ 50-day equivalent trend filter on the weekly chart
                 stop_loss_atr_mult    = 2.0,
                 trail_atr_mult        = 2.5,
                 max_holding_days      = 30,
                 min_confidence        = 0.60):

        self.rsi_bull_min           = rsi_bull_min
        self.rsi_bull_max           = rsi_bull_max
        self.rsi_bear_min           = rsi_bear_min
        self.rsi_bear_max           = rsi_bear_max
        self.require_weekly_confirm = require_weekly_confirm
        self.weekly_sma_window      = weekly_sma_window
        self.stop_loss_atr_mult     = stop_loss_atr_mult
        self.trail_atr_mult         = trail_atr_mult
        self.max_holding_days       = max_holding_days
        self.min_confidence         = min_confidence

    # ── Weekly timeframe construction ────────────────────────

    def _build_weekly_indicators(self, df):
        """
        Resamples daily OHLC into weekly (week ending Friday) bars and
        computes an independent MACD/RSI/trend reading on that higher
        timeframe.
        """
        weekly = df[['Close', 'High', 'Low']].resample('W-FRI').agg(
            {'Close': 'last', 'High': 'max', 'Low': 'min'}
        ).dropna()

        # Weekly RSI(14)
        delta = weekly['Close'].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        weekly['RSI_W'] = (100 - (100 / (1 + rs))).fillna(50)

        # Weekly MACD(12,26,9)
        ema12 = weekly['Close'].ewm(span=12, adjust=False).mean()
        ema26 = weekly['Close'].ewm(span=26, adjust=False).mean()
        macd_w = ema12 - ema26
        signal_w = macd_w.ewm(span=9, adjust=False).mean()
        weekly['MACD_Hist_W'] = macd_w - signal_w

        # Weekly trend filter
        weekly['SMA_W'] = weekly['Close'].rolling(self.weekly_sma_window, min_periods=4).mean()
        weekly['Trend_Up_W']   = (weekly['Close'] > weekly['SMA_W']) & (weekly['MACD_Hist_W'] > 0)
        weekly['Trend_Down_W'] = (weekly['Close'] < weekly['SMA_W']) & (weekly['MACD_Hist_W'] < 0)

        return weekly[['RSI_W', 'MACD_Hist_W', 'SMA_W', 'Trend_Up_W', 'Trend_Down_W']]

    def _attach_weekly(self, df):
        """
        Joins weekly indicators onto the daily index using an as-of
        backward merge — every daily date picks up the most recently
        COMPLETED weekly bar (never a still-forming or future one).
        """
        weekly = self._build_weekly_indicators(df)

        left  = df.reset_index().rename(columns={df.index.name or 'index': 'Date'})
        right = weekly.reset_index().rename(columns={weekly.index.name or 'index': 'Date'})

        merged = pd.merge_asof(
            left.sort_values('Date'), right.sort_values('Date'),
            on='Date', direction='backward'
        )
        merged = merged.set_index('Date')
        merged.index.name = df.index.name
        return merged

    # ── Indicator construction ───────────────────────────────

    def compute_indicators(self, df):
        df = self._attach_weekly(df.copy())

        macd_hist = df['MACD_Hist']
        df['MACD_Bull_Cross'] = (macd_hist.shift(1) <= 0) & (macd_hist > 0)
        df['MACD_Bear_Cross'] = (macd_hist.shift(1) >= 0) & (macd_hist < 0)

        # Normalize histogram magnitude against its own recent volatility —
        # MACD scale varies wildly between a ₹50 stock and a ₹3000 stock.
        hist_vol = macd_hist.rolling(60, min_periods=20).std().replace(0, np.nan)
        df['MACD_Strength'] = (macd_hist.abs() / hist_vol).clip(0, 3) / 3

        df[['RSI_W', 'MACD_Hist_W', 'Trend_Up_W', 'Trend_Down_W']] = \
            df[['RSI_W', 'MACD_Hist_W', 'Trend_Up_W', 'Trend_Down_W']].fillna(
                {'RSI_W': 50, 'MACD_Hist_W': 0, 'Trend_Up_W': False, 'Trend_Down_W': False}
            )
        df['MACD_Strength'] = df['MACD_Strength'].fillna(0)

        return df

    # ── Signal generation ────────────────────────────────────

    def generate_signals(self, df):
        df = self.compute_indicators(df)

        rsi = df['RSI_14']

        rsi_in_bull_zone = (rsi >= self.rsi_bull_min) & (rsi <= self.rsi_bull_max)
        rsi_in_bear_zone = (rsi >= self.rsi_bear_min) & (rsi <= self.rsi_bear_max)

        buy_trigger  = df['MACD_Bull_Cross'] & rsi_in_bull_zone
        sell_trigger = df['MACD_Bear_Cross'] & rsi_in_bear_zone

        if self.require_weekly_confirm:
            buy_trigger  = buy_trigger  & df['Trend_Up_W']
            sell_trigger = sell_trigger & df['Trend_Down_W']

        # ── Confidence composite ──
        bull_mid   = (self.rsi_bull_min + self.rsi_bull_max) / 2
        bull_half  = (self.rsi_bull_max - self.rsi_bull_min) / 2
        rsi_score_buy = (1 - (rsi - bull_mid).abs() / bull_half).clip(0, 1)

        bear_mid   = (self.rsi_bear_min + self.rsi_bear_max) / 2
        bear_half  = (self.rsi_bear_max - self.rsi_bear_min) / 2
        rsi_score_sell = (1 - (rsi - bear_mid).abs() / bear_half).clip(0, 1)

        weekly_bonus_buy  = df['Trend_Up_W'].astype(float)   * 0.15
        weekly_bonus_sell = df['Trend_Down_W'].astype(float) * 0.15

        composite_buy  = (df['MACD_Strength'] * 0.45 + rsi_score_buy  * 0.40 + weekly_bonus_buy).clip(0, 1)
        composite_sell = (df['MACD_Strength'] * 0.45 + rsi_score_sell * 0.40 + weekly_bonus_sell).clip(0, 1)

        confidence = pd.Series(0.5, index=df.index)
        confidence = confidence.where(~buy_trigger,  0.5 + composite_buy  * 0.42)
        confidence = confidence.where(~sell_trigger, 0.5 - composite_sell * 0.42)

        signal = pd.Series('HOLD', index=df.index)
        signal = signal.where(~buy_trigger,  'BUY')
        signal = signal.where(~sell_trigger, 'SELL')

        df['MRC_Signal']     = signal
        df['MRC_Confidence'] = confidence

        return df

    # ── Latest snapshot for live signals / API use ───────────

    def get_latest_signal(self, df):
        scored = self.generate_signals(df)
        latest = scored.iloc[-1]

        signal     = latest['MRC_Signal']
        confidence = float(latest['MRC_Confidence'])
        price      = float(latest['Close'])
        atr        = float(latest['ATR_14'])

        if signal == 'BUY':
            stop_loss = price - self.stop_loss_atr_mult * atr
            reason = (f"🟢 BUY — bullish MACD cross + RSI {latest['RSI_14']:.1f} confirming"
                      f"{', weekly trend agrees (above weekly SMA, weekly MACD bullish)' if latest['Trend_Up_W'] else ''}")
        elif signal == 'SELL':
            stop_loss = price + self.stop_loss_atr_mult * atr
            reason = (f"🔴 SELL — bearish MACD cross + RSI {latest['RSI_14']:.1f} confirming"
                      f"{', weekly trend agrees (below weekly SMA, weekly MACD bearish)' if latest['Trend_Down_W'] else ''}")
        else:
            stop_loss = None
            weekly_state = 'bullish' if latest['Trend_Up_W'] else ('bearish' if latest['Trend_Down_W'] else 'neutral')
            reason = (f"⚪ HOLD — no fresh MACD cross with RSI/weekly confluence "
                      f"(RSI {latest['RSI_14']:.1f}, weekly trend {weekly_state})")

        tradeable = (signal == 'BUY' and confidence >= self.min_confidence) or \
                    (signal == 'SELL' and (1 - confidence) >= self.min_confidence)

        return {
            'strategy'          : 'MACD_RSI_Confluence',
            'signal'            : signal,
            'confidence'        : round(confidence, 4),
            'tradeable'         : bool(tradeable),
            'price'             : round(price, 2),
            'stop_loss'         : round(stop_loss, 2) if stop_loss else None,
            'rsi_daily'         : round(float(latest['RSI_14']), 2),
            'rsi_weekly'        : round(float(latest['RSI_W']), 2),
            'macd_hist_daily'   : round(float(latest['MACD_Hist']), 4),
            'macd_hist_weekly'  : round(float(latest['MACD_Hist_W']), 4),
            'weekly_trend_up'   : bool(latest['Trend_Up_W']),
            'weekly_trend_down' : bool(latest['Trend_Down_W']),
            'reason'            : reason,
        }