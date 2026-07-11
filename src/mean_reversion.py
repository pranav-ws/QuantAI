"""
src/mean_reversion.py

Mean Reversion Strategy — a rule-based (non-ML) trading strategy that
bets on price snapping back to its average after an extreme stretch.

Core idea:
  Stocks oscillate around a "fair value" (their moving average). When
  price gets pulled too far away — usually on a panic sell-off or a
  euphoric spike — gravity tends to pull it back. We look for THREE
  things lining up together before calling it an oversold bounce setup:

    1. Price is statistically extreme    → Z-score vs 20-day mean
    2. Momentum is exhausted              → RSI_14 in oversold territory
    3. Price has pierced the lower band   → Bollinger %B near/below 0

  A "squeeze" (Bollinger Band Width sitting in its lowest percentile
  over the recent past) is treated as a bonus — low-volatility regimes
  that suddenly snap tend to move further once they release.

  Unlike the ML models in this project (which output P(next day UP)),
  this strategy outputs a confidence on the SAME 0–1 scale, so it can
  later be dropped into ensemble_model.py as another voting member:
      confidence > 0.5  → bullish reversion bias (oversold bounce)
      confidence < 0.5  → bearish reversion bias (overbought pullback)
      confidence = 0.5  → no edge / neutral

  This is intentionally NOT a directional momentum strategy — it
  trades AGAINST the recent move, so it complements the trend-following
  ML ensemble rather than duplicating it.
"""
import pandas as pd
import numpy as np


class MeanReversionStrategy:
    """
    Rule-based mean reversion signal generator.
    Works directly on the feature DataFrame produced by src/features.py
    (needs: Close, SMA_20, BB_Upper, BB_Middle, BB_Lower, BB_Width,
    RSI_14, ATR_14).
    """

    def __init__(self,
                 rsi_oversold        = 30,    # RSI below this = oversold
                 rsi_overbought      = 70,    # RSI above this = overbought
                 zscore_entry        = 1.5,   # |z| beyond this = statistically stretched
                 zscore_exit         = 0.3,   # |z| within this = reverted to mean, take profit
                 squeeze_lookback    = 100,    # window to rank BB_Width history
                 squeeze_percentile  = 20,    # bottom 20% width = "squeeze" regime
                 stop_loss_atr_mult  = 1.5,   # stop = entry - 1.5x ATR
                 max_holding_days    = 10,    # time-stop if reversion never happens
                 min_confidence      = 0.62): # confidence floor to actually trade

        self.rsi_oversold       = rsi_oversold
        self.rsi_overbought     = rsi_overbought
        self.zscore_entry       = zscore_entry
        self.zscore_exit        = zscore_exit
        self.squeeze_lookback   = squeeze_lookback
        self.squeeze_percentile = squeeze_percentile
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.max_holding_days   = max_holding_days
        self.min_confidence     = min_confidence

    # ── Indicator construction ───────────────────────────────

    def compute_indicators(self, df):
        """
        Adds mean-reversion-specific columns to a copy of the feature
        DataFrame. Reuses Bollinger Band / RSI / ATR columns that
        src/features.py already computed — no recalculation needed.
        """
        df = df.copy()

        # Band standard deviation, recovered from the existing bands
        # (BB_Upper - BB_Middle) = 2 * rolling_std, since window_dev=2
        band_std = (df['BB_Upper'] - df['BB_Middle']) / 2.0
        band_std = band_std.replace(0, np.nan)

        # Z-score: how many std devs is price away from its 20-day mean?
        df['Zscore_20'] = (df['Close'] - df['BB_Middle']) / band_std

        # %B: 0 = sitting on the lower band, 1 = sitting on the upper band
        band_range = (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)
        df['BB_PctB'] = (df['Close'] - df['BB_Lower']) / band_range

        # Squeeze: is current volatility unusually compressed vs recent history?
        df['BB_Width_Percentile'] = (
            df['BB_Width']
            .rolling(self.squeeze_lookback, min_periods=20)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        )
        df['Squeeze'] = df['BB_Width_Percentile'] <= (self.squeeze_percentile / 100)

        df['Zscore_20'].fillna(0, inplace=True)
        df['BB_PctB'].fillna(0.5, inplace=True)

        return df

    # ── Signal generation ────────────────────────────────────

    def generate_signals(self, df):
        """
        Vectorized signal pass over the whole DataFrame.
        Adds: MR_Signal ('BUY' / 'SELL' / 'HOLD'), MR_Confidence (0-1),
        Reverted (bool — price has snapped back near the mean, used as
        the take-profit trigger in the backtester).
        """
        df = self.compute_indicators(df)

        rsi   = df['RSI_14']
        z     = df['Zscore_20']
        pctb  = df['BB_PctB']
        sqz   = df['Squeeze'].astype(float)

        # ── Component scores (each scaled 0→1, "1" = maximum stretch) ──
        rsi_score_buy  = ((self.rsi_oversold - rsi) / self.rsi_oversold).clip(0, 1)
        rsi_score_sell = ((rsi - self.rsi_overbought) / (100 - self.rsi_overbought)).clip(0, 1)

        band_score_buy  = ((0.05 - pctb) / 0.05).clip(0, 1)
        band_score_sell = ((pctb - 0.95) / 0.05).clip(0, 1)

        z_score_component = (z.abs() / 3.0).clip(0, 1)

        squeeze_bonus = sqz * 0.10

        composite_buy  = (rsi_score_buy  * 0.35 + band_score_buy  * 0.30 +
                          z_score_component * 0.25 + squeeze_bonus)
        composite_sell = (rsi_score_sell * 0.35 + band_score_sell * 0.30 +
                          z_score_component * 0.25 + squeeze_bonus)
        composite_buy  = composite_buy.clip(0, 1)
        composite_sell = composite_sell.clip(0, 1)

        # Trigger conditions — all three legs must line up
        buy_trigger  = (pctb <= 0.05) & (rsi <= self.rsi_oversold)   & (z <= -self.zscore_entry)
        sell_trigger = (pctb >= 0.95) & (rsi >= self.rsi_overbought) & (z >=  self.zscore_entry)

        # Map composite stretch score onto the same 0-1 confidence scale
        # the ML ensemble uses (0.5 = neutral, →1 = bullish, →0 = bearish)
        confidence = pd.Series(0.5, index=df.index)
        confidence = confidence.where(~buy_trigger,  0.5 + composite_buy  * 0.42)
        confidence = confidence.where(~sell_trigger, 0.5 - composite_sell * 0.42)

        signal = pd.Series('HOLD', index=df.index)
        signal = signal.where(~buy_trigger,  'BUY')
        signal = signal.where(~sell_trigger, 'SELL')

        df['MR_Signal']     = signal
        df['MR_Confidence'] = confidence
        df['Reverted']      = z.abs() <= self.zscore_exit

        return df

    # ── Latest snapshot for live signals / API use ───────────

    def get_latest_signal(self, df):
        """
        Returns a dict describing the most recent bar's signal —
        same shape philosophy as get_ensemble_confidence() so it can
        slot into the same dashboards/alerts.
        """
        scored = self.generate_signals(df)
        latest = scored.iloc[-1]

        signal     = latest['MR_Signal']
        confidence = float(latest['MR_Confidence'])
        price      = float(latest['Close'])
        atr        = float(latest['ATR_14'])

        if signal == 'BUY':
            stop_loss   = price - self.stop_loss_atr_mult * atr
            target      = float(latest['BB_Middle'])  # take-profit = reversion to the mean
            reason = (f"🟢 BUY — oversold bounce setup: RSI {latest['RSI_14']:.1f}, "
                      f"Z-score {latest['Zscore_20']:+.2f}, "
                      f"{abs(latest['BB_PctB'])*100:.0f}% through lower band"
                      f"{' · squeeze regime (low vol, primed to move)' if latest['Squeeze'] else ''}")
        elif signal == 'SELL':
            stop_loss   = price + self.stop_loss_atr_mult * atr
            target      = float(latest['BB_Middle'])
            reason = (f"🔴 SELL — overbought pullback setup: RSI {latest['RSI_14']:.1f}, "
                      f"Z-score {latest['Zscore_20']:+.2f}, "
                      f"price stretched above upper band"
                      f"{' · squeeze regime' if latest['Squeeze'] else ''}")
        else:
            stop_loss = target = None
            reason = (f"⚪ HOLD — no statistical extreme: RSI {latest['RSI_14']:.1f}, "
                      f"Z-score {latest['Zscore_20']:+.2f} (need ±{self.zscore_entry})")

        tradeable = signal != 'HOLD' and confidence >= self.min_confidence \
                    if signal == 'BUY' else \
                    signal != 'HOLD' and (1 - confidence) >= self.min_confidence

        return {
            'strategy'    : 'MeanReversion',
            'signal'      : signal,
            'confidence'  : round(confidence, 4),
            'tradeable'   : bool(tradeable),
            'price'       : round(price, 2),
            'stop_loss'   : round(stop_loss, 2) if stop_loss else None,
            'target'      : round(target, 2) if target else None,
            'rsi'         : round(float(latest['RSI_14']), 2),
            'zscore'      : round(float(latest['Zscore_20']), 2),
            'bb_pctb'     : round(float(latest['BB_PctB']), 3),
            'squeeze'     : bool(latest['Squeeze']) if pd.notna(latest['Squeeze']) else False,
            'reason'      : reason,
        }