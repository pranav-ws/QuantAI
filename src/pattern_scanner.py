"""
src/pattern_scanner.py

Detects 8 major chart patterns on daily OHLCV data.
No external libraries needed — pure numpy/pandas math.

Patterns detected:
  1. Head & Shoulders (bearish reversal)
  2. Inverse Head & Shoulders (bullish reversal)
  3. Double Top (bearish reversal)
  4. Double Bottom (bullish reversal)
  5. Cup & Handle (bullish continuation)
  6. Golden / Death Cross (trend change)
  7. Bull / Bear Flag (continuation)
  8. Bullish / Bearish Engulfing (1-day reversal candle)

Each pattern returns:
  - detected   : bool
  - confidence : 0.0 to 1.0
  - direction  : 'BULLISH' or 'BEARISH'
  - description: human-readable explanation
"""

import numpy as np# type: ignore
import pandas as pd# type: ignore
import sqlite3
import os

DB_PATH = os.path.join('data', 'quantai.db')

# ── Load OHLCV data ───────────────────────────────────────
def load_ohlcv(ticker, lookback_days=120):
    """Loads OHLCV data from database."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        """SELECT date, open, high, low, close, volume
           FROM prices WHERE ticker=?
           ORDER BY date DESC LIMIT ?""",
        conn, params=(ticker, lookback_days)
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df.sort_index()  # oldest first

# ── Helper: find local peaks and troughs ──────────────────
def find_peaks(series, window=5):
    """Returns indices of local maxima."""
    peaks = []
    for i in range(window, len(series) - window):
        if series.iloc[i] == series.iloc[i-window:i+window+1].max():
            peaks.append(i)
    return peaks

def find_troughs(series, window=5):
    """Returns indices of local minima."""
    troughs = []
    for i in range(window, len(series) - window):
        if series.iloc[i] == series.iloc[i-window:i+window+1].min():
            troughs.append(i)
    return troughs

# ── Pattern 1: Head & Shoulders ───────────────────────────
def detect_head_and_shoulders(df, tolerance=0.03):
    """
    Classic bearish reversal pattern.
    3 peaks: left shoulder < head > right shoulder
    Shoulders roughly equal height.
    """
    closes = df['Close']
    peaks  = find_peaks(closes, window=8)

    if len(peaks) < 3:
        return {'detected': False}

    # Check last 3 peaks
    for i in range(len(peaks) - 2):
        l = peaks[i]
        h = peaks[i+1]
        r = peaks[i+2]

        left  = float(closes.iloc[l])
        head  = float(closes.iloc[h])
        right = float(closes.iloc[r])

        # Head must be higher than both shoulders
        if not (head > left and head > right):
            continue

        # Shoulders must be roughly equal (within tolerance)
        shoulder_diff = abs(left - right) / left
        if shoulder_diff > tolerance:
            continue

        # Head must be significantly higher than shoulders
        head_prominence = (head - max(left, right)) / head
        if head_prominence < 0.02:
            continue

        confidence = min(1.0, (1 - shoulder_diff/tolerance) * 0.8 +
                         head_prominence * 2)

        return {
            'detected'    : True,
            'pattern'     : 'Head & Shoulders',
            'direction'   : 'BEARISH',
            'confidence'  : round(confidence, 3),
            'description' : f"Bearish reversal — Head at ₹{head:.1f}, "
                           f"shoulders at ₹{left:.1f}/₹{right:.1f}",
            'emoji'       : '🔴'
        }

    return {'detected': False}

# ── Pattern 2: Inverse Head & Shoulders ───────────────────
def detect_inverse_head_and_shoulders(df, tolerance=0.03):
    """
    Bullish reversal — 3 troughs: L > Head < R (head is lowest)
    """
    closes   = df['Close']
    troughs  = find_troughs(closes, window=8)

    if len(troughs) < 3:
        return {'detected': False}

    for i in range(len(troughs) - 2):
        l = troughs[i]
        h = troughs[i+1]
        r = troughs[i+2]

        left  = float(closes.iloc[l])
        head  = float(closes.iloc[h])
        right = float(closes.iloc[r])

        # Head must be LOWER than both shoulders
        if not (head < left and head < right):
            continue

        shoulder_diff = abs(left - right) / left
        if shoulder_diff > tolerance:
            continue

        head_depth = (min(left, right) - head) / min(left, right)
        if head_depth < 0.02:
            continue

        confidence = min(1.0, (1 - shoulder_diff/tolerance) * 0.8 +
                         head_depth * 2)

        return {
            'detected'    : True,
            'pattern'     : 'Inverse Head & Shoulders',
            'direction'   : 'BULLISH',
            'confidence'  : round(confidence, 3),
            'description' : f"Bullish reversal — Head at ₹{head:.1f}, "
                           f"shoulders at ₹{left:.1f}/₹{right:.1f}",
            'emoji'       : '🟢'
        }

    return {'detected': False}

# ── Pattern 3: Double Top ─────────────────────────────────
def detect_double_top(df, tolerance=0.02):
    """
    Bearish reversal: two peaks at roughly the same price level.
    """
    closes = df['Close']
    peaks  = find_peaks(closes, window=8)

    if len(peaks) < 2:
        return {'detected': False}

    p1_idx = peaks[-2]
    p2_idx = peaks[-1]

    p1 = float(closes.iloc[p1_idx])
    p2 = float(closes.iloc[p2_idx])

    price_diff = abs(p1 - p2) / p1

    if price_diff > tolerance:
        return {'detected': False}

    # Make sure peaks are separated enough (at least 10 bars)
    if (p2_idx - p1_idx) < 10:
        return {'detected': False}

    # Current price should be below the peaks
    current = float(closes.iloc[-1])
    if current >= p1 * 0.99:
        return {'detected': False}

    confidence = round(1 - price_diff / tolerance, 3)

    return {
        'detected'    : True,
        'pattern'     : 'Double Top',
        'direction'   : 'BEARISH',
        'confidence'  : confidence,
        'description' : f"Bearish reversal — Two peaks at "
                       f"₹{p1:.1f} & ₹{p2:.1f}",
        'emoji'       : '🔴'
    }

# ── Pattern 4: Double Bottom ──────────────────────────────
def detect_double_bottom(df, tolerance=0.02):
    """
    Bullish reversal: two troughs at roughly the same price level.
    """
    closes  = df['Close']
    troughs = find_troughs(closes, window=8)

    if len(troughs) < 2:
        return {'detected': False}

    t1_idx = troughs[-2]
    t2_idx = troughs[-1]

    t1 = float(closes.iloc[t1_idx])
    t2 = float(closes.iloc[t2_idx])

    price_diff = abs(t1 - t2) / t1

    if price_diff > tolerance:
        return {'detected': False}

    if (t2_idx - t1_idx) < 10:
        return {'detected': False}

    current = float(closes.iloc[-1])
    if current <= t1 * 1.01:
        return {'detected': False}

    confidence = round(1 - price_diff / tolerance, 3)

    return {
        'detected'    : True,
        'pattern'     : 'Double Bottom',
        'direction'   : 'BULLISH',
        'confidence'  : confidence,
        'description' : f"Bullish reversal — Two troughs at "
                       f"₹{t1:.1f} & ₹{t2:.1f}",
        'emoji'       : '🟢'
    }

# ── Pattern 5: Cup & Handle ───────────────────────────────
def detect_cup_and_handle(df):
    """
    Bullish continuation: U-shaped base followed by a small pullback.
    One of the most reliable bullish patterns.
    """
    if len(df) < 60:
        return {'detected': False}

    closes = df['Close']
    recent = closes.iloc[-60:]

    # Left rim: high in first 15 bars
    left_rim  = float(recent.iloc[:15].max())
    # Cup bottom: low in middle 30 bars
    cup_bot   = float(recent.iloc[15:45].min())
    # Right rim: high in last 15 bars
    right_rim = float(recent.iloc[45:].max())

    # Cup depth must be meaningful
    cup_depth = (left_rim - cup_bot) / left_rim
    if cup_depth < 0.08 or cup_depth > 0.50:
        return {'detected': False}

    # Right rim must reach back near left rim
    rim_recovery = (right_rim - cup_bot) / (left_rim - cup_bot)
    if rim_recovery < 0.85:
        return {'detected': False}

    # Handle: small pullback in last 5 bars
    handle = closes.iloc[-5:]
    handle_pullback = (right_rim - float(handle.min())) / right_rim
    if handle_pullback > 0.05:
        return {'detected': False}

    current    = float(closes.iloc[-1])
    confidence = round(rim_recovery * 0.7 + (1 - handle_pullback*10) * 0.3, 3)

    return {
        'detected'    : True,
        'pattern'     : 'Cup & Handle',
        'direction'   : 'BULLISH',
        'confidence'  : min(confidence, 1.0),
        'description' : f"Bullish continuation — Cup depth "
                       f"{cup_depth:.1%}, current ₹{current:.1f}",
        'emoji'       : '🟢'
    }

# ── Pattern 6: Golden / Death Cross ───────────────────────
def detect_cross(df):
    """
    Golden Cross: 50-day MA crosses above 200-day MA → BULLISH
    Death Cross : 50-day MA crosses below 200-day MA → BEARISH
    """
    closes = df['Close']
    if len(closes) < 205:
        return {'detected': False}

    ma50  = closes.rolling(50).mean()
    ma200 = closes.rolling(200).mean()

    # Check if crossover happened in last 5 days
    for i in range(-5, 0):
        prev_diff = float(ma50.iloc[i-1]) - float(ma200.iloc[i-1])
        curr_diff = float(ma50.iloc[i])   - float(ma200.iloc[i])

        if prev_diff < 0 and curr_diff > 0:
            gap = abs(curr_diff) / float(closes.iloc[-1]) * 100
            return {
                'detected'    : True,
                'pattern'     : 'Golden Cross',
                'direction'   : 'BULLISH',
                'confidence'  : min(0.5 + gap * 2, 0.95),
                'description' : f"MA50 crossed above MA200 — "
                               f"strong uptrend signal",
                'emoji'       : '✨'
            }

        if prev_diff > 0 and curr_diff < 0:
            gap = abs(curr_diff) / float(closes.iloc[-1]) * 100
            return {
                'detected'    : True,
                'pattern'     : 'Death Cross',
                'direction'   : 'BEARISH',
                'confidence'  : min(0.5 + gap * 2, 0.95),
                'description' : f"MA50 crossed below MA200 — "
                               f"strong downtrend signal",
                'emoji'       : '💀'
            }

    return {'detected': False}

# ── Pattern 7: Bull / Bear Flag ───────────────────────────
def detect_flag(df):
    """
    Bull Flag: strong upward move (pole) followed by consolidation.
    Bear Flag: strong downward move followed by consolidation.
    """
    closes = df['Close']
    if len(closes) < 30:
        return {'detected': False}

    # Pole: big move in days -30 to -10
    pole_start = float(closes.iloc[-30])
    pole_end   = float(closes.iloc[-10])
    pole_move  = (pole_end - pole_start) / pole_start

    # Flag: tight consolidation in last 10 days
    flag_range = (float(closes.iloc[-10:].max()) -
                  float(closes.iloc[-10:].min()))
    flag_pct   = flag_range / float(closes.iloc[-10])

    if abs(pole_move) < 0.06:    # pole must be >6%
        return {'detected': False}
    if flag_pct > 0.04:          # flag must be tight (<4%)
        return {'detected': False}

    direction  = 'BULLISH' if pole_move > 0 else 'BEARISH'
    confidence = round(min(abs(pole_move) * 3, 0.9), 3)

    return {
        'detected'    : True,
        'pattern'     : 'Bull Flag' if direction == 'BULLISH'
                         else 'Bear Flag',
        'direction'   : direction,
        'confidence'  : confidence,
        'description' : f"{'Bull' if direction == 'BULLISH' else 'Bear'} "
                       f"flag — pole move {pole_move:.1%}, "
                       f"tight consolidation {flag_pct:.1%}",
        'emoji'       : '🟢' if direction == 'BULLISH' else '🔴'
    }

# ── Pattern 8: Engulfing Candle ───────────────────────────
def detect_engulfing(df):
    """
    Bullish Engulfing: big green candle fully covers previous red candle.
    Bearish Engulfing: big red candle fully covers previous green candle.
    Strong 1-2 day reversal signal.
    """
    if len(df) < 3:
        return {'detected': False}

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_body = float(prev['Close']) - float(prev['Open'])
    curr_body = float(curr['Close']) - float(curr['Open'])

    # Bullish engulfing
    if (prev_body < 0 and curr_body > 0 and
        float(curr['Open'])  < float(prev['Close']) and
        float(curr['Close']) > float(prev['Open'])):

        body_ratio = abs(curr_body) / (abs(prev_body) + 0.001)
        if body_ratio >= 1.2:
            return {
                'detected'    : True,
                'pattern'     : 'Bullish Engulfing',
                'direction'   : 'BULLISH',
                'confidence'  : min(0.4 + body_ratio * 0.1, 0.85),
                'description' : f"Bullish engulfing — "
                               f"green candle {body_ratio:.1f}x "
                               f"size of previous red",
                'emoji'       : '🟢'
            }

    # Bearish engulfing
    if (prev_body > 0 and curr_body < 0 and
        float(curr['Open'])  > float(prev['Close']) and
        float(curr['Close']) < float(prev['Open'])):

        body_ratio = abs(curr_body) / (abs(prev_body) + 0.001)
        if body_ratio >= 1.2:
            return {
                'detected'    : True,
                'pattern'     : 'Bearish Engulfing',
                'direction'   : 'BEARISH',
                'confidence'  : min(0.4 + body_ratio * 0.1, 0.85),
                'description' : f"Bearish engulfing — "
                               f"red candle {body_ratio:.1f}x "
                               f"size of previous green",
                'emoji'       : '🔴'
            }

    return {'detected': False}

# ── Master pattern scanner ────────────────────────────────
def scan_patterns(ticker, lookback_days=120):
    """
    Runs all 8 pattern detectors on a stock.
    Returns list of detected patterns and a composite score.
    """
    try:
        df = load_ohlcv(ticker, lookback_days)
        if df.empty or len(df) < 30:
            return [], 0.0
    except Exception:
        return [], 0.0

    detectors = [
        detect_head_and_shoulders,
        detect_inverse_head_and_shoulders,
        detect_double_top,
        detect_double_bottom,
        detect_cup_and_handle,
        detect_cross,
        detect_flag,
        detect_engulfing,
    ]

    found    = []
    bull_score = 0.0
    bear_score = 0.0

    for detector in detectors:
        try:
            result = detector(df)
            if result.get('detected'):
                found.append(result)
                if result['direction'] == 'BULLISH':
                    bull_score += result['confidence']
                else:
                    bear_score += result['confidence']
        except Exception:
            continue

    # Net score: positive = bullish, negative = bearish
    net_score = round(bull_score - bear_score, 3)

    return found, net_score

def get_pattern_confidence_boost(patterns, net_score):
    """
    Converts pattern score into a confidence adjustment for the ensemble.
    Max boost: +10%  Max penalty: -15%
    """
    if net_score > 0.5:
        boost = min(net_score * 0.08, 0.10)
    elif net_score < -0.5:
        boost = max(net_score * 0.10, -0.15)
    else:
        boost = 0.0
    return round(boost, 4)