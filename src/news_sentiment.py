"""
src/news_sentiment.py

Financial news sentiment analysis using VADER
(Valence Aware Dictionary and sEntiment Reasoner).

Why VADER:
  - Built for short social/news text (perfect for headlines)
  - No training required — rule-based dictionary approach
  - Pure Python — works on all Python versions including 3.14
  - Handles financial language, exclamation marks, ALL CAPS etc.

How it integrates with the ensemble:
  - Bullish news  → sentiment score > 0 → confidence multiplier > 1.0 → stronger BUY
  - Bearish news  → sentiment score < 0 → confidence multiplier < 1.0 → weaker signal
  - Neutral/none  → multiplier = 1.0 → no change to ensemble confidence

The multiplier is capped at ±8% so news can influence but never override the ML signal.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime
import time

# Singleton analyser (loaded once, reused)
_analyser = SentimentIntensityAnalyzer()

# ── Simple time-based cache (avoid re-fetching within 30 min) ────
_news_cache: dict = {}
_CACHE_TTL = 30 * 60  # 30 minutes in seconds


def _is_cache_fresh(ticker: str) -> bool:
    if ticker not in _news_cache:
        return False
    cached_at = _news_cache[ticker].get('_ts', 0)
    return (time.time() - cached_at) < _CACHE_TTL


def fetch_news(ticker: str, max_articles: int = 10) -> list:
    """Fetches recent news headlines for a ticker via yfinance."""
    try:
        tk   = yf.Ticker(ticker)
        news = getattr(tk, 'news', None) or []
        # yfinance may return a callable in some versions
        if callable(news):
            news = news()
        return list(news)[:max_articles]
    except Exception:
        return []


def _score_text(text: str) -> float:
    """VADER compound score for one string: -1.0 (very negative) to +1.0 (very positive)."""
    return _analyser.polarity_scores(text)['compound']


def get_sentiment(ticker: str, max_articles: int = 10) -> tuple:
    """
    Returns (avg_score, articles_list) for a ticker.

    avg_score : float  — average compound score across recent headlines
    articles  : list   — dicts with title, publisher, date, score, sentiment, link
    """
    if _is_cache_fresh(ticker):
        cached = _news_cache[ticker]
        return cached['score'], cached['articles']

    raw = fetch_news(ticker, max_articles)
    articles, scores = [], []

    for item in raw:
        title = item.get('title', '').strip()
        if not title:
            continue

        compound = _score_text(title)
        scores.append(compound)

        # Publish time
        ts = item.get('providerPublishTime') or item.get('published', 0)
        try:
            pub_date = datetime.fromtimestamp(int(ts)).strftime('%d %b %Y')
        except Exception:
            pub_date = 'N/A'

        if compound >= 0.15:
            label = 'Bullish'
        elif compound >= 0.05:
            label = 'Mildly Bullish'
        elif compound <= -0.15:
            label = 'Bearish'
        elif compound <= -0.05:
            label = 'Mildly Bearish'
        else:
            label = 'Neutral'

        articles.append({
            'title'    : title,
            'publisher': item.get('publisher', ''),
            'link'     : item.get('link', '#'),
            'score'    : round(compound, 3),
            'sentiment': label,
            'date'     : pub_date,
        })

    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0

    # Cache result
    _news_cache[ticker] = {'score': avg_score, 'articles': articles, '_ts': time.time()}
    return avg_score, articles


def sentiment_label(score: float) -> str:
    """Human-readable label for a compound score."""
    if score >= 0.15:    return '🟢 Bullish'
    if score >= 0.05:    return '🟡 Mildly Bullish'
    if score <= -0.15:   return '🔴 Bearish'
    if score <= -0.05:   return '🟠 Mildly Bearish'
    return '⚪ Neutral'


def get_sentiment_modifier(score: float) -> float:
    """
    Converts an avg compound score to a confidence multiplier.
      score = +1.0 → modifier = 1.08  (boost confidence 8%)
      score =  0.0 → modifier = 1.00  (no change)
      score = -1.0 → modifier = 0.92  (reduce confidence 8%)
    Clamped so news never dominates the ML signal.
    """
    return round(1.0 + (score * 0.08), 4)
