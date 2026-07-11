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
import time, os, json

# Singleton analyser (loaded once, reused)
_analyser = SentimentIntensityAnalyzer()

# ── Simple time-based cache (avoid re-fetching within 30 min) ────
_news_cache: dict = {}
_CACHE_TTL = 30 * 60  # 30 minutes in seconds

# ── Disk-persisted "last known good" news, per ticker ────────────
# yfinance's news feed is thin/inconsistent for many NSE tickers, and
# sometimes returns nothing at all (weekends, temporary hiccups, or a
# stock that simply has no fresh headlines right now). Rather than show
# a dead "No recent headlines found" every time that happens, we keep
# the last successful non-empty fetch on disk and fall back to it,
# clearly labelled with when it was actually fetched.
_LAST_GOOD_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'news_last_good.json')


def _load_last_good_store() -> dict:
    try:
        with open(_LAST_GOOD_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_last_good(ticker: str, score: float, articles: list):
    try:
        os.makedirs(os.path.dirname(_LAST_GOOD_PATH), exist_ok=True)
        store = _load_last_good_store()
        store[ticker] = {
            'score': score,
            'articles': articles,
            'fetched_at': datetime.now().isoformat(),
        }
        with open(_LAST_GOOD_PATH, 'w', encoding='utf-8') as f:
            json.dump(store, f)
    except Exception:
        pass  # persistence is a nice-to-have, never let it break a live fetch


def _get_last_good(ticker: str):
    """Returns (score, articles, fetched_at) from disk, or None if never saved."""
    store = _load_last_good_store()
    entry = store.get(ticker)
    if not entry:
        return None
    return entry.get('score', 0.0), entry.get('articles', []), entry.get('fetched_at')


def _is_cache_fresh(ticker: str) -> bool:
    if ticker not in _news_cache:
        return False
    cached_at = _news_cache[ticker].get('_ts', 0)
    return (time.time() - cached_at) < _CACHE_TTL


def _extract_article_fields(item: dict) -> dict:
    """
    yfinance restructured Ticker.news item shape at some point in 2024,
    moving title/publisher/link/publish-time from top-level keys into a
    nested 'content' dict. Since requirements.txt doesn't pin a yfinance
    version, whichever schema you happen to have installed determines
    which of these works — reading only the old flat keys meant every
    single article silently had an empty title and got filtered out,
    100% of the time, regardless of ticker or real news availability.
    This checks both shapes so it works either way.
    """
    content = item.get('content') if isinstance(item.get('content'), dict) else None

    if content:  # new nested schema
        title     = (content.get('title') or '').strip()
        publisher = ((content.get('provider') or {}).get('displayName')
                     or content.get('publisher') or '')
        link      = ((content.get('canonicalUrl') or {}).get('url')
                     or (content.get('clickThroughUrl') or {}).get('url') or '#')
        raw_date  = content.get('pubDate') or content.get('displayTime') or ''
    else:        # old flat schema
        title     = (item.get('title') or '').strip()
        publisher = item.get('publisher') or ''
        link      = item.get('link') or '#'
        raw_date  = item.get('providerPublishTime') or item.get('published') or ''

    return {'title': title, 'publisher': publisher, 'link': link, 'raw_date': raw_date}


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
    Returns (avg_score, articles_list, meta) for a ticker.

    avg_score : float  — average compound score across recent headlines
    articles  : list   — dicts with title, publisher, date, score, sentiment, link
    meta      : dict   — {'stale': bool, 'fetched_at': iso-str or None}
                 stale=True means these are the last known-good headlines
                 from a previous successful fetch, not fresh ones — the
                 live fetch just now returned nothing.
    """
    if _is_cache_fresh(ticker):
        cached = _news_cache[ticker]
        if cached['articles']:
            return cached['score'], cached['articles'], {'stale': False, 'fetched_at': None}
        # Cached result was itself empty — don't let a 30-min-old "nothing
        # found" block the last-known-good fallback below.
        fallback = _get_last_good(ticker)
        if fallback:
            last_score, last_articles, fetched_at = fallback
            return last_score, last_articles, {'stale': True, 'fetched_at': fetched_at}
        return cached['score'], cached['articles'], {'stale': False, 'fetched_at': None}

    raw = fetch_news(ticker, max_articles)
    articles, scores = [], []

    for item in raw:
        fields = _extract_article_fields(item)
        title  = fields['title']
        if not title:
            continue

        compound = _score_text(title)
        scores.append(compound)

        # Publish time — old schema uses epoch seconds, new schema uses an
        # ISO 8601 string (e.g. "2026-07-10T09:15:00Z"). Handle both.
        raw_date = fields['raw_date']
        pub_date = 'N/A'
        try:
            if isinstance(raw_date, (int, float)) or (isinstance(raw_date, str) and raw_date.isdigit()):
                pub_date = datetime.fromtimestamp(int(raw_date)).strftime('%d %b %Y')
            elif isinstance(raw_date, str) and raw_date:
                pub_date = datetime.fromisoformat(raw_date.replace('Z', '+00:00')).strftime('%d %b %Y')
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
            'publisher': fields['publisher'],
            'link'     : fields['link'],
            'score'    : round(compound, 3),
            'sentiment': label,
            'date'     : pub_date,
        })

    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0

    # Cache result
    _news_cache[ticker] = {'score': avg_score, 'articles': articles, '_ts': time.time()}

    # Also persist the last successful non-empty fetch to disk, so a later
    # request that gets zero articles (weekend, API hiccup, no fresh news
    # today) can still show the last known headlines instead of a dead
    # "No recent headlines found" — the user asked for this explicitly.
    if articles:
        _save_last_good(ticker, avg_score, articles)
        return avg_score, articles, {'stale': False, 'fetched_at': None}

    # Live fetch returned nothing — fall back to the last known-good set.
    fallback = _get_last_good(ticker)
    if fallback:
        last_score, last_articles, fetched_at = fallback
        return last_score, last_articles, {'stale': True, 'fetched_at': fetched_at}

    # Never had any successful fetch for this ticker at all.
    return 0.0, [], {'stale': False, 'fetched_at': None}


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
