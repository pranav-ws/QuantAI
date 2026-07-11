"""
test_sentiment.py
Quick demo: fetch news + sentiment scores for 5 Nifty stocks.
Run: python test_sentiment.py
"""
from src.news_sentiment import get_sentiment, sentiment_label, get_sentiment_modifier

TICKERS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']

print("\n" + "="*70)
print("  QuantAI — News Sentiment Demo")
print("="*70)

for ticker in TICKERS:
    print(f"\n📰  {ticker}")
    score, articles, meta = get_sentiment(ticker, max_articles=5)
    if meta.get('stale'):
        print(f"    (showing last known headlines from {meta.get('fetched_at')})")
    modifier        = get_sentiment_modifier(score)
    label           = sentiment_label(score)

    print(f"    Overall: {label}  (score={score:+.3f}, modifier={modifier:.3f}x)")
    print()

    if articles:
        for a in articles[:5]:
            icon = ('🟢' if 'Bullish'  in a['sentiment'] else
                    '🔴' if 'Bearish'  in a['sentiment'] else '⚪')
            print(f"    {icon} [{a['score']:+.2f}] {a['title'][:65]}")
            print(f"       {a['publisher']} · {a['date']}")
    else:
        print("    (No headlines found — market may be closed or yfinance rate-limited)")

print("\n" + "="*70)
print("  Modifier logic:")
print("  Bullish news  → multiplier > 1.0 → ensemble confidence boosted (max +8%)")
print("  Bearish news  → multiplier < 1.0 → ensemble confidence reduced  (max -8%)")
print("  Neutral/none  → multiplier = 1.0 → no change")
print("="*70 + "\n")
