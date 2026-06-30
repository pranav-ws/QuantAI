import sqlite3, os

DB_PATH = os.path.join('data', 'quantai.db')

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("\n" + "=" * 62)
print("   QuantAI Database — Summary")
print("=" * 62)
print(f"{'Ticker':<20} {'Name':<30} {'Records':>8}")
print("-" * 62)

cursor.execute('''
    SELECT s.ticker, s.name, COUNT(p.id) as cnt
    FROM stocks s
    LEFT JOIN prices p ON s.ticker = p.ticker
    GROUP BY s.ticker
    ORDER BY cnt DESC
''')

for ticker, name, cnt in cursor.fetchall():
    print(f"{ticker:<20} {name:<30} {cnt:>8,}")

cursor.execute('SELECT COUNT(*) FROM prices')
total = cursor.fetchone()[0]

print("=" * 62)
print(f"   Total rows in database: {total:,}")
print("=" * 62 + "\n")

conn.close()
