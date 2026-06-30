from src.database import create_tables, get_prices
from src.data_collector import fetch_all_stocks

def run():
    # 1. Set up database
    create_tables()

    # 2. Fetch all stocks
    fetch_all_stocks()

    # 3. Quick sanity check — show latest 5 rows for Reliance
    print("📊 Sample — Latest 5 rows for RELIANCE.NS:")
    print(f"{'Date':<13} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Volume':>12}")
    print("-" * 58)
    for row in get_prices('RELIANCE.NS', limit=5):
        date, o, h, l, c, v = row
        print(f"{date:<13} {o:>8.1f} {h:>8.1f} {l:>8.1f} {c:>8.1f} {v:>12,}")

if __name__ == "__main__":
    run()
