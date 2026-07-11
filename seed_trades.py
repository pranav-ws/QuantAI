"""seed_trades.py — adds sample closed trades for testing."""
import json, os
from datetime import date, timedelta

trades = []
samples = [
    ('TCS.NS',       2100.0, 2161.0, 13, 0.70, 5),
    ('BHARTIARTL.NS',1780.0, 1822.0, 17, 0.72, 3),
    ('INFY.NS',      1090.0, 1116.0, 23, 0.61, 4),
    ('HDFCBANK.NS',   760.0,  748.0, 25, 0.63, 2),
    ('SUNPHARMA.NS',  905.0,  930.0, 20, 0.66, 6),
    ('TCS.NS',       2200.0, 2180.0, 10, 0.68, 3),
    ('WIPRO.NS',      290.0,  298.0, 60, 0.59, 5),
    ('AXISBANK.NS',  1050.0, 1085.0, 18, 0.64, 4),
]

for i, (ticker, entry, exit_p,
         shares, conf, hold) in enumerate(samples):
    entry_d = (date.today() -
               timedelta(days=30-i*3)).strftime('%Y-%m-%d')
    exit_d  = (date.today() -
               timedelta(days=28-i*3)).strftime('%Y-%m-%d')
    pnl     = (exit_p - entry) * shares
    pnl_pct = (exit_p - entry) / entry * 100
    trades.append({
        'ticker'     : ticker,
        'price'      : entry,
        'exit_price' : exit_p,
        'shares'     : shares,
        'confidence' : conf,
        'trade_value': entry * shares,
        'stop_loss'  : round(entry * 0.97, 2),
        'date'       : entry_d,
        'exit_date'  : exit_d,
        'status'     : 'CLOSED',
        'pnl'        : round(pnl, 2),
        'pnl_pct'    : round(pnl_pct, 2),
        'hold_days'  : hold,
    })

os.makedirs('data', exist_ok=True)
with open('data/paper_trades.json', 'w') as f:
    json.dump(trades, f, indent=2)
print(f"Added {len(trades)} sample trades.")
print("Now run: python3 run_performance.py")