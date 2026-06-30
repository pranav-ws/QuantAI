"""
pairs_trade.py — Daily pairs trading scanner.
Run alongside paper_trade.py every day at 3:30 PM.
"""

from src.pairs_trader import scan_pairs_signals, calculate_pairs_position
from src.alerts import send_message
from datetime import datetime

def run_pairs_session(capital=100000):
    now = datetime.now().strftime('%d %b %Y, %I:%M %p')

    print(f"\n{'='*60}")
    print(f"  QuantAI Pairs Trading — {now}")
    print(f"{'='*60}")

    signals = scan_pairs_signals(entry_z=2.0, exit_z=0.5)

    if not signals:
        print("  No pairs cached. Run find_pairs.py first.")
        return

    print(f"\n  {'PAIR':<25} {'Z-SCORE':>8} {'SIGNAL':<20}")
    print(f"  {'-'*55}")

    active_signals = []

    for s in signals:
        z   = s['zscore']
        sig = s['signal']

        if   sig == 'BUY_1_SHORT_2':
            emoji = '🟢 BUY/SHORT'
        elif sig == 'SHORT_1_BUY_2':
            emoji = '🔴 SHORT/BUY'
        elif sig == 'EXIT':
            emoji = '⚪ EXIT NOW'
        else:
            emoji = '⏸  HOLD'

        pair_name = f"{s['name1']} ↔ {s['name2']}"
        print(f"  {pair_name:<25} {z:>+7.2f}σ  {emoji}")

        if sig in ('BUY_1_SHORT_2', 'SHORT_1_BUY_2'):
            sh1, sh2 = calculate_pairs_position(s, capital)
            if sh1 > 0:
                active_signals.append({**s,
                    'shares1': sh1, 'shares2': sh2})

    # ── Send Telegram alert ────────────────────────────────
    if active_signals:
        msg = f"📊 *QuantAI Pairs Trading*\n📅 {now}\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
        for s in active_signals:
            if s['signal'] == 'BUY_1_SHORT_2':
                msg += (f"🟢 *{s['name1']} ↔ {s['name2']}*\n"
                        f"   BUY  {s['name1']}: {s['shares1']} shares "
                        f"@ ₹{s['price1']:,.1f}\n"
                        f"   SHORT {s['name2']}: {s['shares2']} shares "
                        f"@ ₹{s['price2']:,.1f}\n"
                        f"   Z-Score: {s['zscore']:+.2f}σ  "
                        f"(entry at ±2.0σ)\n"
                        f"   Half-life: {s['half_life']}d  "
                        f"Corr: {s['correlation']:.2f}\n\n")
            else:
                msg += (f"🔴 *{s['name1']} ↔ {s['name2']}*\n"
                        f"   SHORT {s['name1']}: {s['shares1']} shares\n"
                        f"   BUY  {s['name2']}: {s['shares2']} shares\n"
                        f"   Z-Score: {s['zscore']:+.2f}σ\n\n")
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        msg += "_Market-neutral strategy_"
        try:
            send_message(msg)
            print("\n  📱 Telegram alert sent!")
        except Exception:
            pass
    else:
        print("\n  No pairs signal today — all spreads within normal range.")

    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    run_pairs_session()