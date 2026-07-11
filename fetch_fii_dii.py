"""
fetch_fii_dii.py
Fetches latest FII/DII institutional flow data from NSE India.
Run once daily after market close (~4 PM IST):  python fetch_fii_dii.py
"""
from src.fii_dii import fetch_and_store, get_recent_data, get_flow_signal

print("\n" + "="*58)
print("  QuantAI — FII/DII Institutional Flow Fetcher")
print("="*58)

result = fetch_and_store(days=30)

if result['success']:
    print(f"  ✅ Source  : {result['source']}")
    print(f"  ✅ Saved   : {result['rows_saved']} record(s)")
else:
    print(f"  ⚠️  {result.get('message','Fetch failed — NSE may be rate-limiting.')}")

print(f"\n  Recent FII/DII Data (₹ Crore):")
print(f"  {'Date':<13} {'FII Net':>10} {'DII Net':>9} {'Combined':>10}")
print(f"  {'-'*44}")
for r in get_recent_data(days=10):
    icon = '🟢' if r['fii_net'] > 0 else '🔴'
    print(f"  {icon} {r['date']:<12} {r['fii_net']:>+10,.0f} {r['dii_net']:>+9,.0f} {r['combined_net']:>+10,.0f}")

sig = get_flow_signal(days_avg=5)
print(f"\n{'='*58}")
print(f"  Signal       : {sig['signal']}")
print(f"  Modifier     : {sig['modifier']:.2f}× on ensemble confidence")
print(f"  FII Avg      : ₹{sig['fii_net_avg']:+,.0f} Cr")
print(f"  DII Avg      : ₹{sig['dii_net_avg']:+,.0f} Cr")
print(f"  Combined Avg : ₹{sig['combined_avg']:+,.0f} Cr")
print(f"{'='*58}\n")
