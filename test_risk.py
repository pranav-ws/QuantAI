from src.risk import RiskManager

print("\n" + "="*55)
print("  QuantAI — Risk Manager Demo")
print("="*55)

rm = RiskManager(initial_capital=100000)

# ── Test 1: Normal high-confidence trade ────────────────
print("\n📌 Trade 1 — High confidence signal (RELIANCE @ ₹1350)")
shares, sl, reason = rm.calculate_position_size(
    capital=100000, price=1350, confidence=0.67
)
print(f"  {reason}")

# ── Test 2: Low confidence — should be rejected ─────────
print("\n📌 Trade 2 — Low confidence signal (TCS @ ₹3200)")
shares, sl, reason = rm.calculate_position_size(
    capital=100000, price=3200, confidence=0.51
)
print(f"  {reason}")

# ── Test 3: Check drawdown monitor ──────────────────────
print("\n📌 Drawdown Monitor Test:")
test_values = [100000, 97000, 93000, 88000, 85001, 84999]
for val in test_values:
    safe, dd = rm.check_portfolio_risk(val)
    status   = "✅ Trading OK" if safe else "🚫 TRADING HALTED"
    print(f"  Portfolio: ₹{val:,}  |  Drawdown: {dd:+.2%}  |  {status}")

# ── Test 4: Full risk report with open positions ────────
print()
rm2 = RiskManager(initial_capital=100000)
rm2.capital = 55000  # rest is in positions

positions = {
    'RELIANCE.NS' : (30,  1320.0, 1280.0),  # (shares, entry, stop_loss)
    'TCS.NS'      : (10,  3100.0, 3007.0),
    'HDFCBANK.NS' : (25,  1680.0, 1629.6),
}
current_prices = {
    'RELIANCE.NS' : 1348.0,
    'TCS.NS'      : 3085.0,
    'HDFCBANK.NS' : 1710.0,
}
rm2.get_risk_report(positions, current_prices)
