import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
from src.backtest import run_backtest

TICKER = 'RELIANCE.NS'
USE_SLIPPAGE = True              # realistic costs ON (set False to see old zero-cost "paper" results)
SLIPPAGE_PROFILE = 'delivery'    # 'delivery' (overnight) or 'intraday' (same-day square-off)

# Run the backtest
equity_df, trade_df, metrics = run_backtest(
    ticker=TICKER,
    initial_capital=100000,      # Start with ₹1,00,000
    confidence_threshold=0.58,   # Only trade when model is 58%+ confident
    use_slippage=USE_SLIPPAGE,
    slippage_profile=SLIPPAGE_PROFILE
)

# ── Plot results ─────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#0d0d1a')
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)

# Panel 1: Equity curve vs Buy & Hold
ax1 = fig.add_subplot(gs[0])
equity_norm = equity_df['value'] / equity_df['value'].iloc[0] * 100
price_norm  = equity_df['price'] / equity_df['price'].iloc[0] * 100

ax1.plot(equity_df.index, equity_norm, color='#4ecdc4',
         linewidth=1.5, label='QuantAI Strategy')
ax1.plot(equity_df.index, price_norm,  color='#ff6b6b',
         linewidth=1.2, label='Buy & Hold', linestyle='--', alpha=0.8)
ax1.axhline(100, color='white', linewidth=0.5, linestyle=':', alpha=0.5)
ax1.fill_between(equity_df.index, equity_norm, 100,
                 where=(equity_norm >= 100), alpha=0.1, color='#4ecdc4')
ax1.fill_between(equity_df.index, equity_norm, 100,
                 where=(equity_norm < 100),  alpha=0.1, color='red')
ax1.set_title(f'QuantAI vs Buy & Hold — {TICKER} (Test Period)',
              color='white', fontsize=12)
ax1.legend(fontsize=9)
ax1.set_ylabel('Portfolio Value (indexed to 100)', color='white')
ax1.set_facecolor('#1a1a2e')
ax1.tick_params(colors='white')

# Panel 2: Drawdown
ax2 = fig.add_subplot(gs[1])
rolling_max = equity_df['value'].cummax()
drawdown    = (equity_df['value'] - rolling_max) / rolling_max * 100
ax2.fill_between(equity_df.index, drawdown, 0, color='red', alpha=0.4)
ax2.plot(equity_df.index, drawdown, color='red', linewidth=0.8)
ax2.axhline(-20, color='orange', linestyle='--', linewidth=0.8,
            label='-20% danger zone')
ax2.set_ylabel('Drawdown %', color='white')
ax2.set_title('Portfolio Drawdown', color='white', fontsize=10)
ax2.legend(fontsize=8)
ax2.set_facecolor('#1a1a2e')
ax2.tick_params(colors='white')

plt.savefig('models/backtest_results.png', dpi=150,
            bbox_inches='tight', facecolor='#0d0d1a')
plt.show()
print("📊 Chart saved → models/backtest_results.png")