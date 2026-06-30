from src.features import get_feature_dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Build the feature dataset for Reliance
df = get_feature_dataset('RELIANCE.NS')

# Print summary
print("\n=== Feature Dataset Preview ===")
print(df[['Close', 'RSI_14', 'MACD', 'BB_Width', 'Volume_Ratio', 'Target']].tail(10).to_string())
print(f"\nTotal features: {len(df.columns)}")
print(f"Target distribution:\n{df['Target'].value_counts()}")

# Plot 3-panel chart: Price + RSI + MACD
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)

# Panel 1: Price + Bollinger Bands + Moving Averages
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['Close'],     color='white',  linewidth=1.2, label='Close')
ax1.plot(df.index, df['SMA_20'],    color='orange', linewidth=1.0, label='SMA 20', linestyle='--')
ax1.plot(df.index, df['SMA_50'],    color='cyan',   linewidth=1.0, label='SMA 50', linestyle='--')
ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='yellow', label='Bollinger Bands')
ax1.set_title('RELIANCE.NS — Price + Indicators', color='white', fontsize=13)
ax1.legend(loc='upper left', fontsize=8)
ax1.set_facecolor('#1a1a2e')
ax1.tick_params(colors='white')

# Panel 2: RSI
ax2 = fig.add_subplot(gs[1])
ax2.plot(df.index, df['RSI_14'], color='#ff6b6b', linewidth=1.0)
ax2.axhline(70, color='red',   linestyle='--', linewidth=0.8, alpha=0.7)
ax2.axhline(30, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
ax2.fill_between(df.index, df['RSI_14'], 70, where=(df['RSI_14'] >= 70), alpha=0.3, color='red')
ax2.fill_between(df.index, df['RSI_14'], 30, where=(df['RSI_14'] <= 30), alpha=0.3, color='green')
ax2.set_ylabel('RSI', color='white')
ax2.set_ylim(0, 100)
ax2.set_facecolor('#1a1a2e')
ax2.tick_params(colors='white')

# Panel 3: MACD
ax3 = fig.add_subplot(gs[2])
ax3.plot(df.index, df['MACD'],        color='#4ecdc4', linewidth=1.0, label='MACD')
ax3.plot(df.index, df['MACD_Signal'], color='#ff6b6b', linewidth=1.0, label='Signal')
ax3.bar(df.index, df['MACD_Hist'],    color=['green' if v >= 0 else 'red' for v in df['MACD_Hist']], alpha=0.4)
ax3.axhline(0, color='white', linewidth=0.5)
ax3.set_ylabel('MACD', color='white')
ax3.legend(loc='upper left', fontsize=8)
ax3.set_facecolor('#1a1a2e')
ax3.tick_params(colors='white')

fig.patch.set_facecolor('#0d0d1a')
plt.tight_layout()
plt.show()
