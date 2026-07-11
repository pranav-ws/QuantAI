"""
backtest_macd_rsi_confluence.py

Backtests the MACD + RSI Confluence (daily + weekly) strategy on a
single stock.

Exit logic is a hybrid of the other two backtests' ideas, since this
strategy is daily-trend-following but should react fast once daily
momentum actually reverses:

    1. CROSS_EXIT — the opposite MACD crossover fires (the original
                     reason to be in the trade is gone)
    2. TRAIL_STOP — chandelier trailing stop off the highest close
                     since entry (let a strong move run, but protect
                     the gains once it turns)
    3. STOP       — initial hard stop loss (trade failed immediately)
    4. TIME       — max holding period reached

Like the other two strategies, this is rule-based (no training), so
it's backtested across the full history.
"""
import os
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # type: ignore # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
from src.features import get_feature_dataset
from src.macd_rsi_confluence import MACDRSIConfluenceStrategy

TICKER          = 'RELIANCE.NS'
INITIAL_CAPITAL = 100_000
RISK_PER_TRADE  = 0.95   # use 95% of capital per position, same convention as the other backtests


def run_macd_rsi_confluence_backtest(ticker, initial_capital=INITIAL_CAPITAL, strategy=None):
    strategy = strategy or MACDRSIConfluenceStrategy()

    print(f"\n{'='*58}")
    print(f"  QuantAI MACD+RSI Confluence Backtest — {ticker}")
    print(f"  Capital: ₹{initial_capital:,}  |  Min confidence: {strategy.min_confidence}")
    print(f"  Weekly confirmation required: {strategy.require_weekly_confirm}")
    print(f"{'='*58}")

    df = get_feature_dataset(ticker)
    df = strategy.generate_signals(df)

    capital       = initial_capital
    position      = 0
    entry_price   = 0.0
    initial_stop  = 0.0
    highest_close = 0.0
    entry_date    = None
    days_held     = 0
    trades        = []
    equity_curve  = []

    for date, row in df.iterrows():
        price = row['Close']

        # ── Manage open position ─────────────────────────
        if position > 0:
            days_held += 1
            highest_close  = max(highest_close, price)
            trailing_stop  = highest_close - strategy.trail_atr_mult * row['ATR_14']
            effective_stop = max(initial_stop, trailing_stop)

            hit_stop    = price <= effective_stop
            cross_exit  = bool(row['MACD_Bear_Cross'])
            timed_out   = days_held >= strategy.max_holding_days

            if hit_stop or cross_exit or timed_out:
                pnl     = (price - entry_price) * position
                pnl_pct = (price - entry_price) / entry_price * 100
                capital += price * position

                if hit_stop:
                    reason = 'TRAIL_STOP' if effective_stop > initial_stop + 1e-6 else 'STOP'
                elif cross_exit:
                    reason = 'CROSS_EXIT'
                else:
                    reason = 'TIME'

                trades.append({
                    'entry_date'  : entry_date,
                    'exit_date'   : date,
                    'entry_price' : entry_price,
                    'exit_price'  : price,
                    'pnl'         : pnl,
                    'pnl_pct'     : pnl_pct,
                    'days_held'   : days_held,
                    'exit_reason' : reason,
                    'result'      : '✅ WIN' if pnl > 0 else '❌ LOSS',
                })
                position = 0

        # ── Look for new entry ────────────────────────────
        if position == 0 and row['MRC_Signal'] == 'BUY' and \
           row['MRC_Confidence'] >= strategy.min_confidence:
            position = int(capital * RISK_PER_TRADE / price)
            if position > 0:
                entry_price   = price
                entry_date    = date
                initial_stop  = price - strategy.stop_loss_atr_mult * row['ATR_14']
                highest_close = price
                capital      -= position * price
                days_held     = 0

        equity_curve.append({'date': date, 'value': capital + position * price, 'price': price})

    # ── Force-close any position still open when data ends ──
    if position > 0:
        price   = df['Close'].iloc[-1]
        pnl     = (price - entry_price) * position
        pnl_pct = (price - entry_price) / entry_price * 100
        capital += price * position
        trades.append({
            'entry_date'  : entry_date,
            'exit_date'   : df.index[-1],
            'entry_price' : entry_price,
            'exit_price'  : price,
            'pnl'         : pnl,
            'pnl_pct'     : pnl_pct,
            'days_held'   : days_held,
            'exit_reason' : 'END_OF_DATA',
            'result'      : '✅ WIN' if pnl > 0 else '❌ LOSS',
        })
        position = 0

    equity_df = pd.DataFrame(equity_curve).set_index('date')
    trade_df  = pd.DataFrame(trades)

    # ── Metrics ───────────────────────────────────────────
    final_value  = equity_df['value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    bh_return    = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100

    daily_returns = equity_df['value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0

    rolling_max  = equity_df['value'].cummax()
    drawdown     = (equity_df['value'] - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    if not trade_df.empty:
        wins      = (trade_df['pnl'] > 0).sum()
        losses    = (trade_df['pnl'] <= 0).sum()
        win_rate  = wins / len(trade_df) * 100
        avg_win   = trade_df.loc[trade_df['pnl'] > 0,  'pnl_pct'].mean() if wins   else 0
        avg_loss  = trade_df.loc[trade_df['pnl'] <= 0, 'pnl_pct'].mean() if losses else 0
        avg_days  = trade_df['days_held'].mean()
        reasons   = trade_df['exit_reason'].value_counts().to_dict()
    else:
        wins = losses = win_rate = avg_win = avg_loss = avg_days = 0
        reasons = {}

    print(f"\n  📈 PERFORMANCE SUMMARY")
    print(f"  {'Initial Capital':<25} ₹{initial_capital:>10,}")
    print(f"  {'Final Capital':<25} ₹{final_value:>10,.0f}")
    print(f"  {'Total Return':<25} {total_return:>+10.2f}%")
    print(f"  {'Buy & Hold Return':<25} {bh_return:>+10.2f}%")
    print(f"  {'Alpha (vs Buy&Hold)':<25} {total_return - bh_return:>+10.2f}%")
    print(f"\n  📊 RISK METRICS")
    print(f"  {'Sharpe Ratio':<25} {sharpe:>10.3f}  (target > 1.0)")
    print(f"  {'Max Drawdown':<25} {max_drawdown:>+10.2f}%  (target > -20%)")
    print(f"\n  🎯 TRADE STATISTICS")
    print(f"  {'Total Trades':<25} {len(trade_df):>10}")
    print(f"  {'Wins':<25} {wins:>10}")
    print(f"  {'Losses':<25} {losses:>10}")
    print(f"  {'Win Rate':<25} {win_rate:>10.1f}%")
    print(f"  {'Avg Win':<25} {avg_win:>+10.2f}%")
    print(f"  {'Avg Loss':<25} {avg_loss:>+10.2f}%")
    print(f"  {'Avg Holding Period':<25} {avg_days:>10.1f} days")
    print(f"  {'Exit Breakdown':<25} {reasons}")
    print(f"{'='*58}\n")

    return equity_df, trade_df, df, {
        'total_return' : total_return,
        'bh_return'    : bh_return,
        'sharpe'       : sharpe,
        'max_drawdown' : max_drawdown,
        'win_rate'     : win_rate,
        'n_trades'     : len(trade_df),
    }


if __name__ == '__main__':
    equity_df, trade_df, signal_df, metrics = run_macd_rsi_confluence_backtest(TICKER)

    # ── Plot ────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor='#0d0d1a')
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1a1a2e')
    equity_norm = equity_df['value'] / INITIAL_CAPITAL * 100
    price_norm  = equity_df['price'] / equity_df['price'].iloc[0] * 100

    ax1.plot(equity_df.index, equity_norm, color='#a78bfa', linewidth=1.6,
              label='MACD+RSI Confluence Strategy', zorder=3)
    ax1.plot(equity_df.index, price_norm, color='#ff6b6b', linewidth=1.1,
              linestyle='--', label='Buy & Hold', alpha=0.8)

    if not trade_df.empty:
        entry_vals = trade_df['entry_date'].map(
            lambda d: equity_df['value'].loc[d] / INITIAL_CAPITAL * 100 if d in equity_df.index else None)
        ax1.scatter(trade_df['entry_date'], entry_vals, marker='^', color='#22c55e',
                    s=60, zorder=4, label='Entry (confluence)')

    ax1.axhline(100, color='white', linewidth=0.5, linestyle=':', alpha=0.4)
    ax1.set_title(f'MACD+RSI Confluence vs Buy & Hold — {TICKER}', color='white', fontsize=12, pad=12)
    ax1.set_ylabel('Portfolio Value (base 100)', color='white')
    ax1.legend(fontsize=9, facecolor='#12121f', labelcolor='white', edgecolor='#262640')
    ax1.tick_params(colors='white')

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#1a1a2e')
    rolling_max = equity_df['value'].cummax()
    drawdown    = (equity_df['value'] - rolling_max) / rolling_max * 100
    ax2.fill_between(equity_df.index, drawdown, 0, color='red', alpha=0.4)
    ax2.plot(equity_df.index, drawdown, color='red', linewidth=0.8)
    ax2.axhline(-20, color='orange', linestyle='--', linewidth=0.8, alpha=0.7, label='-20% danger zone')
    ax2.set_ylabel('Drawdown %', color='white', fontsize=9)
    ax2.set_title('Strategy Drawdown', color='white', fontsize=10)
    ax2.legend(fontsize=8, facecolor='#12121f', labelcolor='white', edgecolor='#262640')
    ax2.tick_params(colors='white')

    os.makedirs('models', exist_ok=True)
    plt.savefig('models/macd_rsi_confluence_backtest.png', dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.show()
    print("📊 Chart saved → models/macd_rsi_confluence_backtest.png")