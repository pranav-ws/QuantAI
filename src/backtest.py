import pandas as pd # type: ignore
import numpy as np # type: ignore
import sqlite3
import os
import joblib # type: ignore
from src.features import get_feature_dataset
from src.model import FEATURE_COLS
from src.slippage import SlippageModel

DB_PATH = os.path.join('data', 'quantai.db')

def run_backtest(ticker, initial_capital=100000, confidence_threshold=0.58,
                  use_slippage=True, slippage_profile='delivery',
                  slippage_model=None):
    """
    Simulates trading based on ML model signals.
    Only enters a trade when model confidence > threshold.

    Parameters
    ----------
    use_slippage : bool
        If True (default), every entry/exit fill is run through a
        SlippageModel so the backtest reflects realistic costs
        (commission, STT/exchange fees, GST, and volatility/liquidity
        driven slippage) instead of a perfect fill at the signal price.
        Set False to see the old "paper" / zero-cost behaviour.
    slippage_profile : str
        'delivery' (default, overnight swing trades) or 'intraday'
        (same-day square-off) — passed to SlippageModel. Ignored if
        slippage_model is provided directly.
    slippage_model : SlippageModel or None
        Pass a pre-configured SlippageModel instance for full control
        over commission/slippage parameters. If None, a default model
        is built from slippage_profile.
    """
    print(f"\n{'='*55}")
    print(f"  QuantAI Backtest — {ticker}")
    print(f"  Capital: ₹{initial_capital:,}  |  Min confidence: {confidence_threshold}")
    cost_label = f"{slippage_profile} costs" if use_slippage else "ZERO costs (paper fills)"
    print(f"  Execution: {cost_label}")
    print(f"{'='*55}")

    if use_slippage and slippage_model is None:
        slippage_model = SlippageModel(profile=slippage_profile)

    # Load model and feature data
    model_path = os.path.join('models', f'{ticker}_rf_model.pkl')
    model = joblib.load(model_path)
    df = get_feature_dataset(ticker)

    # Use only the TEST period (last 20%)
    split_idx   = int(len(df) * 0.8)
    test_df     = df.iloc[split_idx:].copy()
    X_test      = test_df[FEATURE_COLS]
    probas      = model.predict_proba(X_test)[:, 1]  # P(UP)
    test_df['Signal']     = probas
    test_df['Prediction'] = (probas >= confidence_threshold).astype(int)

    # ── Simulate trading ─────────────────────────────────
    capital      = initial_capital
    position     = 0          # shares held
    entry_price  = 0.0
    trades       = []
    equity_curve = []

    for i, (date, row) in enumerate(test_df.iterrows()):
        price = row['Close']
        atr          = row['ATR_14']      if 'ATR_14' in row else None
        volume_ratio = row['Volume_Ratio'] if 'Volume_Ratio' in row else None

        # Exit: if we hold a position, sell at open of next day
        if position > 0:
            if use_slippage:
                exit_price, exit_costs = slippage_model.get_fill_price(
                    price, 'SELL', atr=atr, volume_ratio=volume_ratio
                )
            else:
                exit_price, exit_costs = price, None

            pnl        = (exit_price - entry_price) * position
            pnl_pct    = (exit_price - entry_price) / entry_price * 100
            capital   += exit_price * position
            trades.append({
                'exit_date'   : date,
                'exit_price'  : exit_price,
                'exit_signal_price': price,
                'exit_cost_pct': exit_costs['cost_pct_of_price'] if exit_costs else 0.0,
                'pnl'         : pnl,
                'pnl_pct'     : pnl_pct,
                'result'      : '✅ WIN' if pnl > 0 else '❌ LOSS'
            })
            position = 0

        # Entry: buy if model is confident stock will go UP
        if row['Prediction'] == 1 and position == 0:
            if use_slippage:
                fill_price, entry_costs = slippage_model.get_fill_price(
                    price, 'BUY', atr=atr, volume_ratio=volume_ratio
                )
            else:
                fill_price, entry_costs = price, None

            position    = int(capital * 0.95 / fill_price)  # use 95% of capital
            entry_price = fill_price
            capital    -= position * fill_price
            if trades:
                trades[-1]['entry_date']        = date
                trades[-1]['entry_price']       = entry_price
                trades[-1]['entry_signal_price'] = price
                trades[-1]['entry_cost_pct']    = entry_costs['cost_pct_of_price'] if entry_costs else 0.0

        # Track equity
        portfolio_value = capital + (position * price)
        equity_curve.append({
            'date'  : date,
            'value' : portfolio_value,
            'price' : price
        })

    equity_df = pd.DataFrame(equity_curve).set_index('date')
    trade_df  = pd.DataFrame(trades)

    # ── Performance Metrics ──────────────────────────────
    final_value = equity_df['value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    # Buy & Hold comparison
    bh_return = (test_df['Close'].iloc[-1] - test_df['Close'].iloc[0]) / test_df['Close'].iloc[0] * 100

    # Sharpe Ratio
    daily_returns = equity_df['value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    # Max Drawdown
    rolling_max  = equity_df['value'].cummax()
    drawdown     = (equity_df['value'] - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    # Win rate
    if not trade_df.empty and 'pnl' in trade_df.columns:
        wins     = (trade_df['pnl'] > 0).sum()
        losses   = (trade_df['pnl'] <= 0).sum()
        win_rate = wins / len(trade_df) * 100 if len(trade_df) > 0 else 0
        avg_win  = trade_df[trade_df['pnl'] > 0]['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = trade_df[trade_df['pnl'] <= 0]['pnl_pct'].mean() if losses > 0 else 0
    else:
        wins = losses = win_rate = avg_win = avg_loss = 0

    # Cost drag from slippage/commissions (₹ spent on fills vs. the zero-cost signal price)
    if use_slippage and not trade_df.empty:
        total_cost_pct_per_trade = (
            trade_df.get('entry_cost_pct', pd.Series(dtype=float)).fillna(0) +
            trade_df.get('exit_cost_pct', pd.Series(dtype=float)).fillna(0)
        )
        avg_cost_pct_per_trade = float(total_cost_pct_per_trade.mean()) if len(total_cost_pct_per_trade) else 0.0
    else:
        avg_cost_pct_per_trade = 0.0

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
    if use_slippage:
        print(f"\n  💸 EXECUTION COST DRAG ({slippage_profile})")
        print(f"  {'Avg round-trip cost/trade':<25} {avg_cost_pct_per_trade:>10.3f}%")
        print(f"  {'Est. total cost drag':<25} {avg_cost_pct_per_trade * len(trade_df):>+10.2f}%  (sum across all trades)")
    print(f"{'='*55}\n")

    return equity_df, trade_df, {
        'total_return'       : total_return,
        'bh_return'          : bh_return,
        'sharpe'             : sharpe,
        'max_drawdown'       : max_drawdown,
        'win_rate'           : win_rate,
        'use_slippage'       : use_slippage,
        'avg_cost_pct_per_trade': avg_cost_pct_per_trade
    }