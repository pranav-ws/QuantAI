"""
src/monte_carlo.py — QuantAI Monte Carlo Simulation Engine

Two complementary simulators:

1. Trade-resampling Monte Carlo (bootstrap)
   Takes the trade-by-trade % returns produced by src.backtest.run_backtest
   and reshuffles / resamples them thousands of times to answer:
   "Given this strategy's historical edge, what is the realistic RANGE of
    outcomes over the next N trades?" — i.e. it stress-tests sequencing
   risk, not just the one historical path we happened to observe.

2. Price-path Monte Carlo (Geometric Brownian Motion)
   Simulates thousands of possible future price paths for a stock from
   its historical daily-return mean/volatility, used to estimate
   probability of hitting a target price, Value-at-Risk (VaR), and
   Conditional VaR (Expected Shortfall) over a future horizon.

Both simulators are pure numpy/pandas — no extra dependencies beyond
what is already in requirements.txt.
"""
import numpy as np# type: ignore# type: ignore# type: ignore
import pandas as pd# type: ignore


# ════════════════════════════════════════════════════════════
#  1. TRADE-RESAMPLING MONTE CARLO (bootstrap on trade returns)
# ════════════════════════════════════════════════════════════

def run_trade_monte_carlo(trade_df, initial_capital=100000,
                           n_simulations=1000, n_trades=None,
                           random_seed=None):
    """
    Bootstraps the strategy's historical trade returns to build a
    distribution of possible equity-curve outcomes.

    Parameters
    ----------
    trade_df : pd.DataFrame
        Output of src.backtest.run_backtest — must contain a 'pnl_pct'
        column (percentage return of each closed trade).
    initial_capital : float
        Starting capital for each simulated path.
    n_simulations : int
        How many random equity paths to generate (1,000 is a good default;
        5,000+ for a smoother distribution).
    n_trades : int or None
        How many trades to simulate per path. Defaults to the number of
        historical trades available (a "replay the same number of bets"
        view). Set higher to project further into the future.
    random_seed : int or None
        Set for reproducible results (useful for tests/CI).

    Returns
    -------
    paths_df : pd.DataFrame
        Shape (n_trades + 1, n_simulations) — each column is one
        simulated equity curve, indexed 0..n_trades.
    summary : dict
        Percentile / risk statistics across all simulated paths.
    """
    if trade_df is None or trade_df.empty or 'pnl_pct' not in trade_df.columns:
        raise ValueError(
            "trade_df must be a non-empty DataFrame with a 'pnl_pct' column "
            "(the output of src.backtest.run_backtest)."
        )

    # Drop any rows where pnl_pct is missing (e.g. an unmatched open trade)
    returns_pct = trade_df['pnl_pct'].dropna().values / 100.0  # convert % → fraction

    if len(returns_pct) == 0:
        raise ValueError("No completed trades with pnl_pct found — nothing to simulate.")

    if n_trades is None:
        n_trades = len(returns_pct)

    rng = np.random.default_rng(random_seed)

    # Sample WITH replacement: each simulated path is a random sequence
    # of historical trade outcomes, capturing "what if the wins and
    # losses had landed in a different order, or repeated differently?"
    sampled_returns = rng.choice(returns_pct, size=(n_simulations, n_trades), replace=True)

    # Build equity curves: capital compounds trade-by-trade
    growth_factors = 1.0 + sampled_returns                       # shape (n_sims, n_trades)
    cum_growth = np.cumprod(growth_factors, axis=1)               # cumulative compounding
    equity_paths = initial_capital * cum_growth                   # actual ₹ value per trade

    # Prepend the starting capital as "trade 0" for every path
    start_row = np.full((n_simulations, 1), initial_capital)
    equity_paths = np.hstack([start_row, equity_paths])           # shape (n_sims, n_trades+1)

    paths_df = pd.DataFrame(equity_paths.T)                       # rows = trade index, cols = sim id
    paths_df.index.name = 'trade_number'

    final_values = equity_paths[:, -1]
    total_returns_pct = (final_values - initial_capital) / initial_capital * 100

    # Max drawdown per simulated path
    running_max = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = (equity_paths - running_max) / running_max * 100
    max_drawdowns = drawdowns.min(axis=1)

    summary = {
        'n_simulations'        : n_simulations,
        'n_trades_per_path'    : n_trades,
        'initial_capital'      : initial_capital,
        'mean_final_value'     : float(np.mean(final_values)),
        'median_final_value'   : float(np.median(final_values)),
        'std_final_value'      : float(np.std(final_values)),
        'best_final_value'     : float(np.max(final_values)),
        'worst_final_value'    : float(np.min(final_values)),
        'mean_return_pct'      : float(np.mean(total_returns_pct)),
        'median_return_pct'    : float(np.median(total_returns_pct)),
        'prob_profit_pct'      : float(np.mean(final_values > initial_capital) * 100),
        'prob_loss_pct'        : float(np.mean(final_values < initial_capital) * 100),
        'percentile_5_value'   : float(np.percentile(final_values, 5)),
        'percentile_25_value'  : float(np.percentile(final_values, 25)),
        'percentile_50_value'  : float(np.percentile(final_values, 50)),
        'percentile_75_value'  : float(np.percentile(final_values, 75)),
        'percentile_95_value'  : float(np.percentile(final_values, 95)),
        'mean_max_drawdown_pct': float(np.mean(max_drawdowns)),
        'worst_max_drawdown_pct': float(np.min(max_drawdowns)),
        'var_95_pct'           : float(np.percentile(total_returns_pct, 5)),   # 5th pct of returns
        'cvar_95_pct'          : float(np.mean(total_returns_pct[total_returns_pct <=
                                       np.percentile(total_returns_pct, 5)])),
    }

    return paths_df, summary


def print_trade_mc_report(summary, ticker="STRATEGY"):
    """Pretty-prints the trade-resampling Monte Carlo summary."""
    print(f"\n{'='*55}")
    print(f"  QuantAI Monte Carlo — Trade Resampling — {ticker}")
    print(f"{'='*55}")
    print(f"  Simulations          : {summary['n_simulations']:,}")
    print(f"  Trades per path      : {summary['n_trades_per_path']}")
    print(f"  Initial Capital      : ₹{summary['initial_capital']:>12,.0f}")
    print(f"\n  📊 FINAL VALUE DISTRIBUTION")
    print(f"  {'Mean':<22} ₹{summary['mean_final_value']:>12,.0f}")
    print(f"  {'Median':<22} ₹{summary['median_final_value']:>12,.0f}")
    print(f"  {'Best case':<22} ₹{summary['best_final_value']:>12,.0f}")
    print(f"  {'Worst case':<22} ₹{summary['worst_final_value']:>12,.0f}")
    print(f"\n  📈 PERCENTILES (final portfolio value)")
    print(f"  {'5th  pct (pessimistic)':<25} ₹{summary['percentile_5_value']:>12,.0f}")
    print(f"  {'25th pct':<25} ₹{summary['percentile_25_value']:>12,.0f}")
    print(f"  {'50th pct (median)':<25} ₹{summary['percentile_50_value']:>12,.0f}")
    print(f"  {'75th pct':<25} ₹{summary['percentile_75_value']:>12,.0f}")
    print(f"  {'95th pct (optimistic)':<25} ₹{summary['percentile_95_value']:>12,.0f}")
    print(f"\n  🎯 PROBABILITIES")
    print(f"  {'P(profit)':<22} {summary['prob_profit_pct']:>11.1f}%")
    print(f"  {'P(loss)':<22} {summary['prob_loss_pct']:>11.1f}%")
    print(f"\n  ⚠️  RISK METRICS")
    print(f"  {'Mean Max Drawdown':<22} {summary['mean_max_drawdown_pct']:>+11.2f}%")
    print(f"  {'Worst Max Drawdown':<22} {summary['worst_max_drawdown_pct']:>+11.2f}%")
    print(f"  {'VaR (95%)':<22} {summary['var_95_pct']:>+11.2f}%   (5% chance of losing more than this)")
    print(f"  {'CVaR (95%)':<22} {summary['cvar_95_pct']:>+11.2f}%   (avg loss IN that worst 5%)")
    print(f"{'='*55}\n")


# ════════════════════════════════════════════════════════════
#  2. PRICE-PATH MONTE CARLO (Geometric Brownian Motion)
# ════════════════════════════════════════════════════════════

def run_price_monte_carlo(price_df, n_days=30, n_simulations=1000,
                           random_seed=None):
    """
    Simulates future price paths using Geometric Brownian Motion (GBM),
    calibrated to the historical daily-return mean and volatility of
    the supplied price series.

    Parameters
    ----------
    price_df : pd.DataFrame
        Must contain a 'Close' column (e.g. from src.features.load_prices
        or get_feature_dataset). Only the 'Close' column is used.
    n_days : int
        Number of future trading days to simulate.
    n_simulations : int
        Number of independent price paths to generate.
    random_seed : int or None
        Set for reproducible results.

    Returns
    -------
    price_paths : pd.DataFrame
        Shape (n_days + 1, n_simulations) — each column is one simulated
        price path, row 0 is today's last close.
    summary : dict
        Percentile prices, probability of profit/target hit, VaR/CVaR
        on the simulated terminal price distribution.
    """
    if 'Close' not in price_df.columns:
        raise ValueError("price_df must contain a 'Close' column.")

    closes = price_df['Close'].dropna()
    if len(closes) < 30:
        raise ValueError("Need at least 30 days of price history to estimate "
                          "drift/volatility reliably.")

    daily_returns = closes.pct_change().dropna()
    mu    = daily_returns.mean()       # average daily return (drift)
    sigma = daily_returns.std()        # daily volatility
    last_price = float(closes.iloc[-1])

    rng = np.random.default_rng(random_seed)

    # GBM discrete-time formula:
    #   S(t+1) = S(t) * exp( (mu - 0.5*sigma^2) + sigma * Z ),  Z ~ N(0,1)
    drift = mu - 0.5 * sigma ** 2
    shocks = rng.normal(loc=0.0, scale=sigma, size=(n_days, n_simulations))
    daily_log_returns = drift + shocks

    log_price_paths = np.cumsum(daily_log_returns, axis=0)
    price_paths_arr = last_price * np.exp(log_price_paths)

    # Prepend today's actual price as day 0
    start_row = np.full((1, n_simulations), last_price)
    price_paths_arr = np.vstack([start_row, price_paths_arr])

    price_paths = pd.DataFrame(price_paths_arr)
    price_paths.index.name = 'day'

    terminal_prices = price_paths_arr[-1, :]
    returns_pct = (terminal_prices - last_price) / last_price * 100

    summary = {
        'last_price'          : last_price,
        'n_days'               : n_days,
        'n_simulations'         : n_simulations,
        'daily_drift_mu'        : float(mu),
        'daily_volatility_sigma': float(sigma),
        'mean_terminal_price'   : float(np.mean(terminal_prices)),
        'median_terminal_price' : float(np.median(terminal_prices)),
        'std_terminal_price'    : float(np.std(terminal_prices)),
        'percentile_5_price'    : float(np.percentile(terminal_prices, 5)),
        'percentile_25_price'   : float(np.percentile(terminal_prices, 25)),
        'percentile_50_price'   : float(np.percentile(terminal_prices, 50)),
        'percentile_75_price'   : float(np.percentile(terminal_prices, 75)),
        'percentile_95_price'   : float(np.percentile(terminal_prices, 95)),
        'prob_profit_pct'       : float(np.mean(terminal_prices > last_price) * 100),
        'var_95_pct'            : float(np.percentile(returns_pct, 5)),
        'cvar_95_pct'           : float(np.mean(returns_pct[returns_pct <=
                                        np.percentile(returns_pct, 5)])),
    }

    return price_paths, summary


def probability_of_target(price_paths, target_price, mode='reach'):
    """
    Estimates the probability of a price target being hit.

    mode='reach'    → probability the path touches target_price at ANY
                       point during the simulated horizon (running max/min).
    mode='terminal' → probability the FINAL simulated price is at/above
                       (or at/below, for a downside target) target_price.
    """
    last_price = price_paths.iloc[0, 0]
    arr = price_paths.values

    if target_price >= last_price:
        if mode == 'reach':
            hit = (arr.max(axis=0) >= target_price)
        else:
            hit = (arr[-1, :] >= target_price)
    else:
        if mode == 'reach':
            hit = (arr.min(axis=0) <= target_price)
        else:
            hit = (arr[-1, :] <= target_price)

    return float(np.mean(hit) * 100)


def print_price_mc_report(summary, ticker="STOCK", target_price=None,
                           price_paths=None):
    """Pretty-prints the price-path Monte Carlo summary."""
    print(f"\n{'='*55}")
    print(f"  QuantAI Monte Carlo — Price Simulation (GBM) — {ticker}")
    print(f"{'='*55}")
    print(f"  Current Price        ₹{summary['last_price']:>12,.2f}")
    print(f"  Horizon              {summary['n_days']:>12} trading days")
    print(f"  Simulations          {summary['n_simulations']:>12,}")
    print(f"  Daily drift (μ)      {summary['daily_drift_mu']*100:>+11.4f}%")
    print(f"  Daily volatility (σ) {summary['daily_volatility_sigma']*100:>11.4f}%")
    print(f"\n  📊 SIMULATED PRICE DISTRIBUTION ({summary['n_days']}d ahead)")
    print(f"  {'5th  pct (pessimistic)':<25} ₹{summary['percentile_5_price']:>10,.2f}")
    print(f"  {'25th pct':<25} ₹{summary['percentile_25_price']:>10,.2f}")
    print(f"  {'50th pct (median)':<25} ₹{summary['percentile_50_price']:>10,.2f}")
    print(f"  {'75th pct':<25} ₹{summary['percentile_75_price']:>10,.2f}")
    print(f"  {'95th pct (optimistic)':<25} ₹{summary['percentile_95_price']:>10,.2f}")
    print(f"\n  🎯 P(price ends higher)  {summary['prob_profit_pct']:>10.1f}%")
    if target_price is not None and price_paths is not None:
        p_reach = probability_of_target(price_paths, target_price, mode='reach')
        p_term  = probability_of_target(price_paths, target_price, mode='terminal')
        direction = "reach/exceed" if target_price >= summary['last_price'] else "drop to/below"
        print(f"  🎯 P({direction} ₹{target_price:,.2f} at any point) {p_reach:>6.1f}%")
        print(f"  🎯 P({direction} ₹{target_price:,.2f} by day {summary['n_days']})    {p_term:>6.1f}%")
    print(f"\n  ⚠️  VaR (95%, {summary['n_days']}d)     {summary['var_95_pct']:>+10.2f}%")
    print(f"  ⚠️  CVaR (95%, {summary['n_days']}d)    {summary['cvar_95_pct']:>+10.2f}%")
    print(f"{'='*55}\n")