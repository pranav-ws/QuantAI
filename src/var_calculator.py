"""
src/var_calculator.py

Value at Risk (VaR) calculator for QuantAI portfolio.

Three methods implemented:
  1. Historical VaR    — uses actual past returns (no assumptions)
  2. Parametric VaR    — assumes normal distribution (fast)
  3. Monte Carlo VaR   — simulates 10,000 future scenarios (most robust)

All three are calculated and the most conservative is used.
Confidence levels: 95% (standard) and 99% (stress test)
"""

import numpy as np# type: ignore# type: ignore
import pandas as pd# type: ignore
import sqlite3
import os
import json
from datetime import datetime, date

DB_PATH      = os.path.join('data', 'quantai.db')
TRADES_PATH  = os.path.join('data', 'paper_trades.json')
CAPITAL_PATH = os.path.join('data', 'paper_capital.json')

# ── Load price data ───────────────────────────────────────
def load_returns(ticker, lookback_days=252):
    """
    Loads daily returns for a ticker from the database.
    252 trading days = 1 year.
    """
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        """SELECT date, close FROM prices
           WHERE ticker=?
           ORDER BY date DESC LIMIT ?""",
        conn, params=(ticker, lookback_days + 1)
    )
    conn.close()

    if df.empty or len(df) < 30:
        return None

    df['date']    = pd.to_datetime(df['date'])
    df            = df.sort_values('date')
    df['returns'] = df['close'].pct_change()
    return df['returns'].dropna()

# ── Method 1: Historical VaR ─────────────────────────────
def historical_var(returns, confidence=0.95):
    """
    Sorts all past daily returns and takes the worst percentile.
    No distribution assumptions — most realistic.

    Example: if 5th percentile of daily returns = -3.2%
             then Historical VaR(95%) = 3.2%
    """
    if returns is None or len(returns) < 30:
        return None
    var_pct = float(np.percentile(returns, (1 - confidence) * 100))
    return abs(var_pct)

# ── Method 2: Parametric VaR ──────────────────────────────
def parametric_var(returns, confidence=0.95):
    """
    Assumes returns follow a normal distribution.
    VaR = mean - z * std
    z = 1.645 for 95%, 2.326 for 99%
    """
    if returns is None or len(returns) < 30:
        return None

    z_scores = {0.95: 1.645, 0.99: 2.326, 0.90: 1.282}
    z        = z_scores.get(confidence, 1.645)

    mu    = float(returns.mean())
    sigma = float(returns.std())
    var   = abs(mu - z * sigma)
    return var

# ── Method 3: Monte Carlo VaR ────────────────────────────
def monte_carlo_var(returns, confidence=0.95,
                    n_simulations=10000):
    """
    Simulates 10,000 possible next-day returns using
    historical mean and volatility.
    Takes the worst percentile of simulated returns.
    More robust than parametric — handles fat tails better.
    """
    if returns is None or len(returns) < 30:
        return None

    mu      = float(returns.mean())
    sigma   = float(returns.std())

    np.random.seed(42)
    simulated = np.random.normal(mu, sigma, n_simulations)
    var       = abs(float(np.percentile(
        simulated, (1 - confidence) * 100
    )))
    return var

# ── Single stock VaR ──────────────────────────────────────
def calculate_stock_var(ticker, position_value,
                        confidence=0.95, lookback=252):
    """
    Calculates VaR for a single stock position.

    Returns dict with:
      historical_var  : worst actual day in past year
      parametric_var  : normal distribution estimate
      monte_carlo_var : simulation estimate
      conservative_var: maximum of all three (most cautious)
      var_rupees      : VaR in rupees based on position value
    """
    returns = load_returns(ticker, lookback)
    if returns is None:
        return None

    h_var  = historical_var(returns,   confidence)
    p_var  = parametric_var(returns,   confidence)
    mc_var = monte_carlo_var(returns,  confidence)

    valid  = [v for v in [h_var, p_var, mc_var] if v is not None]
    if not valid:
        return None

    cons_var = max(valid)   # use most conservative estimate

    return {
        'ticker'         : ticker,
        'position_value' : round(position_value, 2),
        'confidence'     : confidence,
        'historical_var' : round(h_var  * 100, 3) if h_var  else None,
        'parametric_var' : round(p_var  * 100, 3) if p_var  else None,
        'monte_carlo_var': round(mc_var * 100, 3) if mc_var else None,
        'conservative_var': round(cons_var * 100, 3),
        'var_rupees'     : round(cons_var * position_value, 0),
        'daily_volatility': round(float(returns.std()) * 100, 3),
        'annual_volatility': round(float(returns.std()) * np.sqrt(252) * 100, 2),
        'worst_day_pct'  : round(float(returns.min()) * 100, 2),
        'best_day_pct'   : round(float(returns.max()) * 100, 2),
    }

# ── Portfolio VaR ─────────────────────────────────────────
def calculate_portfolio_var(positions, confidence=0.95,
                            lookback=252):
    """
    Calculates VaR for the entire portfolio.

    positions: dict of {ticker: position_value_in_rupees}

    Two approaches:
      1. Individual VaR (simple sum — conservative)
      2. Diversified VaR (accounts for correlation — realistic)
    """
    if not positions:
        return None

    # Get returns for all tickers
    returns_dict = {}
    for ticker in positions:
        ret = load_returns(ticker, lookback)
        if ret is not None and len(ret) >= 30:
            returns_dict[ticker] = ret

    if not returns_dict:
        return None

    # Align all return series
    returns_df = pd.DataFrame(returns_dict).dropna()

    if returns_df.empty:
        return None

    # Individual VaRs
    individual_vars = {}
    total_indiv_var = 0

    for ticker, pos_val in positions.items():
        if ticker not in returns_df.columns:
            continue
        ret   = returns_df[ticker]
        h_var = abs(float(np.percentile(
            ret, (1 - confidence) * 100
        )))
        individual_vars[ticker] = {
            'var_pct'  : round(h_var * 100, 3),
            'var_inr'  : round(h_var * pos_val, 0),
            'position' : round(pos_val, 0)
        }
        total_indiv_var += h_var * pos_val

    # Diversified Portfolio VaR using correlation matrix
    weights  = np.array([positions.get(t, 0)
                          for t in returns_df.columns])
    total_w  = weights.sum()
    if total_w == 0:
        return None

    weights  = weights / total_w
    cov_mat  = returns_df.cov().values
    port_var_pct = float(np.sqrt(
        weights @ cov_mat @ weights
    )) * 1.645  # 95% confidence z-score

    total_value     = sum(positions.values())
    div_var_rupees  = port_var_pct * total_value

    # Diversification benefit
    div_benefit = max(0, total_indiv_var - div_var_rupees)

    # Monte Carlo portfolio simulation
    mu_vec  = returns_df.mean().values
    n_sims  = 10000
    np.random.seed(42)
    try:
        L       = np.linalg.cholesky(cov_mat + np.eye(len(cov_mat))*1e-8)
        z       = np.random.randn(n_sims, len(mu_vec))
        sim_ret = mu_vec + (z @ L.T)
        port_sim = (sim_ret * weights).sum(axis=1)
        mc_var_pct = abs(float(np.percentile(
            port_sim, (1 - confidence) * 100
        )))
        mc_var_rupees = mc_var_pct * total_value
    except Exception:
        mc_var_rupees = div_var_rupees

    conservative_var = max(div_var_rupees, mc_var_rupees)

    return {
        'total_portfolio_value' : round(total_value, 0),
        'confidence'            : confidence,
        'individual_var_sum'    : round(total_indiv_var, 0),
        'diversified_var_inr'   : round(div_var_rupees, 0),
        'diversified_var_pct'   : round(port_var_pct * 100, 3),
        'monte_carlo_var_inr'   : round(mc_var_rupees, 0),
        'conservative_var_inr'  : round(conservative_var, 0),
        'conservative_var_pct'  : round(conservative_var/total_value*100, 3),
        'diversification_benefit': round(div_benefit, 0),
        'individual_vars'       : individual_vars,
        'n_positions'           : len(individual_vars),
    }

# ── Expected Shortfall (CVaR) ─────────────────────────────
def calculate_cvar(returns, confidence=0.95):
    """
    Conditional VaR / Expected Shortfall.
    Average loss on the WORST days beyond the VaR threshold.
    More conservative than VaR — tells you how bad the bad days are.
    """
    if returns is None or len(returns) < 30:
        return None
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= threshold]
    if len(tail_losses) == 0:
        return None
    return abs(float(tail_losses.mean()))

# ── Stress test ───────────────────────────────────────────
def stress_test(ticker, position_value):
    """
    Tests portfolio against 4 historical crisis scenarios.
    How much would you lose if today = March 2020 (COVID crash)?
    """
    scenarios = {
        'COVID Crash (Mar 2020)'   : -0.13,
        'Budget Shock'             : -0.06,
        'Global Selloff'           : -0.08,
        'Circuit Breaker (10%)'    : -0.10,
    }
    results = {}
    for scenario, shock in scenarios.items():
        loss_inr = abs(shock) * position_value
        results[scenario] = {
            'shock_pct'  : shock * 100,
            'loss_inr'   : round(loss_inr, 0),
            'survival'   : position_value - loss_inr
        }
    return results

# ── Full VaR Report ───────────────────────────────────────
def generate_var_report(capital=None, open_trades=None):
    """
    Generates a complete VaR report for the current portfolio.
    Reads from paper_trades.json if positions not provided.
    """
    # Load capital
    if capital is None:
        if os.path.exists(CAPITAL_PATH):
            with open(CAPITAL_PATH) as f:
                cap_data = json.load(f)
                capital  = cap_data.get('capital', 100000)
        else:
            capital = 100000

    # Load open trades
    if open_trades is None:
        if os.path.exists(TRADES_PATH):
            with open(TRADES_PATH) as f:
                all_trades  = json.load(f)
                open_trades = [t for t in all_trades
                               if t.get('status') == 'OPEN']
        else:
            open_trades = []

    # Build positions dict
    positions = {}
    for t in open_trades:
        ticker = t.get('ticker', '')
        value  = float(t.get('trade_value', 0))
        if ticker and value > 0:
            positions[ticker] = positions.get(ticker, 0) + value

    report = {
        'generated_at'  : datetime.now().isoformat(),
        'capital'        : capital,
        'cash_held'      : capital,
        'open_positions' : len(positions),
        'position_details': [],
        'portfolio_var'  : None,
        'stress_tests'   : {},
    }

    # Individual position VaR
    total_invested = 0
    for ticker, pos_val in positions.items():
        stock_var = calculate_stock_var(ticker, pos_val)
        if stock_var:
            cvar_pct = calculate_cvar(
                load_returns(ticker), confidence=0.95
            )
            stock_var['cvar_pct'] = round(
                cvar_pct * 100, 3
            ) if cvar_pct else None
            stock_var['cvar_inr'] = round(
                cvar_pct * pos_val, 0
            ) if cvar_pct else None
            report['position_details'].append(stock_var)
            total_invested += pos_val

    report['total_invested'] = round(total_invested, 0)

    # Portfolio-level VaR
    if len(positions) >= 2:
        port_var = calculate_portfolio_var(positions)
        report['portfolio_var'] = port_var

    # Stress tests on total invested
    if total_invested > 0:
        report['stress_tests'] = stress_test(
            'PORTFOLIO', total_invested
        )

    return report