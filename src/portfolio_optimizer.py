import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
 
warnings.filterwarnings('ignore')
 
RISK_FREE_RATE  = 0.065    # India 10-yr G-sec yield ~6.5% p.a.
TRADING_DAYS    = 252
DB_PATH         = os.path.join('data', 'quantai.db')
 
 
# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — Data helpers
# ══════════════════════════════════════════════════════════════════════
 
def load_prices_for_optimisation(
    tickers: list,
    lookback_days: int = 252,
) -> pd.DataFrame:
    """
    Load closing prices for multiple tickers from the QuantAI SQLite DB.
 
    Returns a wide DataFrame: index = date (DatetimeIndex),
                               columns = tickers,
                               values  = adjusted close price.
    Forward-fills gaps (weekends / NSE holidays) then drops
    rows where ANY ticker is still NaN (e.g. newly listed stocks).
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run pipeline.py first to build the local data store."
        )
 
    frames = {}
    conn   = sqlite3.connect(DB_PATH)
    for ticker in tickers:
        df = pd.read_sql_query(
            "SELECT date, close FROM prices "
            "WHERE ticker=? ORDER BY date DESC LIMIT ?",
            conn, params=(ticker, lookback_days)
        )
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            frames[ticker] = df.set_index('date').sort_index()['close']
    conn.close()
 
    if not frames:
        raise ValueError(
            "No price data found for any of the requested tickers. "
            "Have you run train_all_models.py yet?"
        )
 
    prices = pd.DataFrame(frames)
    prices = prices.ffill().dropna()            # forward-fill then drop incomplete rows
 
    if len(prices) < 60:
        raise ValueError(
            f"Insufficient history: need ≥ 60 trading days, got {len(prices)}. "
            "Try a smaller ticker set or re-run pipeline.py to update prices."
        )
 
    return prices
 
 
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns — more stationary than simple returns for covariance."""
    return np.log(prices / prices.shift(1)).dropna()
 
 
def shrunk_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf analytical shrinkage estimator (annualised).
 
    Why shrinkage?  The sample covariance matrix is ill-conditioned when
    T/N is not huge (e.g. 252 days / 10 stocks ≈ 25 — okay, but at 252/50
    it degrades fast).  L-W pulls the matrix toward a scaled identity,
    dramatically reducing estimation error without sacrificing structure.
    sklearn's implementation is the textbook O(T·N²) algorithm, no extra
    dependencies.
    """
    lw = LedoitWolf()
    lw.fit(returns.values)
    return lw.covariance_ * TRADING_DAYS      # annualise
 
 
def annualised_returns(returns: pd.DataFrame) -> np.ndarray:
    """Simple geometric annualised expected returns per asset."""
    return returns.mean().values * TRADING_DAYS
 
 
def portfolio_stats(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> tuple:
    """Return (annual_return, annual_vol, sharpe) for a weight vector."""
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sr  = (ret - RISK_FREE_RATE) / (vol + 1e-10)
    return ret, vol, sr
 
 
# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — Markowitz Mean-Variance Optimisation
# ══════════════════════════════════════════════════════════════════════
 
def markowitz_weights(
    returns: pd.DataFrame,
    objective: str     = 'max_sharpe',   # 'max_sharpe' | 'min_variance' | 'target_return'
    target_return: float = None,          # used only for 'target_return' objective
    weight_bounds: tuple = (0.0, 0.40),  # (min, max) allocation per stock (max 40%)
    confidence_scores: dict = None,       # optional: boost mu by ML confidence
) -> dict:
    """
    Classic Markowitz optimisation via sequential quadratic programming.
 
    Maximise Sharpe  → finds the tangency portfolio on the efficient frontier.
    Minimise variance→ finds the leftmost (lowest-risk) portfolio.
    Target return    → finds the minimum-risk portfolio for a given return.
 
    confidence_scores (optional):
        Dict mapping ticker → ensemble confidence (0-1). When provided,
        expected returns are nudged upward for high-confidence tickers:
            mu_adjusted = mu_raw * (1 + 0.5 * (conf - 0.5))
        This links the ML signal to the optimiser without overriding it.
 
    weight_bounds: (min_w, max_w) per asset.
        (0, 0.40) = fully long, max 40% in any one stock.
        (0, 1.00) = unconstrained long-only.
        (-0.20, 0.40) = allow mild shorting (requires margin).
 
    Returns dict with weights, stats, and optimisation metadata.
    """
    tickers = list(returns.columns)
    n       = len(tickers)
 
    mu  = annualised_returns(returns)
    cov = shrunk_covariance(returns)
 
    # Optional: blend ML confidence into expected return estimate
    if confidence_scores:
        for i, ticker in enumerate(tickers):
            conf = confidence_scores.get(ticker, 0.5)
            mu[i] *= (1.0 + 0.5 * (conf - 0.5))   # ±25% adjustment at full conf
 
    # ── Objective functions ───────────────────────────────
    def neg_sharpe(w):
        ret, vol, _ = portfolio_stats(w, mu, cov)
        return -(ret - RISK_FREE_RATE) / (vol + 1e-10)
 
    def port_variance(w):
        return float(w @ cov @ w)
 
    obj_fn = neg_sharpe if objective in ('max_sharpe', 'target_return') else port_variance
 
    # ── Constraints ───────────────────────────────────────
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    if objective == 'target_return' and target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w, r=target_return: w @ mu - r
        })
 
    bounds = [weight_bounds] * n
    w0     = np.ones(n) / n   # equal-weight warm start
 
    result = minimize(
        obj_fn, w0,
        method      = 'SLSQP',
        bounds      = bounds,
        constraints = constraints,
        options     = {'ftol': 1e-9, 'maxiter': 2000}
    )
 
    if result.success:
        w = np.clip(result.x, 0, 1)
        w /= w.sum()
    else:
        # Fallback: inverse-volatility portfolio (always feasible)
        vols = np.sqrt(np.diag(cov))
        w    = (1.0 / vols) / (1.0 / vols).sum()
        print(f"  ⚠️  MVO optimisation did not converge ({result.message}). "
              f"Falling back to inverse-volatility weighting.")
 
    ret, vol, sr = portfolio_stats(w, mu, cov)
 
    return {
        'method'                 : 'Markowitz MVO',
        'objective'              : objective,
        'converged'              : bool(result.success),
        'weights'                : {t: round(float(v), 4) for t, v in zip(tickers, w)},
        'expected_annual_return' : round(ret * 100, 2),
        'annual_volatility'      : round(vol * 100, 2),
        'sharpe_ratio'           : round(sr, 3),
    }
 
 
def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int  = 40,
    weight_bounds: tuple = (0.0, 0.40),
) -> list:
    """
    Sweep the efficient frontier by solving minimum-variance at each
    target return level between the min-variance and max-return portfolios.
 
    Returns a list of dicts: [{return, volatility, sharpe, weights}, …]
    suited for plotting a risk-return curve on the dashboard.
    """
    tickers = list(returns.columns)
    n       = len(tickers)
    mu      = annualised_returns(returns)
    cov     = shrunk_covariance(returns)
 
    # ── Find min-variance portfolio return (frontier starts here) ─
    min_var_result = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n,
        method      = 'SLSQP',
        bounds      = [weight_bounds] * n,
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}],
        options     = {'ftol': 1e-9, 'maxiter': 1000}
    )
    min_ret = float(min_var_result.x @ mu) if min_var_result.success else mu.min()
    max_ret = mu.max() * 0.95   # 95% of the max single-stock return
 
    target_rets = np.linspace(min_ret, max_ret, n_points)
    frontier    = []
 
    for tr in target_rets:
        cons = [
            {'type': 'eq', 'fun': lambda w: w.sum() - 1.0},
            {'type': 'eq', 'fun': lambda w, r=tr: w @ mu - r},
        ]
        res = minimize(
            lambda w: w @ cov @ w,
            np.ones(n) / n,
            method      = 'SLSQP',
            bounds      = [weight_bounds] * n,
            constraints = cons,
            options     = {'ftol': 1e-9, 'maxiter': 500}
        )
        if res.success:
            w   = np.clip(res.x, 0, 1); w /= w.sum()
            ret, vol, sr = portfolio_stats(w, mu, cov)
            frontier.append({
                'return'     : round(ret * 100, 2),
                'volatility' : round(vol * 100, 2),
                'sharpe'     : round(sr, 3),
                'weights'    : {t: round(float(v), 4) for t, v in zip(tickers, w)},
            })
 
    return frontier
 
 
# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — Hierarchical Risk Parity (HRP)
# ══════════════════════════════════════════════════════════════════════
 
def _corr_distance(corr: np.ndarray) -> np.ndarray:
    """
    Distance metric for correlation matrices (López de Prado, 2016, eq. 4).
    d(i,j) = √( (1 - ρᵢⱼ) / 2 )  ∈ [0, 1]
    Assets with ρ = +1 → distance 0 (identical).
    Assets with ρ = -1 → distance 1 (perfectly inverse).
    """
    d = np.sqrt(np.clip((1 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return d
 
 
def _quasi_diagonalise(link: np.ndarray, n_items: int) -> list:
    """
    Reorder items so that similar assets are adjacent in the covariance matrix.
    This 'quasi-diagonalisation' is Step 2 of HRP and makes the covariance
    matrix block-diagonal by cluster — enabling recursive bisection in Step 3.
 
    Adapted from López de Prado (2016), Appendix Listing 3.
    """
    link      = link.astype(int)
    sort_ix   = pd.Series([link[-1, 0], link[-1, 1]])
    n_total   = link[-1, 3]   # total items including merged clusters
 
    while sort_ix.max() >= n_items:
        # Space out the index to make room for expansions
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        clusters      = sort_ix[sort_ix >= n_items]
        i             = clusters.index
        j             = clusters.values - n_items
 
        # Replace each cluster reference with its two children
        sort_ix[i]    = link[j, 0]
        children_r    = pd.Series(link[j, 1], index=i + 1)
        sort_ix       = pd.concat([sort_ix, children_r]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
 
    return sort_ix.tolist()
 
 
def _cluster_variance(cov: np.ndarray, indices: list) -> float:
    """
    Variance of the inverse-variance portfolio within a cluster.
    This is the cluster's effective contribution to total portfolio risk.
    """
    sub_cov = cov[np.ix_(indices, indices)]
    inv_var = 1.0 / np.diag(sub_cov)
    w_ivp   = inv_var / inv_var.sum()
    return float(w_ivp @ sub_cov @ w_ivp)
 
 
def _recursive_bisection(cov: np.ndarray, sort_ix: list) -> np.ndarray:
    """
    Step 3 of HRP: allocate weights via recursive bisection.
 
    At each split the capital is divided between the two sub-clusters
    in inverse proportion to their variance:
        left_allocation  = right_cluster_var / (left_var + right_var)
        right_allocation = 1 - left_allocation
 
    This naturally gives more weight to the less volatile cluster.
    """
    w      = pd.Series(1.0, index=range(len(sort_ix)))
    c_items = [list(range(len(sort_ix)))]   # start with all items in one group
 
    while c_items:
        # Split every sublist in half — process left/right pairs
        c_items = [
            sub[j:k]
            for sub in c_items
            for j, k in ((0, len(sub) // 2), (len(sub) // 2, len(sub)))
            if len(sub) > 1
        ]
 
        for i in range(0, len(c_items), 2):
            left  = c_items[i]
            right = c_items[i + 1]
 
            # Map local indices back to the original (sorted) covariance
            left_orig  = [sort_ix[x] for x in left]
            right_orig = [sort_ix[x] for x in right]
 
            var_l = _cluster_variance(cov, left_orig)
            var_r = _cluster_variance(cov, right_orig)
 
            alpha = 1.0 - var_l / (var_l + var_r + 1e-12)
            w[left]  *= alpha
            w[right] *= 1.0 - alpha
 
    return w.values
 
 
def hrp_weights(returns: pd.DataFrame) -> dict:
    """
    Hierarchical Risk Parity allocation.
 
    Three steps:
      1. Tree clustering — group assets by correlation distance using Ward linkage
      2. Quasi-diagonalisation — reorder the covariance matrix by cluster
      3. Recursive bisection — allocate capital top-down within the hierarchy
 
    Key advantage over MVO: HRP never inverts the covariance matrix, so it
    is numerically stable and does NOT require a return estimate (which is
    the main source of Markowitz's out-of-sample underperformance).
    """
    tickers = list(returns.columns)
    n       = len(tickers)
 
    cov  = shrunk_covariance(returns)
    corr = np.corrcoef(returns.values.T)
    mu   = annualised_returns(returns)
 
    # ── Step 1: hierarchical clustering ──────────────────
    dist     = _corr_distance(corr)
    dist_sq  = squareform(dist, checks=False)
    link     = linkage(dist_sq, method='ward')
 
    # ── Step 2: quasi-diagonalise ─────────────────────────
    sort_ix  = _quasi_diagonalise(link, n)
    sort_ix  = [int(x) for x in sort_ix]
 
    # ── Step 3: recursive bisection ───────────────────────
    w_sorted = _recursive_bisection(cov, sort_ix)
 
    # Map sorted weights back to original ticker order
    w_full            = np.zeros(n)
    for local_i, orig_i in enumerate(sort_ix):
        w_full[orig_i] = w_sorted[local_i]
 
    w_full = np.clip(w_full, 0.0, 1.0)
    w_full /= w_full.sum()
 
    ret, vol, sr = portfolio_stats(w_full, mu, cov)
 
    return {
        'method'                 : 'HRP (Hierarchical Risk Parity)',
        'weights'                : {t: round(float(v), 4) for t, v in zip(tickers, w_full)},
        'cluster_order'          : [tickers[i] for i in sort_ix],
        'expected_annual_return' : round(ret * 100, 2),
        'annual_volatility'      : round(vol * 100, 2),
        'sharpe_ratio'           : round(sr, 3),
    }
 
 
# ══════════════════════════════════════════════════════════════════════
#  SECTION 4 — Equal Risk Contribution (ERC)
# ══════════════════════════════════════════════════════════════════════
 
def erc_weights(returns: pd.DataFrame) -> dict:
    """
    Equal Risk Contribution (Risk Parity) portfolio.
 
    Each asset i contributes exactly 1/N of total portfolio variance:
        RC_i = w_i · (Σ·w)_i  =  σ²_p / N  ∀i
 
    Solved numerically. More balanced than MVO (which concentrates risk),
    more constrained than HRP (which can still over-weight low-vol clusters).
    """
    tickers = list(returns.columns)
    n       = len(tickers)
 
    cov = shrunk_covariance(returns)
    mu  = annualised_returns(returns)
 
    def risk_contribution(w):
        """Per-asset risk contribution vector."""
        port_var = float(w @ cov @ w)
        mrc      = cov @ w              # marginal risk contrib (∂σ_p/∂w_i)
        rc       = w * mrc              # total risk contribution per asset
        return rc, port_var
 
    def objective(w):
        """Sum of squared deviations from equal risk contribution."""
        rc, port_var = risk_contribution(w)
        target       = np.full(n, port_var / n)
        return float(np.sum((rc - target) ** 2))
 
    def gradient(w):
        """Analytical gradient — speeds up convergence significantly."""
        rc, port_var = risk_contribution(w)
        target       = np.full(n, port_var / n)
        diff         = rc - target
        # ∂RC_i/∂w_j = cov[i,j]·w_i + (Σw)_i·δ_{ij}
        # ∂σ²_p/∂w_j = 2·(Σw)_j
        grad = np.zeros(n)
        for i in range(n):
            d_rc_i = cov[i] * w[i] + (cov @ w)[i] * np.eye(n)[i]
            d_var  = 2.0 * (cov @ w)
            grad  += 2.0 * diff[i] * (d_rc_i - d_var / n)
        return grad
 
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    bounds      = [(0.001, 0.50)] * n    # slight lb to prevent zeros
    w0          = np.ones(n) / n
 
    result = minimize(
        objective, w0,
        jac         = gradient,
        method      = 'SLSQP',
        bounds      = bounds,
        constraints = constraints,
        options     = {'ftol': 1e-12, 'maxiter': 5000}
    )
 
    w = np.clip(result.x if result.success else w0, 0.0, 1.0)
    w /= w.sum()
 
    rc, port_var     = risk_contribution(w)
    rc_pct           = (rc / (port_var + 1e-12)) * 100
    ret, vol, sr     = portfolio_stats(w, mu, cov)
 
    return {
        'method'                  : 'ERC (Equal Risk Contribution)',
        'converged'               : bool(result.success),
        'weights'                 : {t: round(float(v), 4) for t, v in zip(tickers, w)},
        'risk_contributions_pct'  : {t: round(float(v), 2) for t, v in zip(tickers, rc_pct)},
        'expected_annual_return'  : round(ret * 100, 2),
        'annual_volatility'       : round(vol * 100, 2),
        'sharpe_ratio'            : round(sr, 3),
    }
 
 
# ══════════════════════════════════════════════════════════════════════
#  SECTION 5 — Compare all three methods
# ══════════════════════════════════════════════════════════════════════
 
def compare_all(returns: pd.DataFrame) -> dict:
    """
    Run MVO (max Sharpe), MVO (min variance), HRP, and ERC on the same
    returns DataFrame and return a side-by-side comparison dict.
    """
    results = {}
    methods = [
        ('mvo_max_sharpe',    lambda r: markowitz_weights(r, objective='max_sharpe')),
        ('mvo_min_variance',  lambda r: markowitz_weights(r, objective='min_variance')),
        ('hrp',               hrp_weights),
        ('erc',               erc_weights),
    ]
    for key, fn in methods:
        try:
            results[key] = fn(returns)
        except Exception as exc:
            results[key] = {'error': str(exc)}
 
    return results
 
 
# ══════════════════════════════════════════════════════════════════════
#  SECTION 6 — Convert weights → actual trade sizes
# ══════════════════════════════════════════════════════════════════════
 
def weights_to_allocation(
    weights: dict,
    current_prices: dict,
    capital: float,
) -> dict:
    """
    Convert portfolio weights to share counts given live prices and capital.
 
    Two-pass approach:
      Pass 1 — identify which tickers can actually afford ≥1 share.
               Tickers that can't (price > allocated budget) are dropped and
               their weight is freed up.
      Pass 2 — re-normalise weights across affordable tickers only, then
               compute final share counts. This maximises capital utilisation
               instead of leaving money stranded in 0-share slots.
 
    Args:
        weights        : {ticker: weight_float}  e.g. {'RELIANCE.NS': 0.25}
        current_prices : {ticker: price_float}   live close prices
        capital        : total available capital in ₹
 
    Returns dict:
        {
            ticker: {target_weight_pct, price, shares, value, stop_loss},
            ...,
            '_summary': {capital, total_invested, cash_remaining,
                         utilisation_pct, n_positions, skipped_tickers}
        }
    """
    stop_loss_pct = 0.03   # mirror RiskManager.stop_loss_pct
 
    # ── Validate inputs ────────────────────────────────────────────
    clean_weights = {}
    for ticker, w in weights.items():
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        if not (w >= 0.005):          # drop negligible or NaN weights
            continue
        price = current_prices.get(ticker, 0.0)
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        if not (price > 0):
            print(f"  ⚠️  {ticker}: no valid live price — skipped")
            continue
        clean_weights[ticker] = (w, price)
 
    if not clean_weights:
        print("  ❌ No valid ticker/price pairs — returning empty allocation")
        return {'_summary': {
            'capital': round(capital, 2), 'total_invested': 0.0,
            'cash_remaining': round(capital, 2), 'utilisation_pct': 0.0,
            'n_positions': 0, 'skipped_tickers': [],
        }}
 
    # ── Pass 1: filter out stocks too expensive for their budget ───
    # Normalise to a unit sum so we can compute budgets fairly
    total_w   = sum(w for w, _ in clean_weights.values())
    affordable = {}
    skipped    = []
 
    for ticker, (w, price) in clean_weights.items():
        norm_w = w / total_w
        budget = capital * norm_w
        if budget < price:
            name = ticker.replace('.NS', '')
            print(f"  ⚠️  {name}: budget ₹{budget:,.0f} < 1 share @ ₹{price:,.1f} — skipped, weight redistributed")
            skipped.append(ticker)
        else:
            affordable[ticker] = (w, price)
 
    if not affordable:
        print("  ❌ All tickers were too expensive for their budget slice.")
        return {'_summary': {
            'capital': round(capital, 2), 'total_invested': 0.0,
            'cash_remaining': round(capital, 2), 'utilisation_pct': 0.0,
            'n_positions': 0, 'skipped_tickers': skipped,
        }}
 
    # ── Pass 2: re-normalise over affordable tickers and allocate ──
    total_w2 = sum(w for w, _ in affordable.values())
    allocation     = {}
    total_invested = 0.0
 
    for ticker, (w, price) in sorted(affordable.items(), key=lambda x: -x[1][0]):
        final_w   = w / total_w2          # re-normalised weight
        budget    = capital * final_w
        shares    = int(budget // price)  # whole shares only
 
        # Safety net: re-check after re-normalisation (shouldn't happen, but guard it)
        if shares == 0:
            name = ticker.replace('.NS', '')
            print(f"  ⚠️  {name}: still 0 shares after redistribution (price ₹{price:,.1f}) — skipped")
            skipped.append(ticker)
            continue
 
        value     = shares * price
        stop_loss = price * (1 - stop_loss_pct)
        total_invested += value
 
        allocation[ticker] = {
            'target_weight_pct' : round(final_w * 100, 2),
            'price'             : round(price, 2),
            'shares'            : shares,
            'value'             : round(value, 2),
            'stop_loss'         : round(stop_loss, 2),
        }
 
    cash_remaining = capital - total_invested
    allocation['_summary'] = {
        'capital'          : round(capital, 2),
        'total_invested'   : round(total_invested, 2),
        'cash_remaining'   : round(cash_remaining, 2),
        'utilisation_pct'  : round(total_invested / capital * 100, 1),
        'n_positions'      : len(allocation),
        'skipped_tickers'  : [t.replace('.NS', '') for t in skipped],
    }
    return allocation
 