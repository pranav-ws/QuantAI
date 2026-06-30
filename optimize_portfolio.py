"""
optimize_portfolio.py — QuantAI Portfolio Optimisation CLI
===========================================================
Run this script once after getting your daily signals to
optimise how capital is allocated across your BUY candidates.
 
Usage:
    python optimize_portfolio.py                        # auto-selects BUY signals
    python optimize_portfolio.py --method hrp           # force HRP method
    python optimize_portfolio.py --method mvo           # Markowitz max-Sharpe
    python optimize_portfolio.py --method erc           # Equal Risk Contribution
    python optimize_portfolio.py --method all           # compare all three
    python optimize_portfolio.py --tickers RELIANCE INFY TCS   # custom ticker list
    python optimize_portfolio.py --capital 500000       # custom capital
    python optimize_portfolio.py --lookback 180         # shorter lookback (days)
    python optimize_portfolio.py --frontier             # also print efficient frontier
"""
 
import argparse
import json
import os
import sys
from datetime import date
 
import yfinance as yf # type: ignore
 
# ── Make sure we can import from src/ ────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from src.portfolio_optimizer import (
    load_prices_for_optimisation,
    compute_log_returns,
    markowitz_weights,
    hrp_weights,
    erc_weights,
    compare_all,
    efficient_frontier,
    weights_to_allocation,
)
from src.data_collector import STOCK_UNIVERSE
 
 
# ══════════════════════════════════════════════════════════
#  Helper: fetch BUY signals from the ensemble
# ══════════════════════════════════════════════════════════
 
def get_buy_candidates(min_confidence: float = 0.58) -> dict:
    """
    Run the ensemble model across all Nifty 50 stocks and return
    tickers with a BUY signal above the confidence threshold.
 
    Returns {ticker: confidence} for each BUY candidate.
    """
    import yfinance as yf # type: ignore
    from src.features import add_features
    from src.ensemble_model import get_ensemble_confidence
 
    candidates = {}
    print(f"\n  Scanning {len(STOCK_UNIVERSE)} stocks for BUY signals…")
 
    for ticker in STOCK_UNIVERSE:
        try:
            df = yf.download(ticker, period='120d', progress=False, auto_adjust=True)
            if df.empty or len(df) < 30:
                continue
            if hasattr(df.columns, 'levels'):
                df.columns = [col[0] for col in df.columns]
 
            df = add_features(df)
            conf, _, models_used = get_ensemble_confidence(ticker, df)
 
            if conf is not None and conf >= min_confidence:
                candidates[ticker] = round(conf, 4)
                print(f"  🟢 {ticker:<22} confidence: {conf:.1%}  models: {models_used}")
 
        except Exception:
            pass
 
    return candidates
 
 
def get_live_prices(tickers: list) -> dict:
    """Fetch latest close price for each ticker via yfinance."""
    import math
    prices = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period='5d', progress=False, auto_adjust=True)
            if df.empty:
                continue
            if hasattr(df.columns, 'levels'):
                df.columns = [col[0] for col in df.columns]
            # dropna() removes NaN rows before reading last close
            close_series = df['Close'].dropna()
            if close_series.empty:
                print(f"  ⚠️  No valid close price for {ticker} — skipping")
                continue
            price = float(close_series.iloc[-1])
            if math.isfinite(price) and price > 0:
                prices[ticker] = price
            else:
                print(f"  ⚠️  Invalid price ({price}) for {ticker} — skipping")
        except Exception as e:
            print(f"  ⚠️  Could not fetch price for {ticker}: {e}")
    return prices
 
 
# ══════════════════════════════════════════════════════════
#  Pretty-print helpers
# ══════════════════════════════════════════════════════════
 
def print_header(title: str):
    print(f"\n{'═' * 58}")
    print(f"  {title}")
    print(f"{'═' * 58}")
 
 
def print_weights_table(result: dict, allocation: dict = None):
    """Print a formatted weights table with optional share counts."""
    weights = result['weights']
    print(f"\n  Method    : {result['method']}")
    print(f"  Return    : {result.get('expected_annual_return', '?')}% p.a.")
    print(f"  Volatility: {result.get('annual_volatility', '?')}% p.a.")
    print(f"  Sharpe    : {result.get('sharpe_ratio', '?')}")
    if 'cluster_order' in result:
        print(f"  Clusters  : {' → '.join(result['cluster_order'])}")
    if 'risk_contributions_pct' in result:
        rc = result['risk_contributions_pct']
        print(f"  Risk contributions: ", end='')
        print('  '.join(f"{t.replace('.NS','')}: {v:.1f}%" for t, v in rc.items()))
 
    # Determine which tickers have actual share positions
    positioned = set()
    if allocation:
        positioned = {t for t in allocation if t != '_summary' and allocation[t]['shares'] > 0}
 
    print(f"\n  {'TICKER':<20} {'OPT.WEIGHT':>10}", end='')
    if allocation:
        print(f"  {'ACT.WEIGHT':>10}  {'SHARES':>7}  {'VALUE (₹)':>12}  {'STOP LOSS':>10}", end='')
    print()
    print(f"  {'─' * 72}")
 
    sorted_tickers = sorted(weights.items(), key=lambda x: -x[1])
    for ticker, w in sorted_tickers:
        if w < 0.001:
            continue
        display = ticker.replace('.NS', '')
 
        # Only show allocation columns for tickers with actual shares
        if allocation and ticker in positioned:
            a = allocation[ticker]
            actual_w = a['target_weight_pct']
            print(
                f"  {display:<20} {w * 100:>9.1f}%"
                f"  {actual_w:>9.1f}%"
                f"  {a['shares']:>7}"
                f"  ₹{a['value']:>11,.0f}"
                f"  ₹{a['stop_loss']:>9.1f}"
            )
        else:
            # Show weight-only row for tickers that were skipped
            skip_note = " (too expensive)" if allocation else ""
            print(f"  {display:<20} {w * 100:>9.1f}%{skip_note}")
 
    if allocation and '_summary' in allocation:
        s = allocation['_summary']
        print(f"\n  {'─' * 72}")
        print(f"  Capital        : ₹{s['capital']:>10,.0f}")
        print(f"  Invested       : ₹{s['total_invested']:>10,.0f}  ({s['utilisation_pct']}%)")
        print(f"  Cash remaining : ₹{s['cash_remaining']:>10,.0f}")
        print(f"  Positions      : {s['n_positions']}")
        if s.get('skipped_tickers'):
            print(f"  Skipped        : {', '.join(s['skipped_tickers'])}  (price > budget slice)")
 
 
 
def print_comparison(results: dict):
    """Side-by-side comparison of all optimisation methods."""
    methods = [k for k in results if not results[k].get('error')]
    if not methods:
        print("  ❌ All optimisers failed.")
        return
 
    print(f"\n  {'METHOD':<30} {'RETURN':>8} {'VOL':>8} {'SHARPE':>8}")
    print(f"  {'─' * 58}")
    for key in methods:
        r = results[key]
        print(
            f"  {r['method']:<30} "
            f"{r.get('expected_annual_return', '?'):>7}% "
            f"{r.get('annual_volatility', '?'):>7}% "
            f"{r.get('sharpe_ratio', '?'):>8}"
        )
 
    # Best by Sharpe
    best_key = max(methods, key=lambda k: results[k].get('sharpe_ratio', 0))
    print(f"\n  🏆 Best Sharpe: {results[best_key]['method']}")
 
 
# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════
 
def main():
    parser = argparse.ArgumentParser(
        description="QuantAI Portfolio Optimiser — allocates capital across BUY signals"
    )
    parser.add_argument(
        '--method', default='hrp',
        choices=['mvo', 'hrp', 'erc', 'all'],
        help='Optimisation method (default: hrp)'
    )
    parser.add_argument(
        '--tickers', nargs='+', default=None,
        help='Custom ticker list (e.g. RELIANCE INFY TCS). Overrides auto BUY scan.'
    )
    parser.add_argument(
        '--capital', type=float, default=100_000.0,
        help='Total capital in ₹ (default: 100000)'
    )
    parser.add_argument(
        '--lookback', type=int, default=252,
        help='Days of history used for covariance estimation (default: 252)'
    )
    parser.add_argument(
        '--min-confidence', type=float, default=0.58,
        help='Minimum ensemble confidence to include a stock (default: 0.58)'
    )
    parser.add_argument(
        '--frontier', action='store_true',
        help='Also compute and display the efficient frontier (MVO only)'
    )
    parser.add_argument(
        '--save', type=str, default=None,
        help='Save results to a JSON file (e.g. --save data/optimised_portfolio.json)'
    )
    args = parser.parse_args()
 
    print_header(f"QuantAI Portfolio Optimiser — {date.today()}")
 
    # ── 1. Determine ticker universe ──────────────────────
    if args.tickers:
        tickers = [
            t.upper() if t.upper().endswith('.NS') else t.upper() + '.NS'
            for t in args.tickers
        ]
        confidence_scores = {}
        print(f"\n  Using custom ticker list: {[t.replace('.NS','') for t in tickers]}")
    else:
        confidence_scores = get_buy_candidates(args.min_confidence)
        tickers           = list(confidence_scores.keys())
 
    if len(tickers) < 2:
        print(
            "\n  ❌ Need at least 2 tickers to optimise a portfolio.\n"
            "     Try lowering --min-confidence or passing --tickers manually."
        )
        sys.exit(1)
 
    print(f"\n  Optimising across {len(tickers)} stocks: "
          f"{', '.join(t.replace('.NS','') for t in tickers)}")
    print(f"  Capital: ₹{args.capital:,.0f}  |  Lookback: {args.lookback} days")
 
    # ── 2. Load price history ─────────────────────────────
    print(f"\n  Loading price history from database…")
    try:
        prices  = load_prices_for_optimisation(tickers, lookback_days=args.lookback)
        returns = compute_log_returns(prices)
    except Exception as exc:
        print(f"\n  ❌ Failed to load prices: {exc}")
        sys.exit(1)
 
    print(f"  ✅ {len(returns)} trading days × {len(returns.columns)} stocks loaded")
    print(f"  Period: {returns.index[0].date()} → {returns.index[-1].date()}")
 
    # ── 3. Fetch live prices for allocation ───────────────
    print(f"\n  Fetching live prices…")
    live_prices = get_live_prices(tickers)
 
    # ── 4. Run optimiser(s) ───────────────────────────────
    print_header("Optimisation Results")
 
    all_results = {}
 
    if args.method == 'all':
        results = compare_all(returns)
        all_results.update(results)
        print_comparison(results)
        for key, result in results.items():
            if not result.get('error'):
                w     = result['weights']
                alloc = weights_to_allocation(w, live_prices, args.capital)
                print_weights_table(result, alloc)
                all_results[key]['allocation'] = alloc
 
    else:
        if args.method == 'mvo':
            result = markowitz_weights(
                returns,
                objective='max_sharpe',
                confidence_scores=confidence_scores if confidence_scores else None,
            )
        elif args.method == 'hrp':
            result = hrp_weights(returns)
        else:  # erc
            result = erc_weights(returns)
 
        alloc = weights_to_allocation(result['weights'], live_prices, args.capital)
        print_weights_table(result, alloc)
        all_results[args.method] = {**result, 'allocation': alloc}
 
    # ── 5. Efficient frontier (optional) ─────────────────
    if args.frontier:
        print_header("Efficient Frontier (MVO)")
        print(f"  {'RETURN':>8}  {'VOLATILITY':>11}  {'SHARPE':>8}")
        print(f"  {'─' * 35}")
        frontier_pts = efficient_frontier(returns)
        for pt in frontier_pts[::4]:   # print every 4th point for brevity
            print(f"  {pt['return']:>7.1f}%  {pt['volatility']:>10.1f}%  {pt['sharpe']:>8.3f}")
        all_results['frontier'] = frontier_pts
 
    # ── 6. Save results ───────────────────────────────────
    if args.save:
        os.makedirs(os.path.dirname(args.save) if os.path.dirname(args.save) else '.', exist_ok=True)
        with open(args.save, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  💾 Results saved → {args.save}")
 
    print(f"\n{'═' * 58}\n")
    print("  Next steps:")
    print("  1. Review the allocation table above.")
    print("  2. Enter the trades in your broker / paper trading engine.")
    print("  3. Set stop-losses at the levels shown.")
    print("  4. Re-run this script after market close for rebalancing signals.")
    print()
 
 
if __name__ == '__main__':
    main()
 