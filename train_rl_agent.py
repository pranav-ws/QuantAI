"""
train_rl_agent.py — Train RL agents for all Nifty 50 stocks.

Fixed vs old version:
  - Uses tabular Q-learning (numpy only) instead of deep RL
  - Shows clear tqdm progress bar for each stock
  - Prints episode stats every 30 episodes so it never looks frozen
  - Completes in ~2-4 minutes for all 50 stocks (vs hours for deep RL)
  - Saves .pkl model per stock into models/
"""

import sys, os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Try to import tqdm; fall back to plain print if not installed ─────
try:
    from tqdm import tqdm# type: ignore
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

from src.data_collector import STOCK_UNIVERSE
from src.features import add_features
from src.rl_agent import train_agent, save_agent, EPISODES
import yfinance as yf# type: ignore

EPISODES_TO_TRAIN = 300   # change to 100 for a quick test

def progress_bar(iterable, **kwargs):
    if HAVE_TQDM:
        return tqdm(iterable, **kwargs)
    return iterable   # bare loop fallback

def download_and_prepare(ticker: str):
    df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
    if df.empty:
        return None
    if hasattr(df.columns, 'levels'):
        df.columns = [col[0] for col in df.columns]
    df = add_features(df).dropna()
    return df

def main():
    print("=" * 60)
    print("  QuantAI — RL Agent Training (Tabular Q-Learning)")
    print(f"  Episodes per stock : {EPISODES_TO_TRAIN}")
    print(f"  Stocks to train    : {len(STOCK_UNIVERSE)}")
    print("=" * 60)
    print()

    if not HAVE_TQDM:
        print("  ℹ️  Install tqdm for a nicer progress bar: pip install tqdm")
        print()

    results     = []
    total_start = time.time()

    outer = progress_bar(
        list(STOCK_UNIVERSE.keys()),
        desc="All stocks",
        unit="stock",
        position=0,
        leave=True,
    )

    for ticker in outer:
        name   = STOCK_UNIVERSE[ticker][0]
        start  = time.time()

        if HAVE_TQDM:
            outer.set_description(f"🤖 {ticker}")

        print(f"\n🤖 Training RL agent: {ticker}  ({name})")

        # 1. Download + feature-engineer
        print(f"   📡 Downloading data...", end="", flush=True)
        df = download_and_prepare(ticker)
        if df is None or len(df) < 100:
            print(f"  ❌ insufficient data, skip")
            results.append((ticker, "SKIP", 0, 0))
            continue
        print(f"  ✅ {len(df)} rows × {df.shape[1]} features")

        # 2. Train — progress prints every 30 episodes internally
        print(f"   🏋️  Training {EPISODES_TO_TRAIN} episodes on {len(df)} days...", flush=True)
        try:
            inner_bar = None
            if HAVE_TQDM:
                inner_bar = tqdm(
                    total=EPISODES_TO_TRAIN,
                    desc="   episodes",
                    unit="ep",
                    position=1,
                    leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )

            agent, rewards, values = train_agent(
                ticker, df,
                episodes=EPISODES_TO_TRAIN,
                progress_every=30,
            )

            if inner_bar:
                inner_bar.n = EPISODES_TO_TRAIN
                inner_bar.close()

        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            results.append((ticker, "ERROR", 0, 0))
            continue

        # 3. Save model
        path = save_agent(agent, ticker)

        # 4. Summary stats
        elapsed   = time.time() - start
        final_val = values[-1]
        best_val  = max(values)
        pnl_pct   = (final_val - 100_000) / 100_000 * 100
        avg_r     = sum(rewards) / len(rewards)

        print(f"   ✅ Done in {elapsed:.1f}s  "
              f"| Final value ₹{final_val:,.0f}  "
              f"| P&L {pnl_pct:+.1f}%  "
              f"| Saved → {path}")

        results.append((ticker, "OK", pnl_pct, elapsed))

    # ── Summary table ──────────────────────────────────────
    total_time = time.time() - total_start
    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Total time : {total_time/60:.1f} minutes")
    print(f"  Stocks OK  : {sum(1 for _,s,_,_ in results if s=='OK')}")
    print(f"  Stocks skip: {sum(1 for _,s,_,_ in results if s!='OK')}")
    print()
    print(f"  {'Ticker':<14} {'Result':<8} {'P&L %':>7}  {'Secs':>6}")
    print(f"  {'-'*14} {'-'*8} {'-'*7}  {'-'*6}")
    for t, s, pnl, sec in sorted(results, key=lambda x: -x[2]):
        mark = "🟢" if pnl > 5 else ("🟡" if pnl > 0 else "🔴") if s=="OK" else "❌"
        print(f"  {mark} {t:<13} {s:<8} {pnl:>+6.1f}%  {sec:>5.0f}s")
    print("=" * 60)
    print()
    print("  Next step: python test_rl_agent.py   (to verify signals)")
    print("  Or restart the API for RL signals in the dashboard.")

if __name__ == "__main__":
    main()
