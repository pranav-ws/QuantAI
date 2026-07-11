"""
test_ensemble.py

Demonstrates the full ensemble (ML + rule-based) on 5 stocks.
Shows individual model confidences, rule-based strategy votes,
and the final weighted ensemble confidence side by side.
"""
import yfinance as yf# type: ignore
from src.features import add_features
from src.ensemble_model import get_ensemble_signal_full, get_model_agreement

TEST_TICKERS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']

RB_SHORT = {
    'MeanReversion'       : 'MR',
    'MomentumBreakout'    : 'MB',
    'MACD_RSI_Confluence' : 'MRC',
}

print("\n" + "="*80)
print("  QuantAI — Full Ensemble Demo  (ML + Rule-Based Strategies)")
print("="*80)
print(f"\n  {'Ticker':<14} {'RF':>6} {'XGB':>6} {'SeqNN':>6} │ "
      f"{'MR':>5} {'MB':>5} {'MRC':>5} │ "
      f"{'Ensemble':>9} {'Agreement':>10} {'Signal':>8}")
print(f"  {'─'*76}")

for ticker in TEST_TICKERS:
    try:
        df = yf.download(ticker, period="120d", progress=False, auto_adjust=True)
        if df.empty: continue
        if hasattr(df.columns, 'levels'):
            df.columns = [col[0] for col in df.columns]
        df = add_features(df)
        if df.empty: continue

        ens_conf, individual, models_used, rb_details, ml_count, rb_count = \
            get_ensemble_signal_full(ticker, df)
        if ens_conf is None: continue

        agreement = get_model_agreement(individual)
        signal    = '🟢 BUY' if ens_conf >= 0.58 else '🔴 SKIP'
        short     = ticker.replace('.NS', '')

        rf_s  = f"{individual.get('RF',    0):.0%}" if 'RF'      in individual else ' N/A'
        xgb_s = f"{individual.get('XGBoost',0):.0%}" if 'XGBoost' in individual else ' N/A'
        seq_s = f"{individual.get('SeqNN', individual.get('LSTM', 0)):.0%}" \
                if ('SeqNN' in individual or 'LSTM' in individual) else ' N/A'

        mr_s  = f"{rb_details['MeanReversion']['confidence']:.0%}"        if 'MeanReversion'       in rb_details else '  — '
        mb_s  = f"{rb_details['MomentumBreakout']['confidence']:.0%}"     if 'MomentumBreakout'    in rb_details else '  — '
        mrc_s = f"{rb_details['MACD_RSI_Confluence']['confidence']:.0%}"  if 'MACD_RSI_Confluence' in rb_details else '  — '

        print(f"  {short:<14} {rf_s:>6} {xgb_s:>6} {seq_s:>6} │ "
              f"{mr_s:>5} {mb_s:>5} {mrc_s:>5} │ "
              f"{ens_conf:>8.1%} {agreement:>10} {signal:>8}")

        # Detail line: voters + rule-based reasons
        voter_str = f"ML({ml_count}) + Rule-based({rb_count})" if rb_count else f"ML({ml_count}) only"
        print(f"  {'':14} Voters: {voter_str}")
        for name, d in rb_details.items():
            print(f"  {'':14}   {RB_SHORT.get(name,name)}: {d['reason']}")
        if agreement == 'strong':
            print(f"  {'':14} ✅ Strong agreement — high conviction signal")
        elif agreement == 'weak':
            print(f"  {'':14} ⚠️  Weak agreement — treat with caution")
        print()

    except Exception as e:
        print(f"  {ticker}: Error — {e}\n")

print("="*80)
print("\n  Column guide:")
print("  RF / XGB / SeqNN  — ML model P(UP) confidence (trained, per-ticker)")
print("  MR  (—)           — Mean Reversion: fired if oversold bounce setup detected")
print("  MB  (—)           — Momentum Breakout: fired if 52-week high + volume surge")
print("  MRC (—)           — MACD+RSI Confluence: fired if daily cross + weekly trend agree")
print("  (—) means the strategy is on HOLD today and did not enter the vote")
print("  Ensemble threshold: 58%  |  Rule-based weights: MR=0.6, MB=0.6, MRC=0.7")
print("="*80 + "\n")